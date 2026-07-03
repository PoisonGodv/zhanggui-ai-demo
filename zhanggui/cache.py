"""掌柜军师 AI 引擎 —— 语义缓存 (方案 7.6.2-7.6.5)。

职责: 对「相同业务情境」的 Advice 做语义级缓存复用, 避免重复跑五段管线 /
重复调 LLM, 降本提速 (方案 7.6.2)。

设计要点 (忠实复刻方案 7.6.2-7.6.5):
- 缓存键不是原始文本, 而是「业务情境签名」key_text (make_key_text): 把
  SignalBundle + Routed 的关键字段拼成结构化短文本, 再 embed 成向量。这样
  「雨天+临期2天+沉睡18人」这类情境即便文字描述不同, 向量也高度相似, 能命中。
- 命中判定 (try_hit): 对同 business 且未过期的 entry, 算查询向量与缓存向量
  的余弦距离 (1 - cosine_similarity), 距离 < distance_threshold 即命中。
  distance_threshold=0.15 (方案默认, 约等于 cosine_similarity > 0.85)。
  命中 hit_count+=1, 返回缓存的 Advice + 命中 entry (供 trace/统计)。
- 写入 (put): embedding=embedder.embed(key_text), expires_at=now+ttl_hours
  (默认 24h, 与凌晨预生成周期对齐)。
- 主动失效 (invalidate_tenant): 库存/天气突变或手动刷新时, 把该 tenant 当日
  缓存 expires_at=now (立即过期, 下次查询 miss 触发重算)。
- TTL 过期: try_hit 跳过 expires_at<=now 的 entry (惰性失效, 不主动扫)。
- 存储: 进程内 dict (无 PG); 接口按有 DB 设计, 留好 PGVector 替换点
  (见 _query_entries / _store_entry 的注释, 生产替换为
  `SELECT ... FROM advice_cache WHERE business=$1 AND expires_at>now()
   ORDER BY embedding <=> $2 LIMIT 1` 的向量距离查询)。
- metrics: hit/miss/hit_rate, 供运维看缓存命中率 (方案 7.6.5 可观测)。

零第三方依赖, 仅标准库 (time / dataclasses / typing)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .embedding import Embedder, cosine_similarity
from .types import Advice, Routed, SignalBundle


@dataclass
class CacheEntry:
    """单条语义缓存记录 (方案 7.6.3)。"""
    tenant_id: int
    business: str
    key_text: str
    key_prefix: str           # 强区分字段签名 (weather/actions/days_to_expiry/holiday), 精确匹配
    embedding: List[float]
    advice: Advice
    created_at: float
    expires_at: float
    hit_count: int = 0


class SemanticCache:
    """语义缓存: 同业务情境 (向量近) + 同 business + 未过期 -> 命中复用 Advice。

    Args:
        embedder: Embedder 实例 (LocalEmbedder 零依赖 / QwenEmbedder 有 key)。
        distance_threshold: 余弦距离命中阈值, 距离 < 此值即命中 (默认 0.15)。
        ttl_hours: 缓存存活时长 (默认 24h, 对齐凌晨预生成周期)。
    """

    def __init__(
        self,
        embedder: Embedder,
        distance_threshold: float = 0.15,
        ttl_hours: int = 24,
    ):
        self._embedder = embedder
        self._distance_threshold = float(distance_threshold)
        self._ttl_seconds = float(ttl_hours) * 3600.0
        # 进程内存储: business -> list[CacheEntry]。
        # 生产: PGVector 表 advice_cache(tenant_id,business,key_text,embedding VECTOR(256),
        #        advice JSONB, created_at, expires_at, hit_count), 命中走
        #        SELECT ... WHERE business=$1 AND expires_at>now()
        #          ORDER BY embedding <=> $2 LIMIT 1;
        #        (embedding <=> $2 为余弦距离算子)。
        self._store: Dict[str, List[CacheEntry]] = {}
        # 命中/ miss 计数 (跨所有 tenant 汇总)。
        self._hits = 0
        self._misses = 0

    # ===== 缓存键构造 (方案 7.6.2 情境签名) =====

    def make_key_text(self, bundle: SignalBundle, routed: Routed) -> str:
        """把 bundle + routed 的关键字段拼成结构化短文本 (情境签名)。

        示例:
          "fruit | weather=rain | actions=clearance_coupon,rain_delivery
           | days_to_expiry=2 | sleep_30=18 | holiday=none"

        选取字段的依据 (方案 7.6.2 关键情境维度):
          - business: 业态 (决定剧本骨架)。
          - weather.condition: 天气 (雨天配送 / 高温冷饮等动作触发)。
          - actions: routed.items 的 action_type 去重排序 (决定动作组合)。
          - days_to_expiry: 库存最短临期天数 (决定 clearance 强度)。
          - sleep_30: 沉睡会员数 (决定 wake_dormant 触发与力度)。
          - holiday: 节日 (决定 festival_discount / birthday_care)。

        这些字段相同 (或近似) -> 情境等价 -> Advice 可复用。
        """
        business = (bundle.business or "").strip() or "unknown"

        weather_cond = ""
        if bundle.weather is not None:
            weather_cond = (bundle.weather.condition or "").strip()
        weather_cond = weather_cond or "normal"

        actions: List[str] = []
        if routed is not None and routed.items:
            seen = set()
            for it in routed.items:
                at = (getattr(it, "action_type", "") or "").strip()
                if at and at not in seen:
                    seen.add(at)
                    actions.append(at)
        actions_str = ",".join(sorted(actions)) if actions else "none"

        # 最短临期天数: 仅取 >=0 的临期项, 否则 none。
        expiry_vals = [
            inv.days_to_expiry
            for inv in (bundle.inventory or [])
            if inv is not None and inv.days_to_expiry is not None
            and inv.days_to_expiry >= 0
        ]
        days_to_expiry = str(min(expiry_vals)) if expiry_vals else "none"

        sleep_30 = 0
        if bundle.members is not None:
            sleep_30 = int(getattr(bundle.members, "sleep_30", 0) or 0)

        holiday_str = "none"
        if bundle.holiday is not None:
            name = (bundle.holiday.name or "").strip()
            if name and name != "无" and name.lower() != "none":
                holiday_str = name

        return (
            "{biz} | weather={w} | actions={a} | "
            "days_to_expiry={d} | sleep_30={s} | holiday={h}"
        ).format(
            biz=business,
            w=weather_cond,
            a=actions_str,
            d=days_to_expiry,
            s=sleep_30,
            h=holiday_str,
        )

    # ===== 强区分字段签名 (精确匹配, 修复单维度信号变化误命中) =====

    def make_key_prefix(self, bundle: SignalBundle, routed: Routed) -> str:
        """强区分字段签名: weather / actions / days_to_expiry / holiday 的精确拼接。

        这些字段任一变化即视为不同情境, 必须重算 (不走 embedding 相似度),
        杜绝「天气转晴仍命中雨天 rain_delivery」「临期消失仍命中 clearance」类误命中。
        embedding 相似度只对 sleep_30 等数值微调 (18 vs 19) 做细筛, 不再对强区分字段兜底。
        """
        weather_cond = ""
        if bundle.weather is not None:
            weather_cond = (bundle.weather.condition or "").strip()
        weather_cond = weather_cond or "normal"

        actions: List[str] = []
        if routed is not None and routed.items:
            seen = set()
            for it in routed.items:
                at = (getattr(it, "action_type", "") or "").strip()
                if at and at not in seen:
                    seen.add(at)
                    actions.append(at)
        actions_str = ",".join(sorted(actions)) if actions else "none"

        expiry_vals = [
            inv.days_to_expiry
            for inv in (bundle.inventory or [])
            if inv is not None and inv.days_to_expiry is not None
            and inv.days_to_expiry >= 0
        ]
        days_to_expiry = str(min(expiry_vals)) if expiry_vals else "none"

        holiday_str = "none"
        if bundle.holiday is not None:
            name = (bundle.holiday.name or "").strip()
            if name and name != "无" and name.lower() != "none":
                holiday_str = name

        return "w={w}|a={a}|d={d}|h={h}".format(
            w=weather_cond, a=actions_str, d=days_to_expiry, h=holiday_str,
        )

    # ===== 查询命中 =====

    def try_hit(
        self, tenant_id: int, business: str, key_text: str, key_prefix: str = "",
    ) -> Tuple[Optional[Advice], Optional[CacheEntry]]:
        """语义查询: 同 business + 未过期 + 强区分前缀精确相等 + 余弦距离 < 阈值 -> 命中。

        强区分前缀 (weather/actions/days_to_expiry/holiday) 必须精确相等才进 embedding
        细筛, 杜绝单维度信号变化 (如天气 rain->normal) 仍命中旧 advice 的误命中。
        embedding 相似度只对 sleep_30 等数值微调做细筛。key_prefix 为空时退化为纯向量
        匹配 (向后兼容旧调用, 但不推荐)。

        Returns:
            (advice, entry) 命中; (None, None) 未命中 (调用方应跑管线后 put)。
        """
        business = (business or "").strip()
        query_vec = self._embedder.embed(key_text)
        now = time.time()

        entries = self._store.get(business, [])
        best_entry: Optional[CacheEntry] = None
        best_dist = float("inf")

        for entry in entries:
            # 惰性过期: 已过期跳过 (不主动删, 留给 put 时顺带清理 / TTL 扫描)。
            if entry.expires_at <= now:
                continue
            # 维度不一致 (embedder 换了) 跳过, 避免错误比较。
            if len(entry.embedding) != len(query_vec):
                continue
            # 强区分前缀精确匹配: 任一强字段变化 (天气/动作/临期/节日) 直接排除,
            # 不走 embedding 相似度 (hashing-trick 对单字段变化区分度不足)。
            if key_prefix and getattr(entry, "key_prefix", "") != key_prefix:
                continue
            sim = cosine_similarity(entry.embedding, query_vec)
            dist = 1.0 - sim
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        if best_entry is not None and best_dist < self._distance_threshold:
            # 命中: hit_count 自增, 返回缓存的 Advice。
            best_entry.hit_count += 1
            self._hits += 1
            return best_entry.advice, best_entry

        self._misses += 1
        return None, None

    # ===== 写入 =====

    def put(self, tenant_id: int, business: str, key_text: str, advice: Advice,
            key_prefix: str = "") -> None:
        """写入一条缓存: embedding=embedder.embed(key_text), TTL=now+ttl。

        key_prefix 为强区分字段签名, try_hit 据此精确过滤; 为空则该 entry 不参与
        前缀过滤 (向后兼容, 但不推荐)。

        生产: INSERT INTO advice_cache(...) VALUES ($1,$2,$3,$4,$5,now(),now()+interval'24h',0)
              ON CONFLICT (tenant_id,key_text) DO UPDATE SET advice=... , expires_at=... ;
        """
        business = (business or "").strip()
        now = time.time()
        entry = CacheEntry(
            tenant_id=tenant_id,
            business=business,
            key_text=key_text,
            key_prefix=key_prefix,
            embedding=self._embedder.embed(key_text),
            advice=advice,
            created_at=now,
            expires_at=now + self._ttl_seconds,
            hit_count=0,
        )
        bucket = self._store.setdefault(business, [])
        # 简易去重: 同 tenant + 同 key_text 覆盖旧条目 (幂等写)。
        for i, e in enumerate(bucket):
            if e.tenant_id == tenant_id and e.key_text == key_text:
                bucket[i] = entry
                return
        bucket.append(entry)

    # ===== 主动失效 =====

    def invalidate_tenant(self, tenant_id: int) -> int:
        """主动失效该 tenant 当日所有缓存: expires_at=now (立即过期)。

        库存突变 / 天气转雨 / 手动刷新触发: 下次查询 miss -> 重算 -> put 覆盖。
        生产: UPDATE advice_cache SET expires_at=now() WHERE tenant_id=$1;
        Returns:
            被失效的 entry 条数 (便于 trace)。
        """
        now = time.time()
        count = 0
        for bucket in self._store.values():
            for entry in bucket:
                if entry.tenant_id == tenant_id and entry.expires_at > now:
                    entry.expires_at = now
                    count += 1
        return count

    # ===== 可观测 =====

    def metrics(self) -> dict:
        """缓存命中率: {hit, miss, hit_rate} (方案 7.6.5 可观测)。"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total > 0 else 0.0
        return {
            "hit": self._hits,
            "miss": self._misses,
            "hit_rate": round(hit_rate, 4),
        }
