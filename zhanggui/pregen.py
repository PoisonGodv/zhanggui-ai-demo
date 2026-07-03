"""凌晨预生成 (方案 7.7 Pre-generation Scheduler)。

职责: 在每日凌晨对全量 tenant 批量预跑五段管线, 把 Advice 写入语义缓存
(SemanticCache), 使白天的「今日3件事」查询走缓存命中 (毫秒级) 而非重跑管线。
失败 tenant 自动重试 (指数退避), >20% 失败标记 degraded 供运营介入。
库存/天气/手动刷新可触发单户增量重算 (trigger_incremental) 覆盖当日缓存。

设计要点 (忠实复刻方案 7.7):
- 全量批: 分批 batch_size(默认50)/批, 小并发 ThreadPoolExecutor(max_workers=4);
  单户失败重试 ≤ max_retries(默认3) 次, 退避 0.5*2^n 秒。
- 幂等: cache.put 同 key 覆盖; advice.version = date+'-v1', 同日同规则版本重算
  只覆盖不新增 (语义缓存按 key_text 寻址, 同信号同 key)。
- 降级: 失败率 > fail_threshold(默认0.2) 时 degraded=True, 调用方可据此切降级策略
  (如全户回退 MockBrain 兜底 / 走昨日缓存)。
- 不自起 cron 线程: 测试时手动调 batch_generate(); 另提供 start_cron() 用
  threading.Thread 跑循环 (demo 用, 标注「demo: 手动调用为主」)。

依赖 (注入, 不直接 import 具体类, 便于替换):
  pipeline : zhanggui.pipeline.Pipeline  (有 .generate_today_advice / .collector / .router)
  cache    : zhanggui.cache.SemanticCache (有 .make_key_text / .put)
  tenants  : list[int]  全量商户 id
  scenarios: dict[int, dict]  tenant_id -> scenario (传给 collector 与 pipeline)

零第三方依赖, 仅标准库 (time/threading/concurrent.futures)。

注: key_text 需 bundle+routed 构造, 而 pipeline.generate_today_advice 只返回 Advice
不暴露中间产物; 此处额外调一次 collector.collect + router.route 取 bundle/routed
构造 key (collect/router 为纯函数解析 scenario, 确定性可复现, 与管线内同输入同输出)。
该冗余对零依赖 demo 可忽略; 真实系统可让 pipeline 暴露 bundle/routed 复用。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple


class Scheduler:
    """凌晨预生成调度器 (方案 7.7)。

    Args:
        pipeline        : Pipeline (五段总控, 有 generate_today_advice/collector/router)
        cache           : SemanticCache (有 make_key_text/put)
        tenants         : list[int] 全量商户 id
        scenarios       : dict[int,dict] tenant_id -> scenario dict
        max_workers     : 并发线程数 (默认4, 小并发)
        batch_size      : 分批大小 (默认50/批)
        max_retries     : 单户失败重试上限 (默认3, 退避 0.5*2^n)
        fail_threshold  : 失败率降级阈值 (默认0.2 => >20% 失败 degraded=True)
    """

    def __init__(
        self,
        pipeline,
        cache,
        tenants: List[int],
        scenarios: Dict[int, dict],
        max_workers: int = 4,
        batch_size: int = 50,
        max_retries: int = 3,
        fail_threshold: float = 0.2,
    ) -> None:
        self.pipeline = pipeline
        self.cache = cache
        self.tenants = list(tenants or [])
        self.scenarios = dict(scenarios or {})
        self.max_workers = max(1, int(max_workers))
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self.fail_threshold = float(fail_threshold)

        # 上次 batch_generate 失败的 tenant 列表, 供 incremental_retry 重试。
        # 元素: {"tenant_id":int, "err":str}
        self._failed: List[Dict[str, Any]] = []
        # 增量触发日志 (trigger_incremental 记录), demo 排障用。
        self._incremental_log: List[Dict[str, Any]] = []

        # 共享状态保护: cache.put / _incremental_log 可能被多线程并发写 (批内并发 +
        # start_cron 与手动调用并存), 用一把锁串行化这些写操作。
        # pipeline.generate_today_advice 会重置 pipeline.last_trace (共享可变), 并发下
        # trace 可能交错, 但不影响 Advice 产出正确性 (collect/router 为纯函数);
        # 故不对 generate 加锁, 保留并发收益, trace 仅供 demo 打印。
        self._lock = threading.Lock()

        # cron 线程控制 (start_cron/stop_cron), demo 可选。
        self._cron_thread: Optional[threading.Thread] = None
        self._cron_stop = threading.Event()
        self._cron_interval = 24.0 * 3600.0  # 默认每日一轮 (demo 可调小)

    # ===== 主入口: 全量预生成 =====

    def batch_generate(self) -> Dict[str, Any]:
        """全量预生成: 对每个 tenant 跑 pipeline + put 进 cache (TTL 24h)。

        分批 batch_size/批, 每批 ThreadPoolExecutor(max_workers) 并发; 单户失败
        重试 ≤ max_retries 次 (指数退避 0.5*2^n); >fail_threshold 失败率 degraded=True。
        幂等: put 同 key 覆盖, 同日重算只覆盖不新增。

        Returns:
            {"ok":int, "fail":int, "cached":int, "retries":int, "degraded":bool}
            - ok       : 生成 Advice 成功的 tenant 数
            - fail     : 最终失败的 tenant 数
            - cached   : 成功 put 进 cache 的 tenant 数 (put 自身失败则不计)
            - retries  : 全量累计的重试次数 (成功前的失败尝试 + 失败户的最后一次重试)
            - degraded : 失败率 > fail_threshold 时 True
        """
        tenants_list = list(self.tenants)
        total = len(tenants_list)
        ok = 0
        fail = 0
        cached = 0
        retries = 0
        failed: List[Dict[str, Any]] = []

        if total == 0:
            self._failed = []
            return {"ok": 0, "fail": 0, "cached": 0, "retries": 0, "degraded": False}

        for start in range(0, total, self.batch_size):
            batch = tenants_list[start:start + self.batch_size]
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                future_to_tenant = {
                    ex.submit(self._generate_one, t): t for t in batch
                }
                for fut in as_completed(future_to_tenant):
                    tenant_id = future_to_tenant[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # 防御: _generate_one 内已捕获, 此为兜底
                        res = {
                            "status": "fail", "cached": False,
                            "retries": self.max_retries, "err": str(exc),
                        }
                    if res.get("status") == "ok":
                        ok += 1
                        if res.get("cached"):
                            cached += 1
                    else:
                        fail += 1
                        failed.append({
                            "tenant_id": tenant_id,
                            "err": res.get("err") or "",
                        })
                    retries += int(res.get("retries") or 0)

        degraded = (fail / total) > self.fail_threshold
        # 失败户存下来供 incremental_retry 重试
        self._failed = failed

        return {
            "ok": ok,
            "fail": fail,
            "cached": cached,
            "retries": retries,
            "degraded": degraded,
        }

    # ===== 增量: 重试上次失败户 =====

    def incremental_retry(self) -> Dict[str, Any]:
        """重试上次 batch_generate 失败的 tenant 列表 (self._failed)。

        成功的从 _failed 移除, 失败的留存下轮再试。返回同 batch_generate 结构。
        """
        prev_failed = list(self._failed)
        if not prev_failed:
            return {"ok": 0, "fail": 0, "cached": 0, "retries": 0, "degraded": False}

        # 临时把待重试户作为本批 tenants 跑一遍 (复用 _generate_one),
        # 但不覆盖 self.tenants; 结果单独聚合。
        retry_tenants = [int(item["tenant_id"]) for item in prev_failed]
        total = len(retry_tenants)
        ok = 0
        fail = 0
        cached = 0
        retries = 0
        still_failed: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_tenant = {
                ex.submit(self._generate_one, t): t for t in retry_tenants
            }
            for fut in as_completed(future_to_tenant):
                tenant_id = future_to_tenant[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {
                        "status": "fail", "cached": False,
                        "retries": self.max_retries, "err": str(exc),
                    }
                if res.get("status") == "ok":
                    ok += 1
                    if res.get("cached"):
                        cached += 1
                else:
                    fail += 1
                    still_failed.append({
                        "tenant_id": tenant_id,
                        "err": res.get("err") or "",
                    })
                retries += int(res.get("retries") or 0)

        # 更新 _failed: 仍失败的留存, 成功的移除
        self._failed = still_failed
        degraded = (fail / total) > self.fail_threshold

        return {
            "ok": ok,
            "fail": fail,
            "cached": cached,
            "retries": retries,
            "degraded": degraded,
        }

    # ===== 增量: 单户重算 =====

    def trigger_incremental(self, tenant_id: int, reason: str = "") -> Dict[str, Any]:
        """单户增量重算: 重跑 pipeline + 覆盖 cache; 记 reason。

        库存变更 / 天气突变 / 手动刷新触发。幂等: put 覆盖同日同 key。
        返回 {"ok":bool, "retries":int, "reason":str, "err":Optional[str]}。
        """
        res = self._generate_one(int(tenant_id))
        entry = {
            "tenant_id": int(tenant_id),
            "reason": reason or "",
            "at": time.time(),
            "ok": res.get("status") == "ok",
            "retries": int(res.get("retries") or 0),
            "err": res.get("err"),
        }
        with self._lock:
            self._incremental_log.append(entry)
        return {
            "ok": entry["ok"],
            "retries": entry["retries"],
            "reason": reason or "",
            "err": res.get("err"),
        }

    # ===== cron (demo 可选) =====

    def start_cron(self, interval_seconds: Optional[float] = None) -> None:
        """启动后台守护线程周期跑 batch_generate (demo: 手动调用为主)。

        interval_seconds 缺省 24h; demo 可调小观察。stop_cron() 停止。
        线程为 daemon, 进程退出不阻塞。重复调用幂等 (已在跑则只更新 interval)。
        """
        if interval_seconds is not None:
            self._cron_interval = max(1.0, float(interval_seconds))
        if self._cron_thread is not None and self._cron_thread.is_alive():
            return  # 已在跑
        self._cron_stop.clear()
        self._cron_thread = threading.Thread(
            target=self._cron_loop, name="pregen-cron", daemon=True,
        )
        self._cron_thread.start()

    def stop_cron(self) -> None:
        """停止 cron 线程 (设标志, 等待下一轮 sleep 醒来退出)。"""
        self._cron_stop.set()
        t = self._cron_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._cron_thread = None

    def _cron_loop(self) -> None:
        """cron 循环: 跑一轮 batch_generate, 然后按 interval sleep, 直到 stop。"""
        while not self._cron_stop.is_set():
            try:
                self.batch_generate()
            except Exception:
                # cron 循环永不抛 (demo 守护); 错误已在 batch_generate 内统计
                pass
            # 分段 sleep 以便能及时响应 stop_cron
            self._cron_stop.wait(timeout=self._cron_interval)

    # ===== 内部: 单户生成+缓存 (带重试) =====

    def _generate_one(self, tenant_id: int) -> Dict[str, Any]:
        """单户: 跑 pipeline.generate_today_advice + 构造 key + cache.put, 带重试。

        重试 ≤ max_retries 次, 退避 0.5*2^n (n=0..max_retries-1)。
        generate_today_advice 内部各段已 try/except 降级, 极少抛; 此处重试主要兜
        底 collect/route/encode 整体异常与 cache.put 异常。

        Returns:
            {"status":"ok"|"fail", "cached":bool, "retries":int, "err":Optional[str]}
            retries = 成功前的失败尝试数 (成功则 0..max_retries-1; 全失败为 max_retries)
        """
        scenario = self.scenarios.get(int(tenant_id), {}) or {}
        last_err: Optional[str] = None

        for attempt in range(self.max_retries + 1):  # 1 + max_retries 次尝试
            try:
                advice = self.pipeline.generate_today_advice(int(tenant_id), scenario)
                key_text, key_prefix, business = self._build_key(int(tenant_id), scenario)
                with self._lock:
                    self.cache.put(int(tenant_id), business, key_text, advice, key_prefix)
                return {
                    "status": "ok",
                    "cached": True,
                    "retries": attempt,
                    "err": None,
                }
            except Exception as exc:  # 任意段或 put 异常
                last_err = str(exc)
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))  # 0.5*2^n 退避
                # else 最后一次, 不再 sleep

        return {
            "status": "fail",
            "cached": False,
            "retries": self.max_retries,
            "err": last_err,
        }

    # ===== 内部: 构造缓存 key =====

    def _build_key(self, tenant_id: int, scenario: dict) -> Tuple[str, str, str]:
        """构造 cache key_text + key_prefix + business。

        优先用 cache.make_key_text / make_key_prefix(bundle, routed) (语义 key + 强区分前缀);
        取 bundle/routed 时直接调 pipeline.collector.collect + pipeline.router.route
        (纯函数, 与管线内同输入同输出)。collect/route 失败则降级为 scenario 派生的兜底 key
        (仍可缓存, 命中率略降但不阻断; 兜底时 key_prefix 为空, 不参与前缀过滤)。

        Returns:
            (key_text, key_prefix, business)
        """
        try:
            bundle = self.pipeline.collector.collect(int(tenant_id), scenario)
            routed = self.pipeline.router.route(bundle)
            key_text = self.cache.make_key_text(bundle, routed)
            key_prefix = self.cache.make_key_prefix(bundle, routed)
            business = getattr(bundle, "business", "") or "unknown"
            return key_text, key_prefix, business
        except Exception:
            # 降级: 用 scenario 派生确定性兜底 key (按字段排序拼串, 同 scenario 同 key)
            key_text = "fallback|{tid}|{sig}".format(
                tid=int(tenant_id), sig=self._scenario_sig(scenario),
            )
            business = (scenario or {}).get("business") or "unknown"
            return key_text, "", business

    @staticmethod
    def _scenario_sig(scenario: dict) -> str:
        """scenario 派生确定性签名 (兜底 key 用): 按键排序拼 k=v 串。"""
        if not isinstance(scenario, dict) or not scenario:
            return "empty"
        parts = []
        for k in sorted(scenario.keys()):
            parts.append("{k}={v}".format(k=k, v=scenario[k]))
        return ",".join(parts)

    # ===== 观测 =====

    def failed_tenants(self) -> List[Dict[str, Any]]:
        """返回上次 batch_generate 后仍失败的 tenant 列表 (只读拷贝)。"""
        return list(self._failed)

    def incremental_log(self) -> List[Dict[str, Any]]:
        """返回 trigger_incremental 的历史记录 (只读拷贝)。"""
        return list(self._incremental_log)


__all__ = ["Scheduler"]
