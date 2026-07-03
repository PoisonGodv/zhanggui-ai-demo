"""五段式管线总控 (对应方案 7.1.1 GenerateTodayAdvice)。

Pipeline.generate_today_advice(tenant_id, scenario) -> Advice:
  1) collect : collector.collect(tenant_id, scenario)  -> SignalBundle
  2) route    : router.route(bundle)                   -> Routed (毛利护栏在 rule 段内强制)
  3) enhance  : agent_stage.enhance_or_skip(tid,routed,bundle) -> Enhanced (预算熔断+MockBrain 兜底)
  4) converge : converger.pick_top3(enhanced)          -> list[EnhancedItem]
  5) encode   : validator.encode(top3, tenant_id, date)-> Advice (rank1-3/copy<=40/敏感词/margin_ok)

每段 try/except, 任一段失败降级到可用子集不中断 (方案 7.1.3 降级链路):
  - collect 失败: 用空 SignalBundle 占位 (后续段在空信号上仍产出空 Routed/Enhanced)。
  - route 失败: 用空 Routed (tenant_id 用 bundle.tenant_id)。
  - enhance 失败: 直接对 Routed.items 逐条做模板兜底 EnhancedItem (margin_ok 取护栏判定)。
  - converge 失败: 取 enhanced.items 原序前 3。
  - encode 失败: 用最保守 Advice (空 items, version=date+'-v1')。
段间记录 self.last_trace (dict, 每段产出的可读快照) 供 demo 打印; get_trace() 返回之。

严格遵循 CONTRACT (zhanggui/types.py): 各段产出物用其 dataclass 显式传值。
零第三方依赖, 仅标准库。
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from .types import (
    Advice,
    AdviceItem,
    Enhanced,
    EnhancedItem,
    Routed,
    SignalBundle,
)


class Pipeline:
    """五段式管线总控。

    Args (各段为已构造好的实例, 依赖注入, 便于测试与替换):
        collector    : SignalCollector (有 .collect(tenant_id, scenario)->SignalBundle)
        router       : RuleEngine     (有 .route(bundle)->Routed)
        agent_stage  : AgentStage     (有 .enhance_or_skip(tid,routed,bundle)->Enhanced)
        converger    : Converger      (有 .pick_top3(enhanced)->list[EnhancedItem])
        validator    : SchemaValidator(有 .encode(top3,tenant_id,date)->Advice)
    """

    def __init__(
        self,
        collector,
        router,
        agent_stage,
        converger,
        validator,
        cache: Optional["SemanticCache"] = None,
    ):
        self.collector = collector
        self.router = router
        self.agent_stage = agent_stage
        self.converger = converger
        self.validator = validator
        # 语义缓存 (可选, 默认 None 保持向后兼容; None 时跳过缓存读写)。
        # 命中即直接返回 Advice (trace 记 cache_hit), 不再跑 enhance/converge/encode。
        # 未命中则跑完整五段, 末尾 put 进缓存 (方案 7.6.2 前置缓存)。
        self.cache = cache
        # 每段产出的可读快照, demo 打印用; 每次 generate_today_advice 重置。
        self.last_trace: Dict[str, Any] = {}
        # 本轮 collect/route 构造出的缓存 key (供 encode 后 put 用; cache 为 None 时为 None)。
        self._cache_key_text: Optional[str] = None
        self._cache_key_prefix: Optional[str] = None
        self._cache_business: Optional[str] = None

    # ===== 主入口 =====

    def generate_today_advice(self, tenant_id: int, scenario: dict) -> Advice:
        """顺序五段, 每段 try/except 降级不中断, 最终返回 Advice。

        段间把每段产出物的可读快照写入 self.last_trace (dict), 供 demo 打印与排障。
        任何段异常都被 catch 并降级, 整条管线永不抛 (方案 7.1.3 降级链路)。
        """
        tid = int(tenant_id or 0)
        self.last_trace = {
            "tenant_id": tid,
            "stages": [],
            "degradations": [],
        }

        date_str = self._today_str(scenario)

        # ---- 1) collect ----
        bundle = self._stage_collect(tid, scenario)

        # ---- 2) route ----
        routed = self._stage_route(bundle)

        # ---- 语义缓存前置 (方案 7.6.2): bundle+routed 构造 key_text 命中即返回 ----
        if self.cache is not None:
            try:
                business = getattr(bundle, "business", "") or "unknown"
                key_text = self.cache.make_key_text(bundle, routed)
                key_prefix = self.cache.make_key_prefix(bundle, routed)
                cached_advice, hit_entry = self.cache.try_hit(
                    tid, business, key_text, key_prefix,
                )
                if cached_advice is not None:
                    self._record_stage("cache_hit", {
                        "hit": True,
                        "business": business,
                        "key_text": key_text,
                        "hit_count": getattr(hit_entry, "hit_count", 0),
                        "version": getattr(cached_advice, "version", ""),
                    })
                    return cached_advice
                self._record_stage("cache_hit", {
                    "hit": False,
                    "business": business,
                    "key_text": key_text,
                })
                self._cache_key_text = key_text
                self._cache_key_prefix = key_prefix
                self._cache_business = business
            except Exception as exc:
                # 缓存异常不阻断管线: 记降级, 走完整五段。
                self._record_degradation("cache_hit", str(exc))
                self._cache_key_text = None
                self._cache_key_prefix = None
                self._cache_business = None
        else:
            self._cache_key_text = None
            self._cache_key_prefix = None
            self._cache_business = None

        # ---- 3) enhance (预算熔断在 AgentStage 内) ----
        enhanced = self._stage_enhance(tid, routed, bundle)

        # ---- 4) converge (Top3) ----
        top3 = self._stage_converge(enhanced)

        # ---- 5) encode (rank1-3 / copy<=40 / 敏感词 / margin_ok 强制) ----
        advice = self._stage_encode(top3, tid, date_str, bundle)

        # ---- 语义缓存回写 (方案 7.6.2): 命中失败后 put, 下次同情境命中 ----
        if self.cache is not None and self._cache_key_text is not None:
            try:
                self.cache.put(
                    tid, self._cache_business, self._cache_key_text, advice,
                    self._cache_key_prefix or "",
                )
            except Exception as exc:
                self._record_degradation("cache_put", str(exc))

        return advice

    def get_trace(self) -> Dict[str, Any]:
        """返回最近一次 generate_today_advice 的段间 trace 快照。"""
        return self.last_trace

    # ===== 五段实现 (各自 try/except + trace 记录) =====

    def _stage_collect(self, tenant_id: int, scenario: dict) -> SignalBundle:
        """段1: 信号采集。失败降级为空 SignalBundle (后续段在空信号上仍可产出)。"""
        try:
            bundle = self.collector.collect(tenant_id, scenario)
        except Exception as exc:
            self._record_degradation("collect", str(exc))
            bundle = self._empty_bundle(tenant_id, scenario)
        self._record_stage("collect", {
            "business": getattr(bundle, "business", ""),
            "city": getattr(bundle, "city", ""),
            "date": getattr(bundle, "date", ""),
            "weather": getattr(getattr(bundle, "weather", None), "condition", ""),
            "holiday": getattr(getattr(bundle, "holiday", None), "name", ""),
            "inventory_n": len(getattr(bundle, "inventory", []) or []),
            "members": self._members_snapshot(getattr(bundle, "members", None)),
            "data_quality": getattr(bundle, "data_quality", 0),
        })
        return bundle

    def _stage_route(self, bundle: SignalBundle) -> Routed:
        """段2: 规则路由 (毛利护栏在 router.route 内强制)。失败降级为空 Routed。

        冷启动兜底 (方案 7.10.2 + 6.6): 若常规规则命中 0 条 (routed.items 为空),
        从剧本库 cold_priority 注入不依赖销售数据的满减/关怀动作, 保证「今日3件事」
        永不为空。门控用「命中 0 条」而非 data_quality 数值 —— 新店之所以无建议,
        根因正是无临期/无沉睡/无高毛利等触发信号致规则命中 0; 以 items 空为门控比
        以 dq 为门控更贴近本质, 也避免「dq 高但恰好无触发信号」时给空建议。
        注入动作 cost/price/cost_ceil=0 (满减类, 不绑商品成本), margin_floor=0,
        满减已限故 margin_ok 天然成立。dq 仅在 collect 段记入 trace, 不参与门控。
        """
        try:
            routed = self.router.route(bundle)
        except Exception as exc:
            self._record_degradation("route", str(exc))
            routed = Routed(tenant_id=getattr(bundle, "tenant_id", 0), items=[])
        # 冷启动注入: 常规规则命中 0 条 → 注入 cold_priority 满减动作 (不依赖销售数据)
        if not routed.items:
            injected = self._inject_cold_start(bundle)
            if injected:
                routed = Routed(tenant_id=routed.tenant_id, items=injected)
                self._record_stage("cold_start", {
                    "triggered": True,
                    "reason": "常规规则命中0条, 注入剧本库 cold_priority 冷启动动作",
                    "injected_n": len(injected),
                    "action_types": [it.action_type for it in injected],
                })
        self._record_stage("route", {
            "items_n": len(routed.items),
            "items": [
                {
                    "rule_id": it.rule_id,
                    "action_type": it.action_type,
                    "product": it.product,
                    "discount_range": [
                        it.param_range.discount_min,
                        it.param_range.discount_max,
                    ],
                    "cost_ceil": round(it.cost_ceil, 2),
                    "priority": it.priority,
                }
                for it in routed.items
            ],
        })
        return routed

    def _stage_enhance(
        self, tenant_id: int, routed: Routed, bundle: SignalBundle
    ) -> Enhanced:
        """段3: Agent 增强 (AgentStage 内含预算熔断+MockBrain 兜底)。

        本段已由 AgentStage 保证不抛 (enhance_or_skip 内 try/except 降级 MockBrain),
        但仍包一层 try/except 作最终兜底: 极端情况下用 routed.items 直接造模板
        EnhancedItem, 保证有输出。
        """
        try:
            enhanced = self.agent_stage.enhance_or_skip(tenant_id, routed, bundle)
        except Exception as exc:
            self._record_degradation("enhance", str(exc))
            enhanced = self._fallback_enhanced(tenant_id, routed)
        self._record_stage("enhance", {
            "items_n": len(enhanced.items),
            "items": [
                {
                    "action_type": it.routed.action_type,
                    "product": it.routed.product,
                    "copy": it.copy,
                    "params": it.params,
                    "expected_effect": it.expected_effect,
                    "margin_ok": it.margin_ok,
                }
                for it in enhanced.items
            ],
        })
        return enhanced

    def _stage_converge(self, enhanced: Enhanced) -> List[EnhancedItem]:
        """段4: 收敛 Top3。失败降级为 enhanced.items 原序前 3。"""
        try:
            top3 = self.converger.pick_top3(enhanced)
            if not isinstance(top3, list):
                top3 = list(top3 or [])
        except Exception as exc:
            self._record_degradation("converge", str(exc))
            top3 = list(getattr(enhanced, "items", []) or [])[:3]
        self._record_stage("converge", {
            "top3_n": len(top3),
            "top3": [
                {
                    "action_type": it.routed.action_type,
                    "product": it.routed.product,
                    "expected_effect": it.expected_effect,
                    "margin_ok": it.margin_ok,
                }
                for it in top3
            ],
        })
        return top3

    def _stage_encode(
        self, top3: List[EnhancedItem], tenant_id: int, date_str: str,
        bundle: SignalBundle,
    ) -> Advice:
        """段5: 输出校验 (rank1-3/copy<=40/敏感词/margin_ok 强制)。

        失败降级为最保守 Advice: 空 items, version=date+'-v1'。
        date_str 优先取 bundle.date (collect 已从 scenario 解析当日日期)。
        """
        try:
            advice = self.validator.encode(top3, tenant_id, date_str)
        except Exception as exc:
            self._record_degradation("encode", str(exc))
            advice = Advice(tenant_id=tenant_id, date=date_str, items=[], version=date_str + "-v1")
        self._record_stage("encode", {
            "items_n": len(advice.items),
            "version": advice.version,
            "items": [
                {
                    "rank": it.rank,
                    "action_type": it.action_type,
                    "product": it.product,
                    "copy": it.copy,
                    "expected_effect": it.expected_effect,
                    "margin_ok": it.margin_ok,
                }
                for it in advice.items
            ],
        })
        return advice

    # ===== 降级辅助 =====

    def _inject_cold_start(self, bundle: SignalBundle) -> List["RoutedItem"]:
        """冷启动注入 (方案 7.10.2 + 6.6)。

        常规规则命中 0 条时, 从剧本库 cold_priority 取不依赖销售数据的动作,
        构造占位 RoutedItem。这些动作不绑具体商品成本 (cost/price/cost_ceil=0),
        属满减/关怀类, margin_floor=0, 满减已限故护栏天然通过。

        信号过滤 (避免文案错乱): 仅在对应信号真实存在时才注入依赖信号的动作 —
        rain_delivery 仅雨天注入, festival_discount 仅有节日注入, birthday_care
        仅有当天生日会员注入, wake_dormant/high_value_push 在冷启动降级为
        「新客/老客通用满减」。无任何信号时至少注入 1 条通用满减, 保证新店首日可用。
        """
        from .playbook import get_playbook
        from .types import ParamRange, RoutedItem

        business = getattr(bundle, "business", "") or "fruit"
        pb = get_playbook(business)
        cold_list = pb.get("cold_priority", []) or []
        if not cold_list:
            return []

        weather_cond = getattr(getattr(bundle, "weather", None), "condition", "normal")
        holiday_name = getattr(getattr(bundle, "holiday", None), "name", "无")
        is_festival = getattr(getattr(bundle, "holiday", None), "is_festival", False)
        birthday_n = len(getattr(getattr(bundle, "members", None), "birthday_today", []) or [])

        # 冷启动只注入「满减/关怀类」动作 (product='-', 不依赖商品成本, 文案天然合理)。
        # 商品秒杀类 (rain_delivery/clearance/bundle/high_value_push) 在冷启动无合适
        # 触发条件且需真实商品价格, 不注入, 避免文案错乱 (如「秒杀-0.0元/斤 限0份」)。
        reduce_only = {"wake_dormant", "festival_discount", "birthday_care"}
        signal_gate = {
            "festival_discount": bool(is_festival and holiday_name and holiday_name != "无"),
            "birthday_care": birthday_n > 0,
            "wake_dormant": True,  # 冷启动降级为「新客/老客通用满减」, 无信号依赖
        }
        eligible = [
            a for a in cold_list
            if a in reduce_only and signal_gate.get(a, False)
        ]
        if not eligible:
            eligible = ["wake_dormant"]  # 兜底: 至少一条通用满减, 新店首日可用

        # 冷启动满减参数 (通用, 不绑商品): 满30减5 / 48h
        chosen = eligible[:3]
        items: List[RoutedItem] = []
        for idx, action_type in enumerate(chosen):
            items.append(RoutedItem(
                rule_id="cold_start_{a}".format(a=action_type),
                action_type=action_type,
                target="冷启动全店通用",
                product="-",
                cost=0.0,
                price=0.0,
                cost_ceil=0.0,
                margin_floor=0.0,
                priority=5 - idx,
                reason="冷启动兜底: {a} (不依赖销售数据, 新店首日可用)".format(a=action_type),
                param_range=ParamRange(
                    discount_min=0.0, discount_max=0.0,
                    min_spend=30.0, reduce_max=5.0, expire_hours=48,
                ),
            ))
        return items

    @staticmethod
    def _empty_bundle(tenant_id: int, scenario: dict) -> SignalBundle:
        """collect 失败时的空 SignalBundle 占位 (后续段在空信号上产出空集, 不崩)。"""
        from .types import (
            Weather, Holiday, MemberSig, SignalBundle as _SB,
        )
        date_str = ""
        if isinstance(scenario, dict):
            date_str = str(scenario.get("date") or "")
        return _SB(
            tenant_id=int(tenant_id or 0),
            business="",
            city="",
            date=date_str,
            weather=Weather(condition="normal", temp_c=0.0, forecast=""),
            holiday=Holiday(name="无", days_to=-1, is_festival=False),
            inventory=[],
            members=MemberSig(sleep_30=0, high_value=[], birthday_today=[]),
            data_quality=0,
        )

    def _fallback_enhanced(self, tenant_id: int, routed: Routed) -> Enhanced:
        """enhance 段彻底失败时的模板兜底: 直接对 Routed.items 造 EnhancedItem。

        margin_ok 用护栏二次判定 (cost/price>0 时调 guardrail.guardrail_ok;
        满减类 cost=0 一律 True); copy/expected_effect 用最简模板。
        保证 Enhanced.items 与 Routed.items 一一对应, 不丢候选。
        """
        from .guardrail import guardrail_ok

        items: List[EnhancedItem] = []
        for ri in (routed.items or []):
            if ri.cost > 0 and ri.price > 0:
                # 折扣类: 用区间下限对应的 price_after 做保守判定。
                disc = max(
                    ri.param_range.discount_min,
                    (ri.cost * (1.0 + ri.margin_floor)) / ri.price if ri.price > 0 else 1.0,
                )
                disc = min(max(disc, 0.0), 1.0)
                mok = guardrail_ok(ri.cost, ri.price * disc, ri.margin_floor)
            else:
                mok = True
            items.append(EnhancedItem(
                routed=ri,
                copy="今日推荐{p}".format(p=ri.product) if ri.product and ri.product != "-" else "今日推荐",
                params={},
                expected_effect="",
                margin_ok=mok,
            ))
        return Enhanced(tenant_id=int(tenant_id or 0), items=items)

    # ===== trace 记录 =====

    def _record_stage(self, name: str, snapshot: Dict[str, Any]) -> None:
        """把某段产出物的可读快照追加到 trace.stages。"""
        self.last_trace["stages"].append({"stage": name, "snapshot": snapshot})

    def _record_degradation(self, stage: str, err: str) -> None:
        """记录某段降级 (stage + 错误摘要), demo 可据此标注降级。"""
        self.last_trace["degradations"].append({
            "stage": stage,
            "error": (err or "")[:200],
        })

    # ===== 小工具 =====

    @staticmethod
    def _today_str(scenario: dict) -> str:
        """取当日日期串: 优先 scenario.date, 否则今天 ISO 日期。

        encode 段用此串构造 Advice.date / version。
        """
        if isinstance(scenario, dict):
            d = scenario.get("date")
            if d:
                return str(d)
        return datetime.date.today().isoformat()

    @staticmethod
    def _members_snapshot(members) -> Dict[str, Any]:
        """members 字段可读快照 (sleep_30 / high_value_n / birthday_n)。"""
        if members is None:
            return {"sleep_30": 0, "high_value_n": 0, "birthday_n": 0}
        return {
            "sleep_30": getattr(members, "sleep_30", 0),
            "high_value_n": len(getattr(members, "high_value", []) or []),
            "birthday_n": len(getattr(members, "birthday_today", []) or []),
        }


__all__ = ["Pipeline"]
