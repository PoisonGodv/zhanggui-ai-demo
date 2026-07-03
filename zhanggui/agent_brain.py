"""Agent 段 (对应方案 7.3.2 tool calling 对接 + 7.8.4 单户月成本熔断)。

Brain 协议: enhance(tenant_id, routed, bundle) -> Enhanced。
三实现:
  - MockBrain : 确定性增强 (无 LLM), 基于规则+模板产出, 永远 margin_ok=True。
                模拟「tool calling」: 对 generate_coupon 类动作先调
                TOOLS['generate_coupon'].run 验证, 若 margin_violation 则改用
                min_discount_floor 重算 (方案 7.3.2「护栏不过则返回错误让 Agent
                改用区间下限」)。
  - LLMBrain  : 真模型 (可选, 需 api_key)。openai 兼容协议, 用标准库 urllib
                调 /chat/completions, 带 tools schema, tool-calling 循环最多 3 轮,
                超时 3s, 失败抛 -> 上层降级 MockBrain。不 import openai (未装)。
  - AgentStage: 预算熔断 (超 ¥2/户/月 全降级) + brain 失败降级 MockBrain。
                is_budget_exceeded(tenant_id) -> monthly_cost >= budget_cny;
                enhance_or_skip -> 超预算走 MockBrain, 否则走 self.brain,
                brain.enhance 失败 try/except 降级 MockBrain。

设计要点 (忠实复刻方案 7.3.2 / 7.8.4):
- Agent 不直接定价, 折扣取 param_range 中点但不得低于 min_discount_floor:
    discount = clamp(midpoint, min_discount_floor, discount_max)
    price_after = price * discount
  非商品类 (wake_dormant/festival_discount/birthday_care) 用 min_spend/reduce,
  不走商品折扣 (与 rule_engine 占位动作一致: cost=0/price=0)。
- expected_effect 保守估算 (如 clearance「核销35份 挽回损耗¥180」,
  wake_dormant「召回4-6人」), 数值保守 (方案 7.4.1 硬约束 #4)。
- 调 guardrail.guardrail_ok 二次校验设 margin_ok (方案 7.3.2 护栏二次兜底)。
- copy 从 playbook 模板填充 (MockBrain 直接调 TOOLS['draft_group_message']/
  draft_moments_post 模拟 agent 调工具产出文案)。

严格遵循 CONTRACT (zhanggui/types.py): Enhanced/EnhancedItem/Routed/RoutedItem。
包内相对 import: from .types/.guardrail/.tools/.playbook import。
零第三方依赖, 仅标准库 (LLMBrain 用 urllib.request)。
"""

import json
import math
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .guardrail import guardrail_ok, min_discount_floor
from .playbook import get_playbook
from .tools import TOOLS
from .types import (
    Enhanced,
    EnhancedItem,
    Routed,
    RoutedItem,
    SignalBundle,
)


# === Brain 协议 (对应方案 7.3.2 agent.Agent 的 Enhance 抽象) ===

@runtime_checkable
class Brain(Protocol):
    """Agent 段大脑协议: 把 Routed 增强成 Enhanced。"""

    def enhance(self, tenant_id: int, routed: Routed, bundle: SignalBundle) -> Enhanced:
        ...


# === 7 种 action_type 分类 (与 rule_engine 一致) ===

# 关联具体商品、走商品折扣的动作 (有真实 cost/price)。
_DISCOUNT_ACTIONS = {
    "clearance_coupon",
    "rain_delivery",
    "bundle_slow",
    "high_value_push",
}

# 不关联具体商品、走满减/积分的动作 (占位 product='-', cost/price=0)。
_REDUCE_ACTIONS = {
    "wake_dormant",
    "festival_discount",
    "birthday_care",
}

# 群发类动作 (draft_group_message); 其余走朋友圈 (draft_moments_post)。
# 按方案 7.4.2 clearance/rain_delivery/wake_dormant/festival 均产群发文案。
_GROUP_MSG_ACTIONS = {
    "clearance_coupon",
    "rain_delivery",
    "bundle_slow",
    "wake_dormant",
    "festival_discount",
    "birthday_care",
    "high_value_push",
}

# 营销文案发送时段 (方案 7.4.2「今晚7点」)。
_SEND_WINDOW = "19:00"


def _price_after_floor_aware(
    price: float, discount: float, cost: float, margin_floor: float
) -> float:
    """计算 price_after (保留 2 位), 保证满足护栏 (>= cost_ceil)。

    price*discount 在 discount>=min_discount_floor 时本应 >= cost_ceil, 但
    round(..., 2) 可能向下取整使 price_after 跌破 cost_ceil, 导致二次
    guardrail_ok 误判 margin_ok=False (违反 MockBrain「永远 margin_ok=True」契约)。
    故: 先 round 到分, 若低于 cost_ceil 则向上取整到分, 保证 >= cost_ceil。
    cost_ceil<0 钳为 0; price<=0 返回 0。
    """
    if price <= 0:
        return 0.0
    pa = round(price * discount, 2)
    if cost <= 0:
        return pa
    cost_ceil = cost * (1.0 + margin_floor)
    if cost_ceil < 0:
        cost_ceil = 0.0
    if pa < cost_ceil:
        # 向上取整到分, 保证 >= cost_ceil (不会让顾客多付超过 1 分钱的误差)。
        pa = math.ceil(cost_ceil * 100.0) / 100.0
    return pa


# === MockBrain (确定性, 无 LLM) ===

class MockBrain:
    """确定性增强大脑 (无 LLM, 基于规则+模板)。

    对每个 RoutedItem:
      1. 构造 params:
         - 折扣类动作: discount = clamp(midpoint, min_discount_floor, discount_max);
           price_after = price * discount; limit_qty 取 param_range.limit_qty_max;
           expire_hours 取 param_range.expire_hours。
         - 满减类动作: min_spend/reduce 取 param_range 对应字段; audience_size
           取 bundle.members 相关计数。
      2. 模拟 tool calling: 对 generate_coupon 类动作 (有 cost/price) 先调
         TOOLS['generate_coupon'].run 验证; 若 error=margin_violation 则改用
         min_discount_floor 重算 discount (方案 7.3.2 改用区间下限)。
      3. copy: 调 TOOLS['draft_group_message']/draft_moments_post 模拟 agent
         调工具产出文案 (≤40字模板填充)。
      4. expected_effect: 保守估算 (clearance「核销35份 挽回损耗¥180」等)。
      5. margin_ok: 调 guardrail.guardrail_ok 二次校验; MockBrain 永远保证
         margin_ok=True (改用 min_discount_floor 后必达标)。
    """

    def enhance(self, tenant_id: int, routed: Routed, bundle: SignalBundle) -> Enhanced:
        tenant_id = int(tenant_id or 0)
        items: List[EnhancedItem] = []
        if routed is None:
            return Enhanced(tenant_id=tenant_id, items=items)
        for ri in routed.items or []:
            items.append(self._enhance_one(ri, bundle, tenant_id))
        return Enhanced(tenant_id=tenant_id, items=items)

    # ---- 单条增强 ----

    def _enhance_one(
        self, ri: RoutedItem, bundle: SignalBundle, tenant_id: int
    ) -> EnhancedItem:
        action_type = ri.action_type or ""
        pr = ri.param_range

        # ctx 模拟 agent 调工具时的上下文 (工具从 ctx 读数据)。
        ctx = self._build_ctx(ri, bundle, tenant_id)

        if action_type in _DISCOUNT_ACTIONS:
            params, margin_ok = self._build_discount_params(ri, ctx, action_type)
        else:
            # 满减/积分类占位动作 (wake_dormant/festival/birthday)。
            params, margin_ok = self._build_reduce_params(ri, bundle, action_type)

        # 模拟 tool calling: 调 generate_coupon 验证折扣类动作 (二次兜底)。
        if action_type in _DISCOUNT_ACTIONS and ri.cost > 0 and ri.price > 0:
            params, margin_ok = self._verify_via_coupon_tool(ri, params, ctx, margin_ok)

        # copy: 模拟 agent 调 draft 工具产出文案。
        copy = self._draft_copy(ri, bundle, params, ctx, action_type)

        # expected_effect: 保守估算。
        expected_effect = self._estimate_effect(ri, bundle, params, action_type)

        # margin_ok 二次校验 (满减类 cost=0 一律 True, 已在 _build_reduce_params 设)。
        if action_type in _DISCOUNT_ACTIONS and ri.cost > 0 and ri.price > 0:
            price_after = float(params.get("price_after") or 0.0)
            margin_ok = guardrail_ok(ri.cost, price_after, ri.margin_floor)

        return EnhancedItem(
            routed=ri,
            copy=copy,
            params=params,
            expected_effect=expected_effect,
            margin_ok=bool(margin_ok),
        )

    # ---- ctx 构造 ----

    @staticmethod
    def _build_ctx(ri: RoutedItem, bundle: SignalBundle, tenant_id: int) -> Dict[str, Any]:
        """构造工具调用的 ctx (含 bundle/tenant_id/playbook/margin_floor)。"""
        business = str(getattr(bundle, "business", "") or "") if bundle else ""
        playbook = get_playbook(business)
        return {
            "bundle": bundle,
            "tenant_id": tenant_id,
            "playbook": playbook,
            "margin_floor": float(getattr(ri, "margin_floor", 0.0) or 0.0),
        }

    # ---- 折扣类 params ----

    def _build_discount_params(
        self, ri: RoutedItem, ctx: Dict[str, Any], action_type: str
    ) -> tuple:
        """折扣类 params: discount=clamp(midpoint, min_discount_floor, discount_max)。

        midpoint = (discount_min + discount_max)/2;
        min_disc = min_discount_floor(cost, price, margin_floor) (毛利下限推导);
        discount = max(min_disc, min(midpoint, discount_max)) (不得低于下限,
          不超过上限); 再兜底 clamp 到 [0,1]。
        price_after = price * discount; limit_qty/expire_hours 取 param_range。
        返回 (params dict, margin_ok)。
        """
        pr = ri.param_range
        cost = float(getattr(ri, "cost", 0.0) or 0.0)
        price = float(getattr(ri, "price", 0.0) or 0.0)
        margin_floor = float(getattr(ri, "margin_floor", 0.0) or 0.0)

        d_min = float(getattr(pr, "discount_min", 0.0) or 0.0)
        d_max = float(getattr(pr, "discount_max", 0.0) or 0.0)

        # 中点 (无 LLM 润色, 取区间中点作保守默认)。
        midpoint = (d_min + d_max) / 2.0
        # 毛利下限推导的折扣下限 (防幻觉亏钱)。
        min_disc = min_discount_floor(cost, price, margin_floor) if (cost > 0 and price > 0) else 0.0

        # discount = clamp(midpoint, min_disc, d_max): 不得低于 min_disc, 不超 d_max。
        if midpoint < min_disc:
            discount = min_disc
        else:
            discount = midpoint
        if discount > d_max:
            discount = d_max
        # 下限再取 max(min_disc, d_min) (与 apply_guardrail 一致: 比配置更保守取配置下限)。
        floor = max(min_disc, d_min) if (cost > 0 and price > 0) else d_min
        if discount < floor:
            discount = floor
        # clamp [0,1]。
        if discount < 0.0:
            discount = 0.0
        if discount > 1.0:
            discount = 1.0

        price_after = _price_after_floor_aware(price, discount, cost, margin_floor) if price > 0 else 0.0
        limit_qty = int(getattr(pr, "limit_qty_max", 0) or 0)
        expire_hours = int(getattr(pr, "expire_hours", 0) or 0)
        min_spend = float(getattr(pr, "min_spend", 0.0) or 0.0)

        params: Dict[str, Any] = {
            "discount": round(discount, 4),
            "price_after": price_after,
            "limit_qty": limit_qty,
            "expire_hours": expire_hours,
            "send_window": _SEND_WINDOW,
        }
        if min_spend > 0:
            params["min_spend"] = min_spend

        margin_ok = True
        if cost > 0 and price > 0:
            margin_ok = guardrail_ok(cost, price * discount, margin_floor)
        return params, margin_ok

    # ---- 满减类 params ----

    def _build_reduce_params(
        self, ri: RoutedItem, bundle: SignalBundle, action_type: str
    ) -> tuple:
        """满减/积分类 params: min_spend/reduce/audience_size/expire_hours。

        这些动作不绑定 SKU, 毛利在 action 级别已约束 (满减限额 reduce_max),
        不走商品级 guardrail_ok; margin_floor 仅记录, margin_ok 一律 True。
        """
        pr = ri.param_range
        min_spend = float(getattr(pr, "min_spend", 0.0) or 0.0)
        reduce_max = float(getattr(pr, "reduce_max", 0.0) or 0.0)
        expire_hours = int(getattr(pr, "expire_hours", 0) or 0)

        params: Dict[str, Any] = {}
        if min_spend > 0:
            params["min_spend"] = min_spend
        if reduce_max > 0:
            # 取 reduce_max 作满减额 (保守取上限, 方案区间取值)。
            params["reduce"] = reduce_max
        if expire_hours > 0:
            params["expire_hours"] = expire_hours

        # audience_size: wake_dormant 取 sleep_30; birthday_care 取生日人数;
        # festival_discount 取 high_value 数 (示意受众规模)。
        members = getattr(bundle, "members", None) if bundle else None
        if action_type == "wake_dormant":
            params["audience_size"] = int(getattr(members, "sleep_30", 0) or 0)
        elif action_type == "birthday_care":
            params["audience_size"] = len(getattr(members, "birthday_today", []) or [])
        elif action_type == "festival_discount":
            params["audience_size"] = len(getattr(members, "high_value", []) or [])

        params["send_window"] = _SEND_WINDOW
        # 满减类不走商品护栏, margin_ok 一律 True (MockBrain 保证)。
        return params, True

    # ---- 模拟 tool calling: generate_coupon 二次验证 ----

    def _verify_via_coupon_tool(
        self, ri: RoutedItem, params: Dict[str, Any], ctx: Dict[str, Any],
        margin_ok: bool,
    ) -> tuple:
        """模拟 agent 调 generate_coupon 工具验证折扣 (方案 7.3.2)。

        若工具返回 margin_violation, 则改用 min_discount_floor 重算 discount
        (改用区间下限), price_after 重算, margin_ok 强制 True (MockBrain 保证)。
        """
        coupon_tool = TOOLS.get("generate_coupon")
        if coupon_tool is None:
            return params, margin_ok

        args = {
            "product": ri.product,
            "discount": params.get("discount", 0.0),
            "limit_qty": params.get("limit_qty", 0),
            "expire_hours": params.get("expire_hours", 0),
        }
        try:
            res = coupon_tool.run(args, ctx)
        except Exception:
            # 工具异常不阻断, 保留原 params (已二次校验)。
            return params, margin_ok

        if isinstance(res, dict) and res.get("error") == "margin_violation":
            # 改用区间下限 (min_discount_floor) 重算 (方案 7.3.2 改用区间下限)。
            cost = float(getattr(ri, "cost", 0.0) or 0.0)
            price = float(getattr(ri, "price", 0.0) or 0.0)
            margin_floor = float(getattr(ri, "margin_floor", 0.0) or 0.0)
            # 用全精度 min_discount_floor (不预 round), 避免再次跌破护栏。
            min_disc = min_discount_floor(cost, price, margin_floor)
            # 下限再与 param_range.discount_min 取大 (保守, 与 apply_guardrail 一致)。
            d_min = float(getattr(ri.param_range, "discount_min", 0.0) or 0.0)
            floor = max(min_disc, d_min)
            if floor > 1.0:
                floor = 1.0
            if floor < 0.0:
                floor = 0.0
            # price_after 用 floor-aware 计算, 保证 round 后仍 >= cost_ceil。
            pa = _price_after_floor_aware(price, floor, cost, margin_floor) if price > 0 else 0.0
            params = dict(params)
            params["discount"] = round(floor, 4)
            params["price_after"] = pa
            return params, True  # 改用下限后必达标 (MockBrain 保证 margin_ok=True)
        return params, margin_ok

    # ---- copy: 模拟 agent 调 draft 工具 ----

    def _draft_copy(
        self, ri: RoutedItem, bundle: SignalBundle, params: Dict[str, Any],
        ctx: Dict[str, Any], action_type: str,
    ) -> str:
        """调 draft_group_message / draft_moments_post 模拟 agent 产出文案。"""
        # 冷启动兜底项: 不套 wake_dormant「好久不见」模板 (纯新店无沉睡会员, 口吻
        # 不贴), 用中性首单满减文案。
        rule_id = str(getattr(ri, "rule_id", "") or "")
        if rule_id.startswith("cold_start_"):
            spend = params.get("min_spend") or 30
            reduce = params.get("reduce_max") or params.get("reduce") or 5
            hours = params.get("expire_hours") or 48
            return "新店福利 满{spend}减{reduce} {hours}h内到店可用".format(
                spend=int(spend), reduce=int(reduce), hours=int(hours)
            )
        # 构造工具入参 (从 params + ri 折算)。
        tool_args = self._draft_args(ri, params, action_type)
        tool_name = "draft_group_message" if action_type in _GROUP_MSG_ACTIONS else "draft_moments_post"
        tool = TOOLS.get(tool_name)
        if tool is None:
            return ""
        try:
            res = tool.run(tool_args, ctx)
        except Exception:
            return ""
        if isinstance(res, dict):
            return str(res.get("copy") or "")
        return ""

    @staticmethod
    def _draft_args(ri: RoutedItem, params: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        """折算 draft 工具入参 (product/discount/price/qty/spend/reduce/hours)。"""
        args: Dict[str, Any] = {
            "action_type": action_type,
            "product": ri.product or "-",
        }
        if "discount" in params:
            args["discount"] = params["discount"]
        if "price_after" in params:
            args["price"] = params["price_after"]
        if "limit_qty" in params:
            args["qty"] = params["limit_qty"]
        if "min_spend" in params:
            args["spend"] = params["min_spend"]
        if "reduce" in params:
            args["reduce"] = params["reduce"]
        if "expire_hours" in params:
            args["hours"] = params["expire_hours"]
        if "audience_size" in params:
            args["audience_size"] = params["audience_size"]
        return args

    # ---- expected_effect: 保守估算 ----

    def _estimate_effect(
        self, ri: RoutedItem, bundle: SignalBundle, params: Dict[str, Any],
        action_type: str,
    ) -> str:
        """保守估算预期效果 (方案 7.4.1 #4: 核销X张/到店Y人/挽回¥Z, 数值保守)。

        - clearance_coupon: 核销N份 挽回损耗¥M (N≈limit_qty*0.7, M=N*cost*0.3)。
        - rain_delivery: 核销N份 到店Y人。
        - bundle_slow: 核销N份。
        - wake_dormant: 召回N人 (audience*0.2, 保守)。
        - festival_discount: 到店Y人。
        - birthday_care: 到店Y人。
        - high_value_push: 核销N份。
        未知 action_type -> 空串 (让 converger 记 0 分)。
        """
        limit_qty = int(params.get("limit_qty") or 0)
        audience = int(params.get("audience_size") or 0)
        cost = float(getattr(ri, "cost", 0.0) or 0.0)

        # 冷启动兜底项 (rule_id 以 cold_start_ 开头): audience=0 的纯新店, 不提
        # 「召回N人」(无沉睡会员数据, 口吻不贴), 给中性「到店引流」预期。
        rule_id = str(getattr(ri, "rule_id", "") or "")
        if rule_id.startswith("cold_start_"):
            return "新店引流 到店3-5人"

        if action_type == "clearance_coupon":
            n = max(1, int(limit_qty * 0.7)) if limit_qty > 0 else 35
            # 挽回损耗 ≈ 核销份 * 成本 * 0.3 (临期挽回比例保守)。
            m = round(n * cost * 0.3) if cost > 0 else 180
            return "核销{n}份 挽回损耗¥{m}".format(n=n, m=m)
        if action_type == "rain_delivery":
            n = max(1, int(limit_qty * 0.6)) if limit_qty > 0 else 20
            y = max(2, int(n * 0.5))
            return "核销{n}份 到店{y}人".format(n=n, y=y)
        if action_type == "bundle_slow":
            n = max(1, int(limit_qty * 0.5)) if limit_qty > 0 else 10
            return "核销{n}份".format(n=n)
        if action_type == "high_value_push":
            n = max(1, int(limit_qty * 0.5)) if limit_qty > 0 else 8
            return "核销{n}份".format(n=n)
        if action_type == "wake_dormant":
            # 召回 ≈ audience * 0.2, 保守取下界区间 [ceil(aud*0.16), ceil(aud*0.24)]。
            lo = max(1, int(round(audience * 0.16))) if audience > 0 else 4
            hi = max(lo + 1, int(round(audience * 0.24))) if audience > 0 else 6
            return "召回{lo}-{hi}人".format(lo=lo, hi=hi)
        if action_type in ("festival_discount", "birthday_care"):
            y = max(2, int(round(audience * 0.3))) if audience > 0 else 5
            return "到店{y}人".format(y=y)
        return ""


# === LLMBrain (真模型, 可选, 需 api_key) ===

class LLMBrain:
    """真模型大脑 (openai 兼容, 用标准库 urllib, 需 api_key)。

    __init__: api_key 为空 -> raise ValueError("LLMBrain needs api_key")。
    enhance: 构造 messages 调 /chat/completions (带 tools schema), tool-calling
    循环最多 3 轮, 超时 3s, 失败抛 -> 上层 (AgentStage) 降级 MockBrain。
    不 import openai (未装), 用 urllib.request。

    NOTE: 需 api_key; 无 key 不可用 (构造即抛)。本 demo 默认走 MockBrain。
    """

    # openai 兼容端点默认 (DeepSeek/通义百炼/火山方舟均可改 base_url)。
    _DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    _DEFAULT_MODEL = "deepseek-chat"
    _TIMEOUT = 3.0
    _MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        gateway: Optional[Any] = None,
    ):
        if not api_key and gateway is None:
            raise ValueError("LLMBrain needs api_key")
        self.api_key = api_key or ""
        self.base_url = (base_url or self._DEFAULT_BASE_URL).rstrip("/")
        self.model = model or self._DEFAULT_MODEL
        # 可选模型网关 (方案 7.8): 注入后 _chat 走 gateway.chat_with_tools,
        # 享受分层选层/超时重试/层间降级/月成本熔断; 不注入则走原 urllib 直连。
        self.gateway = gateway
        # 复用 MockBrain 做 tool 结果回灌后的兜底解析 / 护栏兜底。
        self._mock = MockBrain()

    def enhance(self, tenant_id: int, routed: Routed, bundle: SignalBundle) -> Enhanced:
        """真模型增强: 每条 RoutedItem 调一次 LLM 带 tools, 解析 JSON 产出 EnhancedItem。

        失败 (网络/超时/解析) 抛异常, 由 AgentStage 降级 MockBrain。
        """
        tenant_id = int(tenant_id or 0)
        items: List[EnhancedItem] = []
        if routed is None:
            return Enhanced(tenant_id=tenant_id, items=items)
        for ri in routed.items or []:
            items.append(self._enhance_one_llm(ri, bundle, tenant_id))
        return Enhanced(tenant_id=tenant_id, items=items)

    def _enhance_one_llm(
        self, ri: RoutedItem, bundle: SignalBundle, tenant_id: int
    ) -> EnhancedItem:
        """单条: 构造 system+user prompt, tool-calling 循环最多 3 轮, 解析最终 JSON。"""
        ctx = MockBrain._build_ctx(ri, bundle, tenant_id)
        messages = self._build_messages(ri, bundle)

        # tool-calling 循环 (方案 7.3.2: 发请求带 tools -> 模型回 tool_calls ->
        # 本地 Run 工具 -> tool role 回灌 -> 模型输出最终 JSON, 最多 3 轮)。
        from .tools import schemas as _schemas
        tools_schema = _schemas()
        for _ in range(self._MAX_TOOL_ROUNDS):
            resp = self._chat(messages, tools=tools_schema)
            choice = self._first_choice(resp)
            if choice is None:
                break
            msg = choice.get("message", {}) or {}
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # 无 tool_calls, 视为最终输出, 解析 content JSON。
                return self._parse_final(ri, bundle, ctx, msg)
            # 把 assistant 消息回灌, 再逐个执行 tool_calls 并以 tool role 回灌。
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": tool_calls,
            })
            for call in tool_calls:
                tool_msg = self._run_tool_call(call, ctx)
                messages.append(tool_msg)
        # 超过最大轮数仍未拿到最终 JSON: 抛异常让上层降级。
        raise RuntimeError("LLMBrain: exceeded max tool-calling rounds without final JSON")

    # ---- prompt 构造 (方案 7.4.1 system + 7.4.2/7.4.3 user) ----

    def _build_messages(self, ri: RoutedItem, bundle: SignalBundle) -> List[Dict[str, Any]]:
        system = (
            "你是「掌柜军师」的营销文案润色引擎。你只对【规则引擎已选定的动作】做"
            "受众筛选、文案润色、预期效果估算。硬约束:1.禁止从零定价,折扣/满减/限份"
            "必须在 params_range 区间内且折扣后价>=成本*(1+margin_floor),不确定取最保守;"
            "2.copy<=40汉字,口语化,不含绝对化用语(最/第一/永久/包治);3.必须返回严格JSON;"
            "4.expected_effect 给可量化预期(核销X张/到店Y人/挽回¥Z),数值保守。"
            "可用工具:query_nearing_expiry/query_high_value_members/query_weather/"
            "query_holiday/generate_coupon/draft_group_message/draft_moments_post。"
        )
        pr = ri.param_range
        user = (
            "[规则选定的动作]\n"
            "action_type: {at}\nproduct: {p}\ncost: {c}\nprice: {pr}\n"
            "params_range: {{discount_min:{dmin},discount_max:{dmax},"
            "min_spend:{ms},reduce_max:{rm},limit_qty_max:{lq},expire_hours:{eh}}}\n"
            "margin_floor: {mf}\n[商户情境]\nbusiness: {biz}\n"
            "请产出一条建议,返回严格JSON。"
        ).format(
            at=ri.action_type, p=ri.product, c=ri.cost, pr=ri.price,
            dmin=getattr(pr, "discount_min", 0), dmax=getattr(pr, "discount_max", 0),
            ms=getattr(pr, "min_spend", 0), rm=getattr(pr, "reduce_max", 0),
            lq=getattr(pr, "limit_qty_max", 0), eh=getattr(pr, "expire_hours", 0),
            mf=ri.margin_floor, biz=getattr(bundle, "business", ""),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ---- HTTP 调用 (urllib, 不 import openai) ----

    def _chat(self, messages: List[Dict[str, Any]], tools: Optional[List] = None) -> Dict[str, Any]:
        """调 /chat/completions, 超时 3s。失败抛 -> 上层降级。

        注入 gateway 时走 gateway.chat_with_tools (分层/重试/降级/熔断统一),
        否则走原 urllib 直连。两种路径最终都把响应归一成 openai choices 结构。
        """
        if self.gateway is not None:
            # 走模型网关: 用 copy_polish 任务类型 (enhance 主要是文案/params 润色),
            # 享受 lite->v3->r1 降级链; 网关最底层失败抛 -> 本方法抛 -> AgentStage
            # 降级 MockBrain。tools schema 透传。
            from .gateway import ChatMessage, ChatResp
            chat_msgs = [
                ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content") or "",
                    tool_calls=m.get("tool_calls"),
                )
                for m in messages
            ]
            resp = self.gateway.chat_with_tools("copy_polish", chat_msgs, tools=tools)
            return self._gateway_resp_to_openai(resp)
        url = self.base_url + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {k}".format(k=self.api_key),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._TIMEOUT) as resp:  # noqa: S310 (可信 base_url)
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise RuntimeError("LLMBrain HTTP {0}".format(e.code)) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError("LLMBrain network error: {0}".format(e)) from e
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError("LLMBrain bad JSON: {0}".format(e)) from e

    @staticmethod
    def _first_choice(resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if isinstance(choices, list) and choices:
            return choices[0]
        return None

    @staticmethod
    def _gateway_resp_to_openai(resp: Any) -> Dict[str, Any]:
        """把网关 ChatResp 归一成 openai choices 结构, 供 _first_choice/_parse_final 复用。

        网关返回的 content 可能是模型直接给的最终 JSON 文案 (无 tool_calls),
        也可能带 tool_calls (此时上层 _enhance_one_llm 的 tool-calling 循环会继续)。
        """
        if isinstance(resp, dict):
            return resp
        # ChatResp dataclass: content / tool_calls / raw
        msg: Dict[str, Any] = {"role": "assistant", "content": getattr(resp, "content", "") or ""}
        tc = getattr(resp, "tool_calls", None)
        if tc:
            msg["tool_calls"] = tc
        return {"choices": [{"message": msg, "raw": getattr(resp, "raw", {}) or {}}]}

    # ---- tool 执行与回灌 ----

    @staticmethod
    def _run_tool_call(call: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个 tool_call, 以 tool role 回灌结果。"""
        func = (call or {}).get("function", {}) or {}
        name = str(func.get("name") or "")
        args_raw = func.get("arguments")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        except json.JSONDecodeError:
            args = {}
        tool = TOOLS.get(name)
        tool_call_id = str((call or {}).get("id") or "")
        if tool is None:
            content = json.dumps({"error": "unknown_tool", "name": name}, ensure_ascii=False)
        else:
            try:
                res = tool.run(args, ctx)
                content = json.dumps(res, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                content = json.dumps({"error": "tool_failed", "detail": str(e)}, ensure_ascii=False)
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": content,
        }

    # ---- 最终 JSON 解析 ----

    def _parse_final(
        self, ri: RoutedItem, bundle: SignalBundle, ctx: Dict[str, Any],
        msg: Dict[str, Any],
    ) -> EnhancedItem:
        """解析模型最终 content JSON -> EnhancedItem。

        解析失败/护栏不过 -> 用 MockBrain 兜底重算 (方案 7.3.2 护栏二次兜底:
        不达标回退到区间下限, margin_ok 强制)。
        """
        content = str(msg.get("content") or "")
        data: Optional[Dict[str, Any]] = None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # 尝试抽取首个 {...} 片段 (模型可能带前后缀文字)。
            lo = content.find("{")
            hi = content.rfind("}")
            if 0 <= lo < hi:
                try:
                    data = json.loads(content[lo:hi + 1])
                except json.JSONDecodeError:
                    data = None
        if not isinstance(data, dict):
            # 解析失败 -> MockBrain 兜底。
            return self._mock._enhance_one(ri, bundle, int(ctx.get("tenant_id") or 0))

        params = data.get("params") if isinstance(data.get("params"), dict) else {}
        copy = str(data.get("copy") or "")
        expected_effect = str(data.get("expected_effect") or "")
        margin_ok_claim = bool(data.get("margin_ok"))

        # 护栏二次兜底: 折扣类动作若 margin_ok=False 或护栏不过, 改用区间下限。
        if ri.action_type in _DISCOUNT_ACTIONS and ri.cost > 0 and ri.price > 0:
            price_after = float(params.get("price_after") or (ri.price * float(params.get("discount", 0))))
            if not guardrail_ok(ri.cost, price_after, ri.margin_floor):
                # 回退 MockBrain 重算 params (clamp 到区间下限), copy/effect 保留模型产出。
                mock_item = self._mock._enhance_one(ri, bundle, int(ctx.get("tenant_id") or 0))
                params = mock_item.params
                margin_ok = True
            else:
                margin_ok = True
        else:
            margin_ok = True if ri.action_type in _REDUCE_ACTIONS else margin_ok_claim

        return EnhancedItem(
            routed=ri,
            copy=copy,
            params=params,
            expected_effect=expected_effect,
            margin_ok=bool(margin_ok),
        )


# === AgentStage (预算熔断 + 降级) ===

class AgentStage:
    """Agent 段总控 (对应方案 7.3.2 EnhanceOrSkip + 7.8.4 预算熔断)。

    - is_budget_exceeded(tenant_id): monthly_cost >= budget_cny (超 ¥2/户/月)。
    - enhance_or_skip(tenant_id, routed, bundle): 超预算走 MockBrain (等价降级),
      否则走 self.brain; brain.enhance 失败 try/except 降级 MockBrain。
    """

    def __init__(
        self,
        brain: Brain,
        budget_cny: float = 2.0,
        monthly_cost: float = 0.0,
        gateway: Optional[Any] = None,
    ):
        # brain 可为 MockBrain / LLMBrain; 内部保证有兜底 MockBrain。
        self.brain = brain
        self.budget_cny = float(budget_cny)
        self.monthly_cost = float(monthly_cost)
        # 可选模型网关 (方案 7.8): 注入后预算熔断用 gateway.is_budget_exceeded
        # (按月累计, 跨进程内存; 优于本类 monthly_cost 单值), 并把 gateway 下发给
        # LLMBrain 使其 _chat 走网关分层/重试/降级。无 gateway 时保持原状。
        self.gateway = gateway
        if gateway is not None and isinstance(brain, LLMBrain) and brain.gateway is None:
            brain.gateway = gateway
        self._fallback = MockBrain()

    def is_budget_exceeded(self, tenant_id: int) -> bool:
        """超预算熔断 (方案 7.8.4: monthly_cost >= MonthlyLimitCNY)。

        注入 gateway 时优先用 gateway.is_budget_exceeded(tenant_id) (按月累计计数,
        与 add_cost 对齐); 否则用本类 monthly_cost 单值 (向后兼容)。
        """
        if self.gateway is not None:
            try:
                return bool(self.gateway.is_budget_exceeded(int(tenant_id or 0)))
            except Exception:
                # 网关预算查询异常: 回退到 monthly_cost 单值判定, 不阻断。
                pass
        return self.monthly_cost >= self.budget_cny

    def enhance_or_skip(
        self, tenant_id: int, routed: Routed, bundle: SignalBundle
    ) -> Enhanced:
        """超预算走 MockBrain; 否则走 brain, 失败降级 MockBrain (方案 7.3.2)。"""
        tenant_id = int(tenant_id or 0)
        # 预算熔断: 全降级到纯规则+模板文案 (方案 7.8.4)。
        if self.is_budget_exceeded(tenant_id):
            return self._fallback.enhance(tenant_id, routed, bundle)
        # 正常走 brain; brain.enhance 失败 try/except 降级 MockBrain。
        try:
            return self.brain.enhance(tenant_id, routed, bundle)
        except Exception:
            # 降级: 超时/网络/解析失败一律回退确定性 MockBrain, 不中断管线。
            return self._fallback.enhance(tenant_id, routed, bundle)


__all__ = [
    "Brain",
    "MockBrain",
    "LLMBrain",
    "AgentStage",
]
