"""毛利下限护栏 (对应方案 7.2.3 guardrail.go)。

护栏是规则层的硬约束, Agent/LLM 段不得绕过:
  - guardrail_ok(cost, final_price, margin_floor) 判定「折扣后价 ≥ 成本×(1+margin_floor)」。
  - min_discount_floor(cost, price, margin_floor) 推导可允许的折扣下限:
        finalPrice = price*discount >= cost*(1+margin_floor)
      => discount >= cost*(1+margin_floor)/price
      cost<=0 或 price<=0 返回 1.0 (无法给出有意义的折扣, 取最保守 1.0=原价无让利)。
  - apply_guardrail(inv, rule) 在路由段对候选动作做护栏校验并产出 RoutedItem;
    cost<=0 (无成本数据) 或 即使最大让利也亏 (min_discount > discount_max) 都丢弃,
    返回 (None, False)。cost_ceil = cost*(1+margin_floor)。

严格遵循 CONTRACT (zhanggui/types.py): RoutedItem 含 cost/price/cost_ceil/margin_floor/
priority/reason/param_range/product/target/action_type/rule_id。
零第三方依赖, 仅标准库。
"""

from typing import Optional, Tuple

from .types import InventorySig, Rule, RoutedItem


def guardrail_ok(cost: float, final_price: float, margin_floor: float) -> bool:
    """判定「折扣/满减后顾客实付单价 >= 成本×(1+margin_floor)」。

    finalPrice >= floor = cost*(1+margin_floor), floor<0 时钳为 0。
    任意非数 / cost<=0 一律判 False (无成本数据不给出价动作, 防幻觉亏钱)。

    浮点容差: 用 1e-9 抹平 IEEE-754 在 cost*(1+mf) 上的表示误差
    (如 12*1.05 = 12.6000...01, 比 12.6 大 ~1.4e-15), 使顾客实付取「数学真值」
    (12.6) 仍判通过, 与方案 7.4.2 校验「price_after 15.9 >= cost 12 * 1.05 =
    12.6 通过」一致, 也与 agent_brain._price_after_floor_aware「向上取整到分」
    的口径自洽。容差不放大让利空间: 真正低于 floor 的价格 (差 >= 1e-9 元) 仍判
    False, 不构成亏钱风险。
    """
    try:
        c = float(cost)
        p = float(final_price)
        m = float(margin_floor)
    except (TypeError, ValueError):
        return False
    if c <= 0:
        return False
    floor = c * (1.0 + m)
    if floor < 0:
        floor = 0.0
    # 1e-9 容差: 仅抹平浮点表示误差, 不放行任何真实低于 floor 的价格。
    return p >= floor - 1e-9


def min_discount_floor(cost: float, price: float, margin_floor: float) -> float:
    """推导可允许的折扣下限 = cost*(1+margin_floor)/price。

    finalPrice = price*discount >= cost*(1+margin_floor)
    => discount >= cost*(1+margin_floor)/price
    cost<=0 或 price<=0 返回 1.0 (无成本/无售价时取最保守 1.0=原价无让利,
    下游 apply_guardrail 会因 1.0 > discount_max 而丢弃, 与「无成本数据不给出价
    动作」一致; 不会让任何 0<discount<1 的让利通过)。
    """
    try:
        c = float(cost)
        pr = float(price)
        m = float(margin_floor)
    except (TypeError, ValueError):
        return 1.0
    if c <= 0 or pr <= 0:
        return 1.0
    return (c * (1.0 + m)) / pr


def apply_guardrail(inv: InventorySig, rule: Rule) -> Tuple[Optional[RoutedItem], bool]:
    """路由段对单个候选 (inv, rule) 做护栏校验并产出 RoutedItem。

    不达标返回 (None, False):
      - inv 为 None / cost<=0 (无成本数据, 防幻觉亏钱) -> 丢弃。
      - min_discount > discount_max (即使最大让利也亏) -> 丢弃。
    达标返回 (RoutedItem, True):
      - min_discount 若 < discount_min 取 discount_min (比配置更保守时取配置下限)。
      - cost_ceil = cost*(1+margin_floor) (cost_ceil<0 钳为 0)。
      - price 字段用 inv.price (CONTRACT: RoutedItem.price = 商品现售价)。
      - reason 根据 action_type 生成简短中文, 记录命中护栏口径便于老板复核。
      - rule_id / action_type / target / product / cost / margin_floor /
        priority / param_range 均由 rule 与 inv 填充, 与 CONTRACT 一致。
    """
    if inv is None:
        return (None, False)
    cost = float(getattr(inv, "cost", 0.0) or 0.0)
    price = float(getattr(inv, "price", 0.0) or 0.0)
    if cost <= 0:
        # 无成本数据不给出价动作 (方案 7.2.3 防幻觉亏钱)。
        return (None, False)

    pr = rule.param_range
    margin_floor = float(rule.margin_floor)

    min_disc = min_discount_floor(cost, price, margin_floor)
    # price<=0 时 min_disc=1.0; 只要 discount_max<1.0 (正常让利区间) 即丢弃。
    # 即使最大让利也亏 -> 丢弃。
    if min_disc > float(pr.discount_max):
        return (None, False)
    # 比配置更保守时取配置下限 (方案 7.2.3)。
    if min_disc < float(pr.discount_min):
        min_disc = float(pr.discount_min)

    cost_ceil = cost * (1.0 + margin_floor)
    if cost_ceil < 0:
        cost_ceil = 0.0

    product_name = getattr(inv, "name", "") or ""
    reason = _reason_for(rule.action_type, product_name, margin_floor, min_disc)

    item = RoutedItem(
        rule_id=rule.id,
        action_type=rule.action_type,
        target=rule.target,
        product=getattr(inv, "name", "") or "",
        cost=cost,
        price=price,
        cost_ceil=cost_ceil,
        margin_floor=margin_floor,
        priority=int(rule.priority),
        reason=reason,
        param_range=pr,
    )
    return (item, True)


# === reason 文案 (按 action_type 生成简短中文, 便于老板复核) ===

_REASON_TEMPLATES: dict = {
    "clearance_coupon": "临期清仓: {product} 护栏通过(毛利下限{mf:.0%},折扣下限{md:.2f})",
    "bundle_slow": "滞销捆绑: {product} 护栏通过(毛利下限{mf:.0%},折扣下限{md:.2f})",
    "rain_delivery": "雨天秒杀: {product} 护栏通过(毛利下限{mf:.0%},折扣下限{md:.2f})",
    "high_value_push": "高毛利推送: {product} 护栏通过(毛利下限{mf:.0%},折扣下限{md:.2f})",
}

# 通用模板: 不含 {product} (未知 action_type 时不臆测商品语境), 仅记 rule 上下文。
_REASON_FALLBACK: str = "命中规则{rid}: {at} 护栏通过(毛利下限{mf:.0%},折扣下限{md:.2f})"


def _reason_for(
    action_type: str, product_name: str, margin_floor: float, min_disc: float
) -> str:
    """按 action_type 生成简短中文 reason, 占位一次性填实, 不留残留 {占位}。

    - 已登记动作: 用含 {product} 的模板, 填入真实商品名 + 数值。
    - 未登记动作: 回退通用模板 (含 action_type 字面, 不臆测商品语境)。
    任何格式化异常都退化为通用模板, 保证 reason 永远是干净中文串。
    """
    fmt_args = {
        "product": product_name,
        "mf": margin_floor,
        "md": min_disc,
        "rid": "",
        "at": action_type or "",
    }
    template = _REASON_TEMPLATES.get(action_type or "", _REASON_FALLBACK)
    try:
        return template.format(**fmt_args)
    except (KeyError, IndexError, ValueError):
        try:
            return _REASON_FALLBACK.format(**fmt_args)
        except Exception:
            return "护栏通过"


__all__ = ["guardrail_ok", "min_discount_floor", "apply_guardrail"]
