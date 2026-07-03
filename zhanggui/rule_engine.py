"""规则引擎段 (对应方案 7.1.2 Routed + 7.2 Rule Router)。

RuleEngine.route(bundle) -> Routed:
  - 规则按 priority 降序 (越大越先匹配)。
  - 每条规则: 先 business 匹配 (r.business==bundle.business or r.business=='*');
    再检查所有 conds AND 命中 (_match_cond, 字段路径解析如
    'weather.condition' / 'inventory.days_to_expiry' / 'members.sleep_30' /
    'holiday.days_to', op eq/lt/gt/gte/between/in)。
  - 命中后按 action_type 决定关联方式:
      * 关联具体商品的 (clearance_coupon / rain_delivery / bundle_slow / high_value_push):
        对命中规则的 inventory 项逐一调 guardrail.apply_guardrail, 通过的产出 RoutedItem。
      * 不关联具体商品的 (wake_dormant / festival_discount / birthday_care):
        product='-', cost/price/cost_ceil=0, margin_floor 取 rule.margin_floor。
  - 特殊分支: op='in' & value=='nonempty' 表示对应 list 字段非空
    (members.birthday_today / members.high_value)。
  - 每条规则最多产出一组 RoutedItem; 多规则可叠加 (五段管线后续收敛 Top3)。

严格遵循 CONTRACT (zhanggui/types.py): Routed/RoutedItem/ParamRange/Rule/Cond。
零第三方依赖, 仅标准库。
"""

from typing import Any, List, Optional

from .guardrail import apply_guardrail
from .rules_seed import default_rules
from .types import (
    Cond,
    InventorySig,
    MemberSig,
    ParamRange,
    Routed,
    RoutedItem,
    Rule,
    SignalBundle,
)


# 关联具体商品的高价值动作 (每个命中 inv 项产出一条 RoutedItem)。
_INVENTORY_LINKED_ACTIONS = {
    "clearance_coupon",
    "rain_delivery",
    "bundle_slow",
    "high_value_push",
}


class RuleEngine:
    """规则路由器: SignalBundle -> Routed。"""

    def __init__(self, rules: List[Rule]):
        """rules 按 priority 降序 (越大越先匹配); None -> 默认种子规则。"""
        if rules is None:
            rules = default_rules()
        # 过滤掉未启用规则 (后台可视化关闭的规则不参与匹配)。
        enabled = [r for r in rules if getattr(r, "enable", True)]
        # 越大越先, 稳定排序 (priority 相同时保持原顺序, Python sort 稳定)。
        self.rules: List[Rule] = sorted(enabled, key=lambda r: -int(r.priority))

    # ===== 路由主流程 =====

    def route(self, bundle: SignalBundle) -> Routed:
        """对 bundle 逐规则匹配, 收集所有命中+护栏通过的 RoutedItem。

        单条规则匹配失败不影响其他规则; 单个 inv 项护栏不通过被丢弃但不停。
        bundle 为 None 时返回空 Routed (tenant_id 用 0 占位, 上游应保证非空)。
        """
        if bundle is None:
            return Routed(tenant_id=0, items=[])
        items: List[RoutedItem] = []
        for rule in self.rules:
            if not self._business_match(rule.business, bundle.business):
                continue
            if not self._conds_all_match(rule.conds, bundle):
                continue
            items.extend(self._emit_items(rule, bundle))
        return Routed(tenant_id=bundle.tenant_id, items=items)

    # ===== business 匹配 =====

    @staticmethod
    def _business_match(rule_business: str, bundle_business: str) -> bool:
        """r.business=='*' 命中所有业态; 否则精确匹配 (大小写不敏感, 去空白)。"""
        if not rule_business:
            return True
        rb = rule_business.strip().lower()
        if rb == "*":
            return True
        return rb == str(bundle_business or "").strip().lower()

    # ===== conds AND 全匹配 =====

    def _conds_all_match(self, conds: List[Cond], bundle: SignalBundle) -> bool:
        """所有 Cond AND 命中才返回 True; 任一 False 即 False。空 conds 视为命中。"""
        if not conds:
            return True
        for cond in conds:
            if not self._match_cond(cond, bundle):
                return False
        return True

    def _match_cond(self, cond: Cond, bundle: SignalBundle) -> bool:
        """单 Cond 匹配: 字段路径解析 + op 求值。

        字段路径分三段前缀:
          - 'weather.*'   : bundle.weather 的标量字段。
          - 'holiday.*'   : bundle.holiday 的标量字段。
          - 'members.*'   : bundle.members 的标量/列表字段 (sleep_30 标量; high_value/
                            birthday_today 列表, 用 in/nonempty 判非空)。
          - 'inventory.*' : bundle.inventory 列表, 任一项命中即算该 Cond 命中
                            (days_to_expiry/slow_moving/high_margin 等)。
        op:
          eq / lt / gt / gte / between(value=[lo,hi]) / in
          in + value=='nonempty' = 列表非空 (特殊分支)。
        """
        field = (cond.field or "").strip()
        op = (cond.op or "").strip().lower()
        value = cond.value

        if not field:
            return False

        head, _, tail = field.partition(".")
        head = head.lower()
        tail = tail.lower()

        # --- members.* : 单例对象, 标量或列表字段 ---
        if head == "members":
            return self._match_members_cond(bundle.members, tail, op, value)

        # --- weather.* : 单例标量字段 ---
        if head == "weather":
            actual = self._get_scalar(getattr(bundle, "weather", None), tail)
            return self._compare_scalar(actual, op, value)

        # --- holiday.* : 单例标量字段 ---
        if head == "holiday":
            actual = self._get_scalar(getattr(bundle, "holiday", None), tail)
            return self._compare_scalar(actual, op, value)

        # --- inventory.* : 列表, 任一项命中即命中 ---
        if head == "inventory":
            return self._match_inventory_any(bundle.inventory, tail, op, value)

        # 未知字段前缀: 不命中 (保守, 避免幻觉)。
        return False

    # ===== members 字段匹配 =====

    def _match_members_cond(
        self, members: Optional[MemberSig], tail: str, op: str, value: Any
    ) -> bool:
        """members.{tail} 匹配。

        tail:
          - sleep_30: 标量 int, 走 _compare_scalar (gt/gte/lt/eq/between)。
          - high_value / birthday_today: list[int]; op='in'&value=='nonempty' 判非空。
        """
        if members is None:
            return False
        if tail == "sleep_30":
            return self._compare_scalar(getattr(members, "sleep_30", 0), op, value)
        if tail in ("high_value", "birthday_today"):
            lst = getattr(members, tail, None) or []
            if op == "in" and value == "nonempty":
                return len(lst) > 0
            # 也允许 in + 具体值列表 (元素是否在其中), 与方案 op='in' 语义一致。
            if op == "in":
                try:
                    return any(v in (value or []) for v in lst)
                except TypeError:
                    return False
            # 其他 op 退化为「按列表长度当标量」意义不大, 这里对 nonempty 兜底。
            if op == "eq" and value == "nonempty":
                return len(lst) > 0
            return False
        return False

    # ===== inventory 列表「任一命中」匹配 =====

    def _match_inventory_any(
        self, inventory: List[InventorySig], tail: str, op: str, value: Any
    ) -> bool:
        """inventory.{tail}: 列表中任一项的该字段命中即算该 Cond 命中。

        例如 days_to_expiry between [1,3]: 任一临期项在 [1,3] 即命中;
        slow_moving eq True / high_margin eq True 同理。
        列表为空 -> False (没有可命中的项)。
        """
        if not inventory:
            return False
        for inv in inventory:
            actual = self._get_scalar(inv, tail)
            if self._compare_scalar(actual, op, value):
                return True
        return False

    # ===== 标量取值与比较 =====

    @staticmethod
    def _get_scalar(obj: Any, field: str) -> Any:
        """从对象取字段值; obj 为 None 或字段不存在 -> None。"""
        if obj is None or not field:
            return None
        return getattr(obj, field, None)

    def _compare_scalar(self, actual: Any, op: str, value: Any) -> bool:
        """op 求值: eq/lt/gt/gte/between/in。非数比较时做安全降级。

        between: value=[lo,hi], lo<=actual<=hi。
        in: value 为 list/tuple, actual in value。
        """
        if op == "eq":
            return actual == value
        if op == "in":
            # in 通用语义: actual 是否落在 value 列表里 (nonempty 分支已在 members 处理)。
            try:
                return actual in (value or [])
            except TypeError:
                return False
        if op == "between":
            lo, hi = self._unpack_pair(value)
            if lo is None or hi is None:
                return False
            try:
                a = float(actual)
                return lo <= a <= hi
            except (TypeError, ValueError):
                return False
        # 数值比较 op: lt/gt/gte
        try:
            a = float(actual)
            v = float(value)
        except (TypeError, ValueError):
            return False
        if op == "lt":
            return a < v
        if op == "gt":
            return a > v
        if op == "gte":
            return a >= v
        if op == "lte":
            return a <= v
        return False

    @staticmethod
    def _unpack_pair(value: Any):
        """between 的 value 拆成 (lo, hi); 非法 -> (None, None)。"""
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None, None
        return None, None

    # ===== 命中后产出 RoutedItem =====

    def _emit_items(self, rule: Rule, bundle: SignalBundle) -> List[RoutedItem]:
        """规则命中后, 按 action_type 决定产出方式。

        - 关联具体商品的高价值动作: 对命中 inv 项逐一 apply_guardrail, 通过的产出。
        - 不关联具体商品的动作: 单条占位 RoutedItem (product='-', cost/price/cost_ceil=0)。
        """
        action_type = rule.action_type
        if action_type in _INVENTORY_LINKED_ACTIONS:
            return self._emit_inventory_linked(rule, bundle)
        return self._emit_placeholder(rule, bundle)

    def _emit_inventory_linked(
        self, rule: Rule, bundle: SignalBundle
    ) -> List[RoutedItem]:
        """对命中的 inventory 项逐一过护栏, 通过的产出 RoutedItem。

        仅取「真正命中该规则 cond 的 inv 子集」过护栏, 避免把无关商品也推。
        例如 clearance 只对 days_to_expiry in [1,3] 的项; high_value_push 只对
        high_margin=True 的项; rain_delivery / bundle_slow 取相应命中项。
        """
        hit_items = self._select_hit_inventory(rule, bundle.inventory)
        out: List[RoutedItem] = []
        for inv in hit_items:
            item, ok = apply_guardrail(inv, rule)
            if ok and item is not None:
                out.append(item)
        return out

    def _select_hit_inventory(
        self, rule: Rule, inventory: List[InventorySig]
    ) -> List[InventorySig]:
        """从 inventory 中挑出命中该规则 inventory.* conds 的项。

        策略: 收集规则里所有 inventory.* 前缀 cond; 一个 inv 项需命中全部这些
        inventory conds 才入选 (AND)。规则若无 inventory cond (理论上
        linked 动作不应出现), 退化为全部 inventory。
        """
        inv_conds = [
            c for c in rule.conds
            if (c.field or "").strip().lower().startswith("inventory.")
        ]
        if not inv_conds:
            return list(inventory or [])
        selected: List[InventorySig] = []
        for inv in inventory or []:
            if all(
                self._match_inventory_any([inv], (c.field.split(".", 1)[1] or "").lower(),
                                           (c.op or "").lower(), c.value)
                for c in inv_conds
            ):
                selected.append(inv)
        return selected

    def _emit_placeholder(self, rule: Rule, bundle: SignalBundle) -> List[RoutedItem]:
        """不关联具体商品的动作: 单条占位 RoutedItem。

        product='-', cost/price/cost_ceil=0, margin_floor 取 rule.margin_floor,
        priority 取 rule.priority, param_range 取 rule.param_range。
        这些动作 (wake_dormant/festival_discount/birthday_care) 是满减/积分类,
        不绑定单个 SKU, 毛利在 action 级别已约束 (满减限额 reduce_max),
        不走商品级 guardrail_ok。
        """
        item = RoutedItem(
            rule_id=rule.id,
            action_type=rule.action_type,
            target=rule.target,
            product="-",
            cost=0.0,
            price=0.0,
            cost_ceil=0.0,
            margin_floor=float(rule.margin_floor),
            priority=int(rule.priority),
            reason="命中规则{rid}: {at} (非商品绑定动作,满减已限)".format(
                rid=rule.id, at=rule.action_type
            ),
            param_range=rule.param_range,
        )
        return [item]
