"""掌柜军师 AI 引擎 —— 核心数据结构 (CONTRACT 权威定义)。

零第三方依赖, 仅使用标准库 dataclasses/typing。
所有字段严格遵循 07_AI引擎详设.md 的五段式管线契约:
  SignalBundle -> Routed -> Enhanced -> (top3) -> Advice

每个 dataclass 附带 to_dict() 方法 (委托 dataclasses.asdict, 递归展开嵌套
dataclass / list / dict), 便于 demo 打印与 trace 快照序列化。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional


class _AsDictMixin:
    """给所有 dataclass 提供 to_dict(): 委托 dataclasses.asdict 递归转 dict。

    asdict 会递归拷贝 dataclass 实例 / dict / list / tuple, 标量原样返回,
    天然适配本模块的嵌套结构 (Rule 含 ParamRange+list[Cond]; SignalBundle 含
    Weather/Holiday/list[InventorySig]/MemberSig; EnhancedItem.routed 为 RoutedItem)。
    """

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# === 信号采集段 ===

@dataclass
class Weather(_AsDictMixin):
    condition: str          # rain / hot / cold / snow / normal
    temp_c: float
    forecast: str


@dataclass
class Holiday(_AsDictMixin):
    name: str
    days_to: int
    is_festival: bool


@dataclass
class InventorySig(_AsDictMixin):
    product_id: int
    name: str
    cost: float
    price: float
    stock: float
    days_to_expiry: int    # -1 = 非临期
    slow_moving: bool
    high_margin: bool


@dataclass
class MemberSig(_AsDictMixin):
    sleep_30: int
    high_value: List[int]
    birthday_today: List[int]


@dataclass
class SignalBundle(_AsDictMixin):
    tenant_id: int
    business: str
    city: str
    date: str
    weather: Weather
    holiday: Holiday
    inventory: List[InventorySig]
    members: MemberSig
    data_quality: int       # 0-100


# === 规则引擎段 ===

@dataclass
class Cond(_AsDictMixin):
    field: str
    op: str                 # eq / lt / gt / gte / between / in
    value: Any


@dataclass
class ParamRange(_AsDictMixin):
    discount_min: float
    discount_max: float
    min_spend: float = 0.0
    reduce_max: float = 0.0
    limit_qty_max: int = 0
    expire_hours: int = 0


@dataclass
class Rule(_AsDictMixin):
    id: str
    business: str           # 可为 '*'
    priority: int           # 越大越先
    enable: bool
    conds: List[Cond]
    action_type: str
    target: str
    param_range: ParamRange
    margin_floor: float


@dataclass
class RoutedItem(_AsDictMixin):
    rule_id: str
    action_type: str
    target: str
    product: str
    cost: float
    price: float
    cost_ceil: float
    margin_floor: float
    priority: int
    reason: str
    param_range: ParamRange


@dataclass
class Routed(_AsDictMixin):
    tenant_id: int
    items: List[RoutedItem]


# === Agent 段 ===

@dataclass
class EnhancedItem(_AsDictMixin):
    routed: RoutedItem
    copy: str
    params: dict
    expected_effect: str
    margin_ok: bool


@dataclass
class Enhanced(_AsDictMixin):
    tenant_id: int
    items: List[EnhancedItem]


# === 收敛段 / 输出段 ===

@dataclass
class AdviceItem(_AsDictMixin):
    rank: int
    action_type: str
    target: str
    product: str
    params: dict
    copy: str
    reason: str
    expected_effect: str
    margin_ok: bool


@dataclass
class Advice(_AsDictMixin):
    tenant_id: int
    date: str
    items: List[AdviceItem]
    version: str
