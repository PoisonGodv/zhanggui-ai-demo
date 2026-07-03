"""信号采集段 (对应方案 7.1.3 Signal Collector + 7.1.2 signal.Bundle)。

职责: 把「商户当日数据」(模拟的 scenario dict) 解析成强类型的 SignalBundle,
供后续 Rule Router / Agent / Converger / Validator 五段管线消费。

设计要点 (忠实复刻方案 7.1.3):
- 真实系统里天气/节日/库存/会员是并发外部拉取 (errgroup 收敛, 任一失败不阻断);
  本 demo 用纯函数解析 scenario dict 代替外部 IO, 缺字段一律安全降级为默认值,
  体现「信号不全不报错、降级为可用子集」的同义语义。
- inventory 每项 dict -> InventorySig; days_to_expiry 缺省 -1 (非临期)。
- members.high_value / birthday_today 缺省为空 list。
- 提供 two 内置 scenario 工厂:
  scenario_fruit_rain_expiry()   水果店雨天临期 (阳光玫瑰葡萄临期2天 + 草莓临期1天)
  scenario_maternity_dormant()   母婴店沉睡唤醒 (金领冠奶粉高毛利 + 沉睡25人)

严格遵循 CONTRACT (zhanggui/types.py), 包内用相对 import from .types import。
"""

from typing import Any, Dict, List

from .types import (
    Holiday,
    InventorySig,
    MemberSig,
    SignalBundle,
    Weather,
)


# === 安全默认值 ===
# 缺字段时降级, 不让管线因信号不全而中断 (方案 7.1.3 降级链路)。
_DEFAULT_BUSINESS: str = "fruit"
_DEFAULT_CITY: str = "未知城市"
_DEFAULT_DATE: str = "1970-01-01"
_DEFAULT_DATA_QUALITY: int = 60   # 缺省中位数, 不触发极低阈值降级

_DEFAULT_WEATHER_CONDITION: str = "normal"
_DEFAULT_WEATHER_TEMP: float = 20.0
_DEFAULT_WEATHER_FORECAST: str = ""

_DEFAULT_HOLIDAY_NAME: str = "无"
_DEFAULT_HOLIDAY_DAYS_TO: int = -1
_DEFAULT_HOLIDAY_IS_FESTIVAL: bool = False

_DEFAULT_INV_PRODUCT_ID: int = 0
_DEFAULT_INV_COST: float = 0.0
_DEFAULT_INV_PRICE: float = 0.0
_DEFAULT_INV_STOCK: float = 0.0
_DEFAULT_INV_DAYS_TO_EXPIRY: int = -1   # -1 = 非临期
_DEFAULT_INV_SLOW_MOVING: bool = False
_DEFAULT_INV_HIGH_MARGIN: bool = False

_DEFAULT_MEMBERS_SLEEP_30: int = 0


def _as_float(value: Any, default: float) -> float:
    """安全转 float: None / 非数 -> default; bool 不当数值用 (bool 是 int 子类, 需排除)。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> bool:  # type: ignore[override]
    """安全转 int: None / 非数 -> default; bool 排除; float 取整 (临期天数等)。"""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _as_bool(value: Any, default: bool) -> bool:
    """安全转 bool: None -> default; bool 直取; 0/1/字符串 'true'/'false' 解析; 其余 default。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f", ""):
            return False
    return default


def _as_str(value: Any, default: str) -> str:
    """安全转 str: None -> default; 非字符串强制 str() 后若为空回 default。"""
    if value is None:
        return default
    if isinstance(value, str):
        return value if value != "" else default
    try:
        s = str(value)
        return s if s != "" else default
    except Exception:
        return default


def _as_int_list(value: Any) -> List[int]:
    """安全转 int list: None/非 list -> []; 列表元素逐一安全转 int, 失败丢弃。"""
    if not isinstance(value, (list, tuple)):
        return []
    out: List[int] = []
    for item in value:
        if item is None or isinstance(item, bool):
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            try:
                out.append(int(float(item)))
            except (TypeError, ValueError):
                continue
    return out


class SignalCollector:
    """信号采集器: scenario dict -> SignalBundle。"""

    def collect(self, tenant_id: int, scenario: Dict[str, Any]) -> SignalBundle:
        """解析模拟商户当日数据 scenario dict, 产出强类型 SignalBundle。

        Args:
            tenant_id: 租户 id。
            scenario: 商户当日数据 dict, 字段见模块 docstring
                (business / city / date / weather(dict) / holiday(dict)
                 / inventory(list[dict]) / members(dict) / data_quality(int))。
                任一字段缺失均用安全默认值, 不抛异常。

        Returns:
            SignalBundle, inventory 为 list[InventorySig], members 为 MemberSig。
        """
        if not isinstance(scenario, dict):
            scenario = {}

        business = _as_str(scenario.get("business"), _DEFAULT_BUSINESS)
        city = _as_str(scenario.get("city"), _DEFAULT_CITY)
        date = _as_str(scenario.get("date"), _DEFAULT_DATE)
        data_quality = _clamp(
            _as_int(scenario.get("data_quality"), _DEFAULT_DATA_QUALITY),
            0, 100,
        )

        weather = self._parse_weather(scenario.get("weather"))
        holiday = self._parse_holiday(scenario.get("holiday"))
        inventory = self._parse_inventory(scenario.get("inventory"))
        members = self._parse_members(scenario.get("members"))

        return SignalBundle(
            tenant_id=tenant_id,
            business=business,
            city=city,
            date=date,
            weather=weather,
            holiday=holiday,
            inventory=inventory,
            members=members,
            data_quality=data_quality,
        )

    # ---- 子结构解析 (缺字段安全默认) ----

    def _parse_weather(self, raw: Any) -> Weather:
        """weather dict -> Weather。condition 仅取已知枚举之一, 否则降级 normal。"""
        if not isinstance(raw, dict):
            raw = {}
        condition = _as_str(raw.get("condition"), _DEFAULT_WEATHER_CONDITION)
        # 收敛到合法枚举, 未知天气降级为 normal, 避免下游规则误判。
        if condition not in ("rain", "hot", "cold", "snow", "normal"):
            condition = _DEFAULT_WEATHER_CONDITION
        temp_c = _as_float(raw.get("temp_c"), _DEFAULT_WEATHER_TEMP)
        forecast = _as_str(raw.get("forecast"), _DEFAULT_WEATHER_FORECAST)
        return Weather(condition=condition, temp_c=temp_c, forecast=forecast)

    def _parse_holiday(self, raw: Any) -> Holiday:
        """holiday dict -> Holiday。无节日时 name='无' / days_to=-1 / is_festival=False。"""
        if not isinstance(raw, dict):
            raw = {}
        name = _as_str(raw.get("name"), _DEFAULT_HOLIDAY_NAME)
        days_to = _as_int(raw.get("days_to"), _DEFAULT_HOLIDAY_DAYS_TO)
        is_festival = _as_bool(raw.get("is_festival"), _DEFAULT_HOLIDAY_IS_FESTIVAL)
        return Holiday(name=name, days_to=days_to, is_festival=is_festival)

    def _parse_inventory(self, raw: Any) -> List[InventorySig]:
        """inventory list[dict] -> list[InventorySig]。非 list 或空 -> []。"""
        if not isinstance(raw, (list, tuple)):
            return []
        items: List[InventorySig] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            items.append(
                InventorySig(
                    product_id=_as_int(item.get("product_id"), _DEFAULT_INV_PRODUCT_ID),
                    name=_as_str(item.get("name"), ""),
                    cost=_as_float(item.get("cost"), _DEFAULT_INV_COST),
                    price=_as_float(item.get("price"), _DEFAULT_INV_PRICE),
                    stock=_as_float(item.get("stock"), _DEFAULT_INV_STOCK),
                    days_to_expiry=_as_int(
                        item.get("days_to_expiry"), _DEFAULT_INV_DAYS_TO_EXPIRY
                    ),
                    slow_moving=_as_bool(item.get("slow_moving"), _DEFAULT_INV_SLOW_MOVING),
                    high_margin=_as_bool(item.get("high_margin"), _DEFAULT_INV_HIGH_MARGIN),
                )
            )
        return items

    def _parse_members(self, raw: Any) -> MemberSig:
        """members dict -> MemberSig。缺字段 -> sleep_30=0 / 高价值&生日为空。"""
        if not isinstance(raw, dict):
            raw = {}
        sleep_30 = _as_int(raw.get("sleep_30"), _DEFAULT_MEMBERS_SLEEP_30)
        if sleep_30 < 0:
            sleep_30 = 0
        high_value = _as_int_list(raw.get("high_value"))
        birthday_today = _as_int_list(raw.get("birthday_today"))
        return MemberSig(
            sleep_30=sleep_30,
            high_value=high_value,
            birthday_today=birthday_today,
        )


def _clamp(value: int, low: int, high: int) -> int:
    """把 data_quality 钳制到 [low, high] (方案要求 0-100)。"""
    if value < low:
        return low
    if value > high:
        return high
    return value


# === 内置 scenario 工厂 (模块级) ===
# 演示入口 run_demo.py 直接调用, 对应方案 7.4.2 / 7.4.3 两类典型情境。

def scenario_fruit_rain_expiry() -> Dict[str, Any]:
    """水果店: 雨天 + 临期清仓情境。

    - business=fruit, city=某地级市, date=今日, weather rain。
    - holiday 无。
    - inventory: 阳光玫瑰葡萄(cost12/price25.8/stock50/days_to_expiry=2) + 草莓(临期1天)。
    - members: sleep_30=18, high_value=[101,102,103], birthday_today=[]。
    - data_quality=85。

    与方案 7.4.2 user prompt 示例一致 (阳光玫瑰 cost12/price25.8/stock50/临期2天)。
    """
    return {
        "business": "fruit",
        "city": "临沂",
        "date": "2026-07-03",
        "weather": {
            "condition": "rain",
            "temp_c": 22.0,
            "forecast": "小雨转阴,夜间有阵雨",
        },
        "holiday": {
            "name": "无",
            "days_to": -1,
            "is_festival": False,
        },
        "inventory": [
            {
                "product_id": 1001,
                "name": "阳光玫瑰葡萄",
                "cost": 12.0,
                "price": 25.8,
                "stock": 50.0,
                "days_to_expiry": 2,
                "slow_moving": False,
                "high_margin": True,
            },
            {
                "product_id": 1002,
                "name": "奶油草莓",
                "cost": 8.0,
                "price": 19.9,
                "stock": 30.0,
                "days_to_expiry": 1,
                "slow_moving": False,
                "high_margin": False,
            },
        ],
        "members": {
            "sleep_30": 18,
            "high_value": [101, 102, 103],
            "birthday_today": [],
        },
        "data_quality": 85,
    }


def scenario_maternity_dormant() -> Dict[str, Any]:
    """母婴店: 沉睡唤醒 + 母亲节情境。

    - business=maternity, weather normal, holiday 母亲节 days_to=3 is_festival=True。
    - inventory: 金领冠奶粉(cost180/price268/stock40/high_margin=True)。
    - members: sleep_30=25, high_value=[201,202], birthday_today=[203]。
    - data_quality=80。

    对应方案 7.10.2 母婴业态默认剧本 (奶粉会员价/生日关怀/沉睡唤醒)。
    """
    return {
        "business": "maternity",
        "city": "徐州",
        "date": "2026-07-03",
        "weather": {
            "condition": "normal",
            "temp_c": 28.0,
            "forecast": "多云,体感较热",
        },
        "holiday": {
            "name": "母亲节",
            "days_to": 3,
            "is_festival": True,
        },
        "inventory": [
            {
                "product_id": 2001,
                "name": "金领冠奶粉",
                "cost": 180.0,
                "price": 268.0,
                "stock": 40.0,
                "days_to_expiry": -1,
                "slow_moving": False,
                "high_margin": True,
            },
        ],
        "members": {
            "sleep_30": 25,
            "high_value": [201, 202],
            "birthday_today": [203],
        },
        "data_quality": 80,
    }
