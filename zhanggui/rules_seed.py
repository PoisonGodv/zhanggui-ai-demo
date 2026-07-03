"""规则种子库 (对应方案 7.2.1-7.2.2)。

default_rules() 返回冷启动默认 7 条规则, 忠实复刻方案 7.2.2 映射表:

  1. 临期清仓 (clearance_coupon)        fruit, between 1,3, 折扣[0.5,0.7]/限份<=50, 毛利5%
  2. 滞销捆绑 (bundle_slow)             fruit, slow_moving=true, 让利<=8%不单卖, 毛利8%
  3. 雨天外卖 (rain_delivery)           fruit+fresh, weather=rain, 满68送上门/限份<=50, 毛利5%
  4. 沉睡唤醒 (wake_dormant)            *, sleep_30>=10, 满30减5/48h, 毛利0%
  5. 节日满减 (festival_discount)       *, days_to 0,7 & is_festival, 满99减[5,15], 毛利5%
  6. 生日关怀 (birthday_care)           *, birthday_today 非空, 满50减8/双倍积分, 毛利0%
  7. 高毛利push (high_value_push)      *, high_margin=true & high_value 非空, 让利<=10%, 毛利15%

特殊分支: Cond op='in' value='nonempty' 表示对应 list 字段非空
  (birthday_today / high_value); _match_cond 在 rule_engine.py 里实现该分支。
  (high_value_push 触发需 members.high_value 非空, 用 in/nonempty 表达)

严格遵循 CONTRACT (zhanggui/types.py) 的 Rule / Cond / ParamRange dataclass。
零第三方依赖。
"""

from typing import List

from .types import Cond, ParamRange, Rule


def default_rules() -> List[Rule]:
    """冷启动默认规则集 (7 条), 命中 7.2.2 映射表。

    priority 越大越先匹配; 高毛利push / 节日 / 临期 给较高优先级。
    enable 全部 True (种子规则默认开启, 后台可关)。
    """
    return [
        # 7. 高毛利push: 高毛利品->高价值会员, 让利<=10%, 毛利15% (高优先级, 高毛利品最该推)
        Rule(
            id="high_value_push_default",
            business="*",
            priority=70,
            enable=True,
            conds=[
                Cond("inventory.high_margin", "eq", True),
                Cond("members.high_value", "in", "nonempty"),
            ],
            action_type="high_value_push",
            target="高毛利品->高价值会员",
            param_range=ParamRange(
                discount_min=0.90,
                discount_max=0.90,   # 让利<=10% (会员专享价)
                min_spend=0.0,
                reduce_max=0.0,
                limit_qty_max=0,
                expire_hours=0,
            ),
            margin_floor=0.15,
        ),
        # 5. 节日满减: days_to 0..7 & is_festival, 满99减[5,15], 毛利5%
        Rule(
            id="festival_discount_default",
            business="*",
            priority=60,
            enable=True,
            conds=[
                Cond("holiday.days_to", "between", [0, 7]),
                Cond("holiday.is_festival", "eq", True),
            ],
            action_type="festival_discount",
            target="全店主题满减",
            param_range=ParamRange(
                discount_min=0.0,
                discount_max=0.0,
                min_spend=99.0,
                reduce_max=15.0,    # 满99减[5,15], reduce_max 取上限
                limit_qty_max=0,
                expire_hours=0,
            ),
            margin_floor=0.05,
        ),
        # 1. 临期清仓: days_to_expiry 1..3, 折扣[0.5,0.7]/限份<=50, 毛利5%
        Rule(
            id="clearance_coupon_fruit",
            business="fruit",
            priority=50,
            enable=True,
            conds=[
                Cond("inventory.days_to_expiry", "between", [1, 3]),
            ],
            action_type="clearance_coupon",
            target="临期SKU",
            param_range=ParamRange(
                discount_min=0.5,
                discount_max=0.7,
                min_spend=0.0,
                reduce_max=0.0,
                limit_qty_max=50,
                expire_hours=0,
            ),
            margin_floor=0.05,
        ),
        # 3. 雨天外卖: weather=rain, 群内秒杀+满68送上门/限份<=50, 毛利5% (fruit+fresh; demo 用 fruit 即可命中)
        #    「秒杀」含商品折扣, 故给折扣区间[0.7,0.9](比临期清仓[0.5,0.7]保守, 雨天秒杀不亏本);
        #    关联到具体秒杀商品(apply_guardrail 过护栏), discount_max=0 会被护栏误判丢弃,故必须给正区间。
        Rule(
            id="rain_delivery_fresh",
            business="fruit",   # 方案 7.2.2 为 fruit+fresh, demo 业态用 fruit 覆盖主路径
            priority=40,
            enable=True,
            conds=[
                Cond("weather.condition", "eq", "rain"),
            ],
            action_type="rain_delivery",
            target="群内秒杀+本小区配送",
            param_range=ParamRange(
                discount_min=0.7,
                discount_max=0.9,   # 雨天秒杀折扣 7-9 折 (毛利5%护栏内)
                min_spend=68.0,     # 满68送上门
                reduce_max=0.0,
                limit_qty_max=50,
                expire_hours=0,
            ),
            margin_floor=0.05,
        ),
        # 2. 滞销捆绑: slow_moving=true, 让利<=8%不单卖, 毛利8%
        Rule(
            id="bundle_slow_fruit",
            business="fruit",
            priority=30,
            enable=True,
            conds=[
                Cond("inventory.slow_moving", "eq", True),
            ],
            action_type="bundle_slow",
            target="滞销+畅销捆绑",
            param_range=ParamRange(
                discount_min=0.92,
                discount_max=0.92,   # 捆绑价让利<=8% (不单卖)
                min_spend=0.0,
                reduce_max=0.0,
                limit_qty_max=0,
                expire_hours=0,
            ),
            margin_floor=0.08,
        ),
        # 4. 沉睡唤醒: sleep_30>=10, 满30减5/48h, 毛利0%
        Rule(
            id="wake_dormant_default",
            business="*",
            priority=20,
            enable=True,
            conds=[
                Cond("members.sleep_30", "gte", 10),
            ],
            action_type="wake_dormant",
            target="30天未到店高价值会员",
            param_range=ParamRange(
                discount_min=0.0,
                discount_max=0.0,
                min_spend=30.0,
                reduce_max=5.0,    # 满30减5
                limit_qty_max=0,
                expire_hours=48,   # 48h 有效
            ),
            margin_floor=0.0,
        ),
        # 6. 生日关怀: birthday_today 非空, 满50减8/双倍积分, 毛利0%
        Rule(
            id="birthday_care_default",
            business="*",
            priority=10,
            enable=True,
            conds=[
                Cond("members.birthday_today", "in", "nonempty"),
            ],
            action_type="birthday_care",
            target="当天生日会员",
            param_range=ParamRange(
                discount_min=0.0,
                discount_max=0.0,
                min_spend=50.0,
                reduce_max=8.0,    # 生日券满50减8
                limit_qty_max=0,
                expire_hours=0,    # 生日月内有效 (双倍积分)
            ),
            margin_floor=0.0,
        ),
    ]
