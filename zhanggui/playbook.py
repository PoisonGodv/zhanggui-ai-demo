"""冷启动剧本库 (对应方案 7.10.1-7.10.2)。

PLAYBOOKS: 业态 -> 剧本 dict, 每业态含:
  - loss_rate       损耗率默认 (水果0.10/便利0.015/母婴0.02/生鲜0.10/服装0.03)
  - cash_coef       现金系数 (客流反推, 默认1.2, 服装1.3)
  - cold_priority   冷启动期优先 action_type 列表 (不依赖历史销售反推)
  - copy_templates  action_type -> <=40 字文案模板
                    含 {product} {price} {qty} {spend} {reduce} {hours} {discount}
                    {member_tag} {weather_emoji} 等占位。

冷启动期 (前2-4周) 只推 cold_priority 中不依赖销售反推的动作; 新店入驻只问
3 问 (业态/城市/主推品) 即套对应 Playbook, 第一天就有可用建议。

严格遵循 CONTRACT: 不引入新类型, 仅作配置 dict; 文案模板字数 <=40。
零第三方依赖。
"""

from typing import Any, Dict, List

# === 5 业态 key (与 signal_collector 的 business 字段一致) ===
# fruit / convenience / maternity / fresh / apparel (方案 7.1.2 business_type)
BUSINESSES: List[str] = ["fruit", "convenience", "maternity", "fresh", "apparel"]


def _tmpl(s: str) -> str:
    """断言文案模板 <=40 字 (中文每字1, Python str 字符数)。

    超长在导入期直接抛错, 避免运行时产出超长文案被 schema_validator 截断丢词。
    占位符 {product} 等按 1 字宽保守计 (实际填充后可能更长, 运行时仍由
    validator 二次截断到 <=40, 此处只卡模板基线)。
    """
    assert len(s) <= 40, "copy template too long (>40): {s!r}".format(s=s)
    return s


PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    # === 水果 fruit: 临期3天清仓/雨天秒杀/沉睡唤醒/节日满减 ===
    # 损耗率0.10 / 现金系数1.2 / 文案「今晚7点群秒杀{product}{price}/斤 限{qty}份」
    "fruit": {
        "loss_rate": 0.10,
        "cash_coef": 1.2,
        "cold_priority": [
            "rain_delivery",
            "festival_discount",
            "wake_dormant",
            "high_value_push",
        ],
        "copy_templates": {
            "clearance_coupon": _tmpl("临期清仓{product}{price}/份 限{qty}份先到先得"),
            "bundle_slow": _tmpl("滞销{product}搭畅销款 捆绑价{price}不单卖"),
            "rain_delivery": _tmpl("雨天群秒杀{product}{price}/斤 限{qty}份"),
            "wake_dormant": _tmpl("好久不见 满{spend}减{reduce} {hours}h内到店可用"),
            "festival_discount": _tmpl("节日满{spend}减{reduce} 全店主题好货速抢"),
            "birthday_care": _tmpl("生日快乐 满{spend}减{reduce} 生日月双倍积分"),
            "high_value_push": _tmpl("{product}会员价{price} {member_tag}专享速囤"),
        },
    },

    # === 便利 convenience: 滞销捆绑/高频复购/烟酒高毛利push ===
    # 损耗率0.015 / 现金系数1.2 / 文案「{product}满{spend}减{reduce},{hours}h内到店可用」
    "convenience": {
        "loss_rate": 0.015,
        "cash_coef": 1.2,
        "cold_priority": [
            "festival_discount",
            "high_value_push",
            "wake_dormant",
        ],
        "copy_templates": {
            "clearance_coupon": _tmpl("临期{product}{price}/份 限{qty}份速抢"),
            "bundle_slow": _tmpl("滞销{product}搭畅销款 捆绑价{price}"),
            "rain_delivery": _tmpl("雨天满{spend}送上门 {product}{price}/份"),
            "wake_dormant": _tmpl("{product}满{spend}减{reduce} {hours}h内到店可用"),
            "festival_discount": _tmpl("节日满{spend}减{reduce} 速来囤货"),
            "birthday_care": _tmpl("生日满{spend}减{reduce} 双倍积分送您"),
            "high_value_push": _tmpl("烟酒高毛利{product}会员价{price} 专享"),
        },
    },

    # === 母婴 maternity: 奶粉28天复购/生日关怀/储值活动 ===
    # 损耗率0.02 / 现金系数1.2 / 文案「宝宝该补货啦~{product}会员价{price},{member_tag}专享」
    "maternity": {
        "loss_rate": 0.02,
        "cash_coef": 1.2,
        "cold_priority": [
            "birthday_care",
            "high_value_push",
            "wake_dormant",
        ],
        "copy_templates": {
            "clearance_coupon": _tmpl("临期{product}{price}/份 限{qty}份"),
            "bundle_slow": _tmpl("滞销{product}搭畅销款 捆绑{price}"),
            "rain_delivery": _tmpl("雨天满{spend}送上门 {product}{price}"),
            "wake_dormant": _tmpl("宝宝该补货啦 满{spend}减{reduce} {hours}h内可用"),
            "festival_discount": _tmpl("节日满{spend}减{reduce} 母婴好货速囤"),
            "birthday_care": _tmpl("宝宝生日快乐 满{spend}减{reduce}双倍积分"),
            "high_value_push": _tmpl("宝宝该补货啦~{product}会员价{price}专享"),
        },
    },

    # === 生鲜 fresh: 临期清仓/雨天配送/拼团裂变 ===
    # 损耗率0.10 / 现金系数1.2 / 文案「本小区满68免费送上门{product}{price}/份」
    "fresh": {
        "loss_rate": 0.10,
        "cash_coef": 1.2,
        "cold_priority": [
            "rain_delivery",
            "clearance_coupon",
            "wake_dormant",
        ],
        "copy_templates": {
            "clearance_coupon": _tmpl("临期{product}{price}/份 限{qty}份"),
            "bundle_slow": _tmpl("滞销{product}搭畅销款 捆绑{price}"),
            "rain_delivery": _tmpl("本小区满68免费送上门 {product}{price}/份"),
            "wake_dormant": _tmpl("满{spend}减{reduce} {hours}h内到店可用"),
            "festival_discount": _tmpl("节日满{spend}减{reduce} 速抢"),
            "birthday_care": _tmpl("生日满{spend}减{reduce} 双倍积分"),
            "high_value_push": _tmpl("{product}会员价{price} {member_tag}专享"),
        },
    },

    # === 服装 apparel: 季末清仓/换季满减/老带新裂变 ===
    # 损耗率0.03 / 现金系数1.3 / 文案「换季清仓{product}全场{discount}折,老带新再减{reduce}」
    "apparel": {
        "loss_rate": 0.03,
        "cash_coef": 1.3,
        "cold_priority": [
            "festival_discount",
            "clearance_coupon",
            "high_value_push",
        ],
        "copy_templates": {
            "clearance_coupon": _tmpl("季末清仓{product}全场{discount}折 限{qty}件"),
            "bundle_slow": _tmpl("滞销{product}搭畅销款 捆绑{price}"),
            "rain_delivery": _tmpl("雨天满{spend}送上门 {product}{price}"),
            "wake_dormant": _tmpl("老客满{spend}减{reduce} {hours}h内到店可用"),
            "festival_discount": _tmpl("换季清仓{product}全场{discount}折,老带新再减{reduce}"),
            "birthday_care": _tmpl("生日满{spend}减{reduce} 双倍积分"),
            "high_value_push": _tmpl("{product}会员价{price} {member_tag}专享"),
        },
    },
}


def get_playbook(business: str) -> Dict[str, Any]:
    """取业态剧本; 未知业态回退到 fruit (损耗最高、最贴近个体零售默认)。"""
    if business in PLAYBOOKS:
        return PLAYBOOKS[business]
    return PLAYBOOKS["fruit"]


def copy_template(business: str, action_type: str) -> str:
    """取 (业态, action_type) 的文案模板; 缺失回退通用模板。

    回退模板同样 <=40 字, 交给下游 agent_brain / schema_validator 填充与截断。
    """
    pb = get_playbook(business)
    templates = pb.get("copy_templates", {})
    if action_type in templates:
        return templates[action_type]
    # 通用回退: 含 {product} {action_type}, <=40 字。
    return "今日推荐{product} 速来抢购"
