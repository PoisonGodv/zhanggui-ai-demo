"""五段式管线集成测试 (纯标准库 unittest, 不用 pytest)。

覆盖 Pipeline.generate_today_advice 端到端:
  - test_fruit_scenario_produces_advice: 水果雨天临期 -> 非空 <=3, copy<=40,
    clearance/rain 类动作 margin_ok==True。
  - test_maternity_scenario: 母婴沉睡唤醒 -> 产出 wake_dormant 或 festival_discount。
  - test_budget_exceeded_degrades: AgentStage(MockBrain, budget_cny=0.0,
    monthly_cost=1.0) 超预算 -> is_budget_exceeded()==True, enhance_or_skip 仍返回
    非空 Enhanced (降级走 MockBrain 兜底)。
"""

import unittest

from zhanggui.agent_brain import AgentStage, MockBrain
from zhanggui.converger import Converger
from zhanggui.pipeline import Pipeline
from zhanggui.rule_engine import RuleEngine
from zhanggui.rules_seed import default_rules
from zhanggui.schema_validator import SchemaValidator
from zhanggui.signal_collector import (
    SignalCollector,
    scenario_fruit_rain_expiry,
    scenario_maternity_dormant,
)


def _build_pipeline(agent_stage=None) -> Pipeline:
    """组装五段式管线; agent_stage 可注入 (默认 MockBrain, 默认预算)。"""
    if agent_stage is None:
        agent_stage = AgentStage(MockBrain(), budget_cny=2.0, monthly_cost=0.0)
    return Pipeline(
        collector=SignalCollector(),
        router=RuleEngine(default_rules()),
        agent_stage=agent_stage,
        converger=Converger(),
        validator=SchemaValidator(),
    )


class TestFruitScenario(unittest.TestCase):
    """水果店雨天临期情境端到端。"""

    def setUp(self):
        self.pipeline = _build_pipeline()
        self.advice = self.pipeline.generate_today_advice(
            1001, scenario_fruit_rain_expiry()
        )

    def test_fruit_scenario_produces_advice(self):
        adv = self.advice
        # 非空且 <=3。
        self.assertTrue(adv.items, "水果情境应产出非空建议")
        self.assertLessEqual(len(adv.items), 3)
        # 每个 copy 长度 <= 40。
        for it in adv.items:
            self.assertLessEqual(
                len(it.copy), 40,
                "copy 必须截断 <=40: {0!r}".format(it.copy),
            )
        # clearance/rain 类动作 margin_ok 必须为 True (护栏强制通过)。
        discount_actions = ("clearance_coupon", "rain_delivery",
                            "bundle_slow", "high_value_push")
        for it in adv.items:
            if it.action_type in discount_actions:
                self.assertTrue(
                    it.margin_ok,
                    "{0} 动作 margin_ok 必须为 True".format(it.action_type),
                )


class TestMaternityScenario(unittest.TestCase):
    """母婴店沉睡唤醒情境端到端。"""

    def setUp(self):
        self.pipeline = _build_pipeline()
        self.advice = self.pipeline.generate_today_advice(
            2002, scenario_maternity_dormant()
        )

    def test_maternity_scenario(self):
        adv = self.advice
        self.assertTrue(adv.items, "母婴情境应产出非空建议")
        # 至少有 wake_dormant 或 festival_discount 之一 (沉睡唤醒 / 节日满减)。
        actions = {it.action_type for it in adv.items}
        self.assertTrue(
            "wake_dormant" in actions or "festival_discount" in actions,
            "母婴情境应含 wake_dormant 或 festival_discount, 实际: {0}".format(actions),
        )


class TestBudgetExceededDegrades(unittest.TestCase):
    """预算熔断降级: budget_cny=0.0, monthly_cost=1.0 -> 超预算 -> 走 MockBrain 兜底。"""

    def test_budget_exceeded_degrades(self):
        # budget_cny=0.0, monthly_cost=1.0: monthly_cost(1.0) >= budget_cny(0.0) -> 超预算。
        stage = AgentStage(MockBrain(), budget_cny=0.0, monthly_cost=1.0)
        self.assertTrue(stage.is_budget_exceeded(9999))

        # 超预算时 enhance_or_skip 走 MockBrain 兜底, 仍应返回非空 Enhanced。
        from zhanggui.signal_collector import SignalCollector
        from zhanggui.rule_engine import RuleEngine
        from zhanggui.rules_seed import default_rules

        collector = SignalCollector()
        router = RuleEngine(default_rules())
        bundle = collector.collect(1001, scenario_fruit_rain_expiry())
        routed = router.route(bundle)
        enhanced = stage.enhance_or_skip(1001, routed, bundle)
        self.assertIsNotNone(enhanced)
        self.assertTrue(
            enhanced.items,
            "超预算降级走 MockBrain, 仍应返回非空 Enhanced (水果情境有候选)",
        )


class TestColdStartFallback(unittest.TestCase):
    """冷启动兜底 (方案 7.10.2 + 6.6): 真实新店 (无临期/无高毛利/无沉睡/
    无雨无节无生日, data_quality 低) 常规规则命中 0 条时, 应注入 cold_priority
    冷启动动作, 保证新店首日也有可用建议, 且文案合理 (不出现「秒杀-0.0元/斤
    限0份」类占位错乱), margin_ok 全 True。"""

    @staticmethod
    def _cold_scenario(business="fruit", birthday=False):
        return {
            "business": business, "city": "某地级市", "date": "2026-07-03",
            "weather": {"condition": "normal", "temp_c": 25.0, "forecast": "晴"},
            "holiday": {"name": "无", "days_to": 99, "is_festival": False},
            "inventory": [{
                "product_id": 1, "name": "普通商品", "cost": 5.0, "price": 8.0,
                "stock": 100, "days_to_expiry": -1, "slow_moving": False,
                "high_margin": False,
            }],
            "members": {
                "sleep_30": 0, "high_value": [],
                "birthday_today": [301] if birthday else [],
            },
            "data_quality": 30,
        }

    def test_cold_start_produces_advice(self):
        """无任何触发信号的新店仍应产出 >=1 条建议 (冷启动注入)。"""
        p = _build_pipeline()
        advice = p.generate_today_advice(9999, self._cold_scenario())
        self.assertTrue(
            advice.items,
            "冷启动新店应注入 cold_priority 动作, 首日有建议, 不应产出空",
        )
        # 冷启动只注入满减/关怀类 (product='-'), 文案合理无占位错乱。
        for it in advice.items:
            self.assertIn(
                it.action_type,
                ("wake_dormant", "festival_discount", "birthday_care"),
                "冷启动只注入满减/关怀类动作, 不注入商品秒杀类",
            )
            self.assertLessEqual(len(it.copy), 40)
            self.assertTrue(it.margin_ok)
            # 文案不得出现占位错乱 (0.0元/限0份/- 商品)
            self.assertNotIn("0.0", it.copy)
            self.assertNotIn("限0份", it.copy)
            self.assertNotIn("-0.0", it.copy)

    def test_cold_start_trace_recorded(self):
        """冷启动注入应在 trace 留 cold_start 段记录。"""
        p = _build_pipeline()
        p.generate_today_advice(9998, self._cold_scenario())
        stages = [s for s in p.get_trace()["stages"] if s["stage"] == "cold_start"]
        self.assertTrue(stages, "冷启动应记录 cold_start trace 段")
        self.assertTrue(stages[0]["snapshot"]["triggered"])

    def test_cold_start_birthday_injects_care(self):
        """有当天生日会员时, 冷启动应注入 birthday_care (信号准入)。"""
        p = _build_pipeline()
        advice = p.generate_today_advice(9997, self._cold_scenario(birthday=True))
        actions = {it.action_type for it in advice.items}
        self.assertIn("birthday_care", actions)


if __name__ == "__main__":
    unittest.main()
