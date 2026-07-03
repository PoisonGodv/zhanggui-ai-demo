"""M3-S2/S3 集成测试 (纯标准库 unittest): 语义缓存 / 模型网关 / 凌晨预生成 /
效果归因 / bandit 自学习。

覆盖任务契约要求的 7 个用例:
  - test_cache_hit_miss        : 同信号两请求 miss+put 后 hit; 改信号 miss。
  - test_cache_invalidation    : put 后 invalidate_tenant, 再查 miss。
  - test_gateway_fallback      : lite 层失败降级 v3 (MockProvider)。
  - test_gateway_budget_exceeded: add_cost 累计 >= 2.0 -> is_budget_exceeded True。
  - test_pregen_batch          : 3 tenant batch_generate ok=3, cache 有命中。
  - test_bandit_update         : 不同 reward 软更新, weight 朝高 reward 偏移; reweight 回写 priority。
  - test_attribution_reward    : coupon_rate 正确, compute_reward 落在 [0,1]。
"""

import time
import unittest

from zhanggui.agent_brain import AgentStage, MockBrain
from zhanggui.attribution import Outcome, OutcomeCollector
from zhanggui.bandit import Arm, Bandit
from zhanggui.cache import SemanticCache
from zhanggui.converger import Converger
from zhanggui.embedding import LocalEmbedder
from zhanggui.gateway import (
    ChatMessage,
    ChatResp,
    Gateway,
    Layer,
    MockProvider,
    Provider,
)
from zhanggui.pipeline import Pipeline
from zhanggui.pregen import Scheduler
from zhanggui.rule_engine import RuleEngine
from zhanggui.rules_seed import default_rules
from zhanggui.schema_validator import SchemaValidator
from zhanggui.signal_collector import (
    SignalCollector,
    scenario_fruit_rain_expiry,
)
from zhanggui.types import Cond, ParamRange, Rule


# ===== 测试用 Provider: 可控失败/成功, 用于验证网关降级链 =====

class _FailProvider:
    """永远抛异常的 Provider (模拟超时/网络失败), 用于触发层间降级。"""

    def chat(self, messages, tools=None, timeout=3.0):
        raise RuntimeError("simulated provider failure")


def _build_pipeline(cache=None, agent_stage=None) -> Pipeline:
    """组装带可选语义缓存的五段式管线。"""
    if agent_stage is None:
        agent_stage = AgentStage(MockBrain(), budget_cny=2.0, monthly_cost=0.0)
    return Pipeline(
        collector=SignalCollector(),
        router=RuleEngine(default_rules()),
        agent_stage=agent_stage,
        converger=Converger(),
        validator=SchemaValidator(),
        cache=cache,
    )


class TestCacheHitMiss(unittest.TestCase):
    """语义缓存: 同信号 miss+put 后 hit; 改信号 miss。"""

    def setUp(self):
        self.cache = SemanticCache(LocalEmbedder(dim=256))
        self.pipeline = _build_pipeline(cache=self.cache)
        self.scenario = scenario_fruit_rain_expiry()

    def test_cache_hit_miss(self):
        # 第一次: miss -> 跑完整五段 -> put。
        a1 = self.pipeline.generate_today_advice(1001, self.scenario)
        self.assertTrue(a1.items, "首次应产出非空建议")
        m1 = self.cache.metrics()
        self.assertEqual(m1["miss"], 1, "首次应记 1 次 miss")
        self.assertEqual(m1["hit"], 0)

        # 第二次同信号: hit -> 直接返回缓存 (trace 有 cache_hit 段)。
        a2 = self.pipeline.generate_today_advice(1001, self.scenario)
        m2 = self.cache.metrics()
        self.assertEqual(m2["hit"], 1, "第二次同信号应命中")
        # 命中返回的 advice 与首次一致 (version 相同)。
        self.assertEqual(a1.version, a2.version)
        # trace 记录了 cache_hit 段且 hit=True。
        hit_stages = [
            s for s in self.pipeline.get_trace()["stages"] if s["stage"] == "cache_hit"
        ]
        self.assertTrue(hit_stages)
        self.assertTrue(hit_stages[-1]["snapshot"]["hit"])

        # 第三次改信号 (天气转晴 + 无临期): key_text 变化 -> miss。
        sc3 = scenario_fruit_rain_expiry()
        sc3["weather"] = {"condition": "normal", "temp_c": 26.0, "forecast": "晴"}
        sc3["inventory"] = [{
            "product_id": 1, "name": "苹果", "cost": 5.0, "price": 8.0,
            "stock": 100, "days_to_expiry": -1, "slow_moving": False,
            "high_margin": False,
        }]
        self.pipeline.generate_today_advice(1001, sc3)
        m3 = self.cache.metrics()
        self.assertEqual(m3["miss"], 2, "改信号后应再次 miss")
        self.assertEqual(m3["hit"], 1)


class TestCacheInvalidation(unittest.TestCase):
    """主动失效: put 后 invalidate_tenant, 再查 miss。"""

    def test_cache_invalidation(self):
        cache = SemanticCache(LocalEmbedder(dim=256))
        pipe = _build_pipeline(cache=cache)
        sc = scenario_fruit_rain_expiry()
        # 首次 miss+put。
        pipe.generate_today_advice(1001, sc)
        self.assertEqual(cache.metrics()["miss"], 1)
        # 第二次同信号应命中。
        pipe.generate_today_advice(1001, sc)
        self.assertEqual(cache.metrics()["hit"], 1)
        # 主动失效该 tenant 当日缓存。
        n = cache.invalidate_tenant(1001)
        self.assertGreaterEqual(n, 1, "应至少失效 1 条")
        # 失效后再查同信号: miss (entry 已过期, 不命中)。
        pipe.generate_today_advice(1001, sc)
        self.assertEqual(cache.metrics()["miss"], 2, "失效后应再次 miss")
        self.assertEqual(cache.metrics()["hit"], 1)


class TestGatewayFallback(unittest.TestCase):
    """网关降级: lite 层失败 -> 降级 v3 (MockProvider); 全失败抛错。"""

    def test_lite_fail_degrades_to_v3(self):
        # lite 层用 _FailProvider (永远失败), v3 层用 MockProvider (成功)。
        gw = Gateway(
            layers={
                "lite": Layer(name="lite", provider=_FailProvider(), timeout=0.5, retry=1),
                "v3": Layer(name="v3", provider=MockProvider(), timeout=0.5, retry=1),
            },
            budget_cny=2.0,
        )
        resp = gw.chat_with_tools(
            "copy_polish",
            [ChatMessage(role="user", content="润色文案")],
        )
        self.assertIsInstance(resp, ChatResp)
        self.assertTrue(resp.content, "降级到 v3 后应返回非空 content")
        # raw 里 mock=True 说明最终由 MockProvider 兜底。
        self.assertTrue(resp.raw.get("mock"))

    def test_all_layers_fail_raises(self):
        # lite + v3 都失败, 且无更底层 (r1 未配置): 抛 RuntimeError 让上层降级。
        gw = Gateway(
            layers={
                "lite": Layer(name="lite", provider=_FailProvider(), timeout=0.3, retry=0),
                "v3": Layer(name="v3", provider=_FailProvider(), timeout=0.3, retry=0),
            },
            budget_cny=2.0,
        )
        with self.assertRaises(RuntimeError):
            gw.chat_with_tools(
                "copy_polish",
                [ChatMessage(role="user", content="x")],
            )

    def test_pricing_skips_r1(self):
        # pricing 类任务禁止走 r1: r1 用 _FailProvider, v3 用 MockProvider;
        # 即使 r1 配置了也不被调用 (pricing 止于 v3), v3 成功即返回。
        gw = Gateway(
            layers={
                "lite": Layer(name="lite", provider=_FailProvider(), timeout=0.3, retry=0),
                "v3": Layer(name="v3", provider=MockProvider(), timeout=0.3, retry=0),
                "r1": Layer(name="r1", provider=_FailProvider(), timeout=0.3, retry=0),
            },
            budget_cny=2.0,
        )
        resp = gw.chat_with_tools(
            "pricing_tool_call",
            [ChatMessage(role="user", content="定价")],
        )
        self.assertTrue(resp.raw.get("mock"), "pricing 应止于 v3, 不走 r1")


class TestGatewayBudgetExceeded(unittest.TestCase):
    """月成本熔断: add_cost 累计 >= budget_cny -> is_budget_exceeded True。"""

    def test_budget_exceeded(self):
        gw = Gateway(layers={"v3": Layer(name="v3", provider=MockProvider())},
                     budget_cny=2.0)
        self.assertFalse(gw.is_budget_exceeded(5001))
        gw.add_cost(5001, 0.8)
        self.assertFalse(gw.is_budget_exceeded(5001), "0.8 < 2.0 未超")
        gw.add_cost(5001, 1.3)  # 累计 2.1
        self.assertTrue(gw.is_budget_exceeded(5001), "累计 2.1 >= 2.0 应超预算")
        self.assertAlmostEqual(gw.get_budget_used(5001), 2.1, places=6)
        # 不同 tenant 互不影响。
        self.assertFalse(gw.is_budget_exceeded(5002))


class TestPregenBatch(unittest.TestCase):
    """凌晨预生成: 3 tenant batch_generate -> ok=3, cache 有命中。"""

    def test_pregen_batch(self):
        cache = SemanticCache(LocalEmbedder(dim=256))
        pipe = _build_pipeline(cache=cache)
        sc = scenario_fruit_rain_expiry()
        # 3 个 tenant 用同一情境 (business=fruit, 不同 tenant_id)。
        scenarios = {7001: sc, 7002: sc, 7003: sc}
        sched = Scheduler(
            pipeline=pipe, cache=cache,
            tenants=[7001, 7002, 7003], scenarios=scenarios,
            max_workers=2, batch_size=10, max_retries=2,
        )
        res = sched.batch_generate()
        self.assertEqual(res["ok"], 3, "3 户应全部生成成功")
        self.assertEqual(res["fail"], 0)
        self.assertEqual(res["cached"], 3, "3 户应全部写入缓存")
        self.assertFalse(res["degraded"])

        # 预生成后, 同情境查询应命中 (cache metrics 有命中; 同 scenario 跨 tenant
        # 因 key_text 一致也会在批量过程中互相命中, 故 hit >= 1)。
        hit_advice, entry = cache.try_hit(7001, "fruit", cache.make_key_text(
            pipe.collector.collect(7001, sc),
            pipe.router.route(pipe.collector.collect(7001, sc)),
        ))
        self.assertIsNotNone(hit_advice, "预生成后查询应命中")
        self.assertGreaterEqual(cache.metrics()["hit"], 1)


class TestBanditUpdate(unittest.TestCase):
    """bandit: 不同 reward 软更新, weight 朝高 reward 偏移; reweight 回写 priority。"""

    def test_update_weight_shifts_toward_high_reward(self):
        bandit = Bandit(epsilon=0.0, lr=0.5)  # epsilon=0 关探索, 纯利用便于断言
        # 初始 weight=0.5, 高 reward(1.0) 应使 weight 上升趋近 1.0, 低 reward(0.0) 下降趋近 0。
        arm_hi = Arm(rule_id="r_hi", weight=0.5)
        arm_lo = Arm(rule_id="r_lo", weight=0.5)
        for _ in range(5):
            bandit.update(arm_hi, 1.0)
            bandit.update(arm_lo, 0.0)
        self.assertGreater(arm_hi.weight, arm_lo.weight,
                           "高 reward 臂权重应高于低 reward 臂")
        self.assertGreater(arm_hi.weight, 0.5, "高 reward 臂权重应上升 (>初始 0.5)")
        self.assertLess(arm_lo.weight, 0.5, "零 reward 臂权重应下降 (<初始 0.5)")

    def test_select_prefers_high_weight(self):
        bandit = Bandit(epsilon=0.0, lr=0.1)
        arms = [Arm("a", weight=1.0), Arm("b", weight=9.0), Arm("c", weight=5.0)]
        chosen = bandit.select(arms)
        self.assertEqual(chosen.rule_id, "b", "epsilon=0 应选 weight 最大的臂")

    def test_reweight_rules_writes_back_priority(self):
        # 两条规则: rule_hi 高 reward (核销高), rule_lo 零 reward (无核销无到店无营收)。
        rules = [
            Rule(id="rule_hi", business="*", priority=50, enable=True,
                 conds=[Cond("inventory.high_margin", "eq", True)],
                 action_type="high_value_push", target="高毛利push",
                 param_range=ParamRange(0.9, 0.9), margin_floor=0.15),
            Rule(id="rule_lo", business="*", priority=50, enable=True,
                 conds=[Cond("inventory.high_margin", "eq", True)],
                 action_type="bundle_slow", target="滞销捆绑",
                 param_range=ParamRange(0.85, 0.92), margin_floor=0.08),
        ]
        outcomes = OutcomeCollector()
        now = time.time()
        # rule_hi: 发10核销8 (rate=0.8), 到店+4, 营收180; rule_lo: 全零 (无任何效果)。
        for _ in range(3):
            outcomes.record(Outcome(
                tenant_id=1, advice_id="v1", rule_id="rule_hi",
                action_type="high_value_push", coupon_sent=10, coupon_used=8,
                visit_delta=4, group_ok=True, revenue=180.0, measured_at=now,
            ))
            outcomes.record(Outcome(
                tenant_id=1, advice_id="v1", rule_id="rule_lo",
                action_type="bundle_slow", coupon_sent=10, coupon_used=0,
                visit_delta=0, group_ok=False, revenue=0.0, measured_at=now,
            ))
        self.assertAlmostEqual(outcomes.coupon_rate("rule_hi"), 0.8, places=6)
        self.assertAlmostEqual(outcomes.coupon_rate("rule_lo"), 0.0, places=6)
        # reward 差距足够大 (hi≈0.9, lo=0.0), 用较大 lr 使 int 回写后仍能区分。
        self.assertGreater(outcomes.compute_reward("rule_hi"),
                           outcomes.compute_reward("rule_lo"))

        bandit = Bandit(epsilon=0.0, lr=0.8)
        new_rules = bandit.reweight_rules(rules, outcomes)
        prio_map = {r.id: r.priority for r in new_rules}
        self.assertGreater(prio_map["rule_hi"], prio_map["rule_lo"],
                           "高 reward 规则 priority 应被回写为更高")
        # 原规则集不变 (不修改原对象)。
        self.assertEqual(rules[0].priority, 50)
        self.assertEqual(rules[1].priority, 50)


class TestAttributionReward(unittest.TestCase):
    """归因: coupon_rate 正确, compute_reward 落在 [0,1]。"""

    def test_reward_in_range(self):
        oc = OutcomeCollector()
        now = time.time()
        oc.record(Outcome(
            tenant_id=1, advice_id="v1", rule_id="rA",
            action_type="clearance_coupon", coupon_sent=100, coupon_used=30,
            visit_delta=8, group_ok=True, revenue=500.0, measured_at=now,
        ))
        oc.record(Outcome(
            tenant_id=2, advice_id="v1", rule_id="rA",
            action_type="clearance_coupon", coupon_sent=50, coupon_used=25,
            visit_delta=-2, group_ok=False, revenue=120.0, measured_at=now,
        ))
        # coupon_rate = (30+25)/(100+50) = 55/150 ≈ 0.3667
        self.assertAlmostEqual(oc.coupon_rate("rA"), 55.0 / 150.0, places=4)
        r = oc.compute_reward("rA")
        self.assertGreaterEqual(r, 0.0)
        self.assertLessEqual(r, 1.0)
        # 无观测规则 reward=0。
        self.assertEqual(oc.compute_reward("nonexistent"), 0.0)


class TestCacheSingleSignalChange(unittest.TestCase):
    """回归测试: 单维度信号变化 (仅天气 / 仅临期消失) 不得误命中旧 advice。

    修复前 LocalEmbedder hashing-trick 对结构化短 key_text 区分度不足, 单字段变化
    余弦距离 0.05-0.08 < 0.15 阈值, 导致天气转晴仍命中雨天 rain_delivery 建议。
    修复: make_key_prefix 对 weather/actions/days_to_expiry/holiday 强区分字段精确
    匹配, try_hit 前置过滤, 不再依赖向量区分这些字段。
    """

    @staticmethod
    def _rain_scenario():
        return {
            "business": "fruit", "city": "临沂", "date": "2026-07-03",
            "weather": {"condition": "rain", "temp_c": 18.0, "forecast": "雨"},
            "holiday": {"name": "无", "days_to": 99, "is_festival": False},
            "inventory": [{
                "product_id": 1, "name": "阳光玫瑰葡萄", "cost": 12.0,
                "price": 25.8, "stock": 50, "days_to_expiry": 2,
                "slow_moving": False, "high_margin": True,
            }],
            "members": {"sleep_30": 18, "high_value": [101, 102, 103],
                        "birthday_today": []},
            "data_quality": 85,
        }

    def _build(self):
        cache = SemanticCache(LocalEmbedder(dim=256))
        pipeline = _build_pipeline(cache=cache)
        return pipeline, cache

    def test_weather_change_does_not_hit_stale(self):
        """仅天气 rain->normal: 不应命中雨天的 rain_delivery 缓存。"""
        pipeline, cache = self._build()
        sc_rain = self._rain_scenario()
        a1 = pipeline.generate_today_advice(1001, sc_rain)
        # 雨天应有 rain_delivery (规则映射: weather=rain -> rain_delivery)。
        actions1 = {it.action_type for it in a1.items}
        self.assertIn("rain_delivery", actions1, "雨天应产出 rain_delivery")

        sc_sunny = self._rain_scenario()
        sc_sunny["weather"] = {"condition": "normal", "temp_c": 25.0, "forecast": "晴"}
        a2 = pipeline.generate_today_advice(1001, sc_sunny)
        # 关键断言: 不应返回同一缓存对象 (晴天不应命中雨天 advice)。
        self.assertIsNot(a1, a2, "天气变化不得命中旧缓存")
        actions2 = {it.action_type for it in a2.items}
        self.assertNotIn(
            "rain_delivery", actions2,
            "晴天不应收到雨天 rain_delivery 建议 (误命中)",
        )
        m = cache.metrics()
        self.assertEqual(m["hit"], 0, "单维度天气变化应全部 miss")
        self.assertEqual(m["miss"], 2)

    def test_expiry_disappear_does_not_hit_stale(self):
        """仅临期项消失 (其余不变): 不应命中含 clearance 的旧缓存。"""
        pipeline, cache = self._build()
        sc = self._rain_scenario()
        a1 = pipeline.generate_today_advice(1001, sc)
        actions1 = {it.action_type for it in a1.items}
        self.assertIn("clearance_coupon", actions1, "临期应产出 clearance")

        # 去掉临期项 (改成非临期苹果), 其余信号不变 (仍雨天)。
        sc2 = self._rain_scenario()
        sc2["inventory"] = [{
            "product_id": 2, "name": "苹果", "cost": 5.0, "price": 8.0,
            "stock": 100, "days_to_expiry": -1, "slow_moving": False,
            "high_margin": False,
        }]
        a2 = pipeline.generate_today_advice(1001, sc2)
        self.assertIsNot(a1, a2, "临期消失不得命中旧缓存")
        actions2 = {it.action_type for it in a2.items}
        self.assertNotIn(
            "clearance_coupon", actions2,
            "临期已消失不应再收到 clearance 建议 (误命中)",
        )

    def test_same_signal_still_hits(self):
        """回归: 修复后同信号第二次仍应命中 (prefix 一致 + 向量一致)。"""
        pipeline, cache = self._build()
        sc = self._rain_scenario()
        a1 = pipeline.generate_today_advice(1001, sc)
        a2 = pipeline.generate_today_advice(1001, sc)
        self.assertIs(a1, a2, "同信号第二次应命中同一缓存对象")
        self.assertEqual(cache.metrics()["hit"], 1)


if __name__ == "__main__":
    unittest.main()
