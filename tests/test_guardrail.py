"""毛利护栏单元测试 (纯标准库 unittest, 不用 pytest)。

覆盖 zhanggui/guardrail.py 三个公开函数:
  - guardrail_ok(cost, final_price, margin_floor)
  - min_discount_floor(cost, price, margin_floor)
  - apply_guardrail(inv, rule) -> (Optional[RoutedItem], bool)

golden case 对应方案 7.4.2 校验口径:
  cost=12, price=25.8, margin_floor=0.05
  floor = cost*(1+mf) = 12.6
  guardrail_ok(12, 15.9, 0.05) = True   (15.9 >= 12.6)
  guardrail_ok(12, 12.0, 0.05) = False  (12.0 < 12.6)
  min_discount_floor = 12.6/25.8 ≈ 0.488
"""

import unittest

from zhanggui.guardrail import (
    apply_guardrail,
    guardrail_ok,
    min_discount_floor,
)
from zhanggui.types import InventorySig, ParamRange, Rule


class TestGuardrailGolden(unittest.TestCase):
    """方案 7.4.2 黄金用例: cost=12 / price=25.8 / margin_floor=0.05。"""

    def setUp(self):
        self.cost = 12.0
        self.price = 25.8
        self.mf = 0.05
        self.floor = self.cost * (1.0 + self.mf)  # 12.6

    def test_golden_case(self):
        # floor = 12.6; 15.9 >= 12.6 -> 通过; 12.0 < 12.6 -> 拒绝。
        self.assertTrue(guardrail_ok(self.cost, 15.9, self.mf))
        self.assertFalse(guardrail_ok(self.cost, 12.0, self.mf))
        # min_discount_floor = 12.6 / 25.8 ≈ 0.4883...
        md = min_discount_floor(self.cost, self.price, self.mf)
        self.assertAlmostEqual(md, 12.6 / 25.8, places=6)
        self.assertAlmostEqual(md, 0.488, places=2)


class TestApplyGuardrailRejectsBelowFloor(unittest.TestCase):
    """discount_max=0.3 对应 final_price=25.8*0.3=7.74 < 12.6 -> 拒绝 (None, False)。"""

    def test_apply_guardrail_rejects_below_floor(self):
        inv = InventorySig(
            product_id=1001,
            name="阳光玫瑰葡萄",
            cost=12.0,
            price=25.8,
            stock=50.0,
            days_to_expiry=2,
            slow_moving=False,
            high_margin=True,
        )
        rule = Rule(
            id="clearance_coupon_fruit",
            business="fruit",
            priority=50,
            enable=True,
            conds=[],
            action_type="clearance_coupon",
            target="临期SKU",
            param_range=ParamRange(
                discount_min=0.1,
                discount_max=0.3,   # final_price = 25.8*0.3 = 7.74 < 12.6 -> 丢弃
                min_spend=0.0,
                reduce_max=0.0,
                limit_qty_max=50,
                expire_hours=0,
            ),
            margin_floor=0.05,
        )
        item, ok = apply_guardrail(inv, rule)
        self.assertFalse(ok)
        self.assertIsNone(item)


class TestNoCostRejects(unittest.TestCase):
    """cost=0 (无成本数据) -> (None, False), 防幻觉亏钱。"""

    def test_no_cost_rejects(self):
        inv = InventorySig(
            product_id=1002,
            name="无成本商品",
            cost=0.0,
            price=10.0,
            stock=5.0,
            days_to_expiry=-1,
            slow_moving=False,
            high_margin=False,
        )
        rule = Rule(
            id="high_value_push_default",
            business="*",
            priority=70,
            enable=True,
            conds=[],
            action_type="high_value_push",
            target="高毛利品->高价值会员",
            param_range=ParamRange(
                discount_min=0.9,
                discount_max=0.9,
                min_spend=0.0,
                reduce_max=0.0,
                limit_qty_max=0,
                expire_hours=0,
            ),
            margin_floor=0.15,
        )
        item, ok = apply_guardrail(inv, rule)
        self.assertFalse(ok)
        self.assertIsNone(item)


if __name__ == "__main__":
    unittest.main()
