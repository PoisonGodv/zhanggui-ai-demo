"""收敛段：按预期增量价值排序取 Top3。

对应方案 07_AI引擎详设.md 的：
  - 7.1 converge.Top3（PickTop3：按预期增量价值排序取前 3）
  - 7.4 expected_effect（预期效果串：核销X份 / 到店Y人 / 挽回¥Z / 召回N人）

把 Agent 段产出的 Enhanced.items 按各自 expected_effect 里解析出的
「预期增量价值」排序，margin_ok=False 的项排到末尾但不丢弃（让老板看到
护栏拦下的也有记录、只是排名靠后），取前 3 条交给输出校验段
（SchemaValidator.encode）打 rank 1-3。本段不写 rank，只负责排序与截断。
"""

import re
from typing import List

from .types import Enhanced, EnhancedItem


# 方案指定的基础数字提取正则：整数或小数（re.findall(r'\d+\.?\d*')）。
_NUM_RE = r"\d+\.?\d*"

# 四类预期效果数字的「关键词 → 权重」表：
#   核销X份 *1.0   到店Y人 *0.5   挽回¥Z *0.01   召回N人 *0.8
# 每条正则：关键词后接若干非数字字符（如「损耗¥」「份」「人」），再捕获
# 紧随其后的第一个数字。范围类写法（如「召回4-6人」）取下界，与方案
# 「数值保守」一致。无法解析的串记 0 分。
_WEIGHTED_PATTERNS = (
    ("核销", 1.0, re.compile("核销\\D*?(" + _NUM_RE + ")")),
    ("到店", 0.5, re.compile("到店\\D*?(" + _NUM_RE + ")")),
    ("挽回", 0.01, re.compile("挽回\\D*?(" + _NUM_RE + ")")),
    ("召回", 0.8, re.compile("召回\\D*?(" + _NUM_RE + ")")),
)


class Converger:
    """收敛器：从 Enhanced 里挑预期增量价值最高的 3 条。"""

    def _score(self, item: EnhancedItem) -> float:
        """解析 expected_effect，按四类数字加权求和，得到预期增量价值。

        用 re.findall(r'\\d+\\.?\\d*') 的思路提取每个关键词后的数字，按
            核销*1.0 + 到店*0.5 + 挽回¥*0.01 + 召回*0.8
        加权。空串 / 无可解析数字记 0 分。
        """
        text = item.expected_effect or ""
        total = 0.0
        for _kw, weight, pat in _WEIGHTED_PATTERNS:
            for num in pat.findall(text):
                try:
                    total += float(num) * weight
                except (TypeError, ValueError):
                    continue
        return total

    def pick_top3(self, enhanced: Enhanced) -> List[EnhancedItem]:
        """按预期增量价值排序取前 3。

        排序键 (margin_ok, score, priority) 全程降序：
          - margin_ok=True 的项优先；margin_ok=False 的项整体排到末尾、
            不丢弃（让老板看到护栏拦下的也有记录，只是排名靠后）；
          - 同 margin_ok 组内，score 高者在前；
          - score 相同时按 priority 降序。
        不足 3 条返回全部；即便 margin_ok=False 占满也照常返回，保证有输出。
        返回的是新列表，不改动调用方 enhanced.items 的原顺序。
        """
        items = list(enhanced.items or []) if enhanced is not None else []
        if not items:
            return []

        def _key(it: EnhancedItem):
            # bool(True)=1 > bool(False)=0，reverse=True 时 True 组在前。
            return (bool(it.margin_ok), self._score(it), it.routed.priority)

        items.sort(key=_key, reverse=True)
        return items[:3]
