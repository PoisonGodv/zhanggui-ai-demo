"""输出校验段 (对应方案 7.4.4 JSON Schema 约束 + 7.11.2 DoD 文案合规)。

职责:
1. 给 EnhancedItem 编号 rank 1..N, 转成 AdviceItem。
2. 文案 copy:
   - _filter_sensitive: 敏感词/绝对化用语替换为合规词 (如 最->很, 第一->领先, 永久->长期), 其余打码为 '*';
   - _truncate: 截断 <= 40 字符 (中文每字 1, 按 Python str 字符数计);
   - copy 为空 -> 回退模板 "今日推荐:{product}{action_type}"。
3. margin_ok 强制保留原值; 若 False 在 reason 后追加 "[护栏拦截,建议复核]"。
4. 返回 Advice(version = date + '-v1')。

严格遵循 CONTRACT (zhanggui/types.py)。
"""

from .types import Advice, AdviceItem, EnhancedItem

# 7.11.2 DoD #9 文案合规: 敏感词/绝对化用语清单。
SENSITIVE_WORDS: list = [
    "最", "第一", "永久", "包治", "绝对",
    "顶级", "100%", "特效", "神器",
]

# 敏感词 -> 等价合规词映射 (替换而非打码, 文案更自然)。
# 未在映射表的词统一替换为 '*'。
_SENSITIVE_REPLACEMENT: dict = {
    "最": "很",
    "第一": "领先",
    "永久": "长期",
}

# copy 长度硬上限 (7.4.4 schema maxLength: 40)。
_COPY_MAX_LEN: int = 40

# copy 为空时的回退模板。
_FALLBACK_TEMPLATE: str = "今日推荐:{product}{action_type}"


class SchemaValidator:
    """输出校验 / Advice 编码器。"""

    def encode(self, top3: list, tenant_id: int, date: str) -> Advice:
        """把收敛后的 top3 EnhancedItem 编码为对外 Advice。

        Args:
            top3: list[EnhancedItem], 已按预期增量价值排序, 取前 3。
            tenant_id: 租户 id。
            date: 当日日期字符串, 如 "2026-07-03"。

        Returns:
            Advice(version = date + '-v1')。
        """
        items: list = []
        for idx, ei in enumerate(top3):
            rank = idx + 1
            advice_item = self._build_advice_item(ei, rank)
            items.append(advice_item)

        version = "{date}-v1".format(date=date)
        return Advice(
            tenant_id=tenant_id,
            date=date,
            items=items,
            version=version,
        )

    # ---- 文案合规 ----

    def _filter_sensitive(self, copy: str) -> str:
        """敏感词/绝对化用语替换为合规词或 '*'。

        优先按整词匹配并替换为等价合规词 (映射表);
        剩余敏感词逐个打码为 '*'。按词长降序处理, 避免短词先吃掉长词
        (例如 '第一' 不应被 '一' 类规则或单字误伤, '永久' 应整词替换)。
        """
        if not copy:
            return copy
        result = copy
        # 长词优先, 保证 '第一'/'永久'/'包治' 等多字词整词命中。
        for word in sorted(SENSITIVE_WORDS, key=len, reverse=True):
            if word and word in result:
                replacement = _SENSITIVE_REPLACEMENT.get(word, "*")
                result = result.replace(word, replacement)
        return result

    def _truncate(self, copy: str, max_len: int = _COPY_MAX_LEN) -> str:
        """按字符数截断 <= max_len (中文每字 1)。"""
        if len(copy) <= max_len:
            return copy
        return copy[:max_len]

    def _safe_copy(self, copy: str, product: str, action_type: str) -> str:
        """完整文案流水线: 过滤敏感词 -> 截断 -> 空则回退模板。"""
        product = product or ""
        action_type = action_type or ""
        if copy is None:
            copy = ""
        # 先过滤再截断: 防止打码后的 '*' 占位被截断丢词,
        # 顺序与方案 7.4.4 "敏感词过滤二次把关" 一致。
        filtered = self._filter_sensitive(copy)
        truncated = self._truncate(filtered, _COPY_MAX_LEN)
        if not truncated:
            truncated = self._truncate(
                _FALLBACK_TEMPLATE.format(product=product, action_type=action_type),
                _COPY_MAX_LEN,
            )
        return truncated

    # ---- 单条编码 ----

    def _build_advice_item(self, ei, rank: int) -> AdviceItem:
        """单个 EnhancedItem -> AdviceItem。

        margin_ok 强制保留原值 (不篡改); False 时 reason 追加护栏标记。
        """
        routed = ei.routed
        copy = self._safe_copy(ei.copy, routed.product, routed.action_type)

        reason = ei.routed.reason or ""
        margin_ok = bool(ei.margin_ok)
        if not margin_ok:
            marker = "[护栏拦截,建议复核]"
            if reason:
                reason = "{r}{marker}".format(r=reason, marker=marker)
            else:
                reason = marker

        return AdviceItem(
            rank=rank,
            action_type=routed.action_type,
            target=routed.target,
            product=routed.product,
            params=ei.params,
            copy=copy,
            reason=reason,
            expected_effect=ei.expected_effect,
            margin_ok=margin_ok,
        )
