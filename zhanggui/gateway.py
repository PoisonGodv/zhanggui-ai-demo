"""模型网关分层 (对应方案 7.8.1-7.8.4)。

把「调 LLM」从 AgentStage/LLMBrain 里抽出来, 统一成「按任务选层 + 超时重试 +
逐层降级 + 月成本熔断」的网关。

三层 (方案 7.8.2):
  - lite : 轻量模型 (文案润色 copy_polish), 便宜快。
  - v3   : 通用模型 (tool_call), 平衡。
  - r1   : 推理模型 (reasoning), 贵且慢, 仅复杂推理用; 定价类动作禁止走 r1
           (方案 7.8.3: 定价必须可解释/可复算, 不交给黑盒推理模型)。

降级链: lite -> v3 -> r1 (方案 7.8.4「层内重试, 层间降级」)。某层超时/网络失败
重试 retry 次 (指数退避 0.2*2^n); 该层全失败则降级下一层; 最底层 (r1) 失败抛
异常, 由 AgentStage 降级 MockBrain。pricing 类任务在降级链里跳过 r1 (直接止于
v3, v3 再失败则抛)。

成本熔断 (方案 7.8.4): add_cost 累加到月度计数 (进程内 dict, key 形如
"ai:cost:{tenant}:{YYYYMM}"), is_budget_exceeded >= budget_cny。真要持久化/聚合
时把 _cost_store 换成 PG 即可 (接口不变)。

零第三方依赖: OpenAICompatProvider 用标准库 urllib.request, 不 import openai。
无 api_key 时用 MockProvider (确定性模板, 不触网), demo 永远能跑。

严格遵循本模块 CONTRACT (zhanggui/gateway.py 模块契约)。
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = [
    "ChatMessage",
    "ChatResp",
    "Provider",
    "MockProvider",
    "OpenAICompatProvider",
    "Layer",
    "Gateway",
]


# === 消息 / 响应 dataclass ===

@dataclass
class ChatMessage:
    """openai 兼容消息 (role/content/tool_calls)。

    tool_calls 仅 assistant 角色会有; 其余角色为 None。
    """
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class ChatResp:
    """网关统一响应 (provider.chat 的返回)。

    content : 模型最终文本 (openai choices[0].message.content)。
    tool_calls : 模型要求调用的工具列表 (openai message.tool_calls), 空列表表示无。
    raw : 原始响应 dict (调试/trace 用)。
    """
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


# === Provider 协议 (方案 7.8.1) ===

@runtime_checkable
class Provider(Protocol):
    """模型提供方协议: 给定 messages/tools/timeout, 返回 ChatResp。"""

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 3.0,
    ) -> ChatResp:
        ...


# === MockProvider (零依赖, 确定性, 不触网) ===

class MockProvider:
    """确定性 Provider (无 key 时用, 不调网络)。

    返回模板 ChatResp: content 为一段确定性 JSON 文案 (含 copy/params 占位 +
    margin_ok=true), tool_calls 为空 (表示模型直接给出最终答案, 不再调工具)。

    设计意图: 包装 MockBrain「确定性产出文案/params」的语义, 但 MockBrain.enhance
    需要 (tenant_id, routed, bundle) 上下文, 而 Provider.chat 只拿到 messages,
    故此处只产出**确定性占位响应**; 真正的规则+模板兜底由 AgentStage 在 brain
    失败时降级 MockBrain 完成 (本 Provider 是网关最末层, 保证「永远有响应」)。
    """

    _MOCK_CONTENT = (
        '{"copy":"(军师)今日营销建议已就绪,详见下方动作","params":{},'
        '"expected_effect":"到店3-5人","margin_ok":true}'
    )

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 3.0,
    ) -> ChatResp:
        # 确定性: 同输入同输出, 无随机, 无网络, 永不抛错 (是最末降级层)。
        return ChatResp(content=self._MOCK_CONTENT, tool_calls=[], raw={"mock": True})


# === OpenAICompatProvider (真模型, urllib, 需 api_key) ===

class OpenAICompatProvider:
    """openai 兼容 Provider (urllib, 需 api_key)。

    __init__: api_key 为空 -> raise ValueError("OpenAICompatProvider needs api_key")。
    chat: POST {base_url}/chat/completions, 带 messages/tools, 超时 timeout;
          失败 (HTTP/网络/JSON) 抛异常, 由 Gateway 层重试/降级。
    不 import openai (未装), 用 urllib.request。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ):
        if not api_key:
            raise ValueError("OpenAICompatProvider needs api_key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        timeout: float = 3.0,
    ) -> ChatResp:
        url = self.base_url + "/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
            "temperature": 0.3,
        }
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer {k}".format(k=self.api_key),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                raw_text = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            raise RuntimeError("OpenAICompat HTTP {0}".format(e.code)) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RuntimeError("OpenAICompat network error: {0}".format(e)) from e
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise RuntimeError("OpenAICompat bad JSON: {0}".format(e)) from e

        choice = None
        choices = raw.get("choices") if isinstance(raw, dict) else None
        if isinstance(choices, list) and choices:
            choice = choices[0]
        msg = (choice or {}).get("message", {}) or {}
        content = str(msg.get("content") or "")
        tool_calls = list(msg.get("tool_calls") or [])
        return ChatResp(content=content, tool_calls=tool_calls, raw=raw)


# === Layer dataclass ===

@dataclass
class Layer:
    """网关层: 名字 + provider + 超时 + 重试次数。"""
    name: str
    provider: Provider
    timeout: float = 3.0
    retry: int = 2


# === Gateway (方案 7.8.1-7.8.4) ===

# 任务 -> 层 (方案 7.8.2)。copy_polish 走 lite, tool_call 走 v3, reasoning 走 r1。
# pricing 类任务 (task_type 含 'pricing') 禁止走 r1 (见 chat_with_tools)。
_LAYER_BY_TASK = {
    "copy_polish": "lite",
    "tool_call": "v3",
    "reasoning": "r1",
}

# 降级链 (方案 7.8.3: lite -> V3 -> 规则模板)。
# 注意: R1 是「复杂推理升级层」, 不参与降级链 —— 便宜任务失败时升级到更强的 V3 兜底,
# 但绝不升级到最贵的 R1 (成本敏感系统反向操作)。V3 失败直接抛错, 由 AgentStage 降级
# MockBrain (= 方案的规则模板文案)。reasoning 任务选 R1, 失败则降级到 V3。
# 顺序按「兜底方向」: 选中层若失败, 沿链向后找更稳/更强的兜底层, 直到链尾抛错。
_DEGRADE_CHAIN = ["lite", "v3"]

# 重试指数退避基数: 第 n 次重试前 sleep 0.2 * 2^n 秒 (n 从 0 起)。
_BACKOFF_BASE = 0.2


class Gateway:
    """模型网关总控 (方案 7.8)。

    - chat_with_tools(task_type, messages, tools): 按 LAYER_BY_TASK 选层; 调
      provider.chat 带 timeout; 失败重试 retry 次 (指数退避 0.2*2^n); 该层全失败
      降级下一层 (lite->v3->r1); 最底层失败抛异常 (让 AgentStage 降级 MockBrain)。
      task_type 含 'pricing' 时禁止走 r1 (降级链跳过 r1, 止于 v3; v3 再失败则抛)。
    - add_cost/is_budget_exceeded/get_budget_used: 月成本 dict, key=
      "ai:cost:{tenant}:{YYYYMM}", value 累加 cny; is_budget_exceeded >= budget_cny。

    NOTE: LAYER_BY_TASK 在实例上可被覆盖 (类属性 + 实例属性回退), 便于测试/配置。
    """

    LAYER_BY_TASK: Dict[str, str] = _LAYER_BY_TASK

    def __init__(self, layers: Dict[str, Layer], budget_cny: float = 2.0):
        self.layers = layers
        self.budget_cny = float(budget_cny)
        # 月成本计数 (进程内 dict)。真要持久化/聚合时, 把 _cost_store 换成 PG
        # 表 ai_cost(tenant_id, yyyymm, cny) 即可, 接口不变 (方案 7.8.4)。
        self._cost_store: Dict[str, float] = {}

    # ---- 层选择 / 降级链 ----

    def _select_layer_name(self, task_type: str) -> str:
        """按 task_type 查 LAYER_BY_TASK 选层; 未知 task_type 默认 v3 (平衡)。"""
        name = self.LAYER_BY_TASK.get(task_type)
        if not name:
            # 未知任务类型: 默认走 v3 (不冒进用 r1, 也不省到 lite)。
            name = "v3"
        return name

    def _degrade_candidates(self, task_type: str) -> List[str]:
        """构造降级候选层列表 (从选中层起沿降级链向尾)。

        - lite 任务: [lite, v3] (lite 失败升级到更强的 v3 兜底, 再失败抛)。
        - v3 任务: [v3] (v3 失败直接抛, 由 AgentStage 降级 MockBrain)。
        - reasoning(r1) 任务: [r1, v3] (r1 失败降级到更便宜的 v3 兜底, 再失败抛)。
        - pricing 类任务: 剔除 r1 (方案 7.8.3 定价禁走推理模型), 且优先走 v3 而非 lite
          (定价由规则+护栏定, LLM 不定价; 这里仅防御性兜底)。
        """
        selected = self._select_layer_name(task_type)
        is_pricing = "pricing" in (task_type or "")
        # pricing 禁 r1: 强制选 v3。
        if is_pricing and selected == "r1":
            selected = "v3"
        if selected in _DEGRADE_CHAIN:
            start = _DEGRADE_CHAIN.index(selected)
            candidates = _DEGRADE_CHAIN[start:]
        elif selected == "r1":
            # r1 不在降级链: r1 失败降级到 v3 (更便宜、更稳的兜底)。
            candidates = ["r1", "v3"]
        else:
            candidates = ["v3"]
        if is_pricing:
            candidates = [c for c in candidates if c != "r1"]
            if not candidates:
                candidates = ["v3"]
        return candidates

    # ---- 核心调用 ----

    def chat_with_tools(
        self,
        task_type: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatResp:
        """按 task_type 选层调用, 层内重试, 层间降级, 最底层失败抛。

        定价类任务 (task_type 含 'pricing') 禁止走 r1。
        """
        candidates = self._degrade_candidates(task_type)
        last_err: Optional[BaseException] = None
        for layer_name in candidates:
            layer = self.layers.get(layer_name)
            if layer is None:
                # 该层未配置: 跳到下一候选层 (等价降级)。
                continue
            resp, err = self._call_layer_with_retry(layer, messages, tools)
            if resp is not None:
                return resp
            last_err = err
        # 所有候选层 (含最底层) 全失败: 抛异常, 由 AgentStage 降级 MockBrain。
        raise RuntimeError(
            "Gateway: all layers failed for task_type={0} ({1})".format(
                task_type, last_err
            )
        )

    def _call_layer_with_retry(
        self,
        layer: Layer,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]],
    ) -> tuple:
        """单层调用: 最多 retry+1 次 (初试 + retry 次重试), 指数退避 0.2*2^n。

        返回 (ChatResp, None) 成功; (None, err) 全失败。
        """
        attempts = max(0, int(layer.retry)) + 1
        last_err: Optional[BaseException] = None
        for n in range(attempts):
            if n > 0:
                # 指数退避: 第 n 次重试 (n>=1) 前 sleep 0.2 * 2^(n-1)。
                backoff = _BACKOFF_BASE * (2 ** (n - 1))
                time.sleep(backoff)
            try:
                resp = layer.provider.chat(messages, tools=tools, timeout=layer.timeout)
                return resp, None
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        return None, last_err

    # ---- 月成本熔断 (方案 7.8.4) ----

    @staticmethod
    def _cost_key(tenant_id: int) -> str:
        """月成本计数 key: ai:cost:{tenant}:{YYYYMM} (本地时区)。"""
        ym = time.strftime("%Y%m")  # 当前年月 (方案 7.8.4 月度计数)
        return "ai:cost:{t}:{ym}".format(t=int(tenant_id or 0), ym=ym)

    def add_cost(self, tenant_id: int, cny: float) -> None:
        """累加某户当月 AI 成本 (cny 元)。"""
        key = self._cost_key(tenant_id)
        self._cost_store[key] = self._cost_store.get(key, 0.0) + float(cny or 0.0)

    def get_budget_used(self, tenant_id: int) -> float:
        """某户当月已用成本 (cny 元)。"""
        return float(self._cost_store.get(self._cost_key(tenant_id), 0.0))

    def is_budget_exceeded(self, tenant_id: int) -> bool:
        """是否超月度预算 (>= budget_cny)。"""
        return self.get_budget_used(tenant_id) >= self.budget_cny
