"""掌柜军师 AI 引擎 —— Embedder 抽象 (方案 7.6.1)。

职责: 把文本映射成定长稠密向量, 供语义缓存 (zhanggui/cache.py) 计算余弦相似度。

设计要点 (忠实复刻方案 7.6.1):
- Embedder 是 Protocol (鸭子类型), 同 Brain/Provider 一样: 接口 + 默认零依赖实现 +
  有 key 时切真实现。
- LocalEmbedder: 零第三方依赖, 确定性 hashing-trick BoW。
  * 对文本做字符 bigram (中文友好, 无空格也能成段)。
  * 再按 ' ' / '/' / '|' / '=' 切 token (覆盖 make_key_text 拼出来的
    "fruit | weather=rain | actions=..." 这类结构化文本的分词)。
  * 每个 token / ngram 经 hashlib.md5 (跨进程稳定, 不受 PYTHONHASHSEED 影响) 取
    整数, mod dim 映射到某维, 累加 +1/-1 (用 hash 符号位交替以避免全正导致高维
    共线性), 最后 L2 归一化。
  * 同文本 -> 同向量 (无随机); 空文本 -> 零向量 (归一化时除零保护 -> 原样零向量)。
  * 默认 dim=256, 足以区分本 demo 的几十种 key_text 形态。
- QwenEmbedder: urllib 调阿里 DashScope OpenAI 兼容 /embeddings, 1024 维;
  api_key 为空立即 raise ValueError (与 LLMBrain/Gateway 的「无 key 走 mock」约定
  一致: 真 provider 只在有 key 时实例化)。
- cosine_similarity: 标准库实现, 处理零向量 (返回 0.0, 避免除零)。

零第三方依赖, 仅标准库 (hashlib / urllib / json / math)。
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from typing import List, Protocol


class Embedder(Protocol):
    """文本嵌入器接口 (方案 7.6.1)。"""

    def embed(self, text: str) -> List[float]:
        ...  # pragma: no cover

    def dim(self) -> int:
        ...  # pragma: no cover


# === 本地零依赖确定性 Embedder ===

class LocalEmbedder:
    """零依赖确定性文本嵌入: hashing-trick BoW, L2 归一化, dim=256。

    向量构造 (确定性, 无随机):
      1. 字符 bigram: 对全文滑窗取相邻字符对 (中文友好)。
      2. 结构化 token: 按 ' ' / '/' / '|' / '=' 切分 (适配 make_key_text 的结构化文本)。
      3. 每个 token/ngram 用 hashlib.md5 -> int (跨进程稳定, 不受 PYTHONHASHSEED 影响)。
      4. 散列到 dim 维: index = h % dim; 符号 = +1 / -1 (h//dim 奇偶决定), 累加。
      5. L2 归一化 (空文本 / 全零向量保持零向量, 不除零)。

    同文本必同向量; 不同文本因 hashing collision 可能部分重合, 但在 key_text
    这种结构化短文本上区分度足够支撑缓存命中判定 (距离阈值 0.15)。
    """

    def __init__(self, dim: int = 256):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = int(dim)

    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        text = "" if text is None else str(text)
        vec = [0.0] * self._dim
        if not text:
            return vec

        # 收集所有 token: 字符 bigram + 结构化切分 token。
        tokens: List[str] = []

        # (a) 字符 bigram (覆盖中文无空格段, 如 "阳光玫瑰葡萄临期")。
        chars = text
        if len(chars) >= 2:
            for i in range(len(chars) - 1):
                tokens.append(chars[i:i + 2])
        elif len(chars) == 1:
            tokens.append(chars)

        # (b) 结构化 token: 按 ' ' / '/' / '|' / '=' 切分 (覆盖 make_key_text 输出)。
        # 简易手写切分, 避免引入 re 之外依赖 (re 也是标准库, 但这里更显式)。
        buf: List[str] = []
        for ch in text:
            if ch in (" ", "/", "|", "="):
                if buf:
                    tokens.append("".join(buf))
                    buf = []
            else:
                buf.append(ch)
        if buf:
            tokens.append("".join(buf))

        # hashing-trick 累加。
        for tok in tokens:
            if not tok:
                continue
            h = self._stable_hash(tok)
            idx = h % self._dim
            sign = 1.0 if (h // self._dim) % 2 == 0 else -1.0
            vec[idx] += sign

        # L2 归一化 (除零保护)。
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def _stable_hash(token: str) -> int:
        """跨进程稳定的整数 hash: md5 前 8 字节 -> int。

        Python 内置 hash(str) 在进程间会被 PYTHONHASHSEED 随机化, 不能用于
        需要跨进程/重启一致的缓存 key 场景; 故用 hashlib.md5 取整。
        """
        digest = hashlib.md5(token.encode("utf-8")).digest()
        # 取前 8 字节大端转 int, 保证非负。
        return int.from_bytes(digest[:8], "big", signed=False)


# === Qwen 真 Embedder (可选, 需 key) ===

class QwenEmbedder:
    """阿里 DashScope (OpenAI 兼容) 文本嵌入: urllib 调 /embeddings, 1024 维。

    仅在有 api_key 时实例化; api_key 为空立即 raise ValueError (上层据此切回
    LocalEmbedder 的零依赖确定性实现, 保证 demo 永远能跑)。

    生产可换 base_url/model 适配其它 OpenAI 兼容服务 (火山方舟 / 自建 vLLM 等)。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v3",
        timeout: float = 5.0,
    ):
        if not api_key:
            raise ValueError("QwenEmbedder requires non-empty api_key")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        # DashScope text-embedding-v3 默认 1024 维。
        self._dim = 1024

    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        payload = json.dumps(
            {"model": self._model, "input": "" if text is None else str(text)}
        ).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + "/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        # OpenAI 兼容格式: {"data":[{"embedding":[...]}], ...}
        embedding = (data.get("data") or [{}])[0].get("embedding") or []
        # 若服务端返回维度与预期不符, 以实际为准并刷新 dim。
        if embedding and len(embedding) != self._dim:
            self._dim = len(embedding)
        return [float(x) for x in embedding]


# === 余弦相似度 (标准库) ===

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """标准余弦相似度; 任一零向量返回 0.0 (避免除零, 语义上「无信息」)。"""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(n):
        x = a[i]
        y = b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom <= 0.0:
        return 0.0
    return dot / denom
