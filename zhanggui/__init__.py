"""掌柜军师 AI 引擎 —— 包标记 (空, 仅标记为 Python 包)。

五段式管线模块各自独立, 通过相对 import (from .types import ...) 协作:
  signal_collector.SignalCollector.collect -> SignalBundle
  rule_engine.RuleEngine.route             -> Routed (毛利护栏在 guardrail.py)
  agent_brain.AgentStage.enhance_or_skip   -> Enhanced (预算熔断+MockBrain 兜底)
  converger.Converger.pick_top3            -> list[EnhancedItem]
  schema_validator.SchemaValidator.encode  -> Advice
  pipeline.Pipeline.generate_today_advice  -> Advice (五段总控, 段间 try/except 降级)

本文件刻意保持为空 (仅 docstring), 避免在 import 包时拉起整条依赖链
(playbook/tools/agent_brain 等), 子模块按需 import 即可。零第三方依赖。
"""
