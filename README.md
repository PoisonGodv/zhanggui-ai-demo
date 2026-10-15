# 掌柜军师 · AI 引擎最小 Demo

> 忠实复刻《掌柜军师·技术实施方案》第 7 章 [07_AI引擎详设.md](../掌柜军师-技术方案/07_AI引擎详设.md) 的**五段式管线 + 毛利护栏**，用 Python 3 标准库实现，**零第三方依赖、无 LLM API key 亦可跑出完整效果**。

## 这是什么

掌柜军师的产品核心卖点是「**今日 3 件事 + 一键执行**」——每天给个体零售老板 3 条可执行的营销建议（临期清仓/雨天秒杀/沉睡唤醒/节日满减…），并保证不亏钱。本 demo 复刻其 AI 引擎的工程骨架：

```
Signal Collector → Rule Router(毛利护栏) → LLM-Agent(tool-calling) → Converger(Top3) → Schema Validator
       ↑                  ↑                      ↑                      ↑                  ↑
   信号采集           确定性路由+护栏        受众/文案/效果估算        按预期增量排序      copy≤40字+敏感词+margin_ok
```

## 设计要点（忠于方案）

| 方案要求 | 实现位置 | 如何保证效果 |
|---|---|---|
| 五段式管线（7.1） | `pipeline.py` | 五段顺序，每段 try/except 降级不中断 |
| 毛利护栏 `final_price ≥ cost×(1+margin_floor)`（7.2.3） | `guardrail.py` | 路由段 + Agent 段 + 工具段三处校验；浮点 1e-9 容差 |
| 7 个 Agent 工具 + tool-calling（7.3） | `tools.py` + `agent_brain.py` | `generate_coupon` 内嵌护栏，不过返回 `margin_violation` |
| 7 条信号→动作映射规则（7.2.2） | `rules_seed.py` | 临期清仓/滞销捆绑/雨天外卖/沉睡唤醒/节日满减/生日关怀/高毛利push |
| 5 业态剧本库（7.10） | `playbook.py` | fruit/convenience/maternity/fresh/apparel + cold_priority |
| 文案 copy≤40字 + 敏感词过滤（7.4.4） | `schema_validator.py` | 先过滤（最→很/第一→领先…）再截断 |
| 预算熔断 超¥2降级（7.8.4） | `agent_brain.py` AgentStage | `is_budget_exceeded` → 走 MockBrain |
| 冷启动新店首日有建议（7.10.2 + 6.6） | `pipeline.py` `_inject_cold_start` | 常规命中0条时注入 cold_priority 满减动作 |
| LLM 只做最后一公里（7.5） | `MockBrain`/`LLMBrain` 可插拔 | 无 key 用 MockBrain（确定性，永远出效果）；有 key 用 LLMBrain（真模型润色，失败降级 MockBrain） |

## 运行

```bash
cd zhanggui-ai-demo
python3 run_demo.py          # 跑 3 个情境，打印五段 trace + 今日3件事
python3 -m unittest tests.test_guardrail tests.test_pipeline -v   # 9 个测试
```

### 三个演示情境

1. **水果店雨天临期**（tenant=1001）：阳光玫瑰葡萄临期2天 + 草莓临期1天，下雨，沉睡18人 → 产出雨天秒杀/临期清仓等建议
2. **母婴店沉睡唤醒**（tenant=2002）：金领冠奶粉高毛利，母亲节倒计时3天，沉睡25人，生日1人 → 产出高毛利push/唤醒券/节日满减
3. **冷启动新店**（tenant=3003）：无临期/无高毛利/无沉睡/无雨无节，data_quality=30 → 常规规则命中0条，冷启动注入 wake_dormant 通用满减，**新店首日也有建议**

## 毛利护栏黄金案例（DoD #2）

```
cost=12, price=25.8, margin_floor=0.05 → floor=12.6
guardrail_ok(12, 15.9, 0.05) = True   # 15.9 ≥ 12.6 通过
guardrail_ok(12, 12.0, 0.05) = False  # 12.0 < 12.6 拒绝
min_discount_floor ≈ 0.488            # 折扣不得低于此值
```

红队验证（5 个攻击向量 + 2 个补充向量）全部失败：无亏钱建议泄漏、无超长/敏感文案泄漏。

## M3-S2/S3 能力（降本 + 自学习）

在五段式管线之上追加四块能力（方案 7.6-7.9），仍零依赖可跑：

| 能力 | 模块 | 要点 |
|---|---|---|
| 语义缓存（7.6） | `embedding.py` + `cache.py` | `LocalEmbedder` 零依赖确定性 embedding + `SemanticCache`；**强区分前缀**（weather/actions/days_to_expiry/holiday）精确匹配 + embedding 细筛，杜绝单维度信号变化误命中 |
| 模型网关分层（7.8） | `gateway.py` | lite/V3/R1 路由 + 超时/重试/降级链；**R1 不在降级链**（便宜任务失败升级到 V3，绝不升到最贵 R1）；定价禁 R1；预算¥2/户/月熔断 |
| 凌晨预生成（7.7） | `pregen.py` | `Scheduler.batch_generate` 分批+重试+幂等（version 覆盖）；`incremental_retry`/`trigger_incremental` 增量重算；>20%失败降级标记 |
| 效果归因 + bandit（7.9） | `attribution.py` + `bandit.py` | 券核销率/到店反推/拼团成团率采集；reward=0.5·rate+0.3·visit+0.2·rev（clip[0,1]）；epsilon-greedy + 软更新权重回写 rule.priority |

run_demo 情境四演示缓存 miss→hit，情境五演示预生成 3 户 + 核销回填 + bandit 调权对比。

红队再次验证：缓存单维度信号变化（仅天气/仅临期消失）不再误命中；网关降级链符合方案 7.8.3。

## 接真 LLM（可选）

`MockBrain` 已能产出完整效果。若要接真模型润色，编辑 `run_demo.py` 的 `build_pipeline()`：

```python
from zhanggui.agent_brain import LLMBrain
brain = LLMBrain(api_key="sk-...", base_url="https://api.deepseek.com", model="deepseek-chat")
# AgentStage(brain, budget_cny=2.0)  # 超¥2/户/月自动降级 MockBrain
```

支持通义百炼/DeepSeek/火山方舟等 OpenAI 兼容端点（方案 7.8）。

## 文件结构

```
zhanggui-ai-demo/
├── run_demo.py                  # 演示入口（5 情境：3 基础 + 缓存 + 预生成bandit）
├── zhanggui/
│   ├── types.py                 # 14 个 dataclass（管线契约）
│   ├── guardrail.py             # 毛利护栏（7.2.3）
│   ├── signal_collector.py      # 信号采集 + 2 内置情境（7.1.3）
│   ├── rules_seed.py            # 7 条种子规则（7.2.2）
│   ├── rule_engine.py           # 规则路由（7.2.1）
│   ├── playbook.py              # 5 业态剧本库（7.10）
│   ├── tools.py                 # 7 个 Agent 工具（7.3）
│   ├── agent_brain.py           # MockBrain/LLMBrain/AgentStage（7.3.2+7.8.4）
│   ├── converger.py             # Top3 收敛（7.1）
│   ├── schema_validator.py      # 输出校验（7.4.4）
│   ├── pipeline.py              # 五段式总控 + 冷启动注入 + 缓存前置（7.1.1+7.10.2+7.6.2）
│   ├── embedding.py             # Embedder/LocalEmbedder/QwenEmbedder（7.6.1）
│   ├── cache.py                 # 语义缓存 + 强区分前缀（7.6.2-7.6.5）
│   ├── gateway.py               # 模型网关分层 lite/V3/R1 + 降级链 + 预算熔断（7.8）
│   ├── pregen.py                # 凌晨预生成 Scheduler（7.7）
│   ├── attribution.py           # 效果归因 OutcomeCollector（7.9.1）
│   └── bandit.py                # epsilon-greedy bandit 权重学习（7.9.2）
└── tests/
    ├── test_guardrail.py        # 护栏黄金案例
    ├── test_pipeline.py         # 管线集成 + 冷启动 + 预算熔断
    └── test_m3.py               # 语义缓存/网关/预生成/归因bandit + 误命中回归
```

## 与生产方案的差距（demo 简化项）

- 无 PostgreSQL/PGVector/Redis：语义缓存、规则热更新、预算计数用内存占位（接口留好 PG 替换点）
- 无多租户 DB 行级隔离（demo 单进程内存）
- 无真实 cron 守护（预生成用手动调 `batch_generate`；`start_cron` 提供 interval 循环 demo）
- 归因到店反推公式/拼团成团率/T+24h·72h 二次回填/bandit trials 跨日持久化属生产能力，demo 简化为单次采集+单轮 update
- LLM 段默认走 MockBrain（确定性），真模型为可选增强

这些是「最小 demo」的有意裁剪，核心五段式 + 护栏 + 冷启动 + 缓存/网关/预生成/归因bandit 逻辑完整可跑。
