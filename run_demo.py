"""掌柜军师 AI 引擎 —— 五段式管线演示入口。

无参数运行:
    python3 run_demo.py

跑三个典型情境 (方案 7.4.2 / 7.4.3 / 7.10.2):
  1) 水果店雨天临期 (阳光玫瑰葡萄临期2天 + 草莓临期1天, 雨, 沉睡18人)
  2) 母婴店沉睡唤醒 (金领冠奶粉高毛利, 母亲节 days_to=3, 沉睡25人, 生日1人)
  3) 冷启动新店 (无临期/无高毛利/无沉睡/无雨无节, data_quality=30) — 验证
     常规规则命中0条时, 冷启动注入 cold_priority 动作, 新店首日也有建议

对每个 scenario 跑一次 Pipeline.generate_today_advice(tenant_id, scenario),
打印五段 trace (collect/route/enhance/converge/encode 的可读快照) + 最终
今日3件事 (编号 / action_type / product / copy / params / expected_effect /
margin_ok)。中文 print, 用 json.dumps(ensure_ascii=False, indent=2) 排版。

组装 (依赖注入): MockBrain 作为 brain, 默认预算 (budget_cny=2.0, 月成本 0)。
零第三方依赖, 仅标准库。
"""

import json

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


# 段名中文化 (trace.stages[*].stage 字段映射)。
_STAGE_CN = {
    "collect": "1) 信号采集 collect",
    "route": "2) 规则路由 route (毛利护栏在段内强制)",
    "cold_start": "2.5) 冷启动注入 cold_start (常规命中0条时注入剧本库动作)",
    "enhance": "3) Agent 增强 enhance (预算熔断+MockBrain 兜底)",
    "converge": "4) 收敛 converge (Top3 按预期增量价值排序)",
    "encode": "5) 输出校验 encode (rank1-3/copy<=40/敏感词/margin_ok)",
}


def _print_separator(title: str) -> None:
    line = "=" * 70
    print("\n" + line)
    print(title)
    print(line)


def _print_trace(trace: dict) -> None:
    """打印五段 trace 快照 (每段产出物的可读 dict)。"""
    tid = trace.get("tenant_id")
    print("租户 tenant_id = {0}".format(tid))
    degradations = trace.get("degradations") or []
    if degradations:
        print("降级记录: {0}".format(json.dumps(degradations, ensure_ascii=False)))
    else:
        print("降级记录: 无 (五段全部正常)")
    print("-" * 70)
    for st in trace.get("stages") or []:
        name = st.get("stage", "")
        snapshot = st.get("snapshot", {})
        cn = _STAGE_CN.get(name, name)
        print("\n[{0}]".format(cn))
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))


def _print_advice(advice) -> None:
    """打印最终今日3件事 (Advice.items)。"""
    print("\n" + "-" * 70)
    print("今日3件事 (Advice, version={0}, date={1})".format(
        advice.version, advice.date
    ))
    print("-" * 70)
    if not advice.items:
        print("(无建议产出 — 全部被护栏拦截或信号不足)")
        return
    for it in advice.items:
        print("\n#{rank}  {action_type}  |  商品: {product}".format(
            rank=it.rank, action_type=it.action_type, product=it.product
        ))
        print("  copy           : {0}".format(it.copy))
        print("  params         : {0}".format(
            json.dumps(it.params, ensure_ascii=False)
        ))
        print("  expected_effect: {0}".format(it.expected_effect))
        print("  margin_ok      : {0}".format(it.margin_ok))
        print("  reason         : {0}".format(it.reason))


def build_pipeline() -> Pipeline:
    """依赖注入组装五段式管线: MockBrain 作 brain, 默认预算 (¥2/户/月, 月成本0)。"""
    return Pipeline(
        collector=SignalCollector(),
        router=RuleEngine(default_rules()),
        agent_stage=AgentStage(MockBrain(), budget_cny=2.0, monthly_cost=0.0),
        converger=Converger(),
        validator=SchemaValidator(),
    )


def run_scenario(pipeline: Pipeline, tenant_id: int, scenario: dict,
                 title: str) -> None:
    """对单个 scenario 跑一次管线, 打印 trace + 最终3件事。"""
    _print_separator(title)
    advice = pipeline.generate_today_advice(tenant_id, scenario)
    _print_trace(pipeline.get_trace())
    _print_advice(advice)


def scenario_cold_start_new_store() -> dict:
    """冷启动新店情境 (方案 7.10.2 + 6.6): 无临期/无高毛利/无沉睡/无雨无节
    无生日, data_quality=30。常规规则命中 0 条, 触发 cold_start 注入。"""
    return {
        "business": "fruit", "city": "某地级市", "date": "2026-07-03",
        "weather": {"condition": "normal", "temp_c": 25.0, "forecast": "晴"},
        "holiday": {"name": "无", "days_to": 99, "is_festival": False},
        "inventory": [{
            "product_id": 1, "name": "普通苹果", "cost": 5.0, "price": 8.0,
            "stock": 100, "days_to_expiry": -1, "slow_moving": False,
            "high_margin": False,
        }],
        "members": {"sleep_30": 0, "high_value": [], "birthday_today": []},
        "data_quality": 30,
    }


def main() -> None:
    pipeline = build_pipeline()
    print("掌柜军师 AI 引擎 —— 五段式管线演示 (MockBrain, 无 LLM, 零第三方依赖)")
    print("管线: collect -> route(护栏) -> enhance(预算熔断) -> converge(Top3) -> encode")
    run_scenario(
        pipeline, tenant_id=1001,
        scenario=scenario_fruit_rain_expiry(),
        title="情境一: 水果店雨天临期 (tenant=1001, 临沂, 雨天, 阳光玫瑰葡萄临期2天)",
    )
    run_scenario(
        pipeline, tenant_id=2002,
        scenario=scenario_maternity_dormant(),
        title="情境二: 母婴店沉睡唤醒 (tenant=2002, 徐州, 母亲节 days_to=3, 沉睡25人)",
    )
    run_scenario(
        pipeline, tenant_id=3003,
        scenario=scenario_cold_start_new_store(),
        title="情境三: 冷启动新店 (tenant=3003, 无临期/无高毛利/无沉睡, dq=30, 验证新店首日有建议)",
    )
    print("\n" + "=" * 70)
    print("演示完成。三个情境均产出建议, 毛利护栏全程强制, 冷启动新店首日有兜底。")
    print("=" * 70)


if __name__ == "__main__":
    main()
