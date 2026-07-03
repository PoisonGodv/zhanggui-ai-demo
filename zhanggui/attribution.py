"""掌柜军师 AI 引擎 —— 效果归因 (方案 7.9.1)。

忠实复刻 07_AI引擎详设.md 的 7.9.1 节: 收集每条规则触发的动作产出真实业务
结果 (核销率 / 到店增量 / 营收), 归一化后合成单一 reward 信号, 供 bandit
(7.9.2) 做 epsilon-greedy 自学习调权。

存储: 进程内 list (无 PG); 接口保持 DB 友好, 留好 PG 替换点 (record/查询
均按 rule_id 过滤, 可平滑迁到 OUTCOME 表)。零第三方依赖。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Outcome:
    """单次规则触发动作的效果观测 (方案 7.9.1)。

    一条 advice 里的某条 rule 落地后, 由事后回流埋点 / POS 核销 / 私域群
    活跃统计填入本结构; rule_id 与 advice 的 routed item.rule_id 对齐。
    """
    tenant_id: int
    advice_id: str           # 关联 Advice.version / id
    rule_id: str             # 关联 Rule.id
    action_type: str         # clearance_coupon / rain_delivery / ...
    coupon_sent: int         # 发券量
    coupon_used: int         # 核销量
    visit_delta: int         # 到店/到群增量 (正负均可)
    group_ok: bool           # 群触达是否成功 (私域执行约束)
    revenue: float           # 该动作直接带动的营收 (元)
    measured_at: float       # 时间戳 (time.time())


class OutcomeCollector:
    """效果归因收集器 (方案 7.9.1)。

    进程内存储 list[Outcome]; 所有聚合按 rule_id 分组。
    真实环境替换点: record()->INSERT, by_rule()->SELECT WHERE rule_id=?,
    coupon_rate()/visit_delta_norm()/revenue_norm()->GROUP BY 聚合 SQL。
    """

    def __init__(self) -> None:
        # 存储: 进程内 list (无 PG)。接口保持 DB 友好, 留好 PG 替换点:
        #   record -> INSERT INTO outcome (...) VALUES (...)
        #   by_rule -> SELECT * FROM outcome WHERE rule_id=?
        #   coupon_rate -> SELECT SUM(coupon_used)/SUM(coupon_sent) ... GROUP BY rule_id
        # 真实环境切 PG 时只需替换这 4 个方法的实现, 上层 bandit 无感。
        self._store: List[Outcome] = []

    def record(self, outcome: Outcome) -> None:
        """记录一次效果观测 (幂等追加; 真实环境走 INSERT)。"""
        self._store.append(outcome)

    def by_rule(self, rule_id: str) -> List[Outcome]:
        """取出某条规则的全部观测。"""
        return [o for o in self._store if o.rule_id == rule_id]

    def coupon_rate(self, rule_id: str) -> float:
        """核销率 = sum(used) / max(sum(sent), 1)。

        分母用 max(sent,1) 防零除; 未发券 (sent=0) 时返回 0.0 而非异常。
        """
        rows = self.by_rule(rule_id)
        sent = sum(o.coupon_sent for o in rows)
        used = sum(o.coupon_used for o in rows)
        return used / max(sent, 1)

    def visit_delta_norm(self, rule_id: str) -> float:
        """到店增量归一化到 [0,1]。

        按 max-abs 归一化: 除以 max(abs(visit_delta)) (或 1 防零除);
        负增量映射到 <1 的正值 (不clip负数, 但 reward clip 在 compute_reward)。
        方案 7.9.1 用 max-abs 归一, 这里取 max(abs(...)) 作分母。
        """
        rows = self.by_rule(rule_id)
        if not rows:
            return 0.0
        deltas = [o.visit_delta for o in rows]
        m = max((abs(d) for d in deltas), default=0)
        if m == 0:
            return 0.0
        # 取正向贡献: 用 sum(deltas)/m? 方案只说"归一化到[0,1]"。
        # 这里用 平均增量/max-abs, 落在 [-1,1], 再 clip 到 [0,1] 供 reward。
        avg = sum(deltas) / len(deltas)
        val = avg / m
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        return val

    def revenue_norm(self, rule_id: str) -> float:
        """营收归一化到 [0,1]: sum(revenue)/max(sum(revenue) 或 1)。

        单规则内各观测营收之和 / 该规则营收最大值 (这里用 sum 自身除以 max
        单条营收, 与 visit_delta_norm 同构的 max-abs 思路, 防零除)。
        """
        rows = self.by_rule(rule_id)
        if not rows:
            return 0.0
        revs = [o.revenue for o in rows]
        m = max((abs(r) for r in revs), default=0.0)
        if m == 0.0:
            return 0.0
        avg = sum(revs) / len(revs)
        val = avg / m
        if val < 0.0:
            val = 0.0
        if val > 1.0:
            val = 1.0
        return val

    def compute_reward(self, rule_id: str) -> float:
        """合成 reward = clip(0.5*rate + 0.3*visit_norm + 0.2*rev_norm, 0, 1)。

        权重 0.5/0.3/0.2 对应方案 7.9.1: 核销为主信号, 到店次之, 营收再次。
        无观测的规则返回 0.0 (冷启动视为低 reward, 不被 bandit 利用)。
        """
        rows = self.by_rule(rule_id)
        if not rows:
            return 0.0
        rate = self.coupon_rate(rule_id)
        visit = self.visit_delta_norm(rule_id)
        rev = self.revenue_norm(rule_id)
        reward = 0.5 * rate + 0.3 * visit + 0.2 * rev
        if reward < 0.0:
            reward = 0.0
        if reward > 1.0:
            reward = 1.0
        return reward

    # 便捷: 全量规则 reward 快照 (供 bandit / 看板)
    def reward_snapshot(self, rule_ids: List[str]) -> Dict[str, float]:
        return {rid: self.compute_reward(rid) for rid in rule_ids}
