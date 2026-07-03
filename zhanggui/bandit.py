"""掌柜军师 AI 引擎 —— epsilon-greedy bandit 自学习 (方案 7.9.2)。

忠实复刻 07_AI引擎详设.md 的 7.9.2 节: 把每条 Rule 视为一个臂 (arm), 用
epsilon-greedy 在「探索(随机)」与「利用(weight 最大)」之间权衡; 每次观测
到 reward (来自 attribution.OutcomeCollector.compute_reward) 后软更新臂权,
再回写成 Rule.priority, 形成规则优先级的自学习闭环。

零第三方依赖 (仅标准库 random/dataclasses)。注意: demo 直接运行时 random
可用 (只在 workflow 评审脚本里禁随机, 见任务说明)。
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Dict, List

from .attribution import OutcomeCollector
from .types import Rule


@dataclass
class Arm:
    """bandit 臂: 对应一条 Rule。

    weight 随 reward 软更新; trials/reward_sum 维护增量均值 q=reward_sum/trials。
    """
    rule_id: str
    weight: float
    trials: int = 0
    reward_sum: float = 0.0


class Bandit:
    """epsilon-greedy 多臂老虎机 (方案 7.9.2)。

    select: 以 epsilon 概率随机探索, 否则取 weight 最大的臂 (利用)。
    update: trials++, reward_sum+=reward, q=reward_sum/trials,
            weight=weight*(1-lr)+q*lr (软更新, lr 越大越激进)。
    reweight_rules: 用 outcomes 算每条 rule 的 reward, 更新对应 arm, 把
            arm.weight 回写成 Rule.priority, 返回新规则集 (不修改原对象)。
    """

    def __init__(self, epsilon: float = 0.1, lr: float = 0.1) -> None:
        self.epsilon = epsilon
        self.lr = lr

    def select(self, arms: List[Arm]) -> Arm:
        """epsilon-greedy 选臂。

        random.random() < epsilon -> 随机一个臂 (探索);
        否则 -> 取 weight 最大的臂 (利用); 并列时取第一个。
        """
        if not arms:
            raise ValueError("select 需要至少一个 arm")
        if random.random() < self.epsilon:
            # 探索: 均匀随机选一个臂
            return random.choice(arms)
        # 利用: 取 weight 最大 (并列取首个, 保确定性顺序)
        best = arms[0]
        for a in arms[1:]:
            if a.weight > best.weight:
                best = a
        return best

    def update(self, arm: Arm, reward: float) -> None:
        """用观测 reward 软更新臂权。

        trials++, reward_sum+=reward, q=reward_sum/trials,
        weight = weight*(1-lr) + q*lr。
        """
        arm.trials += 1
        arm.reward_sum += reward
        q = arm.reward_sum / arm.trials
        arm.weight = arm.weight * (1.0 - self.lr) + q * self.lr

    def arms_from_rules(self, rules: List[Rule]) -> List[Arm]:
        """规则集 -> 臂集: 用 rule.priority 作初始 weight。"""
        return [
            Arm(
                rule_id=r.id,
                weight=float(r.priority),
                trials=0,
                reward_sum=0.0,
            )
            for r in rules
        ]

    def reweight_rules(
        self,
        rules: List[Rule],
        outcomes: OutcomeCollector,
    ) -> List[Rule]:
        """用效果归因软更新规则优先级, 返回新规则集 (不修改原对象)。

        对每条 rule:
          1. outcomes.compute_reward(rule.id) 取该规则 reward;
          2. arms 表里找/建对应 arm (用 rule.priority 作初值);
          3. update(arm, reward);
          4. dataclasses.replace 把 arm.weight 回写成 Rule.priority (int(round)),
             软更新, 不动其它字段。

        返回新 list[Rule]; 原 rules 对象保持不变 (调用方持有旧快照无感)。
        无任何观测的规则 reward=0, 仍会被 update 一次 (trials+1, weight 下移),
        这符合 bandit「未验证的臂权重随试探衰减」的语义。
        """
        # 维护一个 rule_id -> arm 的临时表 (本轮内累乘多次 update 时臂状态连续)
        arms: Dict[str, Arm] = {}
        # 先用规则集初始化臂 (priority 作 weight)
        for r in rules:
            arms[r.id] = Arm(
                rule_id=r.id,
                weight=float(r.priority),
                trials=0,
                reward_sum=0.0,
            )

        # 软更新: 每条 rule 算一次 reward, 更新对应 arm
        for r in rules:
            reward = outcomes.compute_reward(r.id)
            self.update(arms[r.id], reward)

        # 回写 priority (不修改原对象, 用 dataclasses.replace)
        new_rules: List[Rule] = []
        for r in rules:
            arm = arms[r.id]
            # priority 用 int(round(arm.weight)) 保持与 CONTRACT 一致的整型语义;
            # 软更新使高 reward 规则权重相对上升, 低 reward 规则相对下降。
            new_priority = int(round(arm.weight))
            new_rules.append(dataclasses.replace(r, priority=new_priority))
        return new_rules
