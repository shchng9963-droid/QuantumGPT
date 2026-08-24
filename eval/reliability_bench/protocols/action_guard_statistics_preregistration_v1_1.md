# ActionGuard 方法验证统计预注册 v1.1

状态：在 `method-validation-v2.1` 解封前冻结。

## 假设与计划比较

- H1（Evidence Layer）：Full vs Ledger，以及Full+Guard vs Ledger+Guard。
- H2（ActionGuard）：Ledger+Guard vs Ledger，以及Full+Guard vs Full。
- H3（联合方法）：Full+Guard相对单层方法获得更好的可靠性—覆盖权衡，并相对Global-Revalidate+Guard降低实际重验证成本。

Evidence×Guard交互项预先报告，但不是论文成败的必要条件。只有交互项为正且达到预先规定的统计标准时才声称“超加性协同”；否则只根据四个计划比较讨论条件效应和互补性。

两条路径 `Evidence Layer → stale use → wrong decision` 和 `ActionGuard → valid revalidation → wrong decision` 只称为机制路径分析。没有中介随机化或顺序可忽略性论证时，不作因果中介声明。

## 三个主要终点

1. `task_predicate_correctness`：冻结Task Predicate Evaluator给出的最终动作正确性。
2. `unsafe_or_unknown`：保守证据风险，同时必须拆分报告 `definite_stale_dependence`、`unsupported_decision` 和 `provenance_incomplete`；unknown不得表述为已确认过期依赖。
3. `realized_revalidation_cost`：由真实已接受工具轨迹计算的调用成本单位，并同时报告token、API费用和延迟。

必须同时报告的安全护栏为coverage、feasible completion、correct rejection、selective risk、abstention和unscorable，防止全拒答规避。

## 24条独立方法验证Pilot

`method-validation-v2.1`的24条episode只报告episode配对效应、95%置信区间和方向，不进行或声称正式非劣效检验，不据此确定顶会级总体结论。Global非劣效界值、正式功效目标和最小样本量将在扩展模型、seed、漂移强度和独立模板前另行预注册。

二元终点报告配对风险差及episode配对bootstrap置信区间，并附exact McNemar描述；成本、token和延迟报告配对bootstrap置信区间。保留related/unrelated反事实配对。API失败、timeout、重试耗尽和不可评分不得删除；主要保守分析把不可评分记作失败，并另报complete-case敏感性分析。

## Utility冻结

Utility不是三个主要终点之一，仅作次要综合描述。主权重固定为：

```text
correct = +1.0
wrong = -1.0
unnecessary_abstain = -0.5
cost_unit = -0.01
```

敏感性分析使用预先冻结的笛卡尔网格：`wrong ∈ {0.5,1.0,2.0}`、`unnecessary_abstain ∈ {0.25,0.5,1.0}`、`cost_unit ∈ {0.005,0.01,0.02}`，correct固定为1.0，共27组。不得看结果后增加、删除或替换权重。

## 固定运行规则

六组使用相同模型、temperature、top_p、总token、12轮、10次工具调用、10成本单位、120秒timeout、两次API尝试、1秒固定退避和两次格式修复。Guard最多两次修复且消耗共同轮数。失败trace append-only保存；不得只重跑表现不佳的组。
