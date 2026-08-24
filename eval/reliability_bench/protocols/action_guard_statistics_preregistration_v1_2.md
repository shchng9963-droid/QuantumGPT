# ActionGuard 方法验证统计预注册 v1.2

状态：候选；必须在 `method-validation-v2.1` 解封前冻结。

## 研究假设与比较

- H1：在正确完整依赖图条件下，Full vs Ledger 及 Full+Guard vs Ledger+Guard 估计 Evidence Layer 的条件效应。
- H2：Ledger+Guard vs Ledger 及 Full+Guard vs Full 估计 ActionGuard 的条件效应。
- H3：组合方法寻求最佳可靠性—覆盖权衡，并相对 Global-Revalidate+Guard 降低实际重验证成本。交互项预先报告，但正交互不是“互补性”成立的必要条件。

`Invalidation-Consistency Check` 是实现一致性检查，不是 Oracle 上界。Evidence→stale use→wrong decision 与 Guard→valid revalidation→wrong decision 仅作机制路径分析，不宣称因果中介。

## 主要终点与护栏

三个主要终点固定为：`task_predicate_correctness`、`unsafe_or_unknown`（并拆分 stale/unsupported/provenance incomplete）、`realized_revalidation_cost`。同步报告 coverage、feasible completion、correct rejection、selective risk、abstention、execution failure 和 unscorable。

24 条 v2.1 Pilot 只报告配对效应、区间和方向，不宣称正式非劣效或顶会级总体结论。二元指标报告配对风险差、episode 配对 bootstrap 95% CI 和 exact McNemar 描述；成本、token、延迟和费用报告配对 bootstrap CI。失败与不可评分不删除，主保守分析按失败处理并报告 complete-case 敏感性分析。

## Utility 与扩展实验

Utility 仅为次要描述。主权重固定：correct `+1.0`，wrong `-1.0`，unnecessary abstain `-0.5`，cost unit `-0.01`。敏感性网格：wrong `{0.5,1,2}` × unnecessary abstain `{0.25,0.5,1}` × cost `{0.005,0.01,0.02}`，共 27 组。

正式扩展实验前另行冻结：三个主要终点与四项计划比较的多重性控制/alpha 分配、Global 非劣效界、功效分析、模型、seed、漂移强度与独立模板规模。不得根据 Pilot 结果倒推这些选择。

## 失败、重试与公平性

六组共享同一模型、参数、timeout、API 重试、格式修复、总轮数、工具和成本预算。Guard 修复消耗共同预算。失败 trace append-only 保存，禁止选择性重跑。
