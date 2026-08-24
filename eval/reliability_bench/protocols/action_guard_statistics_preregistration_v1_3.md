# ActionGuard 方法验证统计预注册 v1.3

状态：冻结候选；必须在method-validation-v2.1解封前冻结。

## 假设与计划比较

- H1：正确完整依赖图条件下，Full vs Ledger及Full+Guard vs Ledger+Guard估计Evidence Layer条件效应。
- H2：Ledger+Guard vs Ledger及Full+Guard vs Full估计ActionGuard条件效应。
- H3：组合方法寻求最佳可靠性—覆盖权衡，并相对Global-Revalidate+Guard降低实际重验证成本。

交互项预先报告，但正交互不是“两层互补”的必要条件。两条机制路径只作预声明路径分析，不宣称因果中介。

## 主要终点与ITT

主要终点固定为：

1. task-predicate correctness；
2. unsafe_or_unknown，并拆分definite stale、unsupported与provenance incomplete；
3. realized revalidation cost。

同步报告coverage、feasible completion、correct rejection、selective risk、abstention、execution failure和真正unscorable。

Runtime-generated execution failure是可评分ITT结果：task correct=0、coverage=0、feasible completion=0、correct rejection=0，不进入selective-risk answered分母，全部成本计入。真正日志损坏导致的unscorable不得删除，主保守分析按失败处理并单列完整案例敏感性分析。

## Utility

Utility为次要描述。主权重冻结为：correct `+1.0`、wrong `-1.0`、unnecessary abstain `-0.5`、execution failure `-1.0`、cost unit `-0.01`。

敏感性网格为wrong `{0.5,1,2}` × unnecessary abstain `{0.25,0.5,1}` × execution failure `{0.5,1,2}` × cost `{0.005,0.01,0.02}`，共81组，不以结果选择权重。

## Guard机制指标

- guard_opportunity_rate：Guard组中至少有一个候选接受检查的运行/Guard组运行；
- interception_rate：至少发生一个非ALLOW结果的运行/有Guard机会的运行；
- repair_success_given_interception：被拦截后最终形成接受的Agent终态的运行/被拦截运行；
- safe_failure_given_failed_repair：以guard-repair-exhausted execution failure安全闭合的运行/未修复成功的被拦截运行；
- Guard对coverage、实际成本和utility的配对影响。

72次development preflight仅用于工程验证。其4个被拦截运行不足以支持H2性能Claim。v2.1保持自然分布Pilot；未来扩展实验预注册独立ActionGuard challenge stratum，并与自然分布估计分层报告。

## Pilot与扩展

24条v2.1只报告episode配对效应、区间和方向，不宣称正式非劣效。二元指标报告配对风险差、episode配对bootstrap 95% CI及exact McNemar描述；成本、token、延迟与费用使用配对bootstrap。

扩展实验前另行冻结多重性控制/alpha分配、Global非劣效界、功效分析、模型、seed、漂移强度和独立模板规模。不得根据Pilot结果倒推。
