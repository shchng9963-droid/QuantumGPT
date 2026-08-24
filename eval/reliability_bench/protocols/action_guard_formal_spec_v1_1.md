# ActionGuard v1.1：信念—真值分离规范

状态：冻结候选。本文不授权解封 `method-validation-v2.1`。

## 1. 两类有效性

对证据 \(e\) 和漂移后时刻 \(t\)，定义控制器可见信念：

\[
B_t(e)\in\{\text{believed-valid},\text{believed-invalid},\text{unknown}\}.
\]

它只属于方法内部状态。ActionGuard 的有效性输入严格限定为 \(B_t\)。

独立定义客观有效性：

\[
V_t^*(e)=\mathbf 1[Dep(e)\cap Affected(S_{t-1}\rightarrow S_t)=\varnothing].
\]

\(V_t^*\) 只由冻结 Measurement v2 在运行结束后根据 episode、真实工具轨迹、证据依赖和 snapshot 谱系计算。Guard、Temporal Layer、Recovery Planner和LLM上下文都不能读取 \(V_t^*\)。

因此允许出现 \(B_t(e)=\text{believed-valid}\) 且 \(V_t^*(e)=0\)：这正是 Guard-only 组无法自行解决的客观过期证据。

## 2. 核心2×2

| 组 | 漂移后信念更新 | Guard |
|---|---|---|
| Ledger | 不更新；保留 \(B_t(e)=B_{t-1}(e)\) | 无 |
| Full | Temporal Layer按公开漂移特征与依赖图更新 \(B_t\) | 无 |
| Ledger+Guard | 不更新；Guard仅消费 last-known-valid \(B_t\) | 有 |
| Full+Guard | Temporal Layer更新 \(B_t\)，Guard消费更新结果 | 有 |

Global-Revalidate+Guard将所有旧证据信念设为 invalid，作为成本基线。Oracle-Invalidation+Guard只在该补充组中把真实失效集合写入 \(B_t\)，但Oracle信息仍不能进入Guard代码、公共prompt或其他组。

## 3. 组件边界

- Temporal Evidence Layer：`changed_features + dependency graph → B_t`；它不计算任务答案。
- Recovery Planner：`缺失证据槽 + 抽象恢复动作及成本 → 恢复计划`；它不执行Guard判定。
- ActionGuard：`公共任务契约 + B_t + 候选动作 + 真实已接受调用 → ALLOW/BLOCK/REPAIR_REQUEST`。
- Measurement v2：`完成后的episode + terminal + 真实trace → V*_t及论文指标`；运行期不可访问。

Guard不能自行重新推断漂移、读取评价谓词、读取可接受行动集合、推荐backend/qubit/policy或补工具参数。

## 4. 安全不变量

v1.0的Action Formation、Referential Integrity、Evidence Preconditions、Revalidation Integrity、Safe Finalization和Non-interference保持不变，但其中“有效”统一解释为控制器信念 `believed-valid`，绝不能解释为 \(V_t^*=1\)。

Guard反馈只包含抽象reason code和公共约束。公开实体域、工具参数Schema和工具输出证据槽对六组完全相同；这些是动作语言的公开定义，不是正确动作推荐。

## 5. 正交性反事实门禁

以下测试已程序化并通过：

1. \(B_t=valid,V_t^*=invalid\)：Ledger+Guard允许信念上合法的终止，Measurement判定 definite stale dependence。
2. \(B_t=invalid,V_t^*=invalid\)：Full+Guard阻止引用旧证据。
3. 无关漂移：Full保持 \(B_t=valid\)，Guard不额外阻塞。
4. 成功重验证：新证据在 \(B_t\) 与 \(V_t^*\) 中均有效，两个Guard组均可结束。
5. 固定 \(B_t\) 和候选动作时，改变隐藏 \(V_t^*\) 或可接受答案不会改变Guard输出。

原ActionGuard v1记录永久保留，但状态为“未用于密封验证，因2×2构念混杂被v1.1替代”。
