# ActionGuard 与 Temporal Evidence Layer 形式化规范 v1.2

状态：候选；通过新 Measurement Gold、反事实测试、72 次开发预飞行及冻结清单后冻结。

## F0：信念与真值

对证据 `e` 定义控制器可见信念 `B_t(e) ∈ {valid, invalid, unknown}`，并定义评价器独立重建的客观真值 `V*_t(e) ∈ {valid, invalid}`。`B_t` 不是 `V*_t` 的缓存或别名。

- Ledger 漂移后不更新 `B_t`，保留 last-known-valid；因此允许出现 `B_t(e)=valid ∧ V*_t(e)=invalid`。
- Full 仅由公共 `changed_features` 和方法依赖图更新 `B_t`。
- Guard 只能读取 `B_t`，不能读取或推断 `V*_t`。
- Measurement 仅在运行结束后由 episode 客观资源依赖、漂移和真实轨迹计算 `V*_t`。

关键可证伪反事实为：相同候选终止在 Ledger+Guard 下因 `B_t=valid` 可被允许，但 Measurement 仍因 `V*_t=invalid` 判定 definite stale dependence；Full 更新为 `B_t=invalid` 后，Full+Guard 必须阻止同一引用。无关漂移保持 `B_t=valid` 且不得额外阻塞；成功重验证产生的新证据使两者均恢复 valid。

## 组件边界

Temporal Layer 输出证据信念状态；Recovery Planner 选择恢复哪些证据槽；ActionGuard 只判定候选动作或终止是否满足公开约束。Guard 输出限于 `ALLOW`、`BLOCK(reason_code)`、`REPAIR_REQUEST(reason_code)`，不得给出具体工具参数、backend、qubit、策略或答案。

安全不变量保持为动作 Schema、引用完整性、基于 `B_t` 的证据前置条件、重验证完整性、安全终止与合法动作 non-interference。所有 Guard 组共享修复次数与总预算，Guard 检查、阻塞、修复 token 和延迟均记入成本。

## F1：一致性组

原 `Oracle-Invalidation+Guard` 更名为 `Invalidation-Consistency Check`。在当前合成 Pilot 中，正确完整依赖图使 Full 与客观失效集合信息等价，因此该组仅检查实现一致性，不解释为现实识别上界，也不用于放大 H1。

H1 限定为“依赖图正确且完整的受控条件下，Temporal Evidence Layer 的作用”。未来单独建立缺失依赖、噪声依赖和错误边实验，并以访问真实客观失效集合的组件定义真正 Oracle。

## F2：终止语义

`completion_status ∈ {answered, abstain, execution_failure}` 只描述是否形成任务答案。正确性完全由独立 Task Predicate 判断。对不可达目标，正确 `declare_unreachable` 是 answered 且计为 correct rejection；错误声明计为 wrong decision。`execution_failure` 只表示执行链未产生答案。
