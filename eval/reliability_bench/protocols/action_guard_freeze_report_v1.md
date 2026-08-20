# ActionGuard v1 冻结报告

日期：2026-08-20
状态：方法候选已冻结；`method-validation-v2.1` 仍保持密封；正式六组实验未获授权。

## 1. 形式化规范与组件边界

ActionGuard 被实现为非指示性的运行时 shield。它只返回 `ALLOW`、`BLOCK(reason_code)` 或 `REPAIR_REQUEST(reason_code)`，不返回正确答案、推荐后端、qubit、缓解策略或工具参数。

三个方法组件已经拆分：Temporal Evidence Layer 是证据有效状态的唯一生产者；Recovery Planner 只求解抽象证据槽的最小成本恢复；ActionGuard 只检查候选动作是否满足公共约束。核心 2×2 中，Ledger+Guard 的账本状态保持 unknown，Guard 不执行漂移推断；Full+Guard 只消费 Full 已产生的 validity state。

## 2. 实现与依赖隔离审计

冻结候选为 `7b4264f5c6c904208dc3427f6ed0161fbef1d910`。源码搜索与 AST 单元测试均确认 Guard 运行时不导入或调用 Task Predicate Evaluator、Trace/Evidence Evaluator、Oracle invalidation、ground truth、acceptable actions、Temporal Layer或Recovery Planner。Measurement v2 的五个冻结文件哈希全部与冻结清单一致。

Guard 判断对任务最终正确答案不敏感：保持候选动作、公开实体域和证据支持不变，仅改变隐藏可接受答案不会改变 ALLOW/BLOCK。相同候选动作在 Ledger unknown 与 Full invalid/valid 状态下只因控制器可见 validity state 产生预期差异。

## 3. 安全不变量与反事实测试

37 项 Guard 单元测试全部通过，覆盖工具 Schema、参数类型、实体绑定、Evidence/tool-call ID、证据槽完整性、stale/unknown/missing support、重验证真实调用绑定、成功输出、Non-interference、固定修复预算以及依赖隔离。

另外生成了与密封集不同的 24 个 development smoke episode，形成 120 个合成正例、反例和变形用例；120/120 符合预期。变形维度包括 validity 单变量变化、删除必要证据槽、伪造 tool-call ID、错误实体绑定、合法动作不变、相关/无关漂移以及答案盲性。

ReliabilityBench-Q 全部相关回归测试为 139/139 通过。

## 4. 旧 24 条 development 结果

旧 B1 轨迹只进行了离线、反事实结构复放，不重新调用 LLM，也不使用失效的 judge v1 标签。Ledger 和 Full 各有 24 条可解析轨迹：Ledger 为 0 ALLOW、13 BLOCK、11 REPAIR_REQUEST；Full 为 7 ALLOW、7 BLOCK、10 REPAIR_REQUEST。

这些数字只说明旧输出与新终止/证据约束的结构兼容程度。它们不是可靠性、错误率或方法性能结论；selective risk 保持不可估计。完整开发审计 SHA-256 为 `d23e96a0b8644d677dc8c83379c3ad20901f43a67613d4437766d9ee05ef5b47`。

## 5. 公共 prompt 与预算公平性

冻结的公共运行时、terminal-decision-2.0 Schema和公共 prompt 未被修改，公共运行时哈希仍为 `ab572138d222fcb99037709cc1864b53209e788166431cacd4b298e4905420a8`。六组继续共享工具目录、12 个 LLM 总轮次、10 次工具调用、解析、timeout、重试和 trace 保存规则。

所有 Guard 组最多有两次修复机会，但修复消耗共同的 12 轮上限，不增加隐藏轮次或工具预算。反馈只有抽象 reason code 和被违反的公共约束。无 Guard 组和 Guard 组的总体 LLM/工具上限一致，所有拦截、失败修复、延迟和候选哈希均可由 trace 复核。

## 6. 统计方案

预注册文件固定 H1–H3、四个 2×2 计划比较、主要/次要终点、拒答与不可评分处理、API 失败和重试、episode 配对分析、二元效应与置信区间、成本配对 bootstrap、交互项和两条机制路径。

交互项不是论文成败的唯一条件；只有正且显著时才声称超加性协同。机制路径在无额外中介随机化时只作机制路径分析。24 个 v2.1 episode 仅作为独立方法验证 Pilot；通过后仍须扩模型、seed、漂移强度和独立任务模板。

## 7. v2.1 兼容性复核

本阶段没有修改 episode 内部证据 ID、依赖关系、快照谱系、任务谓词或工具响应语义。ActionGuard 使用外部公共契约、控制器可见 ledger 和真实 trace，终止输出继续使用已冻结 terminal-decision-2.0。因此现有 v2.1 在语义上条件兼容，无需整套作废。

此判断不包含对密封 episode 的读取或重建；本阶段未解封、未运行，也未查看其结果。

## 8. 正式解封前检查表

- [x] Measurement v2 独立 Gold 门禁 1197/1197。
- [x] Measurement v2 源码和公共运行时哈希未变化。
- [x] ActionGuard 形式化规范、组件矩阵和 reason code 冻结。
- [x] Guard 合成不变量、Non-interference 与依赖隔离测试通过。
- [x] 公共 prompt、terminal Schema、预算和重试公平性复核完成。
- [x] H1–H3、2×2 对照、终点和统计处理预注册。
- [x] v2.1 兼容性复核完成，无 episode 语义变化。
- [ ] 独立位置/不同保管人的密钥备份确认。
- [ ] 正式运行模型精确版本、参数、超时和费用配置最终签字。
- [ ] 研究负责人明确批准解封和一次性 144 次运行。

在最后三项完成前，`method_validation_v2_1_unseal_allowed=false`，不得自行解封或启动正式实验。
