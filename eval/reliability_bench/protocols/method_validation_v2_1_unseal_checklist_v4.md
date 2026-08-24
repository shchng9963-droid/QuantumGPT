# method-validation-v2.1 解封前检查表 v4

状态：**继续禁止解封，等待研究负责人书面授权**。

- [x] B_t与V*_t分离；Guard只读B_t；Measurement独立计算V*_t。
- [x] Invalidation-Consistency Check仅作一致性检查，不解释为Oracle上界。
- [x] Measurement v3.1支持runtime-generated execution failure且不伪造decision。
- [x] Measurement v3.1 development Gold `1440/1440`，held-out Gold `1449/1449`。
- [x] 7类reason code、feasible/infeasible、成本与selective-risk语义测试通过。
- [x] Guard v1.2核心哈希保持不变；新版本只增加公共运行时闭环。
- [x] 全新72次preflight：72/72闭合、72/72可评分、0 missing-terminal unscorable。
- [x] 3条execution failure保留为ITT方法结果；未补跑、未增加第三次修复。
- [x] 六组公共prompt、模型、工具与预算公平；无ground-truth或答案泄漏。
- [x] 统计预注册v1.3冻结候选，包含execution-failure惩罚和机制指标。
- [x] v2.1兼容性复核：episode evidence ID、依赖、snapshot谱系、任务谓词与工具响应语义均未改变；变化仅在六组共享的公共输出/运行时适配器与外部Measurement版本，因此v2.1继续密封且无需作废。
- [x] 原ActionGuard v1.2 preflight永久记录为`development-only / failed preflight because repair exhaustion lacked a scoreable runtime terminal state`。
- [ ] 研究负责人明确书面批准解封及24×6正式Pilot。

本清单通过不等于自动授权。不得自行解封、读取或重建method-validation-v2.1，不得启动正式144次。
