# F0—F3 更正报告（候选）

本轮仅修改方法外部接口、Measurement 新版本、开发审计与协议记录；未解封、读取、重建或反向修改 `method-validation-v2.1`。

| 项目 | 更正 | 旧产物处置 |
|---|---|---|
| F0 | 显式类型化分离 `B_t` 与 `V*_t`；Guard 只消费 `B_t` | ActionGuard/Temporal v1.1 保留并 superseded |
| F1 | Oracle 组更名为 Invalidation-Consistency Check；H1 限定正确完整依赖图 | 不再解释为识别上界 |
| F2 | terminal-decision-3.0 使用 answered/abstain/execution_failure；谓词独立判断正确性 | Measurement v2 与旧 Gold 原样保留、标记 superseded |
| F3 | 新增 append-only 哈希勘误 | 六个历史字段不覆盖，未来核验以 erratum 为权威 |

新 Measurement v3 必须用全新 development Gold 修复，并在标签冻结、预测冻结的顺序下通过全新 held-out Gold 的 100% exact agreement。若失败，该 held-out 永久转为 development，修复后换新集合。

语义与泄漏审计使用正向、无歧义字段名；ground-truth payload 及其私有值不得进入六组上下文。`method_validation_v2_1_access.status=not_accessed` 明确仅是 custodian declaration，不冒充技术证明。Windows 换行修复只进入未来 generator，不改 v2.1 的 generator、episode 或哈希。
