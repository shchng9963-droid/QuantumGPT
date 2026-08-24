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

## 执行结果

- F0 反事实、依赖隔离及 189 项 ReliabilityBench 回归测试通过。
- F3 对不透明 archive 与 key 的服务器实算哈希分别为 `4bcd...e132` 与 `d155...2b8f`，与 append-only erratum 一致。
- Measurement v3 首次 held-out 因两处 Gold 声明错误失败并永久转 development；全新替代 held-out 在 44 条轨迹上实现 1329/1329 字段比较零分歧。1329 不是独立样本数。
- 受控 Guard 压力测试覆盖 7 个 reason code，7/7 均完成拦截后合法修复。
- 第一套 72-run 自然预飞行 71/72，未触发自然 Guard 修复；原样留档。
- 第二套全新 72-run 自然预飞行触发 3 次 Guard 干预和 2 次修复，但一条 Global+Guard mitigation 轨迹仍耗尽修复预算，故为 71/72。

因此 Measurement v3 可以冻结，ActionGuard v1.2 仍被自然预飞行门禁阻塞。`method-validation-v2.1` 继续密封，正式 144 次运行继续禁止。
