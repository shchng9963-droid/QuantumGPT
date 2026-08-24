# method-validation-v2.1 解封前检查表 v3

状态：**禁止解封**。

- [x] F0：`B_t` 与 `V*_t` 类型、代码与信息访问分离；关键反事实通过。
- [x] F1：Oracle 组改名为 Invalidation-Consistency Check；H1 限定正确完整依赖图。
- [x] F2：terminal-decision-3.0 与 Measurement v3；不可达任务由谓词判断。
- [x] F3：append-only 哈希勘误；服务器密封 archive 与 key 作为不透明字节重新核验。
- [x] Measurement v3：新 development Gold 42/42；替代 held-out 44 条、1329/1329 字段比较，零分歧。
- [x] 首次 failed held-out 与两套 failed 72-run preflight 均永久留档，门槛未降低。
- [x] 189 项 ReliabilityBench 回归测试通过；Guard 7 类受控压力测试通过。
- [x] 模型、参数、timeout、重试、预算与费用配置签字。
- [x] v2.1 内部语义兼容性复核：无需作废，继续密封。
- [ ] 72 次自然 LLM preflight 的完整率与可评分率达到预注册门禁：当前最佳为 71/72。
- [ ] ActionGuard v1.2 正式冻结：被上一项阻塞。
- [ ] 研究负责人明确书面解封授权。

失败条目不得删除或补跑。下一步只能由研究负责人预先决定：继续坚持严格 72/72；把门禁改为有统计依据且在新数据前冻结的完成率标准；或开发并重新审计一个新的通用恢复机制。不得为了通过门禁增加只针对 mitigation trace 的答案、工具或参数提示。
