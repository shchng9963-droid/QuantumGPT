# method-validation-v2.1 解封门禁追加修订

状态：`append-only protocol amendment / archive remains sealed`

原密封清单记录的 Judge v2 门禁已因独立审计失败而永久失效。该失败记录不删除、不覆盖；本修订只替换未来解封所需的评价门禁，不改变任何 episode、证据 ID、依赖、快照谱系、任务谓词或工具响应语义。

新的评价门禁为：

1. `terminal-decision-2.0`、公共 prompt、运行时适配器及六组公平性合同冻结；
2. Task Predicate Evaluator 和 Trace Evidence Evaluator 的候选源码哈希冻结；
3. 独立 development Gold 允许修复，修复历史完整保留；
4. 新 held-out Gold 在预测前加密，评价器预测及 SHA-256 先冻结；
5. held-out 的唯一真值核心指标达到 100% exact agreement；
6. ActionGuard、六组正式配置、预算、重试和日志协议另行冻结；
7. 研究负责人明确批准解封。

截至本修订：评价门禁第 1—5 项已通过；第 6—7 项未完成。因此 `method-validation-v2.1` 继续保持 `unseal_allowed=false`，不得解密或运行。

兼容性判定：`conditionally compatible; no invalidation required`。候选 commit `40e1a61` 未修改 `generator_v2.py`、`schema.py`、`predicates_v2.py` 或工具响应实现。新增终止 Schema 接入、确定性运行时 observation ID 和外部评价器均位于 episode 外部，不要求重写密封集内部字段。

若后续必须改变密封 episode 的内部证据 ID、资源依赖、快照谱系、任务谓词或工具响应语义，则本兼容性判定立即失效，必须整套作废 v2.1 并重新生成，禁止局部修补。
