# method-validation-v2.1 解封门禁追加修订 v2

状态：append-only；密封包继续禁止解封。

Measurement v2/terminal-decision-2.0 因不可达任务的隐含 status 约定被 supersede。旧 Gold 与 1197 字段比较保持原样，仅作历史溯源，不再承担解封门禁。替代门禁为：

1. Measurement v3、terminal-decision-3.0、公共 prompt 与六组适配器冻结；
2. 新 development Gold 完成修复审计；
3. 与所有旧审计及 v2.1 分离的新 held-out Gold 在预测前冻结标签，在解封标签前冻结预测；
4. 形式上唯一真值字段全部 100% exact agreement；失败集合永久转 development，修复后换全新 held-out；
5. ActionGuard/Temporal v1.2、72 次真实 LLM 开发预飞行、独立 Guard 压力测试和统计预注册 v1.2 冻结；
6. 按 append-only hash erratum 核验密封 archive 与密钥备份；
7. 研究负责人明确授权。

v2.1 当前判定为“条件兼容并继续密封”：新 completion schema、公共 prompt、运行时适配器与外部评价器不改变其内部证据 ID、资源依赖、快照谱系、任务谓词或工具响应语义。若任一内部语义必须改变，则整套作废并重新生成，禁止局部修补。
