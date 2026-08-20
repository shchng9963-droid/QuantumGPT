# 公共运行时接入与独立 Gold-Trace 评价审计报告

## 结论

本阶段门禁通过。公共运行时已以同一 Schema、prompt、工具、预算、解析与失败规则接入六个预注册接口；ActionGuard 相关组仍为无行为占位。独立 development Gold 最终 39/39，通过后冻结评价器候选 commit `40e1a619b0613453246aa8d9c689d237a942486d`。随后一次性 held-out Gold 共 41 条、22 个唯一真值字段、902 次精确比较全部一致。

这证明当前主要测量在受控轨迹上的可识别性和实现正确性，不构成六组方法性能结论，也不授权解封方法验证集。

## 1. 公共 prompt 与信息访问公平性

六组公共 prompt SHA-256 均为：

`4f2c7392cbcbe3f3b4933c159f007d7fc4b93637ca61bfd5ebafc5dc3590a2bb`

六组共享：

- `terminal-decision-2.0`及完全相同的字段说明；
- 八个工具及固定成本；
- 4096 token、12轮、10次工具调用、10成本单位；
- 120秒超时、2次API尝试、固定退避；
- 2次相同的通用格式修复机会；
- append-only失败与trace保存规则；
- 确定性`obs:<trace_id>:<call_index>`证据ID。

唯一差异是预注册组件：Evidence Layer、失效策略、ActionGuard接口状态及Oracle失效私有通道。公共 prompt 不包含失效证据集合、推荐动作或任务谓词。Full不会通过prompt获知应重跑的对象。

## 2. 运行时适配器的语义不干预

适配器只执行：JSON语法解析、严格Schema校验、已知证据ID核验、`tool_call_id`与真实accepted call绑定，以及原样trace保存。

它不能：改写动作、补证据ID、读取任务谓词、推荐可接受行动、把自报重验证变成真实调用。格式错误只收到同一条通用Schema提醒；语义正确与否不触发修复。ID错误保留原始TerminalDecision和错误记录，不生成替代ID。

旧B1中“检测失效引用后要求模型删除引用”的逻辑没有复用到新运行时。

## 3. Gold-Trace覆盖矩阵

| 维度 | held-out覆盖 |
|---|---:|
| 六类任务 | 15 / 4 / 4 / 4 / 8 / 6 |
| related / unrelated drift | 32 / 9 |
| safe / unsafe / unknown | 20 / 7 / 14 |
| success / failure / abstain | 36 / 3 / 2 |
| 合法 / 伪造tool_call_id | 40 / 1 |
| snapshot谱系有效 / 无效 | 40 / 1 |
| evidence closure完整 / 缺失 | 40 / 1 |

此外覆盖工具成功、失败、超时，查询旧证据但不依赖、明确引用旧证据、重验证已执行/未执行/错误输出，mitigation四阶段，重复与无关调用，成本及延迟边界。

反事实/变形对包括：漂移相关性、最终引用ID、工具成功状态、snapshot父子关系、单个父证据缺失、可接受行动成员资格、合法与伪造调用ID。

## 4. development与held-out分离

development Gold 39条，允许发现与修复。首次9条不一致被分类为：8个Gold规范/ID错误、1个延迟浮点规范化实现Bug、0个构念不可识别。修复后39/39。

held-out使用新随机命名空间重写case、证据、资源和snapshot标识，与development、Judge v2审计及未来ActionGuard数据隔离。它不导入任何旧审计数据，也未读取密封v2.1。Gold标签使用AES-256-GCM加密；预测文件SHA-256 `c5508df1...c145b72`先冻结，之后才解密比对。held-out失败时不得在同一集合上修复的规则已编码为一次性预测拒绝。

## 5. 测量可识别性

| construct | required observables | evaluator | ambiguity condition | status | paper claim |
|---|---|---|---|---|---|
| 最终动作正确性 | terminal decision、冻结任务谓词 | Task Predicate | 可接受行动集合未定义 | primary | 当前快照下决策正确性 |
| safe/unsafe/unknown | support IDs、证据图、有效性、Req(d) | Trace Evidence | 出处缺失时只能判unknown | primary | 决策证据安全状态 |
| definite stale dependence | 被引用ID、显式漂移失效 | Trace Evidence | 只查询未引用不构成依赖 | primary | Evidence Layer减少旧证据依赖 |
| support adequacy | Req(d)、当前有效槽覆盖 | Trace Evidence | 未注册证据类型不提供槽 | primary | 决策支持充分性 |
| required revalidation | 初始失效槽、成功当前调用 | Trace Evidence | Req(d)未定义则不可识别 | primary | 选择性恢复必要证据 |
| intervention stages | 策略请求、匹配调用、返回与验证证据 | Trace Evidence | 策略身份无法绑定 | primary | requested→verified |
| coverage / selective risk | status、predicate、feasibility | Task Predicate | feasibility不可执行 | primary | 防止全拒答规避 |
| actual cost | accepted calls、成本表、延迟 | Trace Evidence | 未记录的服务端工作不在估计量内 | primary | 可靠性—成本优势 |
| semantic failure stage | 完整轨迹、人工codebook | 人工/语义诊断分类器 | 多个阶段均合理 | secondary | 仅定性失败分析 |

unknown与unsafe始终分开；`unsafe_or_unknown`仅为保守风险指标。失败/超时的当前工具输出属于unsupported/unknown，不会被误称为过期证据。Agent自报不能覆盖真实工具轨迹。

## 6. 冻结门禁结果

held-out 41条中，下列22个字段均为41/41：Task Predicate、三值状态、stale dependence、support adequacy、必要/有效重验证、四阶段干预、snapshot谱系、证据闭包、调用/失败/重复/无关计数、成本、延迟和运行时ID绑定。

总计902/902精确一致，0 mismatch。没有需要进入“实现Bug / Gold错误 / 构念不可识别”复核的held-out项。

## 7. v2.1兼容性

仅读取了`sealed_manifest.json`；未读取或解密`method_validation_v2.tar.aesgcm`。候选提交对`generator_v2.py`、`schema.py`、`predicates_v2.py`和工具响应语义的diff为空。新增内容均为统一终止输出、外部运行时绑定和外部评价。

因此v2.1为条件兼容，无需整套作废，继续密封。原清单中的Judge v2解封条件通过append-only修订替换为新的Gold评价门禁；但ActionGuard与正式配置尚未冻结，研究负责人也尚未批准，所以当前仍不得解封。

## 8. 研究定义是否需要修改

不需要。development阶段的差异均可归为Gold实现错误或数值规范化Bug，没有出现“给定规定观测仍无法唯一判断”的构念问题。三值证据状态、Req(d)、重验证和干预阶段定义保持v1.1不变。

本阶段没有修改ActionGuard，没有生成新方法验证集，没有运行六组正式LLM实验，也没有解封v2.1。
