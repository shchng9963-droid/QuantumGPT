# ActionGuard v1.1 冻结与真实LLM预飞行报告

日期：2026-08-24

## 结论

ActionGuard v1.1工程门禁通过，核心2×2中的证据信念构念混杂已消除。新的36条真实LLM development preflight达到36/36完整、36/36可评分，返回模型均为`deepseek-v4-flash`，没有API失败、预算耗尽、最终解析失败、全拒答或Guard反馈答案泄漏。

本报告不批准解封`method-validation-v2.1`，也不批准正式144次运行。所有方法性能数字仍为development-only。

## 1. B_t与V*_t分离

方法内部新增controller-visible belief validity \(B_t(e)\)。Ledger在漂移后保留last-known-valid；Full才通过公开变化特征和依赖图更新 \(B_t\)。Ledger+Guard只消费未更新信念，因此能够允许“信念有效、客观失效”的证据。冻结Measurement v2继续独立计算 \(V_t^*(e)\) 和过期依赖，Guard源代码没有Evaluator、Oracle、ground truth或可接受答案依赖。

六项核心反事实/隔离测试全部通过：信念有效但客观过期、Full信念失效阻断、无关漂移Non-interference、新证据双重恢复、隐藏真值不干扰、隐藏正确答案不干扰。全部ReliabilityBench-Q回归测试为148/148通过。

原ActionGuard v1冻结记录保持不变，并新增append-only替代记录：`retained-for-provenance / not-used-for-sealed-method-validation`。v1从未运行密封方法验证集。

## 2. 首轮预飞行失败及版本处理

候选`d0d7792`上的首轮36条保留为`development-only / failed structural preflight`：34/36完整。失败集中在mitigation任务的两个Guard组。原因是公开工具Schema要求`mitigation_policy`属于公共实体域，但共享任务输入没有列出域值，模型连续提出三个自然但未注册的名称并耗尽固定修复预算。

没有局部重跑。协议升级到1.1.1，新增所有组完全相同的公开实体域与工具输出证据槽说明，同时更换开发命名空间、episode seed和schedule seed。失败run的trace、report、config和episode哈希均永久记录。

## 3. 第二轮36条真实LLM预飞行

第二轮使用`AG11DEV2-20260824`、episode seed `20261001`、schedule seed `20261002`，六类任务各运行六组。结果：

- 36/36 trace完整且可由冻结Measurement v2评分；
- 0 API错误尝试、0预算耗尽、0最终解析失败；
- 六组均有6/6 answered decision，未出现全拒答；
- Ledger+Guard 6/6完成，不再结构性0 ALLOW；
- Guard反馈中0个任务实体、证据ID或正确答案泄漏；
- 各episode的六组公共prompt一致，运行配置一致；
- 总计97个LLM turn、213306 token，按冻结费率估算0.0157570728美元；账单为最终权威费用。

开发结果中，Ledger+Guard仍出现客观stale dependence，证明Guard没有暗中获得 \(V_t^*\) 或实现选择性失效。该现象用于组件正交性检查，不作为方法性能估计。

## 4. 统计预注册v1.1

主要终点已收敛为：task-predicate correctness；`unsafe_or_unknown`及stale/unsupported/provenance拆分；realized revalidation cost。Coverage、feasible completion、correct rejection、selective risk、abstention和unscorable固定为同步安全护栏。

24条v2.1仅报告配对效应、置信区间和方向，不作正式非劣效声明。Global非劣效界值与功效分析延后到扩展实验预注册。Utility仅作次要描述，主权重与27组敏感性网格已经冻结。

## 5. 模型、预算与费用配置

内容寻址配置清单固定：DeepSeek `deepseek-v4-flash`，non-thinking，temperature 0，top_p 1，4096总completion token、12轮、10次工具调用、10成本单位、120秒timeout、两次API尝试、1秒固定退避、两次格式修复及两次Guard修复。实际返回模型标识与请求一致。

费率采用2026-08-24核验的DeepSeek官方V4-Flash美元表：每百万token缓存命中输入0.0028美元、未命中输入0.14美元、输出0.28美元。所有估算均保留token依据，provider账单优先。

## 6. 密钥备份与v2.1状态

v2.1密钥已从服务器custodian目录原样复制到D盘独立位置。两端文件均为45字节，SHA-256均为`d1550442b368d9230eb0336e50fd2a3454067212756718eb756e4fdd868f82b8f`。复制和校验过程中未显示或解析密钥内容，也未打开密封archive。

本阶段没有修改v2.1的episode ID、依赖、snapshot谱系、任务谓词或工具响应语义。新增的是方法内部 \(B_t\) 与共享运行交互协议，因此v2.1继续条件兼容并保持密封。

## 7. 当前停止点

- [x] ActionGuard v1.1及组件边界冻结候选完成。
- [x] 2×2正交性反事实测试通过。
- [x] 新36条真实LLM preflight通过。
- [x] 统计预注册v1.1完成。
- [x] 模型、参数、timeout、重试和费用配置内容寻址签字。
- [x] 密钥独立位置备份完成并核验。
- [ ] 研究负责人批准解封v2.1。
- [ ] 研究负责人批准正式144次运行。

最后两项仍为false，因此必须暂停。
