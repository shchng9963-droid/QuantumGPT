# ActionGuard package v1.3 冻结报告

状态：工程与Measurement门禁通过，可提交冻结审核；不得自动解封method-validation-v2.1。

## 组件版本与边界

- Guard安全核心：ActionGuard v1.2，SHA-256 `3cf75ca5045e5e01504597c121a155cbe1ee0086931fa826ff4d8c9b03dec1a2`，保持不变。
- Guard contract v1.2：`91791de7c4b1493572ae8add005f7d77286bfc3ffc678bd73efdc1e2ed25b31a`。
- Temporal Evidence v1.2：`8d3ea320780f1771c5be1def9786379375f16b96c3ad58a2fde39ae797665438`。
- ActionGuard runtime v1.3：`6f9b162a3ba380ad774b2db4d4f17d7934ee27c311203780e32c42564902a378`。
- 公共preflight v1.3：`0ea12f59d431fa51ae4e6153ff8eab28aa41d2051bed35204a5f395eb92be0d3`。
- Runtime terminal v1：`78311ba99d93bcce7daae78c4ef93925986fbc902685197c320f4400b5bf26d4`。
- Public runtime v3.1：`995eacb03f5f5706b38142496b54179bcfdd8dd3a9b34d666483ba34ac62ccd8`。
- Scorer v1.3：`c2034cb29049049d5ebbfa3661badcd2ebe8de645487195cee9a3df74a2967fe`。

新版本只增加统一运行时闭环。Guard的ALLOW/BLOCK/REPAIR_REQUEST逻辑、两次修复上限、抽象反馈、B_t输入和Evaluator隔离边界不变。

## 测试与Gold

- ReliabilityBench服务器回归：197/197通过。
- 7类reason code均能形成decision=null的运行时终态。
- feasible/infeasible均不把execution failure记为correct rejection。
- Agent不能自报execution failure覆盖运行时。
- 非Guard错误路径使用相同运行时失败信封。
- Measurement v3.1 development Gold：1440/1440。
- Measurement v3.1 held-out Gold：1449/1449。

## 新72次preflight冻结信息

- 共享prompt SHA-256：`226eb93f389a668c67731e880b4bab850b799d970e484772830b72259a25afc05`。
- 配置SHA-256：`bca7c9b2aabc5f1a8beba144588a582e2bfda9ba042d6a35699d4809e5686142`。
- Episodes SHA-256：`49ef0a7f22bfa78c3c7b14b5cad52354ac4f5e4dcde26ec9accaa361a22ffcde`。
- Schedule SHA-256：`d5ab5dd3747877490d8efa825030a3ede0054ca2e3d5e7ca36b3464194374abd`。
- Traces SHA-256：`d8586286f0fc52c37cfa85a9eec6a90775b4e7222ade6773d636bf066db700f4b`。
- Scores SHA-256：`29e972f6766d603d8a26e4bb99615c8b1f66a926ca1b821a3aea9de071f1d1b1`。
- Report SHA-256：`e4e246338ee44ed6df54852d708ebf34a6033ffe94ccb109227c21e507f6e43d`。

门禁结果为72/72闭合、72/72可评分、0 missing-terminal unscorable。69条answered和3条execution failure全部保留。报告中API错误、parser failure、共享资源预算耗尽和泄漏均为0。

## 结论边界

该门禁只证明公共运行时、Guard安全失败和Measurement形成闭环。自然preflight只有4个被拦截运行，不能支持H2性能结论，也不能支持正式非劣效或顶会级总体结论。正式实验仍需研究负责人书面授权。
