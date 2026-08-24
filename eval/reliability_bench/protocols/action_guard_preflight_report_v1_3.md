# ActionGuard v1.3 真实LLM development preflight报告

状态：工程门禁通过；仅development，不允许论文性能Claim。

## 设计

全新12个episode：6类任务 × related/unrelated反事实对；每条运行六组，共72次。使用全新namespace、pair、任务参数种子`20264011`和交错顺序种子`20264012`，未补跑旧失败条目。

模型为`deepseek-v4-flash`，temperature 0、top_p 1、最大12轮、10次工具调用、10个成本单位、2次API重试、2次格式修复、Guard最多2次修复。六组公共prompt、模型、工具、Schema和预算一致。

## 门禁结果

- 72/72轨迹正常闭合；
- 72/72进入answered、abstain或execution_failure之一；
- 72/72由Measurement v3.1正式评分；
- 0条因缺少终态而unscorable；
- 0次API错误，0次parser failure，0次共享资源预算耗尽；
- 69条answered、0条abstain、3条runtime-generated execution failure；
- 三条失败的原始候选、两次修复和最终安全阻止均保留；
- ground-truth、答案payload、标识符反馈泄漏均为0；
- B_t/V*_t组件边界审计通过。

3条execution failure分别发生于Global+Guard mitigation、Full+Guard unreachable和Invalidation-Consistency Check unreachable。它们降低coverage和utility，并计入全部实际成本；没有从数据中删除。

## 描述性机制结果

48个Guard组运行均产生至少一次Guard检查，其中4个运行发生拦截，run-level interception rate为`4/48=8.33%`。4个被拦截运行中1个最终形成Agent终态，3个进入安全execution failure：

- `repair_success_given_interception=1/4=25%`；
- `safe_failure_given_failed_repair=3/3=100%`；
- 共10次非ALLOW事件、7次允许的修复尝试；
- 低自然触发率及少量事件不足以支持H2性能结论。

全部72条共177个工具成本单位、370675 tokens，估算API费用`$0.0249244072`。这些是开发流程数据，不进行显著性、非劣效或总体性能声明。

## 研究边界

v2.1继续作为自然分布方法Pilot，不修改密封episode来提高Guard触发率。未来扩展实验另设预注册的ActionGuard challenge stratum；challenge结果与自然分布结果分开报告。
