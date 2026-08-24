# Measurement v3.1 冻结报告

状态：独立Gold门禁通过，冻结候选。

## 升版原因

Measurement v3与terminal-decision-3.0保留不变。其execution-failure分支要求一个decision对象，Trace Evaluator也要求TerminalDecision，因此不能表达“运行时生成、decision=null、仍可评分”的安全失败。追加Measurement v3.1，不覆盖v3历史记录。

## 新语义

- Agent终态：沿用terminal-decision-3.0，但公共运行时只接受`answered | abstain`。
- 运行时终态：runtime-terminal-1.0，`decision=null`且`runtime_generated=true`。
- Task evaluator：execution failure为错误的ITT结果，但不是wrong answered decision、abstain或correct rejection。
- Trace evaluator：决策支持字段不适用；真实工具、干预、成本与延迟继续可计算。
- Utility主权重：correct `+1`、wrong `-1`、unnecessary abstain `-0.5`、execution failure `-1`、cost unit `-0.01`。

## Gold门禁

- Development Gold：48条、30字段/条，`1440/1440` exact agreement。
- Held-out Gold：48条、30字段/条及9个聚合字段，共`1449/1449` exact agreement。
- Held-out标签先以AES-256-GCM冻结；预测文件SHA-256在Gold解封前固定为`bd64b2f36728f6347b1fe576e3e758658fe1bbacd135eb90150125c39aa3ee2b6`。
- Held-out Gold明文SHA-256：`4eef71298065c87187fba3dccda1c7a3290edc8de9badfbb046ccb13383a7cc53`。
- 48条是独立轨迹数；1449是字段比较数，不能表述为1449个独立样本。

Gold覆盖Agent回答、拒答、运行时失败、feasible/infeasible任务、7类Guard reason code、真实成本、失败工具、重复/无关调用及mitigation执行阶段。所有形式上唯一真值字段零分歧。

候选Evaluator commit为`bc29b90a6f77eecd7b6c2e96a1033407c0571b43`。冻结源码哈希见`data/measurement_v3_1_candidate_manifest.json`。后续若修改核心评价语义或这些冻结源码，必须升版并重新执行全套独立Gold门禁。
