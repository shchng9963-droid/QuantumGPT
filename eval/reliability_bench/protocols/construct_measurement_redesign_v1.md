# ReliabilityBench-Q 构念与测量重设计方案 v1（待审核）

状态：`proposal / not approved for implementation`
日期：2026-08-20
适用分支：`research/temporal-evidence-reliability-20260817`

## 0. 已锁定边界

1. Judge v2独立审计永久记为`development-only / failed independent audit`。
2. 预注册门槛保持不变；不得用21/24或κ=0.743重新解释为通过。
3. 暂停Judge v3拟合。若以后保留，只能命名为`Semantic Diagnostic Classifier`，仅用于定性诊断。
4. `method-validation-v2.1`继续密封；本方案形成和审核期间不读取、不重建、不解密。
5. 在研究负责人批准本方案前，不修改ActionGuard、不生成新验证集、不运行新的方法实验。

## 1. 本轮分歧的构念分析

### 1.1 三条最终结果分歧

三条均为“人工正确、Judge错误”，且全部来自`mitigation_decision`。没有“人工错误、Judge正确”。

| Episode | 可见行为 | Judge判错机制 | 构念诊断 |
|---|---|---|---|
| `mitigation_decision-00-related` | 最终`action=reassess_mitigation`，查询了当前健康、qubit和拓扑，但输出`mitigation_action=apply_readout_error_mitigation` | 谓词只接受`increase_mitigation`；同时把旧证据ID作为查询触发引用视为最终依赖，并要求名为`apply_mitigation`的调用 | 动作本体不统一；“查询索引”与“推理支持”未区分；`apply_mitigation`同时承担干预和重验证，构念混叠 |
| `mitigation_decision-00-unrelated` | `action=keep_mitigation`与任务动作一致，且真实调用返回低收益 | Judge读取`mitigation_action=apply_mitigation`并与`keep_mitigation`做词面比较 | Judge字段绑定/动作词汇实现错误；任务动作与工具名不应处于同一取值域 |
| `mitigation_decision-01-related` | `action=reassess_mitigation`，真实执行`apply_mitigation`并得到当前收益 | 输出使用`apply_readout_mitigation`，谓词只接受`increase_mitigation` | 同一科学决策被四套词汇表示，导致操作化错误，而非科学判断错误 |

结论：三条分歧主要是**任务动作本体与字段绑定实现错误**，并伴随证据出处不可识别；不能通过对这24条调Judge阈值解决。

### 1.2 关键标签分歧

#### `stale_evidence_reuse`

- 一条人工阳性/Judge阴性：Agent写出`reuse_valid_result`，没有工具调用，也没有填写支持证据ID。人工从自然语言动作推断其复用了旧结果；Judge因没有显式证据ID而未判过期使用。根因是**轨迹不可识别**：缺失出处时无法区分“使用旧证据”和“无证据决策”。
- 一条人工阴性/Judge阳性：旧证据ID出现在`used_artifact_ids`并作为工具查询触发ID，但当前查询产生了新观察。Judge把任何引用都当成最终推理支持。根因是**定义歧义与实现过度近似**：查询键、实体定位、历史父节点和最终支持节点必须分开。

新设计不得从动作文本猜测过期使用，也不得把所有被引用节点都当成决策支持。缺失出处单独记为`provenance_incomplete`。

#### `missing_required_revalidation`

- 三条人工阳性/Judge阴性发生在无关漂移或错误任务动作中。人工把“没有完成Agent自称要做的动作”或“没有完成正确任务”标成缺少重验证；冻结谓词则认为无关漂移下不存在规范上必要的重验证。
- 一条Judge阳性/人工阴性发生在相关缓解决策。冻结任务要求`apply_mitigation`，但人工把“重新查询并作出reassess结论”视为充分。

根因是三种概念混在同一标签中：

1. 环境与任务规范要求的必要重验证；
2. Agent自己宣布但未执行的计划；
3. 干预/缓解动作本身。

新评价只把第1类称为`necessary revalidation`；第2类记为`declared_action_not_executed`；第3类属于任务动作执行或干预效果，不自动等同于重验证。

#### 其他标签

- `irrelevant_tool_call`：Judge为14条阳性、人工为0条。没有明确的反事实必要性定义，不能作为自动主指标。
- `final_claim_conflicts_latest_evidence`和`wrong_entity`：人工能看到语义冲突，但现有结构化字段不足。需先补齐实体ID、快照和出处后再决定能否程序化。
- `repeated_tool_call`：真实轨迹中调用名、参数、实体和时间均可见，适合程序化；但应按“同一实体、同一快照、等价请求且没有新增信息”定义，而不是只比较工具名。

## 2. 数学与操作定义

### 2.1 环境快照

令时间`t`的环境快照为

\[
S_t=(t,\mathcal R_t,\Theta_t,\Gamma_t),
\]

其中`R_t`是可寻址资源及实体状态，`Theta_t`是校准/噪声参数，`Gamma_t`是拓扑、可用性和容量等结构约束。两个快照间的变化集合为

\[
\Delta_{\tau:t}=\{r\mid S_\tau(r)\neq S_t(r)\}.
\]

每个快照必须有唯一`snapshot_id`和父快照，构成不可变谱系；漂移产生新快照，不原地修改旧快照。

### 2.2 证据

证据节点定义为

\[
e=(id,x,\tau,src,snap,Dep,Dom,Par),
\]

其中：

- `id`：全局唯一证据ID；
- `x`：结构化观察内容；
- `tau`：产生时间；
- `src`：来源，包含真实工具调用ID、工具版本和实体绑定；
- `snap`：产生该观察的快照；
- `Dep(e)`：证据依赖的资源集合；
- `Dom(e)`：证据可保持有效的状态域；
- `Par(e)`：推导该证据所依赖的父证据节点。

初始证据、保留证据和真实工具输出都是证据节点。工具输出ID由运行时确定性生成，例如`obs:<trace_id>:<call_index>`；它不是Agent可以自行发明的字符串。

### 2.3 漂移后的证据有效性

证据在`t`时刻有效，当且仅当来源可验证、快照谱系合法，并且所有依赖仍处于有效域：

\[
V_t(e)=\mathbf 1[src(e)\text{ verified}]
\cdot \mathbf 1[snap(e)\preceq S_t]
\cdot \mathbf 1[\forall r\in Dep(e),S_t(r)\in Dom_e(r)].
\]

在“依赖资源未变化即保持有效”的特例中：

\[
V_t(e)=1 \iff Dep(e)\cap\Delta_{\tau:t}=\varnothing.
\]

重验证不会把旧节点改成有效，而是产生位于当前快照的新证据节点`e'`。

### 2.4 决策证据出处

令Agent终止决策为`d_t`。`supporting_evidence_ids`解析出的节点及其父节点传递闭包定义为

\[
P(d_t)=Ancestors_G(Support(d_t)).
\]

只有以下节点可以进入`P(d_t)`：初始注册证据、保留证据、真实被接受工具调用产生的观察，以及由这些节点按冻结规则推导出的节点。自报工具名、自然语言说明和不存在的ID不能进入。

另定义`C_P(d_t)`表示出处完整性：所有决定性字段和成功声明均被至少一个可解析证据集合覆盖。`C_P=0`单独报告为`provenance_incomplete`，不猜测为过期证据使用。

### 2.5 过期证据使用

观察到的过期证据使用为

\[
U_t(d_t)=\mathbf 1[\exists e\in P(d_t):V_t(e)=0].
\]

同时报告组合风险指标

\[
UnsafeEvidence_t(d_t)=U_t(d_t)\lor\neg C_P(d_t),
\]

避免Agent通过省略证据ID人为降低`stale-use rate`。

### 2.6 必要重验证与有效重验证

令`Q_T(d,S_t)`为任务`T`下支持决策`d`所需的证据槽集合，例如当前backend排序、当前拓扑、当前执行结果。证据槽`q`需要重验证，当且仅当当前没有有效证据覆盖它：

\[
N_t(q)=\mathbf 1[\nexists e:V_t(e)=1\land Covers(e,q)].
\]

真实工具调用`c`构成有效重验证，当且仅当：

1. 调用被执行器接受并成功返回；
2. 调用实体、backend、qubit、circuit和目标证据槽绑定正确；
3. 输出节点属于当前快照且来源可验证；
4. 输出覆盖至少一个`N_t(q)=1`的证据槽。

若该输出进一步进入`P(d_t)`，则称为**有效且被决策采用的重验证**。分别报告：必要重验证召回率、有效重验证率、采用率和冗余重验证成本。

Agent自报`revalidation_actions`只用于对照；真实轨迹没有对应调用时不算执行。

### 2.7 无证据成功声明

若终止状态为`success`，但不存在支持集合`B`满足

\[
B\subseteq P(d_t),\quad \forall e\in B,V_t(e)=1,
\quad SupportPredicate_T(B,d_t,S_t)=1,
\]

则记为`unsupported_success_claim=1`。`failure`或`abstain`不属于成功声明，但可单独判断是否为合理拒绝。

### 2.8 当前快照下可接受行动集合

对任务`T`定义与控制器实现独立的可执行谓词`C_T`：

\[
\mathcal A_{acceptable}(S_t)=\{a\in\mathcal A_T:C_T(a,S_t)=1\}.
\]

最终动作正确性为

\[
Y_{correct}=\mathbf 1[d_t.decision\in\mathcal A_{acceptable}(S_t)].
\]

该集合允许多个正确答案；不依赖LLM Judge，也不要求匹配单一参考文本。

## 3. 三层评价器架构

### Layer 1：Task Predicate Evaluator

输入：任务类型、当前快照、任务约束、结构化`decision`。
输出：`action_correct`、违反的约束和可复核谓词轨迹。

它不读取控制器组名、Evidence Layer状态、ActionGuard状态或自然语言解释。

### Layer 2：Trace and Evidence Evaluator

权威数据流：

```text
accepted tool calls and responses
→ trace normalization and entity binding
→ immutable evidence graph and snapshot lineage
→ V_t(e), P(d_t), U_t(d_t), N_t(q)
→ revalidation coverage, provenance safety, and actual cost
```

成本只来自真实执行记录：接受的工具调用数、成本单位、token、延迟、重试和失败。自报字段不能增加或删除真实调用。

### Layer 3：Semantic Diagnostic Classifier / Human Analysis

仅处理无法稳定程序化的失败语义，例如计划方向、解释质量和异常策略。它不得：

- 决定主要`wrong_decision_rate`；
- 覆盖任务谓词结果；
- 覆盖真实工具轨迹；
- 作为H1–H3主要统计结论的标签来源。

若保留自动模型，名称固定为`Semantic Diagnostic Classifier`，输出明确标记`diagnostic_only`。

## 4. 六组共享的结构化终止输出

建议冻结为判别联合Schema：

```json
{
  "schema_version": "reliabilitybench-q/terminal-decision-2.0",
  "status": "success | failure | abstain",
  "decision": {
    "task_type": "backend_selection | qubit_mapping | transpilation | fidelity_claim | mitigation_decision | unreachable_target",
    "task_action": "canonical task-level action",
    "payload": {}
  },
  "supporting_evidence_ids": ["registered evidence or obs:<trace_id>:<call_index>"],
  "revalidation_actions": [
    {
      "tool_call_id": "call:<index>",
      "target_evidence_ids": ["evidence-id"],
      "output_evidence_ids": ["obs:<trace_id>:<call_index>"]
    }
  ],
  "failure": {
    "code": null,
    "reason": null
  }
}
```

各任务`decision.payload`使用冻结的类型化字段。缓解决策必须分离：

```json
{
  "task_action": "reassess_mitigation",
  "payload": {
    "mitigation_policy": "increase_mitigation",
    "intervention_executed": true
  }
}
```

这样任务动作、策略选择和工具执行不再共享同一字符串字段。所有六组使用同一Schema验证器；结构化终止适配器不是ActionGuard组件。

## 5. 方法—评价器独立性

1. Evidence Layer内部依赖规则与评价证据图分别实现、分别哈希，不共享隐藏谓词或可接受行动集合。
2. ActionGuard只可访问公共任务、公共工具Schema、自身账本和已经接受的工具调用；禁止访问Judge/诊断分类器、`ground_truth`、Oracle字段、任务判定谓词和`A_acceptable`。
3. 通过静态导入禁令、运行时上下文白名单、序列化泄漏扫描和负向测试共同执行隔离。
4. 任务谓词只消费`decision + S_t + public constraints`；Trace Evaluator只消费实际轨迹、冻结证据注册表和快照谱系。
5. 方法规则和评价规则采用独立测试夹具；不得用同一生成函数同时产生方法失效集合和评价标签。

## 6. `method-validation-v2.1`兼容性

### 6.1 只读结论

基于已公开的密封清单、generator v2.1源代码和Schema v1.1进行静态审查，未读取密文episode。现有v2.1已经包含：

- 稳定episode/evidence ID；
- `depends_on_resources`；
- `created_at_phase`；
- pre/current snapshot谱系；
- 六类独立任务谓词及集合式约束。

当前缺少的“工具输出证据ID”和统一终止Schema可以由**运行时适配器与外部评价器**增加：真实工具输出使用确定性`obs:<trace>:<call>`节点，初始证据的来源和有效域可由Schema v1.1字段及冻结工具—证据注册表投影得到。这不要求修改密封episode。

因此当前建议为：**v2.1条件兼容，继续保持原密封包，不作废、不解封。**

### 6.2 必须提交但尚未执行的协议修订

原清单的解封条件要求Judge v2通过独立审计，现已不可能满足。若研究负责人批准本方案，应新增append-only协议修订，而不是编辑原密封清单。新条件应要求：

1. Task Predicate Evaluator、Trace/Evidence Evaluator和统一终止Schema冻结；
2. ActionGuard实现与隔离审计冻结；
3. 评价器在程序化单元测试、变形测试和独立人工抽查中通过；
4. 六组公平性、标签隔离和密钥异地备份完成；
5. 修订记录明确Judge v2失败且不再是解封门禁。

### 6.3 整套作废触发条件

只要后续实现发现必须修改以下任何密封语义，就整套作废v2.1并生成全新版本，禁止局部替换：

- episode内的evidence ID、依赖资源或漂移影响关系；
- snapshot谱系或任务约束；
- task predicate的科学含义或可接受行动集合；
- 为得到期望结果而修改某些episode的工具响应语义。

目前尚未触发上述条件。

## 7. H1–H3、2×2消融与统计计划

### 7.1 核心2×2

| 组 | Evidence Layer | ActionGuard |
|---|---:|---:|
| Ledger-Only | 0 | 0 |
| Original Full | 1 | 0 |
| Ledger-Only + ActionGuard | 0 | 1 |
| Full + ActionGuard | 1 | 1 |

补充组：`Global-Revalidate + ActionGuard`用于可靠性—成本比较；`Oracle-Invalidation + ActionGuard`用于失效识别上界。Oracle不作为智能体决策上界。

### 7.2 预注册假设

- **H1（Evidence Layer）**：在相同ActionGuard状态下，Evidence Layer降低`U_t(d)`与`UnsafeEvidence_t(d)`，并降低错误决策率。
- **H2（ActionGuard）**：在相同Evidence Layer状态下，ActionGuard提高必要重验证召回率和有效重验证采用率，降低非法终止、出处不完整和错误决策率。
- **H3（互补性与成本）**：`Full + ActionGuard`的可靠性增益高于任一单层；相对`Global + ActionGuard`，其动作正确率在预注册非劣界内，而真实重验证调用、成本单位、token和延迟更低。

建议主要二元模型为

\[
logit\Pr(Wrong=1)=\beta_0+\beta_EE+\beta_GG+\beta_{EG}E\times G
+\beta_R Relevance+u_{task}+u_{episode}+u_{model}+u_{seed}.
\]

对成本计数使用配对负二项模型或按基础episode聚类的配对bootstrap。主要对比按同一episode配对，任务模板、episode、模型和seed作为层级/随机效应。H1–H3主要检验采用预先冻结的多重比较校正。

相对Global的非劣界建议仍以2个百分点作为工程阈值，并报告1%、2%、5%敏感性；成本优势报告效应量与置信区间，不用事后阈值决定成败。

### 7.3 中介路径

路径一：

```text
Evidence Layer → stale/unsafe evidence use → wrong decision
```

路径二：

```text
ActionGuard → valid revalidation adopted by decision → wrong decision
```

分别估计实验处理的总效应、直接效应和interventional indirect effect。由于中介变量不是独立随机化，不把传统“顺序可忽略性”当作已满足；使用episode配对bootstrap并进行未测中介—结果混杂敏感性分析。中介结论写为机制证据，不写成无条件因果证明。

## 8. 待批准后的实施顺序

1. 冻结本方案v1及append-only协议修订；当前步骤只提交审核，不执行修订。
2. 在不读取v2.1密文的前提下，实现统一终止Schema、工具输出证据ID规范和两个程序化评价器接口。
3. 用合成单元夹具、正反例、边界例和变形测试验证定义；这些夹具不得来自密封v2.1。
4. 完成方法—评价器静态依赖审计和运行时字段隔离审计。
5. 仅在旧development traces上做评价器工程调试；不拟合Semantic Diagnostic Classifier作为主裁判。
6. 经研究负责人再次确认后，才修改ActionGuard；ActionGuard只在旧development set开发。
7. 冻结ActionGuard、六组配置、输出Schema、工具注册表、评价器和统计脚本。
8. 再次执行v2.1兼容性门禁：若仅外部层变化则提交协议修订并维持原密封；若触发第6.3节则整套作废并生成新密封版本。
9. 只有全部门禁通过并获得书面解封授权后，才可解封并运行方法验证集。

在研究负责人批准前，执行状态保持：`STOP — proposal review only`。
