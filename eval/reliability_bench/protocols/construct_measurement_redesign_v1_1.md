# ReliabilityBench-Q 构念与测量重设计方案 v1.1（有条件批准修订版）

状态：`conditional approval incorporated / implementation not started`
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

缺失出处不直接等同于过期证据。对每个终止决策定义三值证据安全状态：

\[
EvidenceState_t(d)=
\begin{cases}
safe,& Supported_t(d)=1\\
unsafe,& \exists e\in P(d):V_t(e)=0\\
unknown,& C_P(d)=0\land U_t(d)=0.
\end{cases}
\]

其中决策支持充分性为

\[
Supported_t(d)=\mathbf 1\left[
\forall r\in Req(d),\exists e\in P(d):
Covers(e,r)\land V_t(e)=1
\right].
\]

正式报告必须分别给出：

- `definite_stale_dependence`：明确存在无效证据祖先；
- `unsupported_decision`：必要证据槽没有被当前有效证据覆盖；
- `provenance_incomplete`：出处缺失导致无法判定完整支持链；
- 三值`evidence_state={safe, unsafe, unknown}`。

另报告保守风险上界

\[
UnsafeOrUnknown_t(d_t)=U_t(d_t)\lor\neg C_P(d_t),
\]

但它只能作为复合敏感性指标，不能替代三类主指标或被解释为“确定使用了旧证据”。这样既避免Agent通过省略证据ID人为降低`stale-use rate`，也不会把未知错误解释为已确认过期依赖。

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

### 2.9 拒答、覆盖率与选择性风险

`abstain`是合法终止状态，但不能通过全部拒答规避错误。令`Answer_i=1`表示Agent在第`i`个episode提交了可判定的`success`或`failure`决策，而非`abstain`。正式报告至少包括：

\[
Coverage=\frac{\sum_i Answer_i}{N},
\]

\[
SelectiveRisk=\Pr(WrongDecision=1\mid Answer=1).
\]

同时分开报告：

- feasible任务上的完成率`FeasibleCompletionRate`；
- 不可达任务上的正确拒绝率`CorrectRejectionRate`；
- 错误拒绝率和错误接受率；
- coverage—risk曲线，而不只报告单一点；
- overall utility。

建议效用函数在预注册时固定：

\[
Utility=u_c\,Correct-u_w\,Wrong-u_a\,UnnecessaryAbstain-u_{cost}\,Cost,
\]

其中权重必须在查看方法结果前冻结并做敏感性分析。可行任务上的`abstain`不能计为正确，只有任务谓词确认不可达时的拒绝才计为正确决策。

### 2.10 干预请求、尝试、执行与验证

Agent只能声明`intervention_requested`，不能自报`intervention_executed`。四个阶段完全由结构化输出和权威工具轨迹联合推导：

1. `requested`：终止输出请求了某类干预；
2. `attempted`：真实轨迹存在实体与策略匹配的被接受工具调用；
3. `executed`：工具返回成功执行状态，而非仅接受请求；
4. `verified`：执行产生当前快照的新证据，且该证据满足冻结的干预效果验证谓词。

四个变量分别报告，且满足单调约束：

\[
verified\Rightarrow executed\Rightarrow attempted,
\]

但`requested`与`attempted`允许不一致，以暴露未执行声明或执行器主动补救。任何Agent自报字段都不能把未发生的工具调用提升为`attempted/executed/verified`。

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
    "intervention_requested": true
  }
}
```

这样任务动作、策略选择和干预请求不再共享同一字符串字段。`attempted/executed/verified`不出现在Agent可写的事实字段中，只能由真实工具轨迹和冻结的工具响应谓词推导。所有六组使用同一Schema验证器；结构化终止适配器不是ActionGuard组件。

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

1. Task Predicate Evaluator分别通过独立正例、反例、边界和成对反事实测试；每个任务谓词必须覆盖多解集合、拒答和不可达边界。
2. Trace/Evidence Evaluator在与方法验证集隔离的手工gold traces上准确重建快照谱系、实体绑定、证据图、三值证据状态以及`requested/attempted/executed/verified`。
3. `definite_stale_dependence`、`unsupported_decision`、`provenance_incomplete`、必要重验证和干预阶段等核心指标与两名独立人工标注及第三方裁决达到预注册的一致性门槛；门槛、样本和失败后换新集规则必须先冻结。
4. 统一终止Schema、运行时适配器、公共prompt、工具—证据注册表、评价器代码和配置全部冻结并记录commit/SHA-256。
5. 所有六组接收完全相同的Schema、公共prompt、工具响应格式和预算；只有目标控制器机制可以变化。
6. 方法内部规则与评价谓词不存在代码级共享：静态导入图、运行时上下文白名单和隐藏字段泄漏测试全部通过。
7. ActionGuard实现与隔离审计冻结；它仍不得读取Judge、hidden ground truth、Oracle字段或`A_acceptable`。
8. 六组公平性、标签隔离、评分可复现性和密钥异地备份完成。
9. 修订记录明确Judge v2失败且不再是解封门禁；新门禁是对旧门禁的append-only替换，不是取消质量控制。

只有以上门禁全部通过并形成不可变清单后，才允许请求解封授权。

### 6.3 整套作废触发条件

只要后续实现发现必须修改以下任何密封语义，就整套作废v2.1并生成全新版本，禁止局部替换：

- episode内的evidence ID、依赖资源或漂移影响关系；
- snapshot谱系或任务约束；
- task predicate的科学含义或可接受行动集合；
- 为得到期望结果而修改某些episode的工具响应语义。
- 新终止Schema或运行时适配器无法从v2.1现有字段无损生成，必须回写或改造episode。

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

- **H1（Evidence Layer）**：在相同ActionGuard状态下，Evidence Layer降低`definite_stale_dependence`和`unsupported_decision`，改善三值证据状态分布，并降低错误决策率；`unsafe_or_unknown`仅作为保守敏感性指标。
- **H2（ActionGuard）**：在相同Evidence Layer状态下，ActionGuard提高必要重验证召回率和有效重验证采用率，降低非法终止、出处不完整和错误决策率。
- **H3（互补性与成本）**：两层在预注册的条件比较中分别贡献可靠性增益，组合方法总体表现最佳；相对`Global + ActionGuard`，`Full + ActionGuard`的动作正确性、coverage和selective risk满足预注册非劣约束，而真实重验证调用、成本单位、token和延迟更低。

四个主要计划比较为：

1. `Original Full vs Ledger-Only`：无Guard时Evidence Layer作用；
2. `Ledger+Guard vs Ledger-Only`：无选择性失效时Guard作用；
3. `Full+Guard vs Original Full`：有效证据状态下Guard增益；
4. `Full+Guard vs Ledger+Guard`：相同Guard下Evidence Layer增益。

`β_Evidence×Guard`作为预注册交互检验，但不把论文成立条件设为其显著为正。若两个条件效应成立且组合方法总体最佳，可支持“两层互补”；只有交互项和相应效应量均提供充分证据时，才使用“超加性协同”表述。

建议主要二元模型为

\[
logit\Pr(Wrong=1)=\beta_0+\beta_EE+\beta_GG+\beta_{EG}E\times G
+\beta_R Relevance+u_{task}+u_{episode}+u_{model}+u_{seed}.
\]

对成本计数使用配对负二项模型或按基础episode聚类的配对bootstrap。主要对比按同一episode配对，任务模板、episode、模型和seed作为层级/随机效应。H1–H3主要检验采用预先冻结的多重比较校正。

相对Global的非劣界建议仍以2个百分点作为工程阈值，并报告1%、2%、5%敏感性；成本优势报告效应量与置信区间，不用事后阈值决定成败。

### 7.3 机制路径分析

路径一：

```text
Evidence Layer → definite stale/unsupported/unknown evidence state → wrong decision
```

路径二：

```text
ActionGuard → valid revalidation adopted by decision → wrong decision
```

这些分析默认称为“机制路径分析”。报告处理—中间变量关系、条件错误率和路径模型，并使用episode配对bootstrap。由于中间变量不是独立随机化，不默认满足顺序可忽略性，也不直接宣称因果中介效应。只有后续增加对中间变量的随机干预或获得充分识别条件时，才估计并解释因果间接效应。

### 7.4 选择性重验证的优化问题与算法贡献

令`Req(d)`为候选决策所需证据槽，`Q`为可执行重验证动作子集，`c(q)`为工具调用、延迟、token或硬件成本，`Outcomes(Q)`为执行后可产生的证据节点。选择性重验证定义为：

\[
Q^*=\arg\min_{Q\subseteq Actions}\sum_{q\in Q}c(q)
\]

满足

\[
\forall r\in Req(d),\exists e\in Valid(E\cup Outcomes(Q)):
Covers(e,r).
\]

该问题可视为带实体、快照和预算约束的加权集合覆盖/最小恢复计划。计划研究：

- 精确解法：小规模任务上的枚举、整数规划或动态规划，用作Oracle-Action诊断上界；
- 近似策略：按`marginal recovered evidence slots / cost`选择动作，并处理动作结果不确定性；
- 自适应策略：每次工具返回后更新证据图和剩余需求，再选择下一动作；
- 复杂度：一般情形与加权集合覆盖的关系，以及特定层次依赖图上的可解子类。

ActionGuard作为运行时shield维持安全终止不变量：

\[
Finalize(d)\Rightarrow Supported_t(d)\land
d\in\mathcal A_{acceptable}(S_t),
\]

其中Guard运行时不能直接读取评价器的`A_acceptable`；实际实现使用公开约束和证据充分性必要条件，正式正确性由独立评价谓词检查。拟研究或验证以下性质：

1. **Soundness**：缺少有效支持或结构化前置条件时不能以`success`结束；
2. **Selectivity**：无关漂移不会使不相干证据失效；
3. **Non-interference**：对已满足公开安全不变量的合法动作，Guard不改变动作或增加调用；
4. **Cost dominance**：在相同证据需求和确定性工具覆盖模型下，最小选择性方案成本不高于Global-Revalidate；
5. **Approximation quality**：一般情形下给出相对精确上界的经验近似比，理论界只在假设满足时声明。

## 8. 待批准后的实施顺序

1. 冻结本方案v1.1及append-only协议修订；当前提交只纳入有条件批准意见，不包含实现。
2. 在不读取v2.1密文的前提下，实现统一终止Schema、工具输出证据ID规范和两个程序化评价器接口。
3. 用合成单元夹具、正反例、边界例和变形测试验证定义；这些夹具不得来自密封v2.1。
4. 完成方法—评价器静态依赖审计和运行时字段隔离审计。
5. 仅在旧development traces上做评价器工程调试；不拟合Semantic Diagnostic Classifier作为主裁判。
6. 经研究负责人再次确认后，才修改ActionGuard；ActionGuard只在旧development set开发。
7. 冻结ActionGuard、六组配置、输出Schema、工具注册表、评价器和统计脚本。
8. 再次执行v2.1兼容性门禁：若仅外部层变化则提交协议修订并维持原密封；若触发第6.3节则整套作废并生成新密封版本。
9. 只有全部门禁通过并获得书面解封授权后，才可解封并运行方法验证集。

本修订提交后可进入“评价器与运行时适配器实现”，但ActionGuard修改、新验证集生成和v2.1解封仍保持独立阻塞，直到对应门禁和书面授权满足。
