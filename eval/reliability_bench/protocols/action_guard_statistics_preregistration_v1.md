# ActionGuard Method-Validation Statistical Preregistration v1.0

Status: frozen draft before any unsealing of `method-validation-v2.1`.

## Design and hypotheses

The core factorial arms are Ledger, Full, Ledger+Guard, and Full+Guard. Global-Revalidate+Guard is the cost baseline and Oracle-Invalidation+Guard is the invalidation upper-bound diagnostic.

- H1, Evidence Layer: Full improves evidence safety and decision reliability over Ledger; Full+Guard improves them over Ledger+Guard.
- H2, ActionGuard: Ledger+Guard reduces structurally invalid actions and unsupported finalization versus Ledger; Full+Guard does so versus Full.
- H3, combined method: Full+Guard has the best reliability-coverage profile among non-Oracle arms and is no less reliable than Global-Revalidate+Guard while using less actual revalidation cost.

Four planned factorial comparisons are Full vs Ledger, Ledger+Guard vs Ledger, Full+Guard vs Full, and Full+Guard vs Ledger+Guard. The Evidence×Guard interaction is secondary. Complementary conditional effects and the combined method ranking support complementarity; a positive statistically significant interaction is required only for a claim of super-additive synergy.

The paths `Evidence Layer → definite stale dependence → wrong decision` and `ActionGuard → valid revalidation → wrong decision` are preregistered mechanism-path analyses. They are descriptive unless the mediator is independently randomized or the assumptions needed for causal mediation are justified.

## Endpoints

Primary endpoints are task-predicate correctness; definite stale dependence; unsupported decision; feasible-task completion; correct rejection on infeasible tasks; coverage; selective risk among non-abstaining decisions; actual required/effective revalidation; and total realized cost.

Secondary endpoints are provenance incomplete, `unsafe_or_unknown` as a conservative risk bound, abstention, unscorable rate, intervention requested/attempted/executed/verified, irrelevant and repeated calls, latency, token use, API cost, Guard checks, blocks, repair attempts, and Guard latency.

No metric treats `unknown` as confirmed stale use. Agent self-reports do not override the accepted tool trace. A method cannot be judged reliable merely by abstaining.

## Pairing and estimation

All analyses retain episode-level pairing. Binary outcomes report paired risk differences with 95% paired-bootstrap confidence intervals and exact McNemar summaries when applicable. Continuous cost, tokens, tool calls, and latency use episode-paired bootstrap confidence intervals. Expanded experiments use mixed-effects models with method factors as fixed effects and task template, circuit/backend, model, and seed as grouping effects.

The 2×2 model includes Evidence, Guard, and Evidence×Guard terms. Planned contrasts are reported whether or not the interaction is significant. Related/unrelated counterfactual pairs are analyzed jointly and separately.

## Abstention, failure, and scoring rules

`abstain` is not a wrong answered decision, but lowers coverage and feasible completion. Selective risk is conditional on non-abstention and is always reported beside coverage and overall utility. Unscorable runs remain in the dataset; the primary conservative analysis treats them as failures, with a separately labeled complete-case sensitivity analysis.

API timeouts, exhausted retries, parse failures, and tool failures are never deleted. All arms use the same fixed timeout and retry policy; retries are applied by failure type, never by observed method performance. Both attempted and successful costs are retained.

## Pilot scope

The sealed 24-episode `method-validation-v2.1` collection is an independent method-validation pilot only. It cannot by itself support a top-conference population-level claim. Advancement requires independent task templates, at least two models, multiple seeds, multiple drift severities, and a power analysis based on pilot effect and variance estimates.

No endpoint, scale, exclusion, retry rule, or contrast may be changed after unsealing without versioning the protocol and labeling the affected analysis exploratory.
