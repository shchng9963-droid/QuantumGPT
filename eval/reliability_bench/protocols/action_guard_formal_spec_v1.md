# ActionGuard Formal Specification v1.0

Status: frozen candidate for the method-validation pilot. This specification does not authorize unsealing `method-validation-v2.1`.

## Component boundary

The Temporal Evidence Layer is the sole producer of post-drift evidence validity:

\[
(\Delta_t,G_{dep},E_{t-1})\mapsto \widehat V_t(E).
\]

Ledger-Only exposes its existing ledger state and does not infer a drift impact set. Full-Selective applies dependency-scoped invalidation. The Guard consumes only the validity state visible to its controller. It does not receive the drift-to-dependency mapping and cannot manufacture invalidations.

The Recovery Planner solves the separate abstract optimization problem

\[
Q^*=\arg\min_{Q\subseteq \mathcal Q}\sum_{q\in Q}c(q)
\quad\text{s.t.}\quad Req(d)\subseteq \bigcup_{q\in Q} Outcomes(q).
\]

It proposes recovery of evidence slots. It is not part of the Guard. The Guard only decides whether a proposed tool action or terminal action satisfies public constraints.

## Guard contract

The permitted input is:

- public task contract and frozen evidence-slot requirements;
- controller-visible evidence ledger and its current validity values;
- candidate tool action or terminal decision;
- public tool schema;
- accepted real tool calls and their returned evidence bindings.

The only outputs are `ALLOW`, `BLOCK(reason_code)`, and `REPAIR_REQUEST(reason_code)`. The Guard never returns an answer, backend, qubit set, mitigation policy, recovery tool, or concrete tool argument.

The frozen reason codes are `invalid_schema`, `unknown_entity`, `invalid_reference`, `missing_required_support`, `stale_cited_evidence`, `unverified_revalidation`, and `premature_finalization`.

## Safety invariants

1. Action Formation: a proposed tool exists and its arguments exactly satisfy the public schema.
2. Referential Integrity: all public entities, evidence IDs, and tool-call IDs exist in their visible domains and preserve binding.
3. Evidence Preconditions: every evidence slot required by an answered terminal decision is covered by explicitly cited evidence currently marked valid by the controller-visible ledger.
4. Revalidation Integrity: a declared revalidation binds to an accepted real call, a successful response, and the exact new valid output evidence IDs.
5. Safe Finalization: an answered terminal decision cannot finish without explicit, valid, slot-complete support.
6. Non-interference: a syntactically legal action or adequately supported terminal decision is returned unchanged with `ALLOW`.

`abstain` is allowed because it is not an answered success claim. It remains observable and is penalized through coverage, feasible completion, selective risk, and utility; it is never converted into success.

## Fixed repair protocol

Every Guard arm uses at most two repair attempts within the common ceiling of 12 LLM turns and 10 tool calls. A Guard interception adds no hidden turns or tools. Feedback contains only outcome, abstract reason code, and the violated public constraint. Each candidate hash, check, block, retry, latency, token use, and failed repair remains in the trace.

The adapter is non-semantic: it cannot mutate actions, add evidence IDs, synthesize tool calls, or translate self-reported revalidation into execution.

## Isolation requirements

Runtime source code must not import or call Measurement v2 evaluators, task predicates, acceptable action sets, Oracle state, or hidden ground truth. `ActionGuard` must not import the Temporal Layer or Recovery Planner. These conditions are checked through source dependency tests and counterfactual behavior tests.
