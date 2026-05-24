# 30s Demo Talk Track — Drift-Aware Recovery

On a stable simulated FakeBrisbane the agent runs ghz_5 and gets fidelity ≈ 0.906. We then inject sudden hardware drift — T1 collapses and gate errors spike — at t=6h.

The drift-aware agent picks this up from the calibration snapshot, invalidates the stale fidelity result, replans, and re-runs the circuit. Final fidelity: 0.762.

For comparison, the same prompt against a drift-naive ReAct agent reports 0.739 — it never notices the device changed and keeps the stale answer.

The 'recovery gap' (drift-aware − drift-naive = +0.023) is the value-add we sell: continuously valid results on a non-stationary device.
