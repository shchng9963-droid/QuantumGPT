# Drift Detector Accuracy Report

## Benchmark Setup

- **80 scenarios** with **115 labeled changepoints** across 6 categories:
  - Stable (10): no changepoints — false positive control
  - Gradual linear (5): slow decay, no abrupt CPs — specificity test
  - Single sudden shift (25): 1 CP each at varying times and severities
  - Double shift (20): 2 CPs each — degradation then further shift
  - Triple staircase (10): 3 CPs each — progressive degradation
  - Shift + recovery (10): 2 CPs each — degrade then recover

- **Signal**: 5D feature vector extracted from calibration snapshots
  [mean_T1, mean_T2, mean_readout_error, mean_1q_error, mean_2q_error]
- **Matching tolerance**: 1.5 hours (3 time steps at 0.5h resolution)
- **Data source**: SyntheticDriftBackend + PropertiesStream.replay_all()

## Results

| Method   | Precision | Recall | F1    | TP  | FP | FN |
|----------|-----------|--------|-------|-----|----|----|
| **PELT** | **0.907** | **0.930** | **0.918** | 107 | 11 | 8 |
| BOCPD    | 0.202     | 0.678  | 0.311 | 78  | 309| 37 |
| Ensemble | 0.907     | 0.930  | 0.918 | 107 | 11 | 8  |

## Best Parameters

- **PELT**: penalty=3.0, cost_model=rbf
- **BOCPD**: hazard_rate=0.1, threshold=0.1
- **Ensemble**: PELT(pen=3.0) + BOCPD(hz=0.05, th=0.1) — ensemble degenerates
  to PELT when BOCPD threshold is high enough to suppress false positives.

## Analysis

**PELT** is the clear winner with F1=0.918, far exceeding the 0.7 target:
- Detects 107/115 (93%) of labeled changepoints
- Only 11 false positives across 80 scenarios (including 15 no-CP scenarios)
- Robust to penalty tuning: pen=2–5 all yield F1 > 0.87

**BOCPD** suffers from excessive false positives (309 FP) due to:
- Standardized signal with piecewise-constant segments triggers the
  Student-t model at minor fluctuations near segment boundaries
- Better suited for truly streaming data with natural noise

**Recommendation**: Use PELT as the primary offline detector for batch
analysis. BOCPD is available for real-time streaming but should use
aggressive thresholding (threshold >= 0.3) to control false positives.

## Verification

```bash
python -m detection.evaluate --output detection/results.json
# Total time: ~520s
# Full per-scenario results in detection/results.json
```
