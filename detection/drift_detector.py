"""Drift Detector — changepoint detection on quantum hardware telemetry.

Two algorithms:
  1. PELT (Pruned Exact Linear Time) via `ruptures` — offline, finds optimal
     changepoints given a penalty. Best when you have a batch of data.
  2. Bayesian Online Changepoint Detection (BOCPD) — online, processes one
     observation at a time. Best for streaming telemetry.

Both operate on a multivariate signal extracted from CalibrationSnapshot dicts:
  [mean_t1, mean_t2, mean_readout_error, mean_1q_error, mean_2q_error]

Usage:
    from detection.drift_detector import DriftDetector

    detector = DriftDetector()
    report = detector.detect(snapshots)  # list of snapshot dicts
    print(report.changepoints)           # list of ChangePoint
    print(report.summary())
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray


# ──────────────────────────────────────────────
#  Data structures
# ──────────────────────────────────────────────

@dataclass
class ChangePoint:
    """A detected changepoint."""
    index: int               # position in the time series
    time_hours: float        # simulated time (hours)
    severity: float          # 0–1, how large the shift is
    direction: str           # e.g. "degradation" or "recovery"
    features_affected: list[str] = field(default_factory=list)

    def __repr__(self):
        return (f"ChangePoint(idx={self.index}, t={self.time_hours:.1f}h, "
                f"severity={self.severity:.3f}, {self.direction})")


@dataclass
class DriftReport:
    """Full drift analysis report."""
    changepoints: list[ChangePoint]
    algorithm: str
    n_observations: int
    time_range_hours: tuple[float, float]
    signal: NDArray[np.float64]       # (N, D) standardized signal
    raw_signal: NDArray[np.float64]   # (N, D) raw feature values
    feature_names: list[str]
    drift_scores: NDArray[np.float64]  # per-timestep drift score [0,1]
    recalibrate_recommended: bool

    def summary(self) -> str:
        lines = [
            f"DriftReport ({self.algorithm})",
            f"  Observations: {self.n_observations}",
            f"  Time range: {self.time_range_hours[0]:.1f}–{self.time_range_hours[1]:.1f} h",
            f"  Changepoints detected: {len(self.changepoints)}",
            f"  Recalibrate recommended: {self.recalibrate_recommended}",
        ]
        for cp in self.changepoints:
            lines.append(f"    {cp}")
        return "\n".join(lines)


# ──────────────────────────────────────────────
#  Feature extraction from snapshots
# ──────────────────────────────────────────────

FEATURE_NAMES = ["mean_t1", "mean_t2", "mean_readout_err", "mean_1q_err", "mean_2q_err"]


def extract_features(snapshots: list[dict]) -> tuple[NDArray, list[float]]:
    """Extract (N, 5) feature matrix and timestamps from snapshot dicts.

    Handles both CalibrationSnapshot-style dicts (from PropertiesStream)
    and BackendHealth-style dicts (from get_health).
    """
    features = []
    times = []
    for snap in snapshots:
        row = _extract_one(snap)
        features.append(row)
        # Time: prefer stream_sim_time_hours, fallback to index
        t = snap.get("stream_sim_time_hours", snap.get("calibration_age_minutes", 0) / 60.0)
        times.append(float(t))
    return np.array(features, dtype=np.float64), times


def _extract_one(snap: dict) -> list[float]:
    """Extract 5 features from a single snapshot dict."""
    # PropertiesStream snapshot format
    if "qubit_t1_us" in snap:
        t1 = np.mean(snap["qubit_t1_us"])
        t2 = np.mean(snap["qubit_t2_us"])
        re = np.mean(snap["qubit_readout_error"])
        ge1q = np.mean(snap["qubit_gate_error_1q"])
        ge2q_vals = list(snap.get("gate_error_2q", {}).values())
        ge2q = np.mean(ge2q_vals) if ge2q_vals else 0.0
        return [t1, t2, re, ge1q, ge2q]

    # BackendHealth-style
    if "avg_t1_us" in snap:
        return [
            snap["avg_t1_us"],
            snap.get("avg_t2_us", 0),
            snap.get("avg_readout_error", 0),
            snap.get("avg_1q_error", 0),
            snap.get("avg_2q_error", 0),
        ]

    raise ValueError(f"Unknown snapshot format, keys: {list(snap.keys())[:10]}")


# ──────────────────────────────────────────────
#  Standardization
# ──────────────────────────────────────────────

def standardize(X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Z-score standardization. Returns (X_std, mean, std)."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-12] = 1.0  # avoid div-by-zero for constant features
    return (X - mu) / sigma, mu, sigma


# ──────────────────────────────────────────────
#  PELT Detector (offline, batch)
# ──────────────────────────────────────────────

def detect_pelt(
    signal: NDArray,
    pen: float = 3.0,
    model: str = "rbf",
    min_size: int = 2,
) -> list[int]:
    """Run PELT on a (N, D) signal. Returns changepoint indices (0-based).

    Args:
        signal: standardized (N, D) array
        pen: penalty — higher = fewer changepoints
        model: cost model for ruptures ('rbf', 'l2', 'normal', etc.)
        min_size: minimum segment length
    """
    import ruptures as rpt

    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    # ruptures returns 1-based breakpoints, last = N (end sentinel)
    bkps = algo.predict(pen=pen)
    # Convert to 0-based indices, drop the end sentinel
    return [b - 1 for b in bkps if b < len(signal)]


# ──────────────────────────────────────────────
#  Bayesian Online Changepoint Detection
# ──────────────────────────────────────────────

class BOCPD:
    """Bayesian Online Changepoint Detection (Adams & MacKay, 2007).

    Uses sufficient statistics (online mean/var) per run length for O(1) per-run
    update instead of O(r) segment re-computation. The predictive distribution
    is Student-t from the Normal-Inverse-Gamma conjugate prior.

    Parameters:
        hazard_rate: 1 / expected_run_length (prior on segment duration)
        threshold: P(run_length=0) threshold for declaring a changepoint
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 50,
        threshold: float = 0.1,
    ):
        self.hazard = hazard_rate
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._t = 0
        self._changepoints: list[int] = []
        # Sufficient stats arrays — one entry per possible run length
        # Normal-Inverse-Gamma prior: mu0=0, kappa0, alpha0, beta0
        self._kappa0 = 1.0
        self._alpha0 = 1.0
        self._beta0 = 1.0
        self._mu0 = 0.0
        # Per-run-length sufficient stats (vectorised over D features)
        self._kappa: list[NDArray] = []
        self._alpha: list[NDArray] = []
        self._beta: list[NDArray] = []
        self._mu: list[NDArray] = []
        self._rl_probs: NDArray = np.array([1.0])

    def update(self, x: NDArray) -> bool:
        """Process one D-dimensional observation. Returns True if CP detected."""
        x = np.atleast_1d(x).astype(np.float64)
        D = len(x)
        t = self._t
        self._t += 1

        if t == 0:
            # First observation: initialize one run of length 0
            self._kappa = [np.full(D, self._kappa0)]
            self._alpha = [np.full(D, self._alpha0)]
            self._beta = [np.full(D, self._beta0)]
            self._mu = [np.full(D, self._mu0)]
            return False

        n_rl = len(self._kappa)  # number of existing run lengths

        # 1. Evaluate predictive probability for x under each run length
        #    Student-t with df=2*alpha, loc=mu, scale=beta*(kappa+1)/(alpha*kappa)
        log_pred = np.zeros(n_rl)
        for r in range(n_rl):
            kap = self._kappa[r]
            alp = self._alpha[r]
            bet = self._beta[r]
            mu = self._mu[r]

            df = 2 * alp
            scale2 = bet * (kap + 1) / (alp * kap)
            # Log student-t pdf (per feature, then sum)
            z = (x - mu) ** 2 / scale2
            # log Γ((df+1)/2) - log Γ(df/2) - 0.5*log(df*π*scale2) - ((df+1)/2)*log(1+z/df)
            from scipy.special import gammaln
            ll = (gammaln((df + 1) / 2) - gammaln(df / 2)
                  - 0.5 * np.log(df * np.pi * scale2)
                  - ((df + 1) / 2) * np.log(1 + z / df))
            log_pred[r] = np.sum(ll)

        pred = np.exp(log_pred - np.max(log_pred))  # numerical stability

        # 2. Growth and changepoint probabilities
        growth = self._rl_probs * (1 - self.hazard) * pred
        cp_mass = np.sum(self._rl_probs * self.hazard * pred)

        new_rl = np.zeros(n_rl + 1)
        new_rl[0] = cp_mass
        new_rl[1:] = growth

        total = new_rl.sum()
        if total > 0:
            new_rl /= total
        else:
            new_rl[0] = 1.0

        self._rl_probs = new_rl

        # 3. Update sufficient stats — grow existing runs, add new run
        new_kappa = []
        new_alpha = []
        new_beta = []
        new_mu = []

        # New run (run_length = 0): prior stats
        new_kappa.append(np.full(D, self._kappa0))
        new_alpha.append(np.full(D, self._alpha0))
        new_beta.append(np.full(D, self._beta0))
        new_mu.append(np.full(D, self._mu0))

        # Extend existing runs
        for r in range(n_rl):
            kap = self._kappa[r]
            alp = self._alpha[r]
            bet = self._beta[r]
            mu = self._mu[r]

            kap_new = kap + 1
            mu_new = (kap * mu + x) / kap_new
            alp_new = alp + 0.5
            bet_new = bet + 0.5 * kap * (x - mu) ** 2 / kap_new

            new_kappa.append(kap_new)
            new_alpha.append(alp_new)
            new_beta.append(bet_new)
            new_mu.append(mu_new)

        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta
        self._mu = new_mu

        # 4. Prune low-probability run lengths to save memory
        if len(self._rl_probs) > 200:
            keep = self._rl_probs > 1e-6
            keep[0] = True  # always keep run_length=0
            idx = np.where(keep)[0]
            self._rl_probs = self._rl_probs[idx]
            self._rl_probs /= self._rl_probs.sum()
            self._kappa = [self._kappa[i] for i in idx]
            self._alpha = [self._alpha[i] for i in idx]
            self._beta = [self._beta[i] for i in idx]
            self._mu = [self._mu[i] for i in idx]

        # 5. Detect changepoint
        is_cp = new_rl[0] > self.threshold
        if is_cp:
            self._changepoints.append(t)

        return is_cp

    def get_changepoints(self) -> list[int]:
        return list(self._changepoints)

    def get_run_length_probs(self) -> list[NDArray]:
        return [self._rl_probs]


# ──────────────────────────────────────────────
#  Ensemble Detector
# ──────────────────────────────────────────────

class DriftDetector:
    """High-level drift detector combining PELT and BOCPD.

    Modes:
      - 'pelt': offline PELT only
      - 'bocpd': online Bayesian CPD only
      - 'ensemble': both — a changepoint is reported if either flags it
                    within a tolerance window
    """

    def __init__(
        self,
        method: Literal["pelt", "bocpd", "ensemble"] = "ensemble",
        pelt_penalty: float = 3.0,
        pelt_model: str = "rbf",
        bocpd_hazard: float = 0.1,
        bocpd_threshold: float = 0.1,
        merge_window: int = 3,
        severity_window: int = 5,
        recalibrate_threshold: float = 0.4,
    ):
        self.method = method
        self.pelt_penalty = pelt_penalty
        self.pelt_model = pelt_model
        self.bocpd_hazard = bocpd_hazard
        self.bocpd_threshold = bocpd_threshold
        self.merge_window = merge_window
        self.severity_window = severity_window
        self.recalibrate_threshold = recalibrate_threshold

    def detect(self, snapshots: list[dict]) -> DriftReport:
        """Run drift detection on a list of snapshot dicts.

        Args:
            snapshots: list of dicts from PropertiesStream or get_health()

        Returns:
            DriftReport with changepoints, scores, and recommendation
        """
        raw, times = extract_features(snapshots)
        signal, mu, sigma = standardize(raw)
        n = len(signal)

        # Run selected algorithms
        pelt_cps = []
        bocpd_cps = []

        if self.method in ("pelt", "ensemble"):
            pelt_cps = detect_pelt(
                signal, pen=self.pelt_penalty, model=self.pelt_model
            )

        if self.method in ("bocpd", "ensemble"):
            bocpd = BOCPD(
                hazard_rate=self.bocpd_hazard,
                threshold=self.bocpd_threshold,
            )
            for i in range(n):
                bocpd.update(signal[i])
            bocpd_cps = bocpd.get_changepoints()

        # Merge changepoints
        if self.method == "ensemble":
            merged_indices = self._merge_changepoints(pelt_cps, bocpd_cps, n)
        elif self.method == "pelt":
            merged_indices = pelt_cps
        else:
            merged_indices = bocpd_cps

        # Deduplicate: merge close changepoints
        merged_indices = self._deduplicate(merged_indices)

        # Compute per-changepoint severity and direction
        changepoints = []
        for idx in merged_indices:
            cp = self._characterize_changepoint(idx, raw, signal, times)
            changepoints.append(cp)

        # Compute per-timestep drift scores
        drift_scores = self._compute_drift_scores(signal)

        # Recommendation
        recalibrate = (
            len(changepoints) > 0
            and any(cp.severity > self.recalibrate_threshold for cp in changepoints)
        )

        algo_name = self.method
        if self.method == "ensemble":
            algo_name = f"ensemble(PELT+BOCPD, pelt_cps={len(pelt_cps)}, bocpd_cps={len(bocpd_cps)})"

        return DriftReport(
            changepoints=changepoints,
            algorithm=algo_name,
            n_observations=n,
            time_range_hours=(times[0], times[-1]) if times else (0, 0),
            signal=signal,
            raw_signal=raw,
            feature_names=FEATURE_NAMES,
            drift_scores=drift_scores,
            recalibrate_recommended=recalibrate,
        )

    def _merge_changepoints(
        self, pelt_cps: list[int], bocpd_cps: list[int], n: int
    ) -> list[int]:
        """Union merge: a changepoint is kept if either algorithm flags it."""
        all_cps = set(pelt_cps) | set(bocpd_cps)
        return sorted(all_cps)

    def _deduplicate(self, cps: list[int]) -> list[int]:
        """Merge changepoints within merge_window of each other."""
        if not cps:
            return []
        cps = sorted(cps)
        merged = [cps[0]]
        for cp in cps[1:]:
            if cp - merged[-1] <= self.merge_window:
                # Keep the one with higher index (more recent)
                merged[-1] = cp
            else:
                merged.append(cp)
        return merged

    def _characterize_changepoint(
        self, idx: int, raw: NDArray, signal: NDArray, times: list[float]
    ) -> ChangePoint:
        """Compute severity, direction, and affected features for a changepoint."""
        w = self.severity_window
        n, d = raw.shape

        # Compare segments before and after
        before = signal[max(0, idx - w):idx]
        after = signal[idx:min(n, idx + w)]

        if len(before) == 0 or len(after) == 0:
            severity = 0.5
            direction = "unknown"
            affected = FEATURE_NAMES[:]
        else:
            # Severity: max absolute shift in any feature (in std units)
            shift = after.mean(axis=0) - before.mean(axis=0)
            severity = float(np.clip(np.max(np.abs(shift)) / 3.0, 0.0, 1.0))

            # Direction: T1 decrease or gate error increase = degradation
            # T1/T2 are features 0,1 (decrease = bad), errors are 2,3,4 (increase = bad)
            degradation_score = -shift[0] - shift[1] + shift[2] + shift[3] + shift[4]
            direction = "degradation" if degradation_score > 0 else "recovery"

            # Affected features: any with |shift| > 0.5 std
            affected = [
                FEATURE_NAMES[i] for i in range(d) if abs(shift[i]) > 0.5
            ]
            if not affected:
                affected = [FEATURE_NAMES[int(np.argmax(np.abs(shift)))]]

        return ChangePoint(
            index=idx,
            time_hours=times[idx] if idx < len(times) else 0.0,
            severity=severity,
            direction=direction,
            features_affected=affected,
        )

    def _compute_drift_scores(self, signal: NDArray) -> NDArray:
        """Compute per-timestep drift score [0, 1].

        Uses a rolling window: score = max deviation from recent mean.
        """
        n = len(signal)
        scores = np.zeros(n)
        w = max(5, self.severity_window)

        for i in range(n):
            start = max(0, i - w)
            window = signal[start:i + 1]
            if len(window) < 2:
                continue
            deviation = np.abs(signal[i] - window.mean(axis=0))
            scores[i] = float(np.clip(np.max(deviation) / 3.0, 0.0, 1.0))

        return scores
