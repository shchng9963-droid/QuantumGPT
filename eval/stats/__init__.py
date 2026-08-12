"""Paired statistical tests for v2.5 Sprint A benchmark.

Provides four primitives that paper §4 needs:

  wilcoxon_paired(a, b)     — Wilcoxon signed-rank on paired continuous metric (best_score)
  mcnemar_paired(a, b)      — McNemar test on paired binary outcome (target_success)
  bootstrap_ci_paired(a, b) — bootstrap 95% CI of mean(a-b) over n_resamples paired draws
  paired_summary(a, b)      — combines all three into a single dict for reporting

All four enforce: len(a) == len(b), pairs are aligned (no shuffling), and
return NaN/None when n is too small for the test instead of raising. This
keeps reporting code simple — every (system_a, system_b, subset) cell can
call paired_summary() and get a stable shape.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class PairedStats:
    """Lightweight summary returned by paired_summary()."""

    n: int
    mean_delta: float
    median_delta: float
    wilcoxon_stat: float | None
    wilcoxon_p: float | None
    mcnemar_stat: float | None
    mcnemar_p: float | None
    bootstrap_mean: float | None
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "mean_delta": self.mean_delta,
            "median_delta": self.median_delta,
            "wilcoxon_stat": self.wilcoxon_stat,
            "wilcoxon_p": self.wilcoxon_p,
            "mcnemar_stat": self.mcnemar_stat,
            "mcnemar_p": self.mcnemar_p,
            "bootstrap_mean": self.bootstrap_mean,
            "bootstrap_ci_low": self.bootstrap_ci_low,
            "bootstrap_ci_high": self.bootstrap_ci_high,
            **self.extras,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Wilcoxon paired (continuous metric)
# ──────────────────────────────────────────────────────────────────────────────

def wilcoxon_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    zero_method: str = "wilcox",
) -> tuple[float, float] | tuple[None, None]:
    """Wilcoxon signed-rank test on paired continuous samples.

    Returns (stat, p) using scipy.stats.wilcoxon when n_nonzero_diff >= 5;
    returns (None, None) when n is too small (caller decides how to render).

    zero_method: passed through to scipy; 'wilcox' drops zeros, matching v2.4
    ablation reports.
    """

    if len(a) != len(b):
        raise ValueError(f"Length mismatch: len(a)={len(a)} len(b)={len(b)}")
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    nonzero = [d for d in diffs if d != 0]
    if len(nonzero) < 5:
        return None, None
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(diffs, zero_method=zero_method, alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# McNemar paired (binary outcome)
# ──────────────────────────────────────────────────────────────────────────────

def mcnemar_paired(a_success: Sequence[bool], b_success: Sequence[bool]) -> tuple[float, float] | tuple[None, None]:
    """McNemar exact test on paired binary outcomes.

    Counts discordant pairs:
      b01 = (a fails, b succeeds)
      b10 = (a succeeds, b fails)

    Returns (chi2_stat, p) using scipy.stats.binomtest exact two-sided p
    (the exact test, valid for any n; the chi2 statistic is reported for
    paper-friendly formatting). Returns (None, None) when no discordant
    pairs exist (no signal in the data).
    """

    if len(a_success) != len(b_success):
        raise ValueError(
            f"Length mismatch: len(a)={len(a_success)} len(b)={len(b_success)}"
        )
    b01 = sum(1 for x, y in zip(a_success, b_success) if (not x) and y)
    b10 = sum(1 for x, y in zip(a_success, b_success) if x and (not y))
    n_disc = b01 + b10
    if n_disc == 0:
        return None, None

    # Exact two-sided binomial p-value: P(|X - n/2| >= |b10 - n/2|)
    try:
        from scipy.stats import binomtest
        p = float(binomtest(b10, n_disc, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        # Fallback: hand-rolled exact two-sided binomial test
        p = _binom_two_sided_p(b10, n_disc, 0.5)

    # Continuity-corrected chi-squared statistic (paper convention)
    if n_disc > 0:
        chi2 = (abs(b10 - b01) - 1) ** 2 / n_disc if n_disc > 1 else 0.0
    else:
        chi2 = 0.0
    return float(chi2), float(p)


def _binom_two_sided_p(k: int, n: int, p: float) -> float:
    """Two-sided exact binomial p-value, used as scipy-free fallback."""
    from math import comb

    def _pmf(i):
        return comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    obs = _pmf(k)
    return float(sum(_pmf(i) for i in range(n + 1) if _pmf(i) <= obs + 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap paired 95% CI
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci_paired(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10000,
    ci: float = 0.95,
    seed: int = 12345,
) -> tuple[float, float, float] | tuple[None, None, None]:
    """Bootstrap mean(a-b) and CI by resampling paired indices with replacement.

    Returns (mean_delta, ci_low, ci_high) for percentile bootstrap.
    Returns (None, None, None) when len(a) < 2 (bootstrap is degenerate).
    """

    if len(a) != len(b):
        raise ValueError(f"Length mismatch: len(a)={len(a)} len(b)={len(b)}")
    n = len(a)
    if n < 2:
        return None, None, None

    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_resamples)]
    hi = means[min(n_resamples - 1, int((1 - alpha) * n_resamples))]
    return float(sum(means) / len(means)), float(lo), float(hi)


# ──────────────────────────────────────────────────────────────────────────────
# Combined summary
# ──────────────────────────────────────────────────────────────────────────────

def paired_summary(
    a: Sequence[float],
    b: Sequence[float],
    *,
    a_success: Sequence[bool] | None = None,
    b_success: Sequence[bool] | None = None,
    n_bootstrap: int = 10000,
    seed: int = 12345,
) -> PairedStats:
    """Combine Wilcoxon, McNemar (if success vectors given), bootstrap CI.

    a, b: paired continuous metric (e.g., best_post_drift_score).
    a_success, b_success: paired binary outcome (e.g., target_success);
       optional — if either is None, McNemar fields stay None.
    """

    if len(a) != len(b):
        raise ValueError(f"Length mismatch: len(a)={len(a)} len(b)={len(b)}")
    if not a:
        raise ValueError("paired_summary requires non-empty inputs")

    diffs = sorted(float(x) - float(y) for x, y in zip(a, b))
    n = len(diffs)
    mean_d = sum(diffs) / n
    median_d = (diffs[n // 2] if n % 2 else 0.5 * (diffs[n // 2 - 1] + diffs[n // 2]))

    w_stat, w_p = wilcoxon_paired(a, b)

    if a_success is not None and b_success is not None:
        m_stat, m_p = mcnemar_paired(a_success, b_success)
    else:
        m_stat, m_p = None, None

    boot_mean, boot_lo, boot_hi = bootstrap_ci_paired(a, b, n_resamples=n_bootstrap, seed=seed)

    return PairedStats(
        n=n,
        mean_delta=float(mean_d),
        median_delta=float(median_d),
        wilcoxon_stat=w_stat,
        wilcoxon_p=w_p,
        mcnemar_stat=m_stat,
        mcnemar_p=m_p,
        bootstrap_mean=boot_mean,
        bootstrap_ci_low=boot_lo,
        bootstrap_ci_high=boot_hi,
    )
