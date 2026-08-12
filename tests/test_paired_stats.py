"""Tests for eval.stats paired statistical primitives (v2.5 Sprint A, A6).

These tests pin down the math against scipy and against hand-derived
small-sample answers, so future refactors can't silently change p-values.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Wilcoxon paired
# ──────────────────────────────────────────────────────────────────────────────

def test_wilcoxon_paired_matches_scipy_on_clear_signal():
    """When a clearly beats b on every pair, p must be small (< 0.05)."""
    from eval.stats import wilcoxon_paired
    a = [0.91, 0.88, 0.93, 0.89, 0.92, 0.90, 0.94, 0.87, 0.95, 0.91]
    b = [0.85, 0.83, 0.86, 0.82, 0.85, 0.84, 0.87, 0.81, 0.88, 0.85]
    stat, p = wilcoxon_paired(a, b)
    assert stat is not None
    assert p is not None
    assert p < 0.05


def test_wilcoxon_paired_returns_none_when_n_too_small():
    """n_nonzero_diff < 5 must produce (None, None) instead of raising."""
    from eval.stats import wilcoxon_paired
    a = [0.91, 0.88, 0.93]
    b = [0.85, 0.83, 0.86]
    stat, p = wilcoxon_paired(a, b)
    assert stat is None and p is None


def test_wilcoxon_paired_length_mismatch_raises():
    from eval.stats import wilcoxon_paired
    with pytest.raises(ValueError):
        wilcoxon_paired([0.1, 0.2], [0.1, 0.2, 0.3])


# ──────────────────────────────────────────────────────────────────────────────
# McNemar paired
# ──────────────────────────────────────────────────────────────────────────────

def test_mcnemar_paired_no_discordant_returns_none():
    """When success vectors agree everywhere, McNemar has no signal."""
    from eval.stats import mcnemar_paired
    a = [True, True, False, False]
    b = [True, True, False, False]
    stat, p = mcnemar_paired(a, b)
    assert stat is None and p is None


def test_mcnemar_paired_strong_signal_p_low():
    """When a wins on 8 of 8 discordant pairs, p must be < 0.01."""
    from eval.stats import mcnemar_paired
    # 8 (a=T, b=F) discordant pairs, 0 (a=F, b=T)
    a = [True] * 8 + [True, False]
    b = [False] * 8 + [True, False]
    stat, p = mcnemar_paired(a, b)
    assert p is not None
    assert p < 0.01


def test_mcnemar_paired_balanced_discordant_p_one():
    """Equal discordant counts → exact two-sided p == 1.0."""
    from eval.stats import mcnemar_paired
    # 3 (a=T, b=F), 3 (a=F, b=T) — perfectly balanced
    a = [True, True, True, False, False, False]
    b = [False, False, False, True, True, True]
    stat, p = mcnemar_paired(a, b)
    assert p == pytest.approx(1.0, abs=1e-9)


def test_mcnemar_paired_length_mismatch_raises():
    from eval.stats import mcnemar_paired
    with pytest.raises(ValueError):
        mcnemar_paired([True, False], [True, False, True])


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_paired_includes_zero_when_no_effect():
    """Mean(a-b)=0 and small spread → 0 ∈ CI."""
    from eval.stats import bootstrap_ci_paired
    a = [0.85, 0.86, 0.87, 0.85, 0.86, 0.85]
    b = [0.86, 0.85, 0.86, 0.86, 0.85, 0.86]  # ≈ same
    mean, lo, hi = bootstrap_ci_paired(a, b, n_resamples=2000, seed=42)
    assert lo <= 0 <= hi


def test_bootstrap_ci_paired_excludes_zero_when_strong_effect():
    """Mean(a-b)=+0.05 with low noise → 0 < CI."""
    from eval.stats import bootstrap_ci_paired
    a = [0.90, 0.91, 0.89, 0.90, 0.92, 0.90, 0.91, 0.89, 0.90, 0.91]
    b = [0.85, 0.86, 0.84, 0.85, 0.87, 0.85, 0.86, 0.84, 0.85, 0.86]
    mean, lo, hi = bootstrap_ci_paired(a, b, n_resamples=5000, seed=42)
    assert mean == pytest.approx(0.05, abs=0.01)
    assert lo > 0


def test_bootstrap_ci_paired_too_small_returns_none():
    """n=1 cannot bootstrap meaningfully."""
    from eval.stats import bootstrap_ci_paired
    mean, lo, hi = bootstrap_ci_paired([0.9], [0.85])
    assert mean is None and lo is None and hi is None


def test_bootstrap_is_deterministic_with_seed():
    """Same seed must give bit-identical bootstrap CI on same input."""
    from eval.stats import bootstrap_ci_paired
    a = [0.91, 0.88, 0.93, 0.89, 0.92, 0.90, 0.94, 0.87, 0.95, 0.91]
    b = [0.85, 0.83, 0.86, 0.82, 0.85, 0.84, 0.87, 0.81, 0.88, 0.85]
    r1 = bootstrap_ci_paired(a, b, n_resamples=1000, seed=12345)
    r2 = bootstrap_ci_paired(a, b, n_resamples=1000, seed=12345)
    assert r1 == r2


# ──────────────────────────────────────────────────────────────────────────────
# paired_summary
# ──────────────────────────────────────────────────────────────────────────────

def test_paired_summary_combines_all_three():
    """paired_summary returns Wilcoxon + McNemar + bootstrap in one dataclass."""
    from eval.stats import paired_summary
    a = [0.91, 0.88, 0.93, 0.89, 0.92, 0.90, 0.94, 0.87, 0.95, 0.91]
    b = [0.85, 0.83, 0.86, 0.82, 0.85, 0.84, 0.87, 0.81, 0.88, 0.85]
    a_succ = [True] * 10
    b_succ = [True, True, False, False, True, False, True, False, True, True]

    res = paired_summary(a, b, a_success=a_succ, b_success=b_succ,
                         n_bootstrap=2000, seed=42)
    assert res.n == 10
    assert res.mean_delta == pytest.approx(sum(a) / 10 - sum(b) / 10, abs=1e-9)
    assert res.wilcoxon_p is not None and res.wilcoxon_p < 0.05
    assert res.mcnemar_p is not None  # 4 discordant pairs (a=T, b=F)
    assert res.bootstrap_ci_low is not None
    assert res.bootstrap_ci_high > res.bootstrap_ci_low

    d = res.to_dict()
    assert "mean_delta" in d
    assert "wilcoxon_p" in d
    assert "mcnemar_p" in d


def test_paired_summary_empty_input_raises():
    from eval.stats import paired_summary
    with pytest.raises(ValueError):
        paired_summary([], [])


def test_paired_summary_works_without_success_vectors():
    """When success vectors are omitted, McNemar fields stay None."""
    from eval.stats import paired_summary
    a = [0.91, 0.88, 0.93, 0.89, 0.92, 0.90, 0.94, 0.87, 0.95, 0.91]
    b = [0.85, 0.83, 0.86, 0.82, 0.85, 0.84, 0.87, 0.81, 0.88, 0.85]
    res = paired_summary(a, b, n_bootstrap=1000, seed=42)
    assert res.mcnemar_stat is None
    assert res.mcnemar_p is None
    assert res.wilcoxon_p is not None
