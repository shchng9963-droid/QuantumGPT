"""Shared data loaders + plot helpers for the QuantumGPT web demo.

All paths are relative to repo root. Imported by web/app.py and web/pages/*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SHOWCASE = ROOT / "figures" / "showcase"

# ---------------------------------------------------------------- result jsonl
SPRINT_A_JSONL = ROOT / "eval" / "results" / "v3_main_merged" / "result_summary.jsonl"
SPRINT_B_DIRS = [
    ROOT / "eval" / "results" / "v3_orchestrator_pilot_top2_moreseeds",
    ROOT / "eval" / "results" / "v3_orchestrator",
    ROOT / "eval" / "results" / "v3_harder_rerun",
]
RELIABILITY_JSON = ROOT / "eval" / "results" / "verifier_reliability.json"
G3_DIR = ROOT / "eval" / "stats" / "v3_g3"

KEEP_COLS = [
    "system", "circuit", "drift_profile", "seed", "model",
    "best_score", "final_score", "target_fidelity", "target_success",
    "tool_calls", "total_cost_usd", "elapsed_seconds",
    "trace_file", "run_id", "task",
]


def _iter_jsonl(p: Path) -> Iterator[dict]:
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@st.cache_data(show_spinner=False)
def load_sprint_a() -> pd.DataFrame:
    rows = list(_iter_jsonl(SPRINT_A_JSONL))
    df = pd.DataFrame(rows)
    keep = [c for c in KEEP_COLS if c in df.columns]
    return df[keep].copy() if not df.empty else df


@st.cache_data(show_spinner=False)
def load_sprint_b() -> pd.DataFrame:
    rows: list[dict] = []
    for d in SPRINT_B_DIRS:
        f = d / "result_summary.jsonl"
        rows.extend(list(_iter_jsonl(f)))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    keep = [c for c in KEEP_COLS if c in df.columns]
    return df[keep].copy()


@st.cache_data(show_spinner=False)
def load_reliability() -> dict:
    if RELIABILITY_JSON.exists():
        return json.loads(RELIABILITY_JSON.read_text())
    return {}


@st.cache_data(show_spinner=False)
def load_g3() -> dict[str, dict]:
    out = {}
    if G3_DIR.exists():
        for f in sorted(G3_DIR.glob("*.json")):
            try:
                out[f.stem] = json.loads(f.read_text())
            except Exception:
                pass
    return out


# ------------------------------------------------------------------- pairings
def make_pairs(df: pd.DataFrame, system_a: str, system_b: str,
               key=("circuit", "drift_profile", "seed")) -> pd.DataFrame:
    """Return a wide DataFrame with one row per matched (a, b) task."""
    cols = list(key) + ["best_score"]
    a = df[df["system"] == system_a][cols].rename(columns={"best_score": "score_a"})
    b = df[df["system"] == system_b][cols].rename(columns={"best_score": "score_b"})
    merged = a.merge(b, on=list(key), how="inner")
    merged["delta"] = merged["score_a"] - merged["score_b"]
    return merged


def bootstrap_ci(values: np.ndarray, n: int = 5000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(values.mean()), lo, hi


# ----------------------------------------------------------------------- traces
@st.cache_data(show_spinner=False)
def list_full_traces(limit: int = 60) -> pd.DataFrame:
    """Return a DataFrame of available QuantumGPT-Full post_drift_trace.json files."""
    base = ROOT / "eval" / "results" / "public_mqtbench_agent_deepseek_10tasks_3seeds" / "QuantumGPT-Full"
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for trace in sorted(base.glob("*/*/*/post_drift_trace.json"))[:limit]:
        parts = trace.relative_to(base).parts  # profile / circuit / seedX / file
        if len(parts) >= 4:
            rows.append({
                "drift_profile": parts[0],
                "circuit": parts[1],
                "seed": parts[2],
                "trace": str(trace),
            })
    return pd.DataFrame(rows)


def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------- constants
BRAND_INK = "#1a1a2e"
BRAND_TEAL = "#2a9d8f"
BRAND_LAV = "#9d4edd"
BRAND_ROSE = "#e76f51"
BRAND_YEL = "#e9c46a"
BRAND_PANEL = "#fafafa"
BRAND_HAIR = "#e0e0e0"

SYSTEM_COLOR = {
    "QuantumGPT-Full": BRAND_LAV,
    "QuantumGPT-Orchestrator": BRAND_LAV,
    "QuantumGPT-No-Drift": BRAND_TEAL,
    "Static-Pipeline": BRAND_ROSE,
    "ChatLLM-NoTools": BRAND_YEL,
    "Oracle-Adaptive": BRAND_INK,
}
