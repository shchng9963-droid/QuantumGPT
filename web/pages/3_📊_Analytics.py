"""Benchmark explorer — 1,615 行 paired 数据现场重算（中文版）."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qgdata import (
    SYSTEM_COLOR,
    bootstrap_ci,
    load_sprint_a,
    load_sprint_b,
    make_pairs,
)
from theme import theme_css

st.set_page_config(page_title="Benchmark explorer", page_icon="📈", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)

# Plotly dark template tweaked for our palette
PLOT_LAYOUT = dict(
    paper_bgcolor="#101010",
    plot_bgcolor="#101010",
    font=dict(family="Inter, system-ui, sans-serif", color="#f2f2f2", size=12),
    margin=dict(l=140, r=30, t=50, b=40),
)

st.markdown(
    """
<div class="qg-eyebrow">live · paired benchmark explorer</div>
<h1 style="font-size: 38px !important; line-height: 1.15 !important; margin: 8px 0 16px 0 !important;">
📈 把 1,615 行 paired 数据自己切一刀
</h1>
<p style="color: var(--parchment); font-size: 16px; line-height: 1.7; max-width: 900px;">
所有图表都是基于你下方筛选出来的子集<strong style="color: var(--snow);">现场重算</strong>的，
不是预渲染。逐 circuit Δ 与全样本分布会随你切换 drift profile / circuit 立刻更新。
</p>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------- data
df_a = load_sprint_a()
df_b = load_sprint_b()

if df_a.empty and df_b.empty:
    st.error("没找到任何 benchmark 数据。请确认 `eval/results/v3_main_merged/` 存在。")
    st.stop()
elif df_a.empty:
    df = df_b
elif df_b.empty:
    df = df_a
else:
    common = [c for c in df_a.columns if c in df_b.columns]
    df = pd.concat([df_a[common], df_b[common]], ignore_index=True)

all_systems = sorted(df["system"].unique())
all_profiles = sorted(df["drift_profile"].dropna().unique())
all_circuits = sorted(df["circuit"].dropna().unique())

st.markdown(
    """
<div class="help-box" style="background: var(--carbon-2); border: 1px solid var(--charcoal);
     border-radius: 8px; padding: 16px 20px; margin: 16px 0;">
<h4 style="color: var(--signal) !important; font-family: 'JetBrains Mono', monospace; font-size: 12px !important;
     letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 10px 0 !important;">怎么读这页</h4>
<ul style="color: var(--parchment); font-size: 14px; line-height: 1.7;">
<li><strong style="color: var(--snow);">System A / B</strong>：选两个系统做配对比较，A 是 treatment，B 是 control。</li>
<li><strong style="color: var(--snow);">Drift / Circuit</strong>：限定参与配对的样本切片。两个系统在同一切片下的 (circuit, drift, seed) 才会被配上。</li>
<li>下方四个 metric 实时显示当前切片下的 <code>n / 均值 Δ / 95% bootstrap CI / 胜负数</code>。</li>
<li>第一张是 per-circuit Δ；第二张是两系统 best_score 的全样本分布；下方表格列出所有配对原始记录。</li>
</ul>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------- filters
f1, f2, f3 = st.columns(3)
with f1:
    sys_a = st.selectbox(
        "System A (treatment)",
        all_systems,
        index=all_systems.index("QuantumGPT-Full") if "QuantumGPT-Full" in all_systems else 0,
    )
with f2:
    options_b = [s for s in all_systems if s != sys_a]
    sys_b_default = "Static-Pipeline" if "Static-Pipeline" in options_b else options_b[0]
    sys_b = st.selectbox(
        "System B (control)",
        options_b,
        index=options_b.index(sys_b_default),
    )
with f3:
    profiles = st.multiselect("Drift profiles", all_profiles, default=all_profiles)

circuits = st.multiselect("Circuits", all_circuits, default=all_circuits)

slice_df = df[df["drift_profile"].isin(profiles) & df["circuit"].isin(circuits)]
pairs = make_pairs(slice_df, sys_a, sys_b)

# ----------------------------------------------------------------------- metrics
n = len(pairs)
delta_mean, lo, hi = bootstrap_ci(pairs["delta"].to_numpy()) if n else (0, 0, 0)
wins = int((pairs["delta"] > 0).sum())
losses = int((pairs["delta"] < 0).sum())

s1, s2, s3, s4 = st.columns(4)
s1.metric("Paired rows", f"{n}")
s2.metric("Mean Δ", f"{delta_mean:+.4f}")
s3.metric("95% CI", f"[{lo:+.4f}, {hi:+.4f}]")
s4.metric("A wins / losses", f"{wins} / {losses}")

if n == 0:
    st.warning("当前切片下没有可配对样本。请放宽 drift / circuit 选择。")
    st.stop()

# ----------------------------------------------------------------------- forest
per_circuit = (
    pairs.groupby("circuit")["delta"]
    .agg(["mean", "count"]).reset_index()
    .sort_values("mean")
)

forest = go.Figure()
forest.add_vline(x=0, line_color="#3d3a39", line_width=1)
color = SYSTEM_COLOR.get(sys_a, "#00d992")
forest.add_trace(go.Scatter(
    x=per_circuit["mean"],
    y=per_circuit["circuit"],
    mode="markers",
    marker=dict(size=12, color=color, line=dict(color="#101010", width=1)),
    customdata=per_circuit[["count"]].to_numpy(),
    hovertemplate="<b>%{y}</b><br>mean Δ = %{x:+.4f}<br>n = %{customdata[0]}<extra></extra>",
    showlegend=False,
))
forest.update_layout(
    title=f"Per-circuit paired Δ — {sys_a} 减去 {sys_b}",
    xaxis_title="Δ best post-drift fidelity",
    yaxis_title="",
    height=max(360, 34 * len(per_circuit) + 140),
    **PLOT_LAYOUT,
)
forest.update_xaxes(gridcolor="#3d3a39", zerolinecolor="#3d3a39")
forest.update_yaxes(gridcolor="#1a1a1a")
st.plotly_chart(forest, width="stretch")

st.markdown(
    f"""
<div class="qg-takeaway">
当前切片下 <strong>{sys_a}</strong> 在 <strong>{n}</strong> 个配对样本上的均值 Δ 为
<code>{delta_mean:+.4f}</code>，95% bootstrap CI <code>[{lo:+.4f}, {hi:+.4f}]</code>。
A 胜 {wins} / 败 {losses} / 余 {n-wins-losses} 平。CI 不跨 0 即视为强证据。
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------- strip
st.markdown(
    f'<h3 style="margin-top: 2.2rem !important;">两系统的 best post-drift fidelity 分布</h3>',
    unsafe_allow_html=True,
)

strip = go.Figure()
y_map = {sys_a: 1, sys_b: 0}
for sysname, c in [(sys_a, SYSTEM_COLOR.get(sys_a, "#00d992")),
                   (sys_b, SYSTEM_COLOR.get(sys_b, "#8b949e"))]:
    sub = slice_df[slice_df["system"] == sysname]
    if sub.empty:
        continue
    rng = np.random.default_rng(7)
    jitter = rng.normal(0, 0.08, size=len(sub))
    y = np.full(len(sub), y_map[sysname], dtype=float) + jitter
    strip.add_trace(go.Scatter(
        x=sub["best_score"], y=y,
        mode="markers",
        marker=dict(size=8, color=c, opacity=0.7,
                    line=dict(color="#050507", width=0.5)),
        name=sysname,
        customdata=sub[["circuit", "drift_profile", "seed"]].to_numpy(),
        hovertemplate=("%{customdata[0]} · %{customdata[1]} · seed=%{customdata[2]}"
                       "<br>best=%{x:.4f}<extra></extra>"),
    ))
strip.update_layout(
    height=320,
    xaxis_title="best post-drift fidelity",
    yaxis=dict(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=[sys_b, sys_a],
        range=[-0.6, 1.6],
    ),
    legend=dict(orientation="h", y=-0.22, font=dict(color="#f2f2f2")),
    **PLOT_LAYOUT,
)
strip.update_xaxes(gridcolor="#3d3a39")
strip.update_yaxes(gridcolor="#1a1a1a")
st.plotly_chart(strip, width="stretch")

# ----------------------------------------------------------------------- raw
with st.expander("原始 paired 行（按 Δ 降序）"):
    st.dataframe(pairs.sort_values("delta", ascending=False), width="stretch")
