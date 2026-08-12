"""Trace replay — 真实 ReAct trace 逐步回放（中文版）."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qgdata import list_full_traces, load_trace
from theme import theme_css

st.set_page_config(page_title="Trace replay", page_icon="🔍", layout="wide")
st.markdown(theme_css(), unsafe_allow_html=True)

st.markdown(
    """
<style>
.tr-step {
    background: var(--carbon);
    border: 1px solid var(--charcoal);
    border-left: 2px solid var(--charcoal);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: border-color 0.15s ease;
}
.tr-step:hover { border-left-color: var(--signal); }
.tr-tool { color: var(--mint); font-family: 'JetBrains Mono', monospace; font-weight: 600; }
.tr-thought { color: var(--parchment); font-style: italic; font-size: 13.5px; }
.verdict-pass {
    background: rgba(0, 217, 146, 0.08);
    border: 1px solid var(--emerald-line);
    border-left: 3px solid var(--signal);
    padding: 14px 18px; border-radius: 8px;
    color: var(--snow);
}
.verdict-fail {
    background: rgba(251, 86, 91, 0.08);
    border: 1px solid rgba(251, 86, 91, 0.4);
    border-left: 3px solid var(--rose);
    padding: 14px 18px; border-radius: 8px;
    color: var(--snow);
}
.verdict-pass strong, .verdict-fail strong { color: var(--snow) !important; }
.help-box {
    background: var(--carbon-2); border: 1px solid var(--charcoal);
    border-radius: 8px; padding: 16px 20px; margin: 16px 0;
}
.help-box h4 { color: var(--signal) !important; font-family: 'JetBrains Mono', monospace; font-size: 12px !important;
               letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 10px 0 !important; }
.help-box p, .help-box li { color: var(--parchment) !important; font-size: 14px !important; line-height: 1.7 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="qg-eyebrow">live · ReAct trace replay</div>
<h1 style="font-size: 38px !important; line-height: 1.15 !important; margin: 8px 0 16px 0 !important;">
🔍 真实 trace 逐步回放
</h1>
<p style="color: var(--parchment); font-size: 16px; line-height: 1.7; max-width: 900px;">
每一步都是 QuantumGPT-Full 在 30 个 mqtbench 任务上跑出的真实 ReAct 调用。
你看到的工具输入输出全部来自落盘 JSON —— 不是脚本生成的演示数据。
</p>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="help-box">
<h4>怎么用这个页面</h4>
<ul>
<li>顶部三个下拉选定 <strong>(drift profile, circuit, seed)</strong>，每一组都对应一个真实任务。</li>
<li>顶部四个 metric 是这次 run 的全局开销：工具调用数、墙钟、美元成本、token 总量。</li>
<li>"最终回答 (agent claim)"是模型给客户的回复；下面的 verifier 区块给出 trace 一致性判定。</li>
<li>"Steps"列出 ReAct 全部步骤；点开任一步可以看到原始 JSON。</li>
</ul>
</div>
""",
    unsafe_allow_html=True,
)

traces = list_full_traces(limit=80)
if traces.empty:
    st.warning("找不到任何 QuantumGPT-Full trace。请先跑一次 `eval/public_mqtbench_agent_eval.py`。")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    profile = st.selectbox("Drift profile", sorted(traces["drift_profile"].unique()))
with c2:
    sub = traces[traces["drift_profile"] == profile]
    circuit = st.selectbox("Circuit", sorted(sub["circuit"].unique()))
with c3:
    sub = sub[sub["circuit"] == circuit]
    seed = st.selectbox("Seed", sorted(sub["seed"].unique()))

row = sub[sub["seed"] == seed].iloc[0]
trace_path = row["trace"]

st.markdown(
    f'<p style="color: var(--slate); font-family: \'JetBrains Mono\', monospace; font-size: 12px;">'
    f'trace_file = <code>{trace_path}</code></p>',
    unsafe_allow_html=True,
)

trace = load_trace(trace_path)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Tool calls", trace.get("num_tool_calls", "—"))
m2.metric("Wall time", f"{trace.get('elapsed_seconds', 0.0):.1f}s")
cs = trace.get("cost_summary", {}) or {}
m3.metric("Cost", f"${cs.get('total_cost_usd', 0.0):.4f}")
m4.metric("Tokens", f"{trace.get('total_tokens', 0):,}")

st.markdown('<h3 style="margin-top: 2rem !important;">最终回答 (agent claim)</h3>', unsafe_allow_html=True)
final = trace.get("final_answer", "")
st.code(final[:2000] if final else "(empty)", language="markdown")

diag = trace.get("diagnostics", {}) or {}
verifier_data = diag.get("verifier") or diag.get("verifier_verdict") or {}
if not verifier_data and "state_summary" in trace:
    verifier_data = (trace["state_summary"] or {}).get("verifier") or {}

st.markdown('<h3>Verifier 判定</h3>', unsafe_allow_html=True)
if verifier_data:
    passed = (
        verifier_data.get("passed")
        or verifier_data.get("ok")
        or verifier_data.get("verdict") in {"pass", "ok"}
    )
    if passed:
        st.markdown(
            '<div class="verdict-pass"><strong>PASS</strong> · '
            'claim 与 trace 一致，可以交付给客户。</div>',
            unsafe_allow_html=True,
        )
    else:
        reason = verifier_data.get("reason", "claim 缺乏工具调用支持")
        st.markdown(
            f'<div class="verdict-fail"><strong>FAIL</strong> · {reason}</div>',
            unsafe_allow_html=True,
        )
    with st.expander("展开完整 verifier payload"):
        st.json(verifier_data)
else:
    st.info(
        "这条 trace 来自 verifier 加入主流水线之前的批次。"
        "生产 9 / 9 命中数据请见首页图 4 与 `eval/results/verifier_reliability.json`。"
    )

st.markdown('<h3>ReAct steps</h3>', unsafe_allow_html=True)
steps = trace.get("steps", [])
if not steps:
    st.warning("这条 trace 没有 steps 字段。")
    st.stop()

st.markdown(
    f'<p style="color: var(--slate); font-family: \'JetBrains Mono\', monospace; font-size: 12px;">'
    f'{len(steps)} steps · 点击展开 raw JSON</p>',
    unsafe_allow_html=True,
)

for i, s in enumerate(steps, start=1):
    role = s.get("role") or s.get("kind") or s.get("type", "step")
    name = s.get("name") or s.get("tool") or s.get("action") or ""
    summary = s.get("summary") or s.get("content") or ""
    if isinstance(summary, dict):
        summary = json.dumps(summary, indent=2)[:400]
    summary = (summary or "").strip().replace("\n", " ")
    head = f"**{i:02d}** · `{role}`"
    if name:
        head += f" → `{name}`"
    if summary:
        head += f" — {summary[:140]}{'…' if len(summary) > 140 else ''}"
    with st.expander(head, expanded=False):
        st.json(s)
