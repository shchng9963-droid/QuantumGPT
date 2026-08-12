"""QuantumGPT — research demo dashboard (中文版).

Run from repo root:
  streamlit run web/app.py --server.port 8501
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from qgdata import (
    SHOWCASE,
    load_g3,
    load_reliability,
    load_sprint_a,
    load_sprint_b,
)
from theme import theme_css

st.set_page_config(
    page_title="QuantumGPT — 会自我审计的量子智能体",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(theme_css(), unsafe_allow_html=True)


def img_data_uri(path: Path) -> str:
    """Embed PNG inline so we control the figcard layout instead of st.image."""
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


# =============================================================================
# HERO
# =============================================================================
st.markdown(
    """
<div class="qg-hero">
  <div class="qg-tag">Research preview · AAAI 2026 candidate</div>
  <h1>会<span class="accent">自我审计</span>的量子智能体<br/>把幻觉拦在交付之前</h1>
  <p class="lede">
    QuantumGPT 把 18 个量子工具放进一个 ReAct 执行器，再用一个独立的 Verifier
    审视整条 trace —— 任何"嘴上有、动作没有"的最终回答都会被即时拦截。
    在 1,615 条配对评测上，Verifier 在生产数据上 9 / 9 命中幻觉、零误报，
    同时在 best post-drift fidelity 上以严显著优势压过 Static-Pipeline 基线。
  </p>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HEADLINE STATS
# =============================================================================
g3 = load_g3()
rel = load_reliability()
df_a = load_sprint_a()
df_b = load_sprint_b()

st.markdown(
    f"""
<div class="qg-stat-grid">
  <div class="qg-stat">
    <div class="num">9 / 9</div>
    <div class="lbl">production hallucinations caught</div>
    <div class="sub">› 0 false positives</div>
  </div>
  <div class="qg-stat">
    <div class="num">+0.0197</div>
    <div class="lbl">paired Δ vs Static-Pipeline</div>
    <div class="sub">› p &lt; 0.001 · n = 225</div>
  </div>
  <div class="qg-stat">
    <div class="num">$11.46</div>
    <div class="lbl">total cost (450 orchestrator runs)</div>
    <div class="sub">› ≈ $0.025 / task</div>
  </div>
  <div class="qg-stat">
    <div class="num">{len(df_a) + len(df_b):,}</div>
    <div class="lbl">paired benchmark rows</div>
    <div class="sub">› 5 systems · 5 drift profiles · 9 circuits</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# Helper
# =============================================================================
def figure_block(fig_path: Path, eyebrow: str, title: str, summary_html: str,
                 caption_html: str, takeaway: str) -> None:
    uri = img_data_uri(fig_path)
    if not uri:
        st.warning(f"missing: {fig_path}")
        return
    st.markdown(
        f"""
<div class="qg-section">
  <div class="qg-eyebrow">{eyebrow}</div>
  <h2>{title}</h2>
  <p class="lede">{summary_html}</p>

  <div class="qg-figcard"><img src="{uri}" /></div>

  <div class="qg-figcaption">
    <h4>怎么读这张图</h4>
    {caption_html}
  </div>

  <div class="qg-takeaway">{takeaway}</div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# FIG 1 — system overview
# =============================================================================
figure_block(
    fig_path=SHOWCASE / "fig1_overview.png",
    eyebrow="01 · System architecture",
    title="一条闭环：Planner ⇄ Executor ⇄ Verifier",
    summary_html=(
        "智能体由三个角色串成：Planner 给出可执行计划，Executor 在 18 个量子工具里循环 "
        "ReAct，Verifier 单独读取整条 trace 与最终 claim，做一致性审计。所有调用都受 "
        "<code>task_id · max_tool_calls = 12 · max_turns = 6</code> 的预算守卫约束。"
    ),
    caption_html="""
<ul>
  <li>顶部主链路 <strong>Task spec → Planner → Executor → Verifier</strong> 是一次任务的正向流。</li>
  <li>右上 <strong>Shadow backend</strong>（FakeBrisbane + drift）是验证用的影子硬件，
      不向客户暴露；Verifier 在这里把 Executor 声称的指标重跑一次。</li>
  <li>底部 <strong>Trace + budget guard</strong> 是系统贡献中最被忽视的一块：
      它持久化每一步工具调用、用 task_id 做内容寻址；同时在每一轮检查
      <code>max_tool_calls / max_turns</code>，避免 ReAct 进入死循环。</li>
  <li>右下 <strong>Tool registry</strong>（health · transpile · run · ZNE）通过虚线提供
      工具 schema，仅供 Executor 查阅，不直接产出回答。</li>
  <li>红色脚注 <strong>9 / 9 hallucinations caught · 0 false positives</strong> 是这套
      架构在生产数据上的硬指标，和图 4 一一对应。</li>
</ul>
    """,
    takeaway=(
        "Verifier 是唯一同时读取 <strong>final answer</strong> 与 <strong>整条 trace</strong> "
        "的角色。这是把「敢说」和「敢交付」切开的关键边界 —— 也是论文里「系统贡献」那一节的全部内容。"
    ),
)


# =============================================================================
# FIG 2 — paired forest
# =============================================================================
figure_block(
    fig_path=SHOWCASE / "fig2_paired_forest.png",
    eyebrow="02 · Main result",
    title="五条对照下的 paired Δ：贵的更准，便宜的更稳",
    summary_html=(
        "每条横向 whisker 是 QuantumGPT-Full 相对于另一系统在 best post-drift fidelity 上的 "
        "<code>paired Δ</code>，配 5,000 次 bootstrap 95% CI 与 Wilcoxon 显著性符号。"
        "因为是配对实验，每对样本共享同一 (circuit, drift profile, seed)，方差在抵消。"
    ),
    caption_html="""
<ul>
  <li><strong>vs Static-Pipeline</strong>：Δ ≈ <code>+0.0197</code>，<strong>p &lt; 0.001</strong>。
      这是论文最强的卖点 —— 把 ReAct + Verifier 加进来比纯固定流水线高出近 2 个绝对百分点的保真度。</li>
  <li><strong>vs ChatLLM-NoTools</strong>：Δ 同向显著为正，说明"会说量子术语"的模型在没有工具时
      仍然写不出可执行的 transpile / mitigation。</li>
  <li><strong>vs No-Drift</strong>：Δ 仍然显著，意味着 Full 在动态 drift 下额外受益于 verifier 的拦截。</li>
  <li><strong>vs Oracle-Adaptive</strong>：Δ 接近 0 或微弱负向，符合 Oracle 是上限基线的假设 ——
      我们用真实工具 + 黑盒大模型，逼近了对它而言开了"上帝视角"的 oracle。</li>
  <li><strong>vs Orchestrator (Sprint B 子集)</strong>：Δ 几乎为零（n = 225），
      表明在大多数样本上 Full 与 Orchestrator 等价 —— Orchestrator 的增量价值集中在难样本上（见图 5）。</li>
</ul>
    """,
    takeaway=(
        "Full vs Static 的 <code>+0.0197</code> 是<strong>必须保留在 abstract 里的硬数字</strong>；"
        "Orchestrator vs Full 的 ns 是必须诚实写进 limitations 的事实，但反过来恰好支持图 5 那条 hard-regime 论点。"
    ),
)


# =============================================================================
# FIG 3 — Pareto
# =============================================================================
figure_block(
    fig_path=SHOWCASE / "fig3_pareto.png",
    eyebrow="03 · Cost / fidelity frontier",
    title="花得起：在 145 个 oracle-feasible 任务上的成本-保真度前沿",
    summary_html=(
        "横轴是单任务平均美元成本（symlog），纵轴是 best post-drift fidelity。"
        "标准基准设定下取 145 个 oracle-feasible 任务的均值；点的形状区分系统类型，"
        "玫红阴影标出「被 ChatLLM 严格统治」的 Static-Pipeline 区。"
    ),
    caption_html="""
<ul>
  <li><strong>左下角的 Static-Pipeline</strong> 看似「零成本」，但保真度比所有 LLM 路线都低 —— 它处于被严格统治的位置，
      任何下游用户都没有理由再选它。</li>
  <li><strong>ChatLLM-NoTools</strong> 在 LLM 系统里成本最低、保真度最高，是一个非常便宜的 baseline；
      但它没有 trace，也没有可审计性，因此天花板低（图 5 右侧会再次看到）。</li>
  <li><strong>QuantumGPT-Full / No-Drift</strong> 占据保真度上沿，成本比 ChatLLM 高约一个数量级，
      换来 <code>+0.0197</code> 的 paired Δ 和「可验证」的属性。</li>
  <li><strong>Oracle-Adaptive</strong> 不在帕累托线上，因为它要求事实上不存在的 drift 真值，
      这里只作为「上限锚点」给出。</li>
</ul>
    """,
    takeaway=(
        "<strong>每多花一美分换 0.02 个 fidelity</strong>，且换到的是<strong>可被 Verifier 兜底的可信交付</strong>。"
        "对量子云服务而言，这不是优化指标的问题，而是要不要被监管的问题。"
    ),
)


# =============================================================================
# FIG 4 — verifier reliability
# =============================================================================
figure_block(
    fig_path=SHOWCASE / "fig4_reliability.png",
    eyebrow="04 · Verifier reliability",
    title="Verifier 不是花架子：注入测试 + 真实生产 双线命中",
    summary_html=(
        "左面板是 10 例人工注入的对抗用例（凭空捏造 fidelity / 无 trace 支持的回答 / 工具结果与 claim 不一致）；"
        "右面板是 450 次 Sprint B Orchestrator 跑出的 9 例真实幻觉。两边都做到了 100% 召回与 0 误报。"
    ),
    caption_html="""
<ul>
  <li><strong>左面板（注入电池）</strong>：10 个红色 H 表示注入的幻觉用例，全部被 Verifier 拒绝；
      绿色 ✓ 表示注入的诚实用例（有完整 trace 支持），全部通过。这套电池覆盖了
      <strong>无 trace</strong>、<strong>trace 与 claim 数值不一致</strong>、<strong>tool error swallowed</strong> 三类典型失败模式。</li>
  <li><strong>右面板（生产命中）</strong>：450 次跑里 Verifier 共拦截 9 个 final_answer，
      经人工复核全部为真幻觉（agent 声称做了 transpile / 跑了 ZNE，但 trace 里没有相应工具调用）。
      其余 441 条最终回答全部一致通过。</li>
  <li>下方 <strong>FP rate = 0%</strong> / <strong>recall = 100%</strong> 是这两批数据合并的硬指标。</li>
</ul>
    """,
    takeaway=(
        "Verifier 通过的是 <strong>「零 FP + 100% recall」</strong> 的双门槛 —— 这是 G3 reliability gate "
        "在论文里被允许写进 main result 的前提条件。"
    ),
)


# =============================================================================
# FIG 5 — regimes
# =============================================================================
figure_block(
    fig_path=SHOWCASE / "fig5_regimes.png",
    eyebrow="05 · When orchestration matters",
    title="不藏短板：什么时候真值得用 Orchestrator？",
    summary_html=(
        "我们诚实地把样本切成两个 regime。左侧 easy regime 是 mild_gradual 漂移下的 top-2 电路（n = 60 paired），"
        "右侧 harder regime 是 multi_shock + severe_sudden 漂移（n = 30 paired）。"
        "两个面板共同的 y 轴是 NoVerifier vs Orchestrator，散点为单条任务结果，黑色短竖线为均值。"
    ),
    caption_html="""
<ul>
  <li><strong>左面板（easy regime）</strong>：天花板已经被打到 ~0.998，paired Δ ≈
      <code>−0.0004</code>，<strong>Orchestrator 既没赢也没输</strong>。论文必须把这条说在前面 ——
      在容易的样本上 ReAct 加上 verifier 不带额外收益，钱花在了刀背上。</li>
  <li><strong>右面板（harder regime）</strong>：保真度天花板塌到 0.85 附近。Orchestrator 的散点
      整体右移，均值对 NoVerifier 拉开 +0.0089 的差距，越过 <code>target = 0.85</code> 阈值的样本占
      <strong>26 / 30 = 87%</strong>，oracle gap 仅 <code>+0.0089</code>。</li>
  <li>NoVerifier 在 harder regime 出现一个明显落到 0.68 的离群点 —— 那一例就是 Verifier
      原本应当拦下的幻觉性回答，对照图 4 右面板的 9 / 9 完全吻合。</li>
</ul>
    """,
    takeaway=(
        "Orchestrator 的真实价值在<strong>难样本</strong>而不是平均样本。"
        "这恰好是把「防止幻觉」的卖点和「提升保真度」的卖点钉死在同一张图上的最有力证据。"
    ),
)


# =============================================================================
# NAV
# =============================================================================
st.markdown(
    """
<div class="qg-section">
  <div class="qg-eyebrow">06 · Try it yourself</div>
  <h2>把数据自己拖一遍</h2>
  <p class="lede">两个交互页面都基于真实落地的 1,615 条 paired 数据与 30 条 ReAct trace。</p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        """
<div class="qg-nav-card">
  <div class="icon">[ pages › 02 ]</div>
  <h3>📈 Benchmark explorer</h3>
  <p>挑两个系统作为 A / B，随手切换 drift profile、circuit。
     per-circuit Δ forest 与全样本分布会现场重新计算 —— 不是预渲染图。</p>
  <p style="color: var(--slate); font-size: 12px;">› 左侧 sidebar → Benchmark explorer</p>
</div>
""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
<div class="qg-nav-card">
  <div class="icon">[ pages › 01 ]</div>
  <h3>🔍 Trace replay</h3>
  <p>选一条真实 (drift, circuit, seed) trace，按步骤展开 ReAct 调用与 Verifier 判定。
     每一格都是落盘的 JSON，你能验到每一个工具的输入输出。</p>
  <p style="color: var(--slate); font-size: 12px;">› 左侧 sidebar → Trace replay</p>
</div>
""",
        unsafe_allow_html=True,
    )

# =============================================================================
# FOOT
# =============================================================================
st.markdown(
    f"""
<div class="qg-foot">
  data: sprint_a {len(df_a):,} rows + sprint_b {len(df_b):,} rows = {len(df_a) + len(df_b):,} paired ·
  figures regenerable from <code>scripts/make_showcase_figures.py</code> ·
  app v0.2 · QuantumGPT research preview
</div>
""",
    unsafe_allow_html=True,
)
