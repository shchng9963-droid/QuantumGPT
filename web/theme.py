"""Shared CSS theme for the QuantumGPT showcase — VoltAgent-inspired dark.

Inject with:  st.markdown(theme_css(), unsafe_allow_html=True)
"""

THEME_CSS = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">

<style>
:root {
    --abyss: #050507;
    --carbon: #101010;
    --carbon-2: #161617;
    --charcoal: #3d3a39;
    --charcoal-soft: #2a2826;
    --signal: #00d992;
    --mint: #2fd6a1;
    --emerald-fade: rgba(0, 217, 146, 0.08);
    --emerald-line: rgba(0, 217, 146, 0.35);
    --snow: #f2f2f2;
    --parchment: #b8b3b0;
    --slate: #8b949e;
    --rose: #fb565b;
    --amber: #ffba00;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: var(--abyss) !important;
    color: var(--snow) !important;
    font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif !important;
    font-feature-settings: "calt", "rlig";
}

[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] {
    background: var(--carbon) !important;
    border-right: 1px solid var(--charcoal) !important;
}

.block-container {
    padding-top: 1.4rem !important;
    padding-bottom: 4rem !important;
    max-width: 1280px !important;
}

h1, h2, h3, h4 {
    color: var(--snow) !important;
    font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif !important;
    letter-spacing: -0.02em !important;
    font-weight: 700 !important;
}
h1 { font-size: 60px !important; line-height: 1.0 !important; letter-spacing: -0.025em !important; }
h2 { font-size: 36px !important; line-height: 1.11 !important; margin-top: 2.4rem !important; }
h3 { font-size: 22px !important; line-height: 1.3 !important; }

p, li, span, label {
    color: var(--parchment) !important;
    font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif !important;
    line-height: 1.65 !important;
}

a { color: var(--signal) !important; text-decoration: none !important; }
a:hover { color: var(--mint) !important; text-decoration: underline !important; }

code, pre, kbd, samp {
    font-family: 'JetBrains Mono', SFMono-Regular, Menlo, monospace !important;
    background: var(--carbon-2) !important;
    color: var(--mint) !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 0.92em !important;
}
pre { padding: 14px 16px !important; border: 1px solid var(--charcoal) !important; }

hr {
    border: 0 !important;
    border-top: 1px solid var(--charcoal) !important;
    margin: 3rem 0 !important;
}

/* ---- streamlit metric ---------------------------------------------------- */
[data-testid="stMetric"] {
    background: var(--carbon) !important;
    border: 1px solid var(--charcoal) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricValue"] {
    color: var(--snow) !important;
    font-size: 30px !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricLabel"] {
    color: var(--slate) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricDelta"] svg { display: none; }

/* ---- buttons / selects --------------------------------------------------- */
.stButton button {
    background: var(--carbon) !important;
    color: var(--mint) !important;
    border: 1px solid var(--charcoal) !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    transition: all 0.15s ease;
}
.stButton button:hover {
    border-color: var(--signal) !important;
    box-shadow: 0 0 12px rgba(0, 217, 146, 0.25);
}

[data-baseweb="select"] > div {
    background: var(--carbon) !important;
    border-color: var(--charcoal) !important;
    color: var(--snow) !important;
}

[data-testid="stExpander"] {
    background: var(--carbon) !important;
    border: 1px solid var(--charcoal) !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary { color: var(--snow) !important; font-weight: 500; }

/* ---- dataframe ----------------------------------------------------------- */
[data-testid="stDataFrame"] {
    background: var(--carbon) !important;
    border: 1px solid var(--charcoal) !important;
    border-radius: 8px !important;
}

/* ---- our custom classes -------------------------------------------------- */
.qg-eyebrow {
    color: var(--signal);
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.qg-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border: 1px solid var(--emerald-line);
    background: var(--emerald-fade);
    color: var(--mint);
    border-radius: 9999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.qg-tag::before {
    content: "";
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--signal);
    box-shadow: 0 0 8px var(--signal);
    animation: pulse 2.6s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 0.4; transform: scale(0.85); }
    50% { opacity: 1; transform: scale(1.0); }
}

.qg-hero {
    padding: 64px 0 32px 0;
    border-bottom: 1px solid var(--charcoal);
    margin-bottom: 32px;
}
.qg-hero h1 {
    font-size: 60px !important;
    line-height: 1.02 !important;
    margin: 18px 0 16px 0 !important;
    color: var(--snow) !important;
}
.qg-hero h1 .accent {
    color: var(--signal);
    font-style: normal;
}
.qg-hero p.lede {
    font-size: 18px !important;
    line-height: 1.6 !important;
    color: var(--parchment) !important;
    max-width: 780px;
}

.qg-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 36px 0 12px 0;
}
.qg-stat {
    background: var(--carbon);
    border: 1px solid var(--charcoal);
    border-left: 2px solid var(--signal);
    border-radius: 8px;
    padding: 20px 22px;
}
.qg-stat .num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--snow);
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.qg-stat .lbl {
    color: var(--slate);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 10.5px;
    font-weight: 600;
    margin-top: 6px;
}
.qg-stat .sub {
    color: var(--mint);
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin-top: 12px;
    font-weight: 500;
}

.qg-section {
    margin: 64px 0 32px 0;
}
.qg-section h2 {
    margin: 8px 0 12px 0 !important;
}
.qg-section .lede {
    color: var(--parchment) !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    max-width: 820px;
}

.qg-figcard {
    background: var(--carbon);
    border: 1px solid var(--charcoal);
    border-radius: 10px;
    padding: 18px;
    margin: 28px 0 8px 0;
    box-shadow: rgba(0, 0, 0, 0.35) 0px 16px 40px,
                rgba(148, 163, 184, 0.05) 0px 0px 0px 1px inset;
}
.qg-figcard img {
    border-radius: 6px;
    display: block;
    width: 100%;
}

.qg-figcaption {
    background: var(--carbon-2);
    border: 1px solid var(--charcoal);
    border-radius: 8px;
    padding: 18px 22px;
    margin: 16px 0;
}
.qg-figcaption h4 {
    color: var(--signal) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    margin: 0 0 12px 0 !important;
}
.qg-figcaption p, .qg-figcaption li {
    color: var(--parchment) !important;
    font-size: 14.5px !important;
    line-height: 1.75 !important;
    margin: 6px 0 !important;
}
.qg-figcaption strong { color: var(--snow) !important; font-weight: 600 !important; }
.qg-figcaption code {
    background: var(--abyss) !important;
    color: var(--mint) !important;
    border: 1px solid var(--charcoal-soft);
    padding: 1px 6px !important;
    font-size: 13px !important;
}
.qg-figcaption ul { padding-left: 18px; margin: 6px 0 !important; }

.qg-takeaway {
    border: 1px solid var(--emerald-line);
    background: var(--emerald-fade);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 12px;
    color: var(--snow) !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
}
.qg-takeaway::before {
    content: "结论 ▸  ";
    color: var(--signal);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.qg-row { display: grid; grid-template-columns: 1.05fr 1fr; gap: 24px; align-items: start; }
.qg-row.reverse { grid-template-columns: 1fr 1.05fr; }
@media (max-width: 900px) {
    .qg-row, .qg-row.reverse { grid-template-columns: 1fr; }
    .qg-stat-grid { grid-template-columns: repeat(2, 1fr); }
    .qg-hero h1 { font-size: 40px !important; }
}

.qg-nav-card {
    background: var(--carbon);
    border: 1px solid var(--charcoal);
    border-radius: 10px;
    padding: 22px 24px;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
.qg-nav-card:hover {
    border-color: var(--signal);
    box-shadow: 0 0 16px rgba(0, 217, 146, 0.2);
}
.qg-nav-card .icon {
    font-family: 'JetBrains Mono', monospace;
    color: var(--signal);
    font-size: 12px;
    letter-spacing: 0.12em;
}
.qg-nav-card h3 {
    margin: 6px 0 8px 0 !important;
    font-size: 20px !important;
}
.qg-nav-card p {
    color: var(--parchment) !important;
    font-size: 14px !important;
}

.qg-foot {
    margin-top: 4rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--charcoal);
    color: var(--slate);
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""


def theme_css() -> str:
    return THEME_CSS
