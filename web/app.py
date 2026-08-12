"""QuantumGPT Mission Control — single-page monitor screen.

Embeds web/static/monitor.html with web/static/missions.json injected inline.
Run: streamlit run web/app.py --server.port 8501
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "static" / "monitor.html"
MISSIONS_PATH = ROOT / "static" / "missions.json"

st.set_page_config(
    page_title="QuantumGPT — Mission Control",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Strip Streamlit chrome so the monitor can fill the viewport edge to edge.
st.markdown(
    """
    <style>
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
    [data-testid="stStatusWidget"], #MainMenu, footer { display: none !important; }
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    [data-testid="stAppViewContainer"] {
        background: #050507 !important;
    }
    [data-testid="stSidebarContent"] {
        background: #0c0c0d !important;
        border-right: 1px solid #2c2a29 !important;
    }
    [data-testid="stSidebarContent"] * { color: #f2f2f2 !important; }
    [data-testid="stSidebarNav"] a {
        color: #b8b3b0 !important;
        font-family: 'JetBrains Mono', SFMono-Regular, monospace !important;
        font-size: 12px !important;
    }
    [data-testid="stSidebarNav"] a:hover { color: #00d992 !important; }
    iframe[title="streamlit_components.v1.html.html"] { border: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Build the inlined HTML: replace the fetch() call with an embedded constant.
html = HTML_PATH.read_text(encoding="utf-8")
missions = json.loads(MISSIONS_PATH.read_text(encoding="utf-8"))
inject = "window.__MISSIONS__ = " + json.dumps(missions, ensure_ascii=False) + ";"

# Patch the boot() to use the inline data instead of fetch.
html = html.replace(
    'const data = await fetch("./missions.json", { cache: "no-cache" }).then(r => r.json());',
    'const data = window.__MISSIONS__;',
)
html = html.replace(
    "<script>",
    f"<script>\n{inject}\n",
    1,
)

components.html(html, height=900, scrolling=False)
