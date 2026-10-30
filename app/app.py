from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
sys.path.insert(0, str(APP_DIR))
from logic import (  # noqa: E402
    COUNT_COLUMNS, DURATION_COLUMNS, FEATURE_COLUMNS, InputValidationError,
    RATE_COLUMNS, TECHNICAL_COLUMNS, ensure_artifact_files, score_sessions,
    validate_frame,
)

MODEL_PATH = ROOT / "models" / "xgb_no_pagevalues.joblib"
META_PATH = ROOT / "models" / "app_meta.json"
RESULTS_PATH = ROOT / "reports" / "supplemental_results.json"
DATA_PATH = ROOT / "data" / "processed" / "shoppers_clean.csv"
VERSION = "2.2"
FIREBASE_API_KEY = "AIzaSyBWlZmKjZbFcVMgfYQaY8G2gE38PgyhnxQ"
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:{}?key={}"
SESSION_SECRET = "visitor_intel_safe_token_key_v2"
_SESSION_SERIALIZER = URLSafeTimedSerializer(SESSION_SECRET, salt="visitor_intel_salt")
SESSION_MAX_AGE = 86400 * 7  # 7 days
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("visitor_intent_app")


def create_session_token(user: dict) -> str:
    """Generate a compact, cryptographically signed session token for refresh persistence."""
    clean_user = {
        "email": user.get("email", ""),
        "name": user.get("name", ""),
        "uid": user.get("uid", ""),
        "provider": user.get("provider", "password"),
    }
    return _SESSION_SERIALIZER.dumps(clean_user)


def decode_session_token(token: str) -> dict | None:
    """Verify and decode a session token, returning user dict or None."""
    if not token or not isinstance(token, str):
        return None
    try:
        data = _SESSION_SERIALIZER.loads(token.strip(), max_age=SESSION_MAX_AGE)
        if isinstance(data, dict) and "email" in data:
            return data
    except (BadSignature, SignatureExpired, Exception):
        pass
    return None

# ─── Streamlit Component for Firebase Auth ───
FIREBASE_AUTH_DIR = APP_DIR / "firebase_auth"
_firebase_auth_component = components.declare_component(
    "firebase_auth",
    path=str(FIREBASE_AUTH_DIR),
)

# ─── Page Configuration ───
st.set_page_config(
    page_title="Visitor Intelligence · AI Purchase Intent Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Design System & CSS Tokens ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-main: #090d16;
    --bg-surface: #0f1424;
    --bg-card: rgba(20, 26, 45, 0.85);
    --bg-card-hover: rgba(26, 34, 58, 0.95);
    --border-card: rgba(99, 102, 241, 0.16);
    --border-card-hover: rgba(99, 102, 241, 0.38);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-pill: 9999px;
    --shadow-card: 0 12px 32px -4px rgba(0, 0, 0, 0.45), 0 0 16px rgba(99, 102, 241, 0.06);
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, [data-testid="stToolbar"],
header[data-testid="stHeader"], .stDeployButton,
[data-testid="manage-app-button"], [data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}

.block-container {
    max-width: 1440px !important;
    padding: 1rem 2rem 3.5rem !important;
}

/* Public Header & Nav */
.pub-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.85rem 1.6rem;
    background: linear-gradient(180deg, rgba(20, 26, 45, 0.96) 0%, rgba(15, 20, 36, 0.92) 100%);
    border: 1px solid rgba(99, 102, 241, 0.22);
    border-radius: var(--radius-lg);
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.45);
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
}
.brand-group {
    display: flex;
    align-items: center;
    gap: 12px;
}
.brand-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
    font-weight: 900;
    color: #fff;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
}
.brand-title {
    font-size: 1.12rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.brand-sub {
    font-size: 0.68rem;
    font-weight: 700;
    color: #818cf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Hero Section */
.hero-wrapper {
    background: linear-gradient(135deg, #121833 0%, #1a1e48 50%, #152244 100%);
    border: 1px solid rgba(99, 102, 241, 0.24);
    border-radius: var(--radius-lg);
    padding: 3.2rem 3rem;
    margin-bottom: 2.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.hero-wrapper::after {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 480px;
    height: 480px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.22) 0%, transparent 70%);
    pointer-events: none;
}
.pill-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(99, 102, 241, 0.15);
    color: #a5b4fc;
    padding: 0.35rem 0.95rem;
    border-radius: var(--radius-pill);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.1rem;
    border: 1px solid rgba(99, 102, 241, 0.3);
}
.hero-headline {
    color: #ffffff !important;
    font-size: 2.7rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.04em;
    line-height: 1.2;
    margin: 0 0 1rem;
    max-width: 820px;
}
.hero-description {
    color: #cbd5e1;
    font-size: 1.12rem;
    line-height: 1.65;
    max-width: 720px;
    margin: 0 0 1.6rem;
}

/* Interactive Preview Box */
.preview-card {
    background: rgba(15, 20, 36, 0.9);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: var(--radius-md);
    padding: 1.6rem 1.8rem;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(16px);
}
.preview-status {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1);
}

/* Section Header */
.sec-hd {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin: 2.4rem 0 1.1rem;
}
.sec-hd .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.6);
}
.sec-hd h3 {
    margin: 0 !important;
    font-size: 1.28rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    letter-spacing: -0.02em;
}

/* Cards */
.vi-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1.45rem 1.55rem;
    box-shadow: var(--shadow-card);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}
.vi-card:hover {
    border-color: var(--border-card-hover);
    transform: translateY(-3px);
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5), 0 0 22px rgba(99, 102, 241, 0.12);
}
.vi-card .ic {
    font-size: 1.75rem;
    margin-bottom: 0.75rem;
    display: block;
}
.vi-card h4 {
    color: #f8fafc !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.45rem !important;
}
.vi-card p {
    color: #94a3b8 !important;
    font-size: 0.88rem !important;
    line-height: 1.55 !important;
    margin: 0 !important;
}

/* Stepper Workflow */
.step-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.5rem;
    position: relative;
    box-shadow: var(--shadow-card);
    transition: all 0.25s ease;
    height: 100%;
}
.step-card:hover {
    border-color: var(--border-card-hover);
    transform: translateY(-2px);
}
.step-num {
    position: absolute;
    top: -11px;
    left: 18px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #ffffff;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 0.18rem 0.65rem;
    border-radius: var(--radius-pill);
    box-shadow: 0 2px 10px rgba(99, 102, 241, 0.4);
}
.step-card h4 {
    color: #f8fafc !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin: 0.4rem 0 0.4rem !important;
}
.step-card p {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}

/* Prediction Result Cards */
.int-res {
    border-radius: var(--radius-lg);
    padding: 2.2rem 2.4rem;
    border: 1px solid var(--border-card);
    position: relative;
    overflow: hidden;
    margin: 1.2rem 0 1.5rem;
    box-shadow: var(--shadow-card);
}
.int-res::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
}
.int-res.high {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-color: rgba(16, 185, 129, 0.3);
}
.int-res.high::before {
    background: linear-gradient(90deg, #10b981, #34d399);
}
.int-res.medium {
    background: linear-gradient(145deg, rgba(245, 158, 11, 0.08) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-color: rgba(245, 158, 11, 0.3);
}
.int-res.medium::before {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
}
.int-res.low {
    background: linear-gradient(145deg, rgba(100, 116, 139, 0.1) 0%, rgba(15, 23, 42, 0.95) 100%);
    border-color: rgba(100, 116, 139, 0.3);
}
.int-res.low::before {
    background: linear-gradient(90deg, #64748b, #94a3b8);
}
.int-tag {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-pill);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.int-res.high .int-tag { background: rgba(16, 185, 129, 0.16); color: #34d399; }
.int-res.medium .int-tag { background: rgba(245, 158, 11, 0.16); color: #fbbf24; }
.int-res.low .int-tag { background: rgba(100, 116, 139, 0.16); color: #cbd5e1; }
.int-band {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem;
    line-height: 1.1;
}
.int-res.high .int-band { color: #10b981; }
.int-res.medium .int-band { color: #f59e0b; }
.int-res.low .int-band { color: #94a3b8; }
.int-desc {
    color: #cbd5e1;
    font-size: 0.96rem;
    margin: 0.35rem 0 0.65rem;
    line-height: 1.5;
}

/* Metric Boxes */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.1rem 1.25rem !important;
    box-shadow: var(--shadow-card) !important;
    transition: all 0.25s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--border-card-hover) !important;
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] div {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

/* Chart Wrappers */
.ch-wrap {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.35rem 0.6rem;
    box-shadow: var(--shadow-card);
    margin-bottom: 0.75rem;
}
.ch-wrap h4 {
    color: #f8fafc !important;
    font-size: 0.96rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.15rem !important;
}
.ch-wrap .ch-sub {
    color: #64748b;
    font-size: 0.78rem;
    margin: 0 0 0.5rem;
}

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stFormSubmitButton > button, .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #7c3aed 100%) !important;
    border: none !important;
    color: #ffffff !important;
    min-height: 3.1rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
}
.stFormSubmitButton > button:hover, .stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}
.stButton > button:not([kind="primary"]) {
    background: rgba(30, 41, 59, 0.65) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    color: #e2e8f0 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    background: rgba(30, 41, 59, 0.9) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Breadcrumb Bar */
.crumb-bar {
    display: flex;
    align-items: center;
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: var(--radius-md);
    padding: 0.65rem 1.2rem;
    margin-bottom: 1.1rem;
}
.crumb-left {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.86rem;
    color: #94a3b8;
}
.crumb-active {
    color: #a5b4fc;
    font-weight: 700;
}

/* Footer */
.app-ft {
    text-align: center;
    color: #64748b;
    font-size: 0.78rem;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid rgba(99, 102, 241, 0.12);
    margin-top: 3.5rem;
    line-height: 1.8;
}
.app-ft a {
    color: #818cf8;
    text-decoration: none;
    font-weight: 500;
}

/* Sample Scenario Preset Cards */
.smp-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1rem 1.1rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--shadow-card);
    min-height: 72px;
}
.smp-card h4 {
    color: #f8fafc !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.3rem !important;
}
.smp-card p {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    line-height: 1.45 !important;
    margin: 0 !important;
}

/* Prediction Result Score Line */
.int-score {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
}
.int-score strong {
    color: #e2e8f0;
    font-weight: 700;
}

/* EDA Chart Labels */
.eda-label {
    font-size: 0.72rem;
    color: #64748b;
    font-style: italic;
    margin-top: 0.2rem;
    margin-bottom: 0.8rem;
    padding-left: 0.2rem;
}

/* Segment Profile Cards */
.seg-card {
    background: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.3rem;
    box-shadow: var(--shadow-card);
    height: 100%;
}
.seg-card h4 {
    color: #f8fafc !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.5rem !important;
}
.seg-card .seg-rate {
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.seg-card p {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Navigation Helper Callbacks (Safe State Machine) ───
PUBLIC_PAGES = {"Home", "Methodology", "About", "Login"}
PROTECTED_PAGES = {"Overview", "Predict Intent", "Insights"}
PAGE_SLUG_MAP = {
    "home": "Home",
    "login": "Login",
    "dashboard": "Overview",
    "overview": "Overview",
    "predict": "Predict Intent",
    "insights": "Insights",
    "methodology": "Methodology",
    "about": "About",
}
SLUG_INV_MAP = {
    "Home": "home",
    "Login": "login",
    "Overview": "dashboard",
    "Predict Intent": "predict",
    "Insights": "insights",
    "Methodology": "methodology",
    "About": "about",
}


def on_top_nav_change() -> None:
    """Callback executed when user selects an item on the top segmented control."""
    selected = st.session_state.get("top_main_segmented_control")
    if selected:
        if "user" not in st.session_state and selected in PROTECTED_PAGES:
            st.session_state["redirect_target"] = selected
            dest = "Login"
            st.session_state["top_main_segmented_control"] = "Login"
        else:
            dest = selected
        st.session_state["navigation"] = dest
        st.query_params["page"] = SLUG_INV_MAP.get(dest, "home")
        if "user" in st.session_state:
            st.query_params["session"] = create_session_token(st.session_state["user"])


def nav_to(page_name: str) -> None:
    """Updates navigation, handles authentication guards, and syncs query params."""
    if "user" not in st.session_state and page_name in PROTECTED_PAGES:
        st.session_state["redirect_target"] = page_name
        dest = "Login"
    else:
        dest = page_name
    st.session_state["navigation"] = dest
    st.session_state["top_main_segmented_control"] = dest
    st.query_params["page"] = SLUG_INV_MAP.get(dest, "home")
    if "user" in st.session_state:
        st.query_params["session"] = create_session_token(st.session_state["user"])


def do_logout() -> None:
    """Purges session, deletes authentication tokens, and resets component nonces."""
    if "user" in st.session_state:
        del st.session_state["user"]
    if "session" in st.query_params:
        del st.query_params["session"]
    st.session_state["logged_out"] = True
    st.session_state["auth_nonce"] = st.session_state.get("auth_nonce", 0) + 1
    for k in list(st.session_state.keys()):
        if (
            k.startswith("field_")
            or k.startswith("firebase_")
            or k.startswith("fb_")
            or k in ("last_result", "last_input", "redirect_target", "top_main_segmented_control")
        ):
            del st.session_state[k]
    st.session_state["navigation"] = "Home"
    st.session_state["top_main_segmented_control"] = "Home"
    st.query_params["page"] = "home"


# ─── Asset Loading (PRESERVED) ───
@st.cache_resource(show_spinner="Initializing fitted pipeline & intelligence models…")
def load_assets() -> tuple[Any, dict, dict, pd.DataFrame, dict]:
    """Load and check artifacts only; never trains or mutates a model."""
    ensure_artifact_files([MODEL_PATH, META_PATH, RESULTS_PATH, DATA_PATH])
    metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)
    if metadata.get("model_file") != MODEL_PATH.name:
        raise ValueError("Metadata points to a different model file.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "named_steps"):
        raise TypeError("Model is not the expected fitted sklearn Pipeline.")
    transformer = model.named_steps.get("prep")
    if transformer is None:
        raise ValueError("Model is missing preprocessing pipeline.")
    # Verify PageValues is not used in the deployed model
    selected_features = set()
    for _name, _trans, cols in transformer.transformers:
        if isinstance(cols, (list, tuple, np.ndarray)):
            selected_features.update(cols)
    if "PageValues" in selected_features:
        raise ValueError("Deployment artifact unexpectedly uses PageValues.")
    raw = pd.read_csv(DATA_PATH)
    numeric = COUNT_COLUMNS + DURATION_COLUMNS + ["BounceRates", "ExitRates", "SpecialDay"]
    limits = {col: (float(raw[col].min()), float(raw[col].max())) for col in numeric}
    return model, metadata, results, raw, limits


# ─── Robust Layout & Chart Helpers ───
def _get_layout(h: int = 340, yt: str = "", xt: str = "", horiz: bool = False, showlegend: bool = False) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=12),
        margin=dict(l=45, r=25, t=35, b=45),
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)", title=xt),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)", title=yt),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, font_family="Inter, sans-serif", bordercolor="#334155"),
        bargap=0.28,
        height=h,
        showlegend=showlegend,
    )
    if horiz:
        layout["yaxis"]["autorange"] = "reversed"
        layout["yaxis"]["gridcolor"] = "rgba(0,0,0,0)"
    return layout


def _bar(labels, values, color="#6366f1", yt="", h=340, horiz=False, tfmt=".3f"):
    if horiz:
        fig = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=color, cornerradius=8),
            text=[f"{v:{tfmt}}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
        fig.update_layout(**_get_layout(h=h, xt=yt, horiz=True, showlegend=False))
    else:
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            marker=dict(color=color, cornerradius=8),
            text=[f"{v:{tfmt}}" for v in values],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
        fig.update_layout(**_get_layout(h=h, yt=yt, horiz=False, showlegend=False))
    return fig


def _grouped(labels, groups, colors, yt="", h=360):
    fig = go.Figure()
    for i, (name, vals) in enumerate(groups.items()):
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker=dict(color=colors[i % len(colors)], cornerradius=8),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
    layout = _get_layout(h=h, yt=yt, showlegend=True)
    layout["barmode"] = "group"
    layout["legend"] = dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="center", x=0.5, font=dict(size=12),
    )
    fig.update_layout(**layout)
    return fig


def _donut(labels, values, colors, h=280):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.65,
        marker=dict(colors=colors, line=dict(color="#090d16", width=2.5)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#f1f5f9"),
        hovertemplate="<b>%{label}</b><br>Sessions: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=h,
        showlegend=False,
    )
    return fig


def _monthly_spline_chart(months, rates, h=320):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=rates,
        mode="lines+markers+text",
        line=dict(color="#6366f1", width=3.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.18)",
        marker=dict(size=8, color="#8b5cf6", line=dict(color="#ffffff", width=1.5)),
        text=[f"{r:.1%}" for r in rates],
        textposition="top center",
        textfont=dict(size=11, color="#cbd5e1"),
        hovertemplate="<b>Month: %{x}</b><br>Purchase Rate: %{y:.1%}<extra></extra>",
    ))
    layout = _get_layout(h=h, yt="Conversion Rate", showlegend=False)
    layout["yaxis"]["tickformat"] = ".0%"
    fig.update_layout(**layout)
    return fig


def _radar_dna_chart(inp: dict, limits: dict) -> go.Figure:
    def norm(val, key):
        lo, hi = limits[key]
        return min(max((float(val) - lo) / (hi - lo + 1e-9), 0.0), 1.0)

    p_dur = norm(inp.get("ProductRelated_Duration", 0), "ProductRelated_Duration")
    p_cnt = norm(inp.get("ProductRelated", 0), "ProductRelated")
    exit_score = max(0.0, 1.0 - norm(inp.get("ExitRates", 0), "ExitRates") * 3)
    bounce_score = max(0.0, 1.0 - norm(inp.get("BounceRates", 0), "BounceRates") * 3)
    admin_score = norm(inp.get("Administrative_Duration", 0), "Administrative_Duration")

    categories = ["Product Dwell Time", "Product Pages", "Exit Resistance", "Bounce Resistance", "Admin Dwell"]
    session_vals = [p_dur, p_cnt, exit_score, bounce_score, admin_score]
    buyer_benchmark = [0.45, 0.40, 0.85, 0.88, 0.25]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=session_vals, theta=categories, fill="toself",
        fillcolor="rgba(99, 102, 241, 0.35)", line=dict(color="#6366f1", width=2.5),
        name="Current Session",
    ))
    fig.add_trace(go.Scatterpolar(
        r=buyer_benchmark, theta=categories, fill="toself",
        fillcolor="rgba(16, 185, 129, 0.15)", line=dict(color="#10b981", width=2, dash="dot"),
        name="Avg Buyer Benchmark",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(148, 163, 184, 0.12)"),
            angularaxis=dict(gridcolor="rgba(148, 163, 184, 0.12)", linecolor="rgba(148, 163, 184, 0.12)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    )
    return fig


def band_info(b: str) -> tuple[str, str, str, str]:
    mapping = {
        "High": (
            "high",
            "Strong browsing engagement signals associated with high conversion probability.",
            "Trigger high-value personalized offers, priority live-chat assistance, or VIP incentives.",
            "High Priority Intervention",
        ),
        "Medium": (
            "medium",
            "Moderate interest demonstrated. Visitor is actively evaluating products.",
            "Show social proof, customer testimonials, or gentle exit-intent reassurance.",
            "Targeted Reassurance",
        ),
        "Low": (
            "low",
            "Minimal intent signals. Casual exploration or high probability of immediate exit.",
            "Maintain passive navigation experience. Avoid wasteful aggressive discounts.",
            "Passive Monitoring",
        ),
    }
    return mapping.get(b, mapping["Low"])


def hero(title: str, sub: str, tag: str = "VISITOR INTELLIGENCE") -> None:
    st.markdown(
        f'<div class="hero-wrapper"><div class="pill-badge">⚡ {tag}</div>'
        f'<h1 class="hero-headline">{title}</h1><p class="hero-description">{sub}</p></div>',
        unsafe_allow_html=True,
    )


def sec(title: str) -> None:
    st.markdown(
        f'<div class="sec-hd"><div class="dot"></div><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def vcard(icon: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="vi-card"><span class="ic">{icon}</span>'
        f'<h4>{title}</h4><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )


def pstep(n: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="step-card"><div class="step-num">STAGE {n}</div>'
        f'<h4>{title}</h4><p>{desc}</p></div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════
# 🌐 PUBLIC HOMEPAGE (Root Landing Page)
# ═══════════════════════════════════════
def pg_home(raw: pd.DataFrame, meta: dict, results: dict):
    # Public Navbar
    st.markdown("""
    <div class="pub-nav">
        <div class="brand-group">
            <div class="brand-icon">VI</div>
            <div>
                <div class="brand-title">Visitor Intelligence</div>
                <div class="brand-sub">AI Purchase Intent Engine</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Secondary Navigation Bar for Public Visitors
    p_c1, p_c2, p_c3, p_c4, p_c5, p_c6 = st.columns([1, 1, 1, 1, 1.2, 1.3], vertical_alignment="center")
    with p_c1:
        st.button("🏠 Home", key="pnav_home", on_click=nav_to, args=("Home",), use_container_width=True)
    with p_c2:
        st.button("🔬 Methodology", key="pnav_meth", on_click=nav_to, args=("Methodology",), use_container_width=True)
    with p_c3:
        st.button("ℹ️ About", key="pnav_about", on_click=nav_to, args=("About",), use_container_width=True)
    with p_c4:
        st.button("📊 Insights", key="pnav_ins", on_click=nav_to, args=("Insights",), use_container_width=True)
    with p_c5:
        if "user" in st.session_state:
            st.button("🚪 Logout", key="pnav_logout", on_click=do_logout, use_container_width=True)
        else:
            st.button("🔑 Sign In", key="pnav_login", on_click=nav_to, args=("Login",), use_container_width=True)
    with p_c6:
        label = "🚀 Open Dashboard" if "user" in st.session_state else "🚀 Get Started"
        st.button(label, key="pnav_getstart", on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)

    st.markdown("<div style='margin-bottom: 1.8rem;'></div>", unsafe_allow_html=True)

    # ── Section 1: Hero Section ──
    h_col1, h_col2 = st.columns([1.35, 1.0], vertical_alignment="center")
    with h_col1:
        st.markdown("""
        <div class="pill-badge">🚀 NEXT-GEN BEHAVIORAL MACHINE LEARNING</div>
        <h1 class="hero-headline">Understand Visitor Intent.<br><span style="color:#818cf8;">Predict What Converts.</span></h1>
        <p class="hero-description">
            Turn website browsing behavior into actionable purchase-intent insights with machine learning,
            velocity diagnostics, and cost-calibrated visitor intervention.
        </p>
        """, unsafe_allow_html=True)

        btn_c1, btn_c2, _ = st.columns([1.1, 1.1, 0.8])
        with btn_c1:
            st.button("🔮 Explore Dashboard", key="hero_cta_dash", on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)
        with btn_c2:
            st.button("⚡ Live Model Demo", key="hero_cta_demo", on_click=nav_to, args=("Predict Intent",), use_container_width=True)

    with h_col2:
        # Miniature Dashboard Product Preview Card
        st.markdown(f"""
        <div class="preview-card">
            <div class="preview-status">
                <span style="font-size:0.75rem;font-weight:700;color:#10b981;">🟢 PIPELINE ONLINE</span>
                <span style="font-size:0.72rem;color:#818cf8;font-weight:600;">XGBoost v{VERSION}</span>
            </div>
            <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;margin-bottom:0.2rem;">Live Inference Monitor</div>
            <p style="font-size:0.78rem;color:#94a3b8;margin-bottom:1rem;">Real-time session classification preview</p>
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:0.9rem;margin-bottom:0.8rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Predicted Intent Band</div>
                        <div style="font-size:1.25rem;font-weight:900;color:#10b981;">HIGH INTENT</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase;font-weight:700;">Ranking Score</div>
                        <div style="font-size:1.25rem;font-weight:900;color:#f8fafc;">0.774</div>
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:8px;">
                <div style="flex:1;background:rgba(16,185,129,0.1);border-radius:8px;padding:0.5rem;text-align:center;">
                    <div style="font-size:0.65rem;color:#34d399;font-weight:700;">CONVERSION RATE</div>
                    <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;">33.2%</div>
                </div>
                <div style="flex:1;background:rgba(99,102,241,0.1);border-radius:8px;padding:0.5rem;text-align:center;">
                    <div style="font-size:0.65rem;color:#a5b4fc;font-weight:700;">BUYER LIFT</div>
                    <div style="font-size:0.95rem;font-weight:800;color:#f8fafc;">2.85×</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Section 2: Platform Benchmark Metrics ──
    sec("Platform Benchmark Metrics (UCI Repository Ground Truth)")
    m1, m2, m3, m4 = st.columns(4)
    total_sess = len(raw)
    base_rate = float(raw["Revenue"].mean())
    m1.metric("Verified Sessions", f"{total_sess:,}", help="Clean deduplicated sessions from UCI Online Shoppers benchmark.")
    m2.metric("Observed Base Conversion", f"{base_rate:.1%}", help="Empirical conversion rate across all analyzed sessions.")
    m3.metric("Live Predictors", "16 Features", help="PageValues strictly excluded to avoid lookahead data leakage.")
    m4.metric("Model Architecture", "Class-Weighted XGBoost", help="265 estimators with depth=4 and calibrated cost thresholding.")

    # ── Section 3: Core Platform Features ──
    sec("Enterprise-Grade Intelligence Capabilities")
    f_r1 = st.columns(4)
    with f_r1[0]:
        vcard("⚡", "Real-Time Intent Prediction", "Instant classification into High, Medium, and Low intent bands for active web sessions.")
    with f_r1[1]:
        vcard("📈", "Behavioral Session Velocity", "Deep parsing of product dwell time, pageview depths, and exit velocity patterns.")
    with f_r1[2]:
        vcard("👥", "Visitor Cohort Segmentation", "Comparative conversion dynamics for New Visitors (24.9%) versus Returning Shoppers (14.1%).")
    with f_r1[3]:
        vcard("📊", "Conversion & Traffic Analytics", "Longitudinal monthly distribution mapping and multi-channel acquisition diagnostics.")

    st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
    f_r2 = st.columns(4)
    with f_r2[0]:
        vcard("📄", "CSV Batch Scoring Engine", "Upload high-volume historical logs to predict, rank, and download enriched conversion datasets.")
    with f_r2[1]:
        vcard("🧠", "Model Performance Insights", "Transparent held-out evaluation (0.376 PR-AUC), 5-fold cross-validation, and decile lift analysis.")
    with f_r2[2]:
        vcard("🛡️", "Leakage-Aware Pipeline", "Safely eliminates PageValues to ensure authentic production realism without lookahead contamination.")
    with f_r2[3]:
        vcard("🎯", "Actionable Decision Support", "Cost-sensitive thresholding (R=3, 5, 10) to guide targeted discount incentives and VIP intervention.")

    # ── Section 4: How It Works Workflow ──
    sec("How the Intelligence Engine Works")
    st.caption("A 4-stage machine learning pipeline engineered for reliable, leakage-free operational decision support:")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        pstep("01", "Capture Session Signals", "Ingest page counts, cumulative durations, bounce/exit rates, and technical metadata.")
    with w2:
        pstep("02", "Feature Preprocessing", "Apply log1p transformation for skewed numerics and OneHotEncoding for categories.")
    with w3:
        pstep("03", "XGBoost Scoring", "Evaluate session through class-weighted trees tuned specifically for imbalanced conversion paths.")
    with w4:
        pstep("04", "Intent Classification", "Map score to validated intent bands and trigger cost-optimal business actions.")

    # ── Section 5: Real-World Data Visualization Preview ──
    sec("Interactive Visual Analytics Preview")
    st.caption("Rendered directly from the verified 12,205-session e-commerce benchmark:")
    prev_c1, prev_c2 = st.columns(2)
    with prev_c1:
        pc = int(raw["Revenue"].sum())
        npc = total_sess - pc
        fig_don = _donut(["Non-Purchaser", "Purchaser"], [npc, pc], ["#1e293b", "#6366f1"])
        fig_don.update_layout(annotations=[dict(
            text=f"<b>{base_rate:.1%}</b><br><span style='font-size:11px;color:#94a3b8'>BASE CONVERSION</span>",
            x=0.5, y=0.5, font_size=20, font_color="#f8fafc", showarrow=False,
        )])
        st.markdown('<div class="ch-wrap"><h4>Verified Conversion Prevalence</h4>'
                    f'<div class="ch-sub">{pc:,} converting sessions out of {total_sess:,} total</div></div>', unsafe_allow_html=True)
        st.plotly_chart(fig_don, use_container_width=True, key="home_don")

    with prev_c2:
        mo = meta["observed_months"]
        mr = raw.groupby("Month")["Revenue"].mean().reindex(mo).dropna()
        fig_spl = _monthly_spline_chart(mr.index.tolist(), mr.values.tolist(), 280)
        st.markdown('<div class="ch-wrap"><h4>Observed Monthly Conversion Rate</h4>'
                    '<div class="ch-sub">November & May volume leaders; calendar year distribution</div></div>', unsafe_allow_html=True)
        st.plotly_chart(fig_spl, use_container_width=True, key="home_spl")

    # ── Section 6: Responsible AI Disclosure ──
    sec("Responsible AI & Leakage Prevention")
    st.info(
        "🛡️ **Production Reality Over Inflated Scores:** While including `PageValues` generates an artificial 0.76 PR-AUC, "
        "it represents post-purchase path data unavailable at early intervention. Our deployed pipeline excludes it, "
        "delivering a genuine **0.376 PR-AUC** (2.85× top-decile lift) for real-world reliability.",
        icon="🛡️",
    )

    # ── Section 7: Final CTA & Footer ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,0.18),rgba(139,92,246,0.15));
        border:1px solid rgba(99,102,241,0.3);border-radius:20px;padding:2.5rem;text-align:center;margin:3rem 0 1.5rem;">
        <h2 style="font-size:1.9rem;font-weight:900;color:#ffffff;margin-bottom:0.6rem;">Turn Visitor Behavior Into Smarter Decisions.</h2>
        <p style="color:#cbd5e1;font-size:0.98rem;max-width:650px;margin:0 auto 1.4rem;">
            Experience high-performance purchase intent intelligence built on verified machine learning architecture.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cta_c1, cta_c2, cta_c3 = st.columns([1.5, 1.2, 1.5])
    with cta_c2:
        st.button("🚀 Launch Dashboard Workspace", key="home_final_cta", on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)


# ═══════════════════════════════════════
# 🔐 AUTHENTICATION PAGE
# ═══════════════════════════════════════
def show_auth_page() -> None:
    # Header with Return to Home button
    top_c1, top_c2 = st.columns([4, 1.2], vertical_alignment="center")
    with top_c1:
        st.markdown("""
        <div class="brand-group" style="margin-bottom: 1rem;">
            <div class="brand-icon">VI</div>
            <div>
                <div class="brand-title">Visitor Intelligence</div>
                <div class="brand-sub">Platform Authentication Gateway</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with top_c2:
        st.button("← Return to Homepage", key="auth_return_home", on_click=nav_to, args=("Home",), use_container_width=True)

    target_dest = st.session_state.get("redirect_target", "Overview")
    if target_dest != "Overview":
        st.caption(f"🔒 Authenticate to access **{target_dest}**")

    is_logged_out = st.session_state.get("logged_out", False)
    auth_nonce = st.session_state.get("auth_nonce", 0)
    auth_key = f"firebase_auth_box_{auth_nonce}"

    # Render Firebase Interactive Component (Google One-Click + Email/Password + Guest Demo)
    auth_event = _firebase_auth_component(key=auth_key, logout_signal=is_logged_out, default=None)
    if is_logged_out:
        auth_event = None
        st.session_state["logged_out"] = False

    if auth_event and isinstance(auth_event, dict) and "email" in auth_event:
        st.session_state["user"] = auth_event
        st.session_state["logged_out"] = False
        st.query_params["session"] = create_session_token(auth_event)
        dest = st.session_state.pop("redirect_target", "Overview")
        nav_to(dest)
        st.rerun()

    # Native Streamlit Fallback / Direct Login Tabs
    with st.expander("🛠️ Direct Login / Network Fallback (If Popups are Blocked)", expanded=False):
        st.caption("Direct HTTPS connection to Firebase Authentication API.")
        t_in, t_up, t_guest = st.tabs(["Direct Sign In", "Create Account", "⚡ Guest Access"])

        with t_in:
            with st.form("fallback_signin"):
                em = st.text_input("Email", key="fb_em_in")
                pw = st.text_input("Password", type="password", key="fb_pw_in")
                btn = st.form_submit_button("Sign In with Email", type="primary", use_container_width=True)
            if btn:
                _handle_firebase_direct_auth("signInWithPassword", em, pw)

        with t_up:
            with st.form("fallback_signup"):
                nem = st.text_input("Email", key="fb_em_up")
                npw = st.text_input("Password", type="password", key="fb_pw_up")
                cbtn = st.form_submit_button("Create Firebase Account", type="primary", use_container_width=True)
            if cbtn:
                _handle_firebase_direct_auth("signUp", nem, npw)

        with t_guest:
            st.info("Explore the entire dashboard instantly as a Guest Analyst.")
            if st.button("Enter as Guest Analyst", key="fb_guest_btn", use_container_width=True):
                guest_data = {
                    "email": "guest.analyst@webannage.ai",
                    "name": "Guest Analyst",
                    "uid": "guest-demo",
                    "provider": "demo",
                }
                st.session_state["user"] = guest_data
                st.session_state["logged_out"] = False
                st.query_params["session"] = create_session_token(guest_data)
                dest = st.session_state.pop("redirect_target", "Overview")
                nav_to(dest)
                st.rerun()


def _handle_firebase_direct_auth(endpoint: str, email: str, password: str) -> None:
    email = email.strip()
    if not email or not password:
        st.error("Please enter both email address and password.")
        return

    payload = json.dumps({
        "email": email,
        "password": password,
        "returnSecureToken": True,
    }).encode("utf-8")
    request = urllib.request.Request(
        FIREBASE_AUTH_URL.format(endpoint, FIREBASE_API_KEY),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            auth_result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            details = json.loads(error.read().decode("utf-8"))
            code = details.get("error", {}).get("message", "")
        except Exception:
            code = ""
        messages = {
            "EMAIL_EXISTS": "An account already exists with this email address.",
            "INVALID_LOGIN_CREDENTIALS": "Invalid email or password.",
            "EMAIL_NOT_FOUND": "No account found with this email.",
            "INVALID_PASSWORD": "Incorrect password.",
            "INVALID_EMAIL": "Please provide a valid email format.",
            "WEAK_PASSWORD": "Password must be at least 6 characters.",
            "OPERATION_NOT_ALLOWED": "Email/Password sign-in is disabled in Firebase Console.",
            "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Please try again later.",
        }
        st.error(messages.get(code, f"Firebase authentication error ({code or error.code})."))
        return
    except Exception as error:
        LOGGER.warning("Firebase request failed: %s", error)
        st.error("Could not connect to Firebase Authentication. Check your network.")
        return

    user_payload = {
        "email": auth_result.get("email", email),
        "name": auth_result.get("displayName") or email.split("@", 1)[0],
        "uid": auth_result.get("localId", ""),
        "provider": "password",
    }
    st.session_state["user"] = user_payload
    st.session_state["logged_out"] = False
    st.query_params["session"] = create_session_token(user_payload)
    dest = st.session_state.pop("redirect_target", "Overview")
    nav_to(dest)
    st.rerun()


# ═══════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ═══════════════════════════════════════
def pg_overview(model, meta, results, raw):
    hero(
        "Website Visitor Intelligence Workspace",
        "Predict visitor purchase intent in real time from browsing patterns and technical context. "
        "Engineered with class-weighted XGBoost on the UCI Online Shoppers benchmark.",
        "ANALYTICS & EXECUTIVE DASHBOARD",
    )

    c1, c2, _, _ = st.columns([1.3, 1.3, 1, 1])
    c1.button("🔮  Predict Live Session", key="cta1", on_click=nav_to, args=("Predict Intent",), type="primary", use_container_width=True)
    c2.button("📊  Explore Model Insights", key="cta2", on_click=nav_to, args=("Insights",), use_container_width=True)

    # ── KPI Metrics (Dynamic from real data) ──
    sec("Key Benchmark Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    total = len(raw)
    pr = float(raw["Revenue"].mean())
    n_buyers = int(raw["Revenue"].sum())
    m1.metric("Total Sessions", f"{total:,}")
    m2.metric("Base Purchase Rate", f"{pr:.1%}")
    m3.metric("Purchasing Sessions", f"{n_buyers:,}")
    m4.metric("Model Architecture", "XGBoost")
    m5.metric("Held-Out PR-AUC", "0.376", delta="2.85× top-decile lift")

    # ── Row 1: Conversion Distribution + Monthly Trend ──
    sec("Conversion & Traffic Analytics")
    ch1, ch2 = st.columns(2)
    with ch1:
        npc = total - n_buyers
        fig = _donut(["Non-Purchaser", "Purchaser"], [npc, n_buyers], ["#334155", "#6366f1"])
        fig.update_layout(annotations=[dict(
            text=f"<b>{pr:.1%}</b><br><span style='font-size:11px;color:#94a3b8'>CONVERSION</span>",
            x=0.5, y=0.5, font_size=22, font_color="#f8fafc", showarrow=False,
        )])
        st.markdown(
            f'<div class="ch-wrap"><h4>Purchase Conversion Distribution</h4>'
            f'<div class="ch-sub">{n_buyers:,} converting sessions out of {total:,} total · '
            f'Majority-class baseline accuracy = {1-pr:.1%} (predicting "no purchase" always)</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d1")

    with ch2:
        mo = meta["observed_months"]
        mr = raw.groupby("Month")["Revenue"].mean().reindex(mo).dropna()
        fig = _monthly_spline_chart(mr.index.tolist(), mr.values.tolist(), 320)
        st.markdown(
            '<div class="ch-wrap"><h4>Observed Monthly Conversion Trend</h4>'
            '<div class="ch-sub">Spline curve of purchase rate per month · Jan & Apr absent from dataset</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d2")

    # ── Row 2: Visitor Cohort + Intent Band Distribution ──
    ch3, ch4 = st.columns(2)
    with ch3:
        vr = raw.groupby("VisitorType", observed=True)["Revenue"].agg(["mean", "count"]).reset_index()
        vr["label"] = vr["VisitorType"].str.replace("_", " ")
        colors_vr = ["#6366f1", "#06b6d4", "#8b5cf6"]
        fig = go.Figure()
        for i, row_vr in vr.iterrows():
            fig.add_trace(go.Bar(
                x=[row_vr["label"]],
                y=[row_vr["mean"] * 100],
                name=row_vr["label"],
                marker=dict(color=colors_vr[i % len(colors_vr)], cornerradius=8),
                text=[f'{row_vr["mean"]:.1%}'],
                textposition="outside",
                textfont=dict(size=12, color="#94a3b8"),
            ))
        fig.update_layout(**_get_layout(h=300, yt="Conversion Rate (%)", showlegend=False))
        st.markdown(
            '<div class="ch-wrap"><h4>Conversion Rate by Visitor Cohort</h4>'
            '<div class="ch-sub">New visitors (24.9%) convert at nearly 2× returning visitors (14.1%)</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d3")

    with ch4:
        bd = meta["score_bands"]["rates"]
        band_colors = ["#10b981", "#f59e0b", "#64748b"]
        fig = go.Figure(go.Bar(
            x=list(bd.keys()),
            y=[v["sessions"] for v in bd.values()],
            marker=dict(color=band_colors, cornerradius=8),
            text=[f'{v["sessions"]:,}\n({v["conversion_rate"]:.1%} conv)' for v in bd.values()],
            textposition="outside",
            textfont=dict(size=11, color="#cbd5e1"),
        ))
        fig.update_layout(**_get_layout(h=300, yt="Sessions", showlegend=False))
        st.markdown(
            '<div class="ch-wrap"><h4>5-Fold Cross-Fitted Intent Band Distribution</h4>'
            '<div class="ch-sub">High (33.2% conv) · Medium (14.8% conv) · Low (3.1% conv)</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d4")

    # ── Row 3: EDA — PageValues Leakage + BounceRates vs Revenue ──
    sec("Exploratory Data Analysis — Key EDA Findings")
    st.caption("All charts computed live from the verified 12,205-session dataset:")
    e1, e2 = st.columns(2)
    with e1:
        # PageValues leakage bar
        pv_zero = raw[raw["PageValues"] == 0]["Revenue"].mean()
        pv_nonzero = raw[raw["PageValues"] > 0]["Revenue"].mean()
        pv_cnt_z = int((raw["PageValues"] == 0).sum())
        pv_cnt_nz = int((raw["PageValues"] > 0).sum())
        fig_pv = go.Figure(go.Bar(
            x=["PageValues = 0\n(77.6% of sessions)", "PageValues > 0\n(22.4% of sessions)"],
            y=[pv_zero * 100, pv_nonzero * 100],
            marker=dict(color=["#475569", "#f43f5e"], cornerradius=8),
            text=[f"{pv_zero:.1%}", f"{pv_nonzero:.1%}"],
            textposition="outside",
            textfont=dict(size=13, color="#cbd5e1"),
        ))
        fig_pv.update_layout(**_get_layout(h=300, yt="Conversion Rate (%)", showlegend=False))
        st.markdown(
            '<div class="ch-wrap"><h4>⚠️ PageValues Leakage Proof</h4>'
            '<div class="ch-sub">56% conversion when PageValues > 0 — this is post-purchase data, NOT available at session start</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_pv, use_container_width=True, key="e1")

    with e2:
        # Weekend vs Weekday conversion
        we = raw.groupby("Weekend")["Revenue"].agg(["mean", "count"]).reset_index()
        we["label"] = we["Weekend"].map({0: "Weekday", 1: "Weekend"})
        fig_we = go.Figure(go.Bar(
            x=we["label"],
            y=we["mean"] * 100,
            marker=dict(color=["#6366f1", "#06b6d4"], cornerradius=8),
            text=[f"{v:.1%}" for v in we["mean"]],
            textposition="outside",
            textfont=dict(size=13, color="#94a3b8"),
        ))
        fig_we.update_layout(**_get_layout(h=300, yt="Conversion Rate (%)", showlegend=False))
        st.markdown(
            '<div class="ch-wrap"><h4>Conversion Rate: Weekday vs Weekend</h4>'
            '<div class="ch-sub">Weekday sessions slightly outperform weekend browsing in conversion probability</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_we, use_container_width=True, key="e2")

    # ── Row 4: Traffic Type conversion + Bounce/Exit correlation ──
    e3, e4 = st.columns(2)
    with e3:
        tt = raw.groupby("TrafficType", observed=True)["Revenue"].agg(["mean", "count"]).reset_index()
        tt = tt[tt["count"] >= 80].sort_values("mean", ascending=False).head(8)
        fig_tt = go.Figure(go.Bar(
            x=tt["TrafficType"].astype(str),
            y=tt["mean"] * 100,
            marker=dict(color="#8b5cf6", cornerradius=8),
            text=[f"{v:.1%}" for v in tt["mean"]],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
        ))
        fig_tt.update_layout(**_get_layout(h=300, yt="Conversion Rate (%)", showlegend=False))
        st.markdown(
            '<div class="ch-wrap"><h4>Conversion by Traffic Source (Top Types)</h4>'
            '<div class="ch-sub">Different acquisition channels show 3–4× variation in conversion rates</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_tt, use_container_width=True, key="e3")

    with e4:
        # BounceRates vs ExitRates scatter colored by Revenue
        sample = raw.sample(min(800, len(raw)), random_state=42)
        fig_sc = go.Figure()
        for rev, color, label in [(0, "#475569", "No Purchase"), (1, "#10b981", "Purchase")]:
            mask = sample["Revenue"] == rev
            fig_sc.add_trace(go.Scatter(
                x=sample.loc[mask, "BounceRates"],
                y=sample.loc[mask, "ExitRates"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=5, opacity=0.65),
                hovertemplate="Bounce: %{x:.3f}<br>Exit: %{y:.3f}<extra></extra>",
            ))
        layout_sc = _get_layout(h=300, xt="Bounce Rate", yt="Exit Rate", showlegend=True)
        layout_sc["legend"] = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        fig_sc.update_layout(**layout_sc)
        st.markdown(
            '<div class="ch-wrap"><h4>BounceRates vs ExitRates (Corr = 0.91)</h4>'
            '<div class="ch-sub">Near-collinear features — buyers cluster at low bounce/exit values</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="e4")

    # ── Row 5: Session segments ──
    seg_data = results.get("segmentation", {}).get("profiles", [])
    if seg_data:
        sec("K-Means Customer Segmentation (k=3, Silhouette = 0.434)")
        st.caption("Sessions clustered by engagement behavior — PageValues and Revenue excluded from clustering:")
        seg_names = ["🟢 Focused Buyers", "🟡 Casual Browsers", "🔴 Bouncers"]
        seg_colors = ["#10b981", "#f59e0b", "#f43f5e"]
        seg_cols = st.columns(3)
        for col, seg, name, color in zip(seg_cols, seg_data, seg_names, seg_colors):
            with col:
                cr = seg["conversion_rate"]
                st.markdown(
                    f'<div class="seg-card">'
                    f'<h4>{name}</h4>'
                    f'<div class="seg-rate" style="color:{color};">{cr:.1%}</div>'
                    f'<p style="color:#94a3b8;font-size:0.75rem;margin-bottom:0.6rem !important;">Conversion Rate</p>'
                    f'<p>Sessions: <strong style="color:#e2e8f0">{seg["sessions"]:,}</strong></p>'
                    f'<p>Median Product Pages: <strong style="color:#e2e8f0">{int(seg["median_product_pages"])}</strong></p>'
                    f'<p>Median Dwell Time: <strong style="color:#e2e8f0">{seg["median_product_duration"]:.0f}s</strong></p>'
                    f'<p>Median Exit Rate: <strong style="color:#e2e8f0">{seg["median_exit_rate"]:.1%}</strong></p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Takeaways ──
    sec("Strategic Machine Learning Takeaways")
    i1, i2, i3 = st.columns(3)
    with i1:
        vcard("📦", "Product Duration is Top Signal",
              "Time spent on product pages + low exit velocity are the sharpest indicators of purchase intent per SHAP analysis.")
    with i2:
        vcard("🛡️", "Leakage-Aware Production Model",
              "PageValues stripped out. Including it gives 0.76 PR-AUC but causes severe lookahead leakage — the honest model scores 0.376.")
    with i3:
        vcard("🎯", "Ranking Over Raw Probabilities",
              "The pipeline ranks sessions by intent to optimize targeting decisions under cost budgets R=3, 5, 10.")



# ═══════════════════════════════════════
# PAGE 2 — PREDICT INTENT
# ═══════════════════════════════════════
def pg_predict(model, meta, limits, ratio, results):
    hero(
        "Real-Time Purchase Intent Predictor",
        "Score active sessions directly through the production XGBoost pipeline. "
        "Excludes PageValues to ensure legitimate, leakage-free operational decision support.",
        "PREDICTIVE INFERENCE ENGINE",
    )
    st.info("🛡️ **Leakage Protection Enabled:** `PageValues` is omitted to preserve production realism.", icon="🛡️")

    tab1, tab2 = st.tabs(["⚡ Single Session Inference", "📄 CSV Batch Scoring Engine"])
    with tab1:
        _single_predict_view(model, meta, limits, ratio)
    with tab2:
        _batch_predict_view(model, meta, limits, ratio)


def _single_predict_view(model, meta, limits, ratio):
    df_defaults = meta["default_values"]
    for feat in FEATURE_COLUMNS:
        st.session_state.setdefault("field_" + feat, df_defaults[feat])

    presets = {
        "High-Intent Buyer": (
            "🟢", "Heavy product browsing, minimal bounce, returning shopper",
            {"ProductRelated": 45, "ProductRelated_Duration": 2400.0, "ExitRates": 0.005, "BounceRates": 0.001, "VisitorType": "Returning_Visitor", "Month": "Nov"},
        ),
        "Hesitant Evaluator": (
            "🟡", "Moderate product activity, looking around catalog",
            {"ProductRelated": 12, "ProductRelated_Duration": 350.0, "ExitRates": 0.025, "BounceRates": 0.01, "VisitorType": "Returning_Visitor", "Month": "Nov"},
        ),
        "Casual / Bounce Risk": (
            "🔴", "Single-page arrival, rapid exit velocity",
            {"ProductRelated": 2, "ProductRelated_Duration": 18.0, "ExitRates": 0.18, "BounceRates": 0.15, "VisitorType": "Returning_Visitor", "Month": "Nov"},
        ),
        "New Explorer": (
            "🟣", "First-time visitor researching offerings",
            {"ProductRelated": 18, "ProductRelated_Duration": 650.0, "ExitRates": 0.015, "BounceRates": 0.005, "VisitorType": "New_Visitor", "Month": "Dec"},
        ),
    }

    sec("Quick-Load Sample Scenarios")
    st.caption("Select a scenario to populate session features with realistic browsing aggregates:")
    cols_pre = st.columns(4)
    for col, (lbl, (ic, desc, vals)) in zip(cols_pre, presets.items()):
        with col:
            st.markdown(f'<div class="smp-card"><h4>{ic} {lbl}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
            if st.button(f"Load {lbl}", key=f"btn_pre_{lbl}", use_container_width=True):
                for k, v in {**df_defaults, **vals}.items():
                    st.session_state["field_" + k] = v
                st.rerun()

    with st.form("session_predict_form"):
        cm = {c: int(limits[c][1]) for c in COUNT_COLUMNS}
        dm = {c: float(limits[c][1]) for c in DURATION_COLUMNS}

        sec("1. Page Engagement Aggregates")
        st.caption("Session page counts and cumulative dwell durations.")
        p1, p2, p3 = st.columns(3)
        with p1:
            admin = st.number_input("Administrative Pages", 0, cm["Administrative"], step=1, key="field_Administrative")
            admin_d = st.number_input("Admin Duration (sec)", 0.0, dm["Administrative_Duration"], step=10.0, key="field_Administrative_Duration")
        with p2:
            info_p = st.number_input("Informational Pages", 0, cm["Informational"], step=1, key="field_Informational")
            info_d = st.number_input("Info Duration (sec)", 0.0, dm["Informational_Duration"], step=10.0, key="field_Informational_Duration")
        with p3:
            prod = st.number_input("Product Pages", 0, cm["ProductRelated"], step=1, key="field_ProductRelated")
            prod_d = st.number_input("Product Duration (sec)", 0.0, dm["ProductRelated_Duration"], step=30.0, key="field_ProductRelated_Duration")

        sec("2. Behavioral Velocity Signals")
        st.caption("Google Analytics engagement signals for this session.")
        b1, b2, b3 = st.columns(3)
        bounce = b1.number_input("Bounce Rate", float(limits["BounceRates"][0]), float(limits["BounceRates"][1]), step=0.001, format="%.4f", key="field_BounceRates")
        exit_r = b2.number_input("Exit Rate", float(limits["ExitRates"][0]), float(limits["ExitRates"][1]), step=0.001, format="%.4f", key="field_ExitRates")
        special = b3.select_slider("Special Day Closeness", options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], key="field_SpecialDay")

        sec("3. Visitor & Technical Context")
        v1, v2, v3 = st.columns(3)
        month = v1.selectbox("Month of Session", meta["observed_months"], key="field_Month")
        visitor = v2.selectbox("Visitor Cohort", meta["visitor_types"], format_func=lambda x: x.replace("_", " "), key="field_VisitorType")
        weekend = v3.selectbox("Weekend Session?", [0, 1], format_func=lambda x: "Yes (Weekend)" if x else "No (Weekday)", key="field_Weekend")

        tech = meta["technical_category_options"]
        t1, t2, t3, t4 = st.columns(4)
        sel = {}
        for col, (feat, title) in zip([t1, t2, t3, t4], [("OperatingSystems", "OS Code"), ("Browser", "Browser Code"), ("Region", "Region Code"), ("TrafficType", "Traffic Source Code")]):
            sel[feat] = col.selectbox(title, tech[feat], key="field_" + feat)

        submitted = st.form_submit_button("🔮  Compute Purchase Intent Score", type="primary", use_container_width=True)

    if submitted:
        row = {
            "Administrative": admin, "Administrative_Duration": admin_d,
            "Informational": info_p, "Informational_Duration": info_d,
            "ProductRelated": prod, "ProductRelated_Duration": prod_d,
            "BounceRates": bounce, "ExitRates": exit_r, "SpecialDay": special,
            "Month": month, "VisitorType": visitor, "Weekend": weekend, **sel,
        }
        try:
            with st.spinner("Executing pipeline transformation and XGBoost scoring…"):
                clean = validate_frame(pd.DataFrame([row]), meta, limits)
                result = score_sessions(model, clean, meta, ratio).iloc[0].to_dict()
                st.session_state["last_result"] = result
                st.session_state["last_input"] = row
        except InputValidationError as exc:
            st.error(str(exc), icon="⚠️")
        except Exception:
            LOGGER.exception("Prediction failed")
            st.error("Error executing model inference. Please verify inputs.", icon="❌")

    if "last_result" in st.session_state:
        _render_prediction_output(st.session_state["last_result"], st.session_state.get("last_input", {}), ratio, meta, limits)


def _render_prediction_output(result, inp, ratio, meta, limits):
    band = result["intent_band"]
    css, desc, rec, action = band_info(band)
    score = result["model_score"]
    thr = result["threshold"]

    sec("Inference Results & Actionable Decision")
    st.markdown(
        f'<div class="int-res {css}">'
        f'<div class="int-tag">PURCHASE INTENT CLASSIFICATION</div>'
        f'<div class="int-band">{band.upper()} INTENT BAND</div>'
        f'<div class="int-desc">{desc}</div>'
        f'<div class="int-score">Model Ranking Score: <strong>{score:.4f}</strong> (Uncalibrated ranking signal)</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns([1, 1.4])
    with d1:
        st.metric("Historical Cohort Conversion", f"{result['historical_band_reference']:.1%}",
                  help="Observed conversion rate among similar sessions in validation dataset.")
        if result["intervention_flag"]:
            st.success(f"✅ Flagged for Proactive Action (Score ≥ {thr:.2f} threshold at R={ratio})", icon="🚀")
        else:
            st.info(f"ℹ️ Below Intervention Cutoff (Score < {thr:.2f} threshold at R={ratio})", icon="💤")

    with d2:
        emoji = {"high": "🔥", "medium": "🎯", "low": "🛡️"}[css]
        st.markdown(
            f'<div class="vi-card" style="border-left: 4px solid var(--accent-indigo);">'
            f'<h4>{emoji} Strategic Recommendation: {action}</h4>'
            f'<p>{rec}</p></div>',
            unsafe_allow_html=True,
        )

    # Visual Diagnostics: Speedometer Gauge + Radar DNA Profile
    sec("Behavioral Profile Diagnostics")
    g_col1, g_col2 = st.columns([1, 1])

    with g_col1:
        st.markdown('<div class="ch-wrap"><h4>Threshold Position Speedometer</h4>'
                    f'<div class="ch-sub">Current score relative to score bands and R={ratio} cutoff</div></div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(size=32, color="#f8fafc"), valueformat=".3f"),
            gauge=dict(
                axis=dict(range=[0, 1], tickcolor="#64748b", tickfont=dict(color="#94a3b8", size=11)),
                bar=dict(color="#6366f1", thickness=0.32),
                bgcolor="rgba(30, 41, 59, 0.5)",
                borderwidth=0,
                steps=[
                    dict(range=[0, meta["score_bands"]["medium_cutoff"]], color="rgba(100, 116, 139, 0.15)"),
                    dict(range=[meta["score_bands"]["medium_cutoff"], meta["score_bands"]["high_cutoff"]], color="rgba(245, 158, 11, 0.15)"),
                    dict(range=[meta["score_bands"]["high_cutoff"], 1.0], color="rgba(16, 185, 129, 0.18)"),
                ],
                threshold=dict(line=dict(color="#f43f5e", width=3.5), thickness=0.85, value=thr),
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            height=250,
            margin=dict(l=30, r=30, t=30, b=15),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge")

    with g_col2:
        st.markdown('<div class="ch-wrap"><h4>Session Behavioral DNA (Radar Analysis)</h4>'
                    '<div class="ch-sub">Current session vs benchmark converting visitor profile</div></div>', unsafe_allow_html=True)
        fig_radar = _radar_dna_chart(inp, limits)
        st.plotly_chart(fig_radar, use_container_width=True, key="radar_dna")


def _batch_predict_view(model, meta, limits, ratio):
    sec("Batch CSV Session Scoring")
    st.caption("Upload a CSV file containing multiple sessions with the 16 model features:")

    tmpl = pd.DataFrame([meta["default_values"]], columns=FEATURE_COLUMNS).to_csv(index=False).encode("utf-8")
    b1, b2, _ = st.columns([1.2, 1.4, 1.4])
    b1.download_button("📥 Download CSV Template", tmpl, "visitor_batch_template.csv", "text/csv", use_container_width=True)

    # Load sample demo file
    sample_path = ROOT / "data" / "sample_batch_scoring.csv"
    if sample_path.is_file():
        with open(sample_path, "rb") as f:
            b2.download_button("📋 Download 20-Row Demo File", f.read(), "sample_batch_scoring.csv", "text/csv", use_container_width=True)

    st.markdown("**Required columns:** " + ", ".join(f"`{c}`" for c in FEATURE_COLUMNS))
    uploaded = st.file_uploader("Upload CSV sessions file", type=["csv"], key="batch_csv_file")
    if not uploaded:
        st.info("💡 Download the 20-row demo file above, then re-upload it here to see the scoring engine in action.", icon="💡")
        return

    try:
        batch = pd.read_csv(uploaded)
        if batch.empty:
            st.error("Uploaded CSV contains no records.", icon="⚠️")
            return
        with st.spinner(f"Validating and scoring {len(batch):,} sessions…"):
            clean = validate_frame(batch, meta, limits)
            scored = score_sessions(model, clean, meta, ratio)
            output = batch.copy()
            for col in scored.columns:
                output[col] = scored[col].to_numpy()

        st.success(f"Successfully processed {len(output):,} sessions through XGBoost pipeline!", icon="✅")

        s1, s2, s3, s4 = st.columns(4)
        band_counts = output["intent_band"].value_counts()
        s1.metric("Scored Sessions", f"{len(output):,}")
        s2.metric("High Intent", f"{band_counts.get('High', 0):,} ({band_counts.get('High', 0)/len(output):.1%})")
        s3.metric("Medium Intent", f"{band_counts.get('Medium', 0):,} ({band_counts.get('Medium', 0)/len(output):.1%})")
        s4.metric("Flagged for Action", f"{int(output['intervention_flag'].sum()):,} ({output['intervention_flag'].mean():.1%})")

        # Visualize distribution of scored bands
        bc1, bc2 = st.columns(2)
        with bc1:
            bands_order = ["High", "Medium", "Low"]
            band_c = [band_counts.get(b, 0) for b in bands_order]
            fig_bd = go.Figure(go.Bar(
                x=bands_order, y=band_c,
                marker=dict(color=["#10b981", "#f59e0b", "#64748b"], cornerradius=8),
                text=band_c, textposition="outside",
                textfont=dict(size=12, color="#cbd5e1"),
            ))
            fig_bd.update_layout(**_get_layout(h=260, yt="Sessions", showlegend=False))
            st.markdown('<div class="ch-wrap"><h4>Intent Band Distribution</h4><div class="ch-sub">Batch results classified by model score</div></div>', unsafe_allow_html=True)
            st.plotly_chart(fig_bd, use_container_width=True, key="batch_bands")
        with bc2:
            fig_sc = go.Figure(go.Histogram(
                x=output["model_score"],
                nbinsx=20,
                marker=dict(color="#6366f1", line=dict(color="#090d16", width=1)),
            ))
            thr_val = float(output["threshold"].iloc[0]) if "threshold" in output.columns else 0.49
            fig_sc.add_vline(x=thr_val, line=dict(color="#f43f5e", dash="dash", width=2),
                             annotation_text=f"Threshold={thr_val:.2f}", annotation_position="top right",
                             annotation_font_color="#f43f5e")
            fig_sc.update_layout(**_get_layout(h=260, xt="Model Ranking Score", yt="Sessions", showlegend=False))
            st.markdown('<div class="ch-wrap"><h4>Score Distribution Histogram</h4><div class="ch-sub">Red line = intervention cutoff at selected R</div></div>', unsafe_allow_html=True)
            st.plotly_chart(fig_sc, use_container_width=True, key="batch_hist")

        st.dataframe(output.head(100), hide_index=True, use_container_width=True)
        st.download_button(
            "📥 Download Scored Batch Results",
            output.to_csv(index=False).encode("utf-8"),
            "visitor_intent_scored_results.csv",
            "text/csv",
            type="primary",
            use_container_width=True,
        )
    except InputValidationError as exc:
        st.error(str(exc), icon="⚠️")
    except Exception as exc:
        LOGGER.exception("Batch error: %s", exc)
        st.error("Error processing batch CSV. Please check formatting against the template.", icon="❌")


# ═══════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ═══════════════════════════════════════
def pg_insights(meta, results, raw):
    hero(
        "Model Performance & Behavioral Analytics",
        "Rigorous held-out test evaluation, cross-validation benchmarks, "
        "feature importance diagnostics, PageValues ablation, and segmentation analysis.",
        "BENCHMARKS & DIAGNOSTICS",
    )
    st.info(
        "📊 **PR-AUC is Primary:** With only ~15.6% positive conversion prevalence, "
        "precision-recall AUC is the decisive metric. Accuracy = 84.4% by predicting 'no purchase' always — completely useless.",
        icon="📊",
    )

    # ── Held-Out Test Results ──
    sec("Held-Out Test Results (Leakage-Free Deployed Model)")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("PR-AUC (Primary)", "0.376", delta="vs 0.15 random baseline")
    s2.metric("ROC-AUC", "0.795", delta="vs 0.50 random baseline")
    s3.metric("Precision (R=5)", "29.6%")
    s4.metric("Recall (R=5)", "75.9%")
    s5.metric("Top Decile Lift", "2.85×", delta="44.5% conv in top 10%")

    # ── PageValues Ablation ──
    sec("⚠️ PageValues Ablation Study — The Most Important Finding")
    ab1, ab2 = st.columns([1.3, 0.7])
    with ab1:
        # Pull from supplemental results if available
        comp = results.get("matched_model_comparison", [])
        xgb_with = next((r["mean_pr_auc"] for r in comp if r["feature_set"] == "with PageValues" and r["model"] == "XGBoost"), 0.7492)
        xgb_without = next((r["mean_pr_auc"] for r in comp if r["feature_set"] == "without PageValues" and r["model"] == "XGBoost"), 0.3795)
        fig = _grouped(
            ["PR-AUC (CV Mean)", "ROC-AUC"],
            {
                "With PageValues (Leaky — 0.76 test)": [xgb_with, 0.9317],
                "Without PageValues (Deployed — 0.376 test)": [xgb_without, 0.7949],
            },
            ["#f43f5e", "#6366f1"], "Metric Score", 350,
        )
        st.markdown(
            '<div class="ch-wrap"><h4>Ablation: XGBoost With vs Without PageValues</h4>'
            '<div class="ch-sub">PageValues is computed from completed transactions — it is unavailable at prediction time and leaks the outcome</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="pv_ab")
    with ab2:
        pv_z = results.get("conversion_pagevalues_zero", 0.039)
        pv_nz = results.get("conversion_pagevalues_positive", 0.563)
        st.markdown(
            f'<div class="vi-card">'
            f'<span class="ic">🛡️</span>'
            f'<h4>Why PageValues Leaks</h4>'
            f'<p>When PageValues = 0: <strong style="color:#94a3b8">{pv_z:.1%}</strong> conversion<br>'
            f'When PageValues &gt; 0: <strong style="color:#f43f5e">{pv_nz:.1%}</strong> conversion<br><br>'
            f'This 14× gap exists because PageValues is derived from <em>post-transaction</em> analytics. '
            f'At session start, it does not exist yet. Our deployed model safely excludes it.</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.metric("PR-AUC Gap (Leakage)", f"−{xgb_with - xgb_without:.3f}", delta=f"−{(xgb_with-xgb_without)/xgb_with:.0%} signal lost (honest model)", delta_color="inverse")

    # ── Matched Model Comparison ──
    sec("Matched-Settings Cross-Validation (5-Fold, All 3 Models)")
    comp = results.get("matched_model_comparison", [])
    if comp:
        model_names = ["Logistic Regression", "Random Forest", "XGBoost"]
        with_pv = [r["mean_pr_auc"] for r in comp if r["feature_set"] == "with PageValues"]
        without_pv = [r["mean_pr_auc"] for r in comp if r["feature_set"] == "without PageValues"]
        fig_comp = _grouped(
            model_names,
            {"With PageValues (Leaky)": with_pv, "Without PageValues (Deployed)": without_pv},
            ["#f59e0b", "#6366f1"], "Mean CV PR-AUC", 360,
        )
        st.markdown(
            '<div class="ch-wrap"><h4>All Models: PR-AUC Comparison</h4>'
            '<div class="ch-sub">XGBoost wins both with and without PageValues — selected as final model</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="cv_mc")

        # Show fold-level detail for XGBoost without PageValues
        xgb_folds = next((r["fold_pr_auc"] for r in comp if r["feature_set"] == "without PageValues" and r["model"] == "XGBoost"), [])
        if xgb_folds:
            st.caption(f"XGBoost (no PageValues) per-fold PR-AUC: {[round(x,3) for x in xgb_folds]} · Mean: {sum(xgb_folds)/len(xgb_folds):.4f} · Std: {(sum((x-sum(xgb_folds)/len(xgb_folds))**2 for x in xgb_folds)/len(xgb_folds))**0.5:.4f}")

    # ── Imbalance Handling Comparison ──
    sec("Class Imbalance Handling: Class Weights vs SMOTE")
    imb = results.get("imbalance_comparison", {})
    if imb:
        ic1, ic2 = st.columns(2)
        w_mean = imb.get("class_weighted_xgb_mean", 0.379)
        s_mean = imb.get("smote_in_cv_xgb_mean", 0.355)
        ic1.metric("Class-Weighted XGBoost (Deployed)", f"{w_mean:.4f} CV PR-AUC", delta="Winner ✅")
        ic2.metric("SMOTE Inside CV + XGBoost", f"{s_mean:.4f} CV PR-AUC", delta=f"−{w_mean-s_mean:.4f} vs weighted", delta_color="inverse")
        st.info(
            f"⚡ **SMOTE correctly applied inside each CV fold** — never before splitting. "
            f"Class weighting outperforms SMOTE here (+{w_mean-s_mean:.3f} PR-AUC) because SMOTE on "
            f"one-hot encoded categoricals creates fractional synthetic coordinates. Note: {imb.get('note', '')}",
            icon="⚡",
        )

    # ── Cost Threshold Optimization ──
    sec("Cost-Sensitive Threshold Optimization")
    st.markdown(
        r"**Business Cost Formula:** $\text{Total Cost} = \text{False Positives} + R \times \text{False Negatives}$"
        "\n\nWhere **R** = cost of missing a real buyer vs. wasting an intervention. Thresholds calibrated on Out-Of-Fold predictions only — test set never touched during tuning."
    )
    cost_data = results.get("app_oof_cost_sensitivity", meta.get("cost_sensitivity", []))
    if cost_data:
        cost_rows = []
        for c in cost_data:
            cost_rows.append({
                "Cost Ratio R": f"R={c['R']}",
                "Threshold": f"{c['threshold']:.2f}",
                "Precision": f"{c['precision']:.1%}",
                "Recall": f"{c['recall']:.1%}",
                "Sessions Flagged": f"{c['flagged_percent']:.1f}%",
                "Cost vs Do-Nothing": f"{c['cost_vs_do_nothing_percent']:.1f}%",
            })
        st.dataframe(pd.DataFrame(cost_rows), hide_index=True, use_container_width=True)
        st.caption("At R=5: Catching 75.9% of buyers while only flagging 40% of sessions = 40% fewer wasted interventions vs flagging everyone.")

    # ── Decile Lift ──
    sec("Decile Lift Analysis — Ranking Quality")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Top 10% Conversion Rate", "44.5%", "2.85× base rate")
    l2.metric("Top 20% Conversion Rate", "35.0%", "2.24× base rate")
    l3.metric("Top 30% Buyer Capture", "64.7%", "247 of 382 buyers")
    l4.metric("Bottom 40% Conversion", "6.0%", "Safe non-intervention zone")

    fig_lift = go.Figure()
    fig_lift.add_trace(go.Bar(
        x=["Top 10%", "Top 20%", "Top 30%", "Top 40%", "Top 50%", "Bottom 50%"],
        y=[44.5, 35.0, 21.6, 17.2, 14.1, 6.0],
        marker=dict(color=["#10b981", "#34d399", "#6366f1", "#8b5cf6", "#f59e0b", "#64748b"], cornerradius=8),
        text=["44.5%", "35.0%", "21.6%", "17.2%", "14.1%", "6.0%"],
        textposition="outside",
        textfont=dict(size=12, color="#cbd5e1"),
    ))
    base_line = [15.6] * 6
    fig_lift.add_trace(go.Scatter(
        x=["Top 10%", "Top 20%", "Top 30%", "Top 40%", "Top 50%", "Bottom 50%"],
        y=base_line, mode="lines",
        line=dict(color="#f43f5e", dash="dash", width=2),
        name=f"Base Rate (15.6%)",
    ))
    layout_lift = _get_layout(h=320, yt="Conversion Rate (%)", showlegend=True)
    layout_lift["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    fig_lift.update_layout(**layout_lift)
    st.markdown(
        '<div class="ch-wrap"><h4>Conversion Rate by Decile vs Base Rate</h4>'
        '<div class="ch-sub">Top 10% of sessions ranked by model score show 2.85× the base conversion rate</div></div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_lift, use_container_width=True, key="lift_chart")

    # ── Feature Importance ──
    sec("Global Feature Importance (Test Set)")
    fi1, fi2 = st.columns(2)
    with fi1:
        shap_feats = ["Product Duration", "Month (Seasonal)", "Exit Rate", "Browser", "Traffic Type",
                      "Bounce Rate", "Admin Duration", "OS", "Product Pages", "Special Day"]
        shap_vals  = [0.444, 0.428, 0.410, 0.207, 0.160, 0.111, 0.109, 0.108, 0.089, 0.071]
        fig_shap = _bar(shap_feats, shap_vals, "#8b5cf6", "Mean |SHAP| Value", 400, horiz=True)
        st.markdown(
            '<div class="ch-wrap"><h4>SHAP Feature Importance (TreeExplainer)</h4>'
            '<div class="ch-sub">ProductRelated_Duration is the dominant purchase signal</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")

    with fi2:
        perm_feats = ["Product Duration", "Exit Rate", "Month (Seasonal)", "Product Pages",
                      "Visitor Type", "Bounce Rate", "Admin Duration", "Traffic Type", "OS", "Admin Pages"]
        perm_vals  = [0.0794, 0.0738, 0.0588, 0.0184, 0.0151, 0.0100, 0.0093, 0.0082, 0.0080, 0.0048]
        fig_perm = _bar(perm_feats, perm_vals, "#06b6d4", "PR-AUC Drop on Shuffle", 400, horiz=True)
        st.markdown(
            '<div class="ch-wrap"><h4>Permutation Importance (Robustness Check)</h4>'
            '<div class="ch-sub">Consistent with SHAP — Duration & Exit Rate confirmed as top features</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_perm, use_container_width=True, key="perm_chart")

    st.info(
        "🔬 **SHAP shows association, not causation.** These findings describe behavioral correlations "
        "observed in historical session data — we cannot claim that increasing product page time *causes* purchase.",
        icon="🔬",
    )

    # ── Learning Curve + Calibration Curve ──
    sec("Diagnostic Curves")
    d_c1, d_c2 = st.columns(2)
    lc = ROOT / "reports" / "figures" / "learning_curve.png"
    cc = ROOT / "reports" / "figures" / "calibration_curve.png"
    with d_c1:
        st.markdown(
            '<div class="ch-wrap"><h4>Sample Efficiency Learning Curve</h4>'
            '<div class="ch-sub">CV PR-AUC improves as training size grows — model is not saturated</div></div>',
            unsafe_allow_html=True,
        )
        if lc.is_file():
            st.image(str(lc), use_container_width=True)
        else:
            # Draw from supplemental_results if image missing
            lc_data = results.get("learning_curve", [])
            if lc_data:
                fig_lc = go.Figure()
                sizes = [d["training_rows"] for d in lc_data]
                cv_means = [d["cv_pr_auc_mean"] for d in lc_data]
                cv_stds = [d["cv_pr_auc_std"] for d in lc_data]
                fig_lc.add_trace(go.Scatter(x=sizes, y=cv_means, mode="lines+markers",
                    line=dict(color="#6366f1", width=2.5),
                    marker=dict(size=8),
                    error_y=dict(type="data", array=cv_stds, visible=True, color="rgba(99,102,241,0.3)"),
                    name="CV PR-AUC"))
                fig_lc.update_layout(**_get_layout(h=300, xt="Training Rows", yt="CV PR-AUC"))
                st.plotly_chart(fig_lc, use_container_width=True, key="lc_fallback")

    with d_c2:
        st.markdown(
            '<div class="ch-wrap"><h4>Calibration Reliability Curve</h4>'
            '<div class="ch-sub">Raw scores serve as ranking signal, not calibrated probabilities</div></div>',
            unsafe_allow_html=True,
        )
        if cc.is_file():
            st.image(str(cc), use_container_width=True)
        else:
            cal = results.get("calibration", {})
            if cal:
                frac = cal.get("reliability_fraction_positive", [])
                pred = cal.get("reliability_mean_score", [])
                if frac and pred:
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=pred, y=frac, mode="lines+markers",
                        line=dict(color="#10b981", width=2.5), name="Calibrated",
                        marker=dict(size=8)))
                    fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(color="#64748b", dash="dash"), name="Ideal"))
                    fig_cal.update_layout(**_get_layout(h=300, xt="Mean Predicted Score", yt="Observed Purchase Rate", showlegend=True))
                    st.plotly_chart(fig_cal, use_container_width=True, key="cal_fallback")
                raw_brier = cal.get("raw_oof_brier", 0.185)
                cal_brier = cal.get("sigmoid_calibrated_oof_brier", 0.115)
                bc1, bc2 = st.columns(2)
                bc1.metric("Raw Brier Score", f"{raw_brier:.4f}")
                bc2.metric("Calibrated Brier Score", f"{cal_brier:.4f}", delta=f"−{raw_brier-cal_brier:.4f} improvement")

    # ── Segmentation Summary ──
    seg = results.get("segmentation", {})
    if seg:
        sec("K-Means Segmentation Summary")
        st.caption(f"Optimal k={seg['selected_k']} selected by silhouette score = {seg['silhouette']:.3f}")
        k_vals = [s["k"] for s in seg.get("candidate_silhouettes", [])]
        s_vals = [s["silhouette"] for s in seg.get("candidate_silhouettes", [])]
        if k_vals:
            fig_sil = go.Figure(go.Bar(
                x=[f"k={k}" for k in k_vals], y=s_vals,
                marker=dict(color=["#10b981" if k == seg["selected_k"] else "#475569" for k in k_vals], cornerradius=8),
                text=[f"{s:.3f}" for s in s_vals], textposition="outside",
                textfont=dict(size=12, color="#94a3b8"),
            ))
            fig_sil.update_layout(**_get_layout(h=260, yt="Silhouette Score"))
            st.markdown(
                '<div class="ch-wrap"><h4>Silhouette Score by Number of Clusters</h4>'
                '<div class="ch-sub">k=3 achieves highest silhouette — elbow method confirms</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_sil, use_container_width=True, key="sil_chart")


# ═══════════════════════════════════════
# PAGE 4 — METHODOLOGY
# ═══════════════════════════════════════
def pg_methodology(meta, raw):
    hero(
        "Methodology & End-to-End ML Pipeline",
        "Architectural specifications, preprocessing pipelines, class imbalance handling, and decision criteria.",
        "TECHNICAL METHODOLOGY & SPECS",
    )

    sec("Inference Pipeline Stages")
    steps = [
        ("01", "Source Data", "12,330 historical e-commerce sessions with 17 features & Revenue target."),
        ("02", "Deduplication", "125 exact duplicates eliminated → 12,205 verified unique sessions."),
        ("03", "Data Preprocessing", "log1p scaling for skewed numerics; OneHotEncoding for categoricals."),
        ("04", "Leakage Audit", "PageValues explicitly excluded to avoid live prediction leakage."),
        ("05", "XGBoost Engine", "Class-weighted booster (265 estimators, lr=0.037, max_depth=4)."),
        ("06", "Decision Layer", "Mapping to High/Medium/Low bands and cost-optimal R cutoff."),
    ]
    for i in range(0, 6, 3):
        cols = st.columns(3)
        for col, (n, t, d) in zip(cols, steps[i:i + 3]):
            with col:
                pstep(n, t, d)

    sec("Detailed Specifications")
    a1, a2 = st.columns(2)
    with a1:
        with st.expander("Dataset Distribution & Target", expanded=True):
            st.markdown(
                f"- **Data Repository:** UCI Online Shoppers (Sakar & Kastro, 2018)\n"
                f"- **Clean Records:** {len(raw):,} sessions\n"
                f"- **Purchasing Sessions:** {raw.Revenue.sum():,} ({raw.Revenue.mean():.1%})\n"
                f"- **Non-Purchasers:** {(~raw.Revenue.astype(bool)).sum():,} ({(1-raw.Revenue.mean()):.1%})"
            )
        with st.expander("Feature Preprocessing Pipeline"):
            st.markdown(
                "- **Skewed Numerics:** `log1p` transformation followed by `StandardScaler`\n"
                "- **Categorical Variables:** `OneHotEncoder(handle_unknown='ignore')`\n"
                "- **Passthrough Features:** `SpecialDay`, `Weekend`\n"
                "- **Deliberately Excluded:** `PageValues`"
            )
    with a2:
        with st.expander("XGBoost Classifier Parameters", expanded=True):
            st.dataframe(pd.DataFrame([
                {"Hyperparameter": "n_estimators", "Value": "265"},
                {"Hyperparameter": "learning_rate", "Value": "0.037"},
                {"Hyperparameter": "max_depth", "Value": "4"},
                {"Hyperparameter": "min_child_weight", "Value": "9"},
                {"Hyperparameter": "subsample", "Value": "0.741"},
                {"Hyperparameter": "colsample_bytree", "Value": "0.953"},
            ]), hide_index=True, use_container_width=True)

        with st.expander("Cost Minimization Criterion"):
            st.markdown(
                "Cost optimization formula on Out-Of-Fold predictions:\n\n"
                r"$$\text{Total Cost} = \text{False Positives} + R \times \text{False Negatives}$$"
                "\n\nWhere $R$ represents the cost penalty of missing a genuine purchaser relative to unnecessary intervention."
            )

    sec("Score Band Mapping")
    bands = meta["score_bands"]
    for col, (lbl, info, thr, clr) in zip(st.columns(3), [
        ("🟢 High Intent", bands["rates"]["High"], f"≥ {bands['high_cutoff']:.3f}", "#10b981"),
        ("🟡 Medium Intent", bands["rates"]["Medium"], f"≥ {bands['medium_cutoff']:.3f}", "#f59e0b"),
        ("🔵 Low Intent", bands["rates"]["Low"], f"< {bands['medium_cutoff']:.3f}", "#64748b"),
    ]):
        with col:
            st.markdown(
                f'<div class="vi-card">'
                f'<h4>{lbl}</h4>'
                f'<p style="font-size:0.95rem;font-weight:700;color:{clr};margin-bottom:0.35rem!important">Score {thr}</p>'
                f'<p>Sessions: {info["sessions"]:,} · Observed Conversion: {info["conversion_rate"]:.1%}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════
def pg_about(meta, raw):
    hero(
        "About the Visitor Intelligence Project",
        "A rigorous, leakage-aware decision-support machine learning solution for e-commerce engagement.",
        "PROJECT INFORMATION & CITATION",
    )

    sec("System Capabilities")
    o1, o2 = st.columns(2)
    with o1:
        vcard("🎯", "Business Objective", "Provide automated intent ranking for website browsing sessions to empower targeted personalization and discount optimization.")
    with o2:
        vcard("📊", "Cleaned UCI Dataset", "Trained and benchmarked on 12,205 verified session records from the UCI Machine Learning Repository.")

    st.markdown("")
    t1, t2 = st.columns(2)
    with t1:
        vcard("⚡", "Full Tech Stack", "Python 3.12 · scikit-learn · XGBoost · Streamlit · Plotly · Google Firebase Auth")
    with t2:
        vcard("🧠", "Preserved Model Safety", "Strict isolation guarantees: The app consumes pre-trained joblib weights and never retrains on startup.")

    sec("Known Limitations & Academic Scope")
    lims = [
        ("📅", "Temporal Single-Site Scope", "Dataset represents a single calendar year for one retailer without explicit microsecond timestamps."),
        ("📉", "Session Aggregates", "Features represent completed session sums rather than live clickstream sequences."),
        ("🔗", "PageValues Exclusion", "PageValues produces 0.76 PR-AUC but is tied to the post-purchase funnel; safely excluded here."),
        ("💰", "Assumed Cost Parameter", "The cost ratio R is illustrative for decision-theoretic trade-offs."),
        ("🎲", "Ranking Signal", "Model predictions are treated as relative ranking indices rather than strictly calibrated probabilities."),
        ("🔬", "Non-Causal Association", "Feature signals reflect behavioral correlations, not established causal mechanisms."),
    ]
    for i in range(0, 6, 3):
        cols = st.columns(3)
        for col, (ic, ti, de) in zip(cols, lims[i:i + 3]):
            with col:
                vcard(ic, ti, de)

    sec("Academic Citation")
    st.markdown(
        "> **UCI Online Shoppers Purchasing Intention Dataset**  \n"
        "> C. Sakar & S. O. Kastro (2018). *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository.  \n"
        "> DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)"
    )
    st.markdown("---")
    v1, v2, v3 = st.columns(3)
    v1.metric("Platform Version", f"v{VERSION}")
    v2.metric("Analyzed Sessions", f"{len(raw):,}")
    v3.metric("Deployment Pipeline", "XGBoost + sklearn")


# ═══════════════════════════════════════
# MAIN APP ROUTER & AUTH GATEWAY
# ═══════════════════════════════════════
try:
    model, metadata, results, dataset, input_limits = load_assets()
except Exception:
    LOGGER.exception("Critical error loading model assets")
    st.error("⚠️ **ML Artifacts Missing:** Ensure models and clean data files exist in the project repository.", icon="🚨")
    st.stop()

# ── Restore Authentication from Token (Across Browser Refreshes) ──
if "user" not in st.session_state:
    token_in_url = st.query_params.get("session")
    if token_in_url:
        restored = decode_session_token(token_in_url)
        if restored:
            st.session_state["user"] = restored
            st.session_state["logged_out"] = False
        else:
            if "session" in st.query_params:
                del st.query_params["session"]

# ── URL Query Param & Routing Synchronization ──
query_page = st.query_params.get("page", None)
if query_page and query_page.lower() in PAGE_SLUG_MAP:
    resolved_target = PAGE_SLUG_MAP[query_page.lower()]
    if "user" not in st.session_state and resolved_target in PROTECTED_PAGES:
        st.session_state["redirect_target"] = resolved_target
        st.session_state["navigation"] = "Login"
    else:
        st.session_state["navigation"] = resolved_target

# Default to "Home" for unauthenticated visitors, or "Overview" for logged-in users
if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Overview" if "user" in st.session_state else "Home"

current_page = st.session_state["navigation"]

# Guard Protected Pages
if current_page in PROTECTED_PAGES and "user" not in st.session_state:
    st.session_state["redirect_target"] = current_page
    current_page = "Login"
    st.session_state["navigation"] = "Login"

# ═══════════════════════════════════════
# 🌟 TOP NAVBAR (Rendered for Authenticated Workspace)
# ═══════════════════════════════════════
if current_page not in ("Home", "Login"):
    nav_pages = ["Overview", "Predict Intent", "Insights", "Methodology", "About", "Home"]
    nav_icons = {
        "Overview": "🏠", "Predict Intent": "⚡", "Insights": "📊",
        "Methodology": "🔬", "About": "ℹ️", "Home": "🌐"
    }

    with st.container():
        col_brand, col_nav, col_user = st.columns([1.8, 4.4, 2.0], vertical_alignment="center")

        with col_brand:
            st.markdown("""
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);width:40px;height:40px;
                    border-radius:12px;display:flex;align-items:center;justify-content:center;
                    font-size:18px;color:#fff;font-weight:900;box-shadow:0 4px 16px rgba(99,102,241,0.35);">VI</div>
                <div>
                    <div style="font-size:1.05rem;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;line-height:1.2;">Visitor Intelligence</div>
                    <div style="font-size:0.67rem;color:#818cf8;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;">AI Purchase Intent Engine</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_nav:
            if "top_main_segmented_control" not in st.session_state or st.session_state["top_main_segmented_control"] not in nav_pages:
                st.session_state["top_main_segmented_control"] = current_page

            st.segmented_control(
                "Top Navigation",
                options=nav_pages,
                format_func=lambda p: f"{nav_icons[p]} {p}",
                key="top_main_segmented_control",
                on_change=on_top_nav_change,
                label_visibility="collapsed",
            )

        with col_user:
            user = st.session_state.get("user")
            if user:
                u_info_col, u_btn_col = st.columns([1.4, 1.0], vertical_alignment="center")
                display_name = user.get("name") or user.get("email", "Analyst").split("@")[0]
                with u_info_col:
                    st.markdown(f"""
                    <div style="text-align:right;">
                        <div style="font-size:0.83rem;font-weight:700;color:#f1f5f9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{display_name}</div>
                        <div style="font-size:0.68rem;color:#94a3b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{user.get("email","")}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with u_btn_col:
                    st.button("🚪 Logout", key="top_navbar_logout_btn", on_click=do_logout, use_container_width=True)
            else:
                st.button("🔑 Sign In", key="top_navbar_signin_btn", on_click=nav_to, args=("Login",), use_container_width=True)

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(99,102,241,0.3) 0%,rgba(139,92,246,0.1) 60%,transparent 100%);margin:0.7rem 0 1.2rem;"></div>', unsafe_allow_html=True)

    # Subpage Breadcrumb
    if current_page != "Overview":
        c_crumb1, c_crumb2 = st.columns([4, 1.2], vertical_alignment="center")
        with c_crumb1:
            st.markdown(f"""
            <div class="crumb-bar">
                <div class="crumb-left">
                    <span>🏠 Dashboard</span>
                    <span style="color:#64748b;">›</span>
                    <span class="crumb-active">{nav_icons.get(current_page, '')} {current_page}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_crumb2:
            st.button("← Back to Dashboard", key="btn_crumb_back", on_click=nav_to, args=("Overview",), use_container_width=True)

    # Sidebar Companion Controls
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.75rem 0 0.5rem;">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);width:46px;height:46px;
                border-radius:14px;display:inline-flex;align-items:center;justify-content:center;
                font-size:20px;color:#fff;font-weight:900;letter-spacing:-0.02em;
                box-shadow:0 6px 20px rgba(99,102,241,0.35);margin-bottom:8px;">VI</div>
            <div style="font-size:1.02rem;font-weight:800;color:#f8fafc;letter-spacing:-0.02em;">Control Sidebar</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        user = st.session_state.get("user")
        if user:
            display_name = user.get("name") or user.get("email", "Analyst").split("@")[0]
            provider_type = user.get("provider", "password")
            icon_map = {"google.com": "🔵 Google OAuth", "password": "📧 Firebase Auth", "demo": "⚡ Guest Demo"}
            prov_badge = icon_map.get(provider_type, "🔑 Authenticated")

            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                border-radius:14px;padding:0.85rem 1rem;margin-bottom:0.7rem;">
                <div style="font-size:0.88rem;font-weight:700;color:#f1f5f9;">{display_name}</div>
                <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;word-break:break-all;">{user.get("email","")}</div>
                <div style="display:inline-block;margin-top:6px;font-size:0.65rem;font-weight:700;color:#a5b4fc;
                    background:rgba(99,102,241,0.18);padding:2px 8px;border-radius:99px;">{prov_badge}</div>
            </div>
            """, unsafe_allow_html=True)

            st.button("🚪 Logout (Sidebar)", key="sidebar_signout", on_click=do_logout, use_container_width=True)
        else:
            st.button("🔑 Sign In", key="sidebar_signin", on_click=nav_to, args=("Login",), use_container_width=True)

        st.markdown("---")
        st.markdown("<p style='font-size:0.8rem;font-weight:700;color:#cbd5e1;margin-bottom:4px;'>Cost Sensitivity Scenario</p>", unsafe_allow_html=True)
        ratio = st.select_slider(
            "Cost Scenario (R)",
            options=[3, 5, 10],
            value=5,
            help="Assumed cost ratio of missed buyer vs unnecessary intervention.",
            label_visibility="collapsed",
        )
        current_thr = metadata["cost_thresholds"][str(ratio)]
        st.metric("Intervention Cutoff", f"{current_thr:.2f}", help=f"Sessions scoring ≥ {current_thr:.2f} are flagged at R={ratio}.")

        st.markdown("---")
        st.caption(f"**Model** · Class-Weighted XGBoost  \n**Dataset** · UCI Online Shoppers (12,205 rows)  \n**Status** · 🟢 Online · v{VERSION}")
else:
    ratio = 5

# ── Dynamic Route Dispatcher ──
if current_page == "Home":
    pg_home(dataset, metadata, results)
elif current_page == "Login":
    show_auth_page()
elif current_page == "Overview":
    pg_overview(model, metadata, results, dataset)
elif current_page == "Predict Intent":
    pg_predict(model, metadata, input_limits, ratio, results)
elif current_page == "Insights":
    pg_insights(metadata, results, dataset)
elif current_page == "Methodology":
    pg_methodology(metadata, dataset)
else:
    pg_about(metadata, dataset)

# ── Global Application Footer ──
st.markdown(
    '<div class="app-ft">'
    '<strong>Website Visitor Engagement Analytics Platform</strong> · AI-Powered Purchase Intent Engine<br>'
    'Benchmarked on <a href="https://archive.ics.uci.edu/dataset/468/online" target="_blank">UCI Online Shoppers Dataset</a> '
    '· Sakar &amp; Kastro (2018) · Deliberately leakage-aware production deployment · v' + VERSION +
    '</div>',
    unsafe_allow_html=True,
)
