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

# ─── Design System & CSS Tokens (v3) ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Tokens ── */
:root{
    --bg:        #0b0f1a;
    --surface:   #111827;
    --card:      #161d2e;
    --card-h:    #1c2540;
    --border:    #1f2a44;
    --bdr-act:   rgba(99,102,241,.45);
    --primary:   #6366f1;
    --pdim:      rgba(99,102,241,.12);
    --success:   #34d399;
    --warning:   #fbbf24;
    --danger:    #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --soft:      #94a3b8;
    --r-sm:      8px;
    --r-md:      12px;
    --r-lg:      18px;
    --r-pill:    9999px;
    --sh:        0 4px 24px rgba(0,0,0,.4);
    --sh-lg:     0 12px 40px rgba(0,0,0,.55);
}

/* ── Base ── */
html,body,.stApp,[data-testid="stAppViewContainer"]{
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif!important;
    background:var(--bg)!important;
    color:var(--text)!important;
}
#MainMenu,footer,[data-testid="stToolbar"],
header[data-testid="stHeader"],.stDeployButton,
[data-testid="manage-app-button"],[data-testid="stStatusWidget"]{
    display:none!important;visibility:hidden!important;height:0!important;
}
.block-container{max-width:1380px!important;padding:.5rem 2rem 4rem!important;}

/* ── PUBLIC NAV ── */
.pub-nav{
    display:flex;align-items:center;justify-content:space-between;
    padding:.65rem 1.4rem;
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--r-lg);
    margin-bottom:1.8rem;
    box-shadow:var(--sh);
}
.brand-group{display:flex;align-items:center;gap:10px;}
.brand-icon{
    width:36px;height:36px;border-radius:9px;
    background:var(--primary);
    display:flex;align-items:center;justify-content:center;
    font-size:14px;font-weight:800;color:#fff;
}
.brand-title{font-size:.95rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;}
.brand-sub{font-size:.60rem;font-weight:600;color:#818cf8;text-transform:uppercase;letter-spacing:.06em;}

/* ── AUTH NAV TOP NAVBAR (authenticated pages) ── */
.top-navbar{
    display:flex;align-items:center;
    padding:.5rem 1.2rem;
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--r-lg);
    margin-bottom:1.1rem;
    box-shadow:var(--sh);
}
.nb-brand{display:flex;align-items:center;gap:8px;flex-shrink:0;}
.nb-logo{
    width:32px;height:32px;border-radius:8px;
    background:var(--primary);
    display:flex;align-items:center;justify-content:center;
    font-size:12px;font-weight:800;color:#fff;flex-shrink:0;
}
.nb-title{font-size:.88rem;font-weight:700;color:#f1f5f9;white-space:nowrap;}
.nb-sub{font-size:.58rem;font-weight:600;color:#818cf8;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;}
.nb-sep{width:1px;height:24px;background:var(--border);flex-shrink:0;margin:0 8px;}
.nb-user{display:flex;align-items:center;gap:7px;flex-shrink:0;margin-left:auto;}
.nb-avatar{
    width:28px;height:28px;border-radius:50%;
    background:linear-gradient(135deg,#6366f1,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    font-size:11px;font-weight:700;color:#fff;flex-shrink:0;
}
.nb-uname{font-size:.78rem;font-weight:600;color:#e2e8f0;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.nb-uemail{font-size:.63rem;color:var(--muted);max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.nb-badge{
    font-size:.58rem;font-weight:600;padding:2px 7px;
    border-radius:var(--r-pill);
    background:var(--pdim);color:#a5b4fc;
    border:1px solid rgba(99,102,241,.2);white-space:nowrap;
}

/* ── HERO ── */
.hero-wrapper{
    background:linear-gradient(135deg,#0f1629 0%,#151d3a 60%,#111828 100%);
    border:1px solid var(--border);
    border-left:3px solid var(--primary);
    border-radius:var(--r-lg);
    padding:2.6rem 2.8rem;
    margin-bottom:1.8rem;
    overflow:hidden;
    box-shadow:var(--sh-lg);
}
.pill-badge{
    display:inline-flex;align-items:center;gap:6px;
    background:var(--pdim);color:#a5b4fc;
    padding:.28rem .8rem;border-radius:var(--r-pill);
    font-size:.68rem;font-weight:700;letter-spacing:.10em;text-transform:uppercase;
    margin-bottom:.9rem;border:1px solid rgba(99,102,241,.25);
}
.hero-headline{
    color:#f8fafc!important;font-size:2.3rem!important;font-weight:800!important;
    letter-spacing:-.03em;line-height:1.22;margin:0 0 .85rem;max-width:760px;
}
.hero-description{
    color:var(--soft);font-size:.97rem;line-height:1.65;max-width:660px;margin:0 0 1.3rem;
}

/* ── PREVIEW CARD ── */
.preview-card{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:1.3rem 1.5rem;box-shadow:var(--sh-lg);
}
.preview-status{
    display:flex;align-items:center;justify-content:space-between;
    margin-bottom:.85rem;padding-bottom:.5rem;border-bottom:1px solid var(--border);
}

/* ── SECTION HEADER ── */
.sec-hd{display:flex;align-items:center;gap:.6rem;margin:2rem 0 .9rem;}
.sec-hd .dot{width:3px;height:18px;border-radius:99px;background:var(--primary);flex-shrink:0;}
.sec-hd h3{
    margin:0!important;font-size:1.05rem!important;font-weight:700!important;
    color:#f1f5f9!important;letter-spacing:-.02em;
}

/* ── CARDS ── */
.vi-card{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:1.25rem 1.35rem;
    box-shadow:var(--sh);transition:border-color .2s,transform .2s;height:100%;
}
.vi-card:hover{border-color:var(--bdr-act);transform:translateY(-2px);}
.vi-card .ic{font-size:1.5rem;margin-bottom:.55rem;display:block;}
.vi-card h4{color:#f1f5f9!important;font-size:.93rem!important;font-weight:700!important;margin:0 0 .35rem!important;}
.vi-card p{color:var(--soft)!important;font-size:.82rem!important;line-height:1.5!important;margin:0!important;}

/* ── STEP CARDS ── */
.step-card{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:1.25rem 1.35rem;
    position:relative;box-shadow:var(--sh);
    transition:border-color .2s,transform .2s;height:100%;
}
.step-card:hover{border-color:var(--bdr-act);transform:translateY(-2px);}
.step-num{
    position:absolute;top:-10px;left:14px;
    background:var(--primary);color:#fff;
    font-size:.62rem;font-weight:800;padding:.14rem .55rem;border-radius:var(--r-pill);
}
.step-card h4{color:#f1f5f9!important;font-size:.92rem!important;font-weight:700!important;margin:.32rem 0 .32rem!important;}
.step-card p{color:var(--soft)!important;font-size:.80rem!important;line-height:1.45!important;margin:0!important;}

/* ── SAMPLE SCENARIO CARDS ── */
.smp-card{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:.9rem 1rem;
    margin-bottom:.5rem;min-height:72px;
}
.smp-card h4{color:#f1f5f9!important;font-size:.85rem!important;font-weight:700!important;margin:0 0 .22rem!important;}
.smp-card p{color:var(--muted)!important;font-size:.75rem!important;margin:0!important;line-height:1.4!important;}

/* ── PREDICTION RESULT ── */
.int-res{
    border-radius:var(--r-lg);padding:1.7rem 2rem;
    border:1px solid var(--border);
    position:relative;overflow:hidden;margin:1rem 0 1.2rem;box-shadow:var(--sh);
}
.int-res::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.int-res.high  {background:rgba(52,211,153,.05); border-color:rgba(52,211,153,.25);}
.int-res.medium{background:rgba(251,191,36,.05);  border-color:rgba(251,191,36,.25);}
.int-res.low   {background:rgba(100,116,139,.05); border-color:rgba(100,116,139,.25);}
.int-res.high::before  {background:var(--success);}
.int-res.medium::before{background:var(--warning);}
.int-res.low::before   {background:var(--soft);}
.int-tag{
    display:inline-block;padding:.18rem .65rem;
    border-radius:var(--r-pill);font-size:.62rem;font-weight:700;
    letter-spacing:.10em;text-transform:uppercase;margin-bottom:.5rem;
}
.int-res.high .int-tag  {background:rgba(52,211,153,.12); color:var(--success);}
.int-res.medium .int-tag{background:rgba(251,191,36,.12);  color:var(--warning);}
.int-res.low .int-tag   {background:rgba(100,116,139,.12); color:var(--soft);}
.int-band{font-size:1.75rem;font-weight:800;letter-spacing:-.02em;margin-bottom:.3rem;line-height:1.1;}
.int-res.high .int-band  {color:var(--success);}
.int-res.medium .int-band{color:var(--warning);}
.int-res.low .int-band   {color:var(--soft);}
.int-desc {color:var(--soft);font-size:.91rem;margin:.28rem 0 .6rem;line-height:1.5;}
.int-score{color:var(--muted);font-size:.78rem;}

/* ── METRICS ── */
[data-testid="stMetric"]{
    background:var(--card)!important;border:1px solid var(--border)!important;
    border-radius:var(--r-md)!important;padding:1rem 1.2rem!important;
    box-shadow:var(--sh)!important;transition:border-color .2s,transform .2s!important;
}
[data-testid="stMetric"]:hover{border-color:var(--bdr-act)!important;transform:translateY(-2px);}
[data-testid="stMetricLabel"] p{color:var(--soft)!important;font-size:.73rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:.05em!important;}
[data-testid="stMetricValue"] div{color:#f1f5f9!important;font-weight:800!important;}

/* ── CHART WRAPPERS ── */
.ch-wrap{
    background:var(--card);border:1px solid var(--border);
    border-radius:var(--r-md);padding:1.05rem 1.15rem .45rem;
    box-shadow:var(--sh);margin-bottom:.65rem;
}
.ch-wrap h4{color:#f1f5f9!important;font-size:.88rem!important;font-weight:700!important;margin:0 0 .1rem!important;}
.ch-wrap .ch-sub{color:var(--muted);font-size:.73rem;margin:0 0 .4rem;}

/* ── BUTTONS ── */
.stButton>button,.stFormSubmitButton>button{
    border-radius:var(--r-sm)!important;
    font-family:'Inter',sans-serif!important;font-weight:600!important;
    font-size:.87rem!important;transition:all .18s ease!important;
}
.stFormSubmitButton>button,.stButton>button[kind="primary"]{
    background:var(--primary)!important;border:none!important;
    color:#fff!important;min-height:2.7rem!important;
    font-size:.93rem!important;font-weight:700!important;
}
.stFormSubmitButton>button:hover,.stButton>button[kind="primary"]:hover{
    background:#5254cc!important;transform:translateY(-1px)!important;
}
.stButton>button:not([kind="primary"]){
    background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;
}
.stButton>button:not([kind="primary"]):hover{
    border-color:var(--primary)!important;color:#a5b4fc!important;transform:translateY(-1px)!important;
}

/* ── BREADCRUMB ── */
.crumb-bar{
    display:flex;align-items:center;
    background:var(--surface);border:1px solid var(--border);
    border-radius:var(--r-md);padding:.5rem 1rem;margin-bottom:.9rem;
}
.crumb-left{display:flex;align-items:center;gap:7px;font-size:.80rem;color:var(--soft);}
.crumb-active{color:#a5b4fc;font-weight:600;}

/* ── SIDEBAR ── */
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}

/* ── TABS ── */
[data-testid="stTabs"] [role="tablist"]{
    background:var(--surface)!important;border-radius:10px!important;
    padding:4px!important;gap:4px!important;border:1px solid var(--border)!important;
}
[data-testid="stTabs"] [role="tab"]{border-radius:7px!important;font-weight:600!important;font-size:.85rem!important;}

/* ── DATAFRAME / EXPANDER ── */
[data-testid="stDataFrame"]{border:1px solid var(--border)!important;border-radius:var(--r-md)!important;overflow:hidden;}
[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:var(--r-md)!important;background:var(--card)!important;}

/* ── FOOTER ── */
.app-ft{
    text-align:center;color:var(--muted);font-size:.74rem;
    padding:2.2rem 0 1rem;border-top:1px solid var(--border);
    margin-top:3rem;line-height:1.8;
}
.app-ft a{color:#818cf8;text-decoration:none;font-weight:500;}

/* ═══════════════════════════════════
   PROFESSIONAL NAVBAR OVERRIDE
   Makes st.columns-based nav look like
   a real top navigation bar
═══════════════════════════════════ */

/* Hide Streamlit's own top padding so our nav sits flush */
.block-container { padding-top: 0.2rem !important; }

/* Nav link buttons — reset to plain text style */
div[data-testid="column"].nav-col > div > div > div > button {
    background: transparent !important;
    border: none !important;
    color: var(--soft) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.5rem !important;
    min-height: unset !important;
    border-radius: 6px !important;
    width: 100% !important;
    transition: color 0.15s, background 0.15s !important;
    box-shadow: none !important;
}
div[data-testid="column"].nav-col > div > div > div > button:hover {
    background: var(--pdim) !important;
    color: #e2e8f0 !important;
    transform: none !important;
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
    expected = set(FEATURE_COLUMNS)
    present = set(getattr(model, "feature_names_in_", []))
    if not expected.issubset(present):
        raise ValueError("Model and feature schema do not match.")
    transformer = model.named_steps.get("prep")
    if transformer is None:
        raise ValueError("Model is missing preprocessing pipeline.")
    selected_features = set()
    for name, _, cols in transformer.transformers:
        if isinstance(cols, (list, tuple, np.ndarray)):
            selected_features.update(cols)
    if "PageValues" in selected_features:
        raise ValueError("Deployment artifact unexpectedly uses PageValues.")
    if selected_features != set(FEATURE_COLUMNS):
        raise ValueError("Unexpected prediction feature schema.")
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
    # ── Professional Public Navbar ──
    # HTML shell (brand on left, nav links text in centre, CTA on right)
    st.markdown("""
    <div style="
        display:flex; align-items:center; justify-content:space-between;
        padding:.6rem 1.4rem;
        background:var(--surface);
        border:1px solid var(--border);
        border-radius:14px;
        margin-bottom:1.5rem;
        box-shadow:0 2px 12px rgba(0,0,0,.3);
    ">
        <div style="display:flex;align-items:center;gap:9px;">
            <div style="width:34px;height:34px;border-radius:9px;background:var(--primary);
                display:flex;align-items:center;justify-content:center;
                font-size:13px;font-weight:800;color:#fff;">VI</div>
            <div>
                <div style="font-size:.9rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;">Visitor Intelligence</div>
                <div style="font-size:.58rem;color:#818cf8;font-weight:600;text-transform:uppercase;letter-spacing:.07em;">Purchase Intent Engine</div>
            </div>
        </div>
        <div style="color:var(--muted);font-size:.75rem;">
            Navigate using the links below &nbsp;▾
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Streamlit buttons row (nav links + CTAs)
    pc1, pc2, pc3, pc4, _sp, pc5, pc6 = st.columns([.9, 1.1, .8, .7, 1.2, 1.0, 1.2], vertical_alignment="center")
    pc1.button("Home",        key="pnav_home",  on_click=nav_to, args=("Home",),        use_container_width=True)
    pc2.button("Methodology", key="pnav_meth",  on_click=nav_to, args=("Methodology",), use_container_width=True)
    pc3.button("Insights",    key="pnav_ins",   on_click=nav_to, args=("Insights",),    use_container_width=True)
    pc4.button("About",       key="pnav_about", on_click=nav_to, args=("About",),       use_container_width=True)
    # spacer column intentionally empty
    if "user" in st.session_state:
        pc5.button("Sign out",  key="pnav_logout", on_click=do_logout, use_container_width=True)
        pc6.button("Open dashboard", key="pnav_dash", on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)
    else:
        pc5.button("Sign in",    key="pnav_login",    on_click=nav_to, args=("Login",),    use_container_width=True)
        pc6.button("Get started",key="pnav_getstart", on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)
    st.markdown("<div style='height:1px;background:var(--border);margin:.5rem 0 1.4rem;'></div>",
                unsafe_allow_html=True)

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

    sec("Key Benchmark Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    total = len(raw)
    pr = float(raw["Revenue"].mean())
    m1.metric("Total Sessions", f"{total:,}")
    m2.metric("Base Purchase Rate", f"{pr:.1%}")
    m3.metric("Live Predictors", f"{len(FEATURE_COLUMNS)}")
    m4.metric("Model Architecture", "XGBoost")
    m5.metric("Held-Out PR-AUC", "0.376")

    sec("Conversion & Traffic Analytics")
    ch1, ch2 = st.columns(2)
    with ch1:
        pc = int(raw["Revenue"].sum())
        npc = total - pc
        fig = _donut(["Non-Purchaser", "Purchaser"], [npc, pc], ["#1e293b", "#6366f1"])
        fig.update_layout(annotations=[dict(
            text=f"<b>{pr:.1%}</b><br><span style='font-size:11px;color:#94a3b8'>CONVERSION</span>",
            x=0.5, y=0.5, font_size=22, font_color="#f8fafc", showarrow=False,
        )])
        st.markdown(
            f'<div class="ch-wrap"><h4>Purchase Conversion Distribution</h4>'
            f'<div class="ch-sub">{pc:,} converting sessions out of {total:,} total analyzed</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d1")

    with ch2:
        mo = meta["observed_months"]
        mr = raw.groupby("Month")["Revenue"].mean().reindex(mo).dropna()
        fig = _monthly_spline_chart(mr.index.tolist(), mr.values.tolist(), 320)
        st.markdown(
            '<div class="ch-wrap"><h4>Observed Monthly Conversion Trend</h4>'
            '<div class="ch-sub">Smooth spline visualization of conversion probability across active months</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d2")

    ch3, ch4 = st.columns(2)
    with ch3:
        vr = results.get("visitor_rates", {})
        if vr:
            fig = _bar(
                [k.replace("_", " ") for k in vr],
                [v["count"] for v in vr.values()],
                "#8b5cf6", "Sessions", 300, tfmt=",",
            )
            st.markdown(
                '<div class="ch-wrap"><h4>Visitor Distribution by Cohort</h4>'
                '<div class="ch-sub">New visitors convert at 24.9% vs returning at 14.1%</div></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True, key="d3")
    with ch4:
        bd = meta["score_bands"]["rates"]
        fig = go.Figure(go.Bar(
            x=list(bd.keys()),
            y=[v["sessions"] for v in bd.values()],
            marker=dict(color=["#10b981", "#f59e0b", "#64748b"], cornerradius=8),
            text=[f'{v["sessions"]:,}' for v in bd.values()],
            textposition="outside",
            textfont=dict(size=12, color="#cbd5e1"),
        ))
        fig.update_layout(**_get_layout(h=300, yt="Sessions", showlegend=False))
        st.markdown(
            '<div class="ch-wrap"><h4>5-Fold Cross-Fitted Intent Bands</h4>'
            '<div class="ch-sub">High Intent (33.2% conv) · Medium (14.8%) · Low (3.1%)</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="d4")

    sec("Strategic Machine Learning Takeaways")
    i1, i2, i3 = st.columns(3)
    with i1:
        vcard("📦", "Product Duration is Top Signal",
              "Time spent browsing product pages and low exit velocity provide the sharpest indicators of impending transaction.")
    with i2:
        vcard("🛡️", "Leakage-Aware Production Model",
              "PageValues was deliberately stripped out. While giving inflated 0.76 PR-AUC, it causes severe lookahead leakage in production.")
    with i3:
        vcard("🎯", "Ranking Over Raw Probabilities",
              "The pipeline prioritizes ranking sessions to optimize targeting interventions under operational cost budgets (R=3, 5, 10).")


# ═══════════════════════════════════════
# PAGE 2 — PREDICT INTENT
# ═══════════════════════════════════════
def pg_predict(model, meta, limits, ratio, results, raw: pd.DataFrame | None = None):
    hero(
        "Real-Time Purchase Intent Predictor",
        "Score active sessions directly through the production XGBoost pipeline. "
        "Excludes PageValues to ensure legitimate, leakage-free operational decision support.",
        "PREDICTIVE INFERENCE ENGINE",
    )
    st.info(
        "🛡️ **Leakage Protection Enabled:** `PageValues` is omitted to preserve production realism. "
        "Bounce & exit rates are derived automatically from real dataset statistics — they cannot be entered manually.",
        icon="🛡️",
    )

    tab1, tab2 = st.tabs(["⚡ Dynamic Session Scorer", "📄 CSV Batch Scoring Engine"])
    with tab1:
        _single_predict_view(model, meta, limits, ratio, raw)
    with tab2:
        _batch_predict_view(model, meta, limits, ratio)


def _derive_session_features(
    browsing_depth: str,
    avg_time_per_product: float,
    admin_pages: int,
    info_pages: int,
    is_focused: bool,
    raw: pd.DataFrame,
) -> dict[str, float]:
    """
    Derive behavioral session aggregates from high-level session journey inputs.
    Bounce and exit rates are derived from real dataset statistics for the
    corresponding browsing depth bucket — the user cannot inject arbitrary values.
    """
    # Map browsing depth labels to product page count ranges (sampled from dataset percentiles)
    depth_map = {
        "Just arrived (1–2 pages)":       (1,   2),
        "Light browsing (3–6 pages)":      (3,   6),
        "Active browsing (7–15 pages)":    (7,  15),
        "Deep research (16–30 pages)":     (16, 30),
        "Intensive session (31+ pages)":   (31, 999),
    }
    low, high = depth_map[browsing_depth]
    # Use midpoint deterministically; cap at the dataset observed maximum
    display_high = min(high, 45)  # dataset max for ProductRelated is ~705 but representative cap is 45
    prod_count = int(round((low + min(high, 45)) / 2))
    prod_count = max(prod_count, low)  # ensure at least the lower bound
    prod_duration = round(prod_count * avg_time_per_product, 1)

    admin_duration = round(admin_pages * 45.0, 1)   # ~45 sec avg admin page dwell
    info_duration  = round(info_pages  * 60.0, 1)   # ~60 sec avg info page dwell

    # Derive bounce & exit rates from the real dataset's statistics
    # for sessions in the same product-page-count bucket.
    # This prevents gamification — rates come from observed data, not user input.
    mask = (raw["ProductRelated"] >= low) & (raw["ProductRelated"] <= high)
    bucket = raw[mask]
    if len(bucket) >= 10:
        bounce_rate = float(bucket["BounceRates"].median())
        exit_rate   = float(bucket["ExitRates"].median())
    else:
        # Fallback to full-dataset median if bucket is too thin
        bounce_rate = float(raw["BounceRates"].median())
        exit_rate   = float(raw["ExitRates"].median())

    # Focused sessions (add-to-cart intent) have lower exit velocity
    if is_focused:
        exit_rate   = round(exit_rate   * 0.55, 5)
        bounce_rate = round(bounce_rate * 0.45, 5)

    return {
        "Administrative":            admin_pages,
        "Administrative_Duration":   admin_duration,
        "Informational":             info_pages,
        "Informational_Duration":    info_duration,
        "ProductRelated":            prod_count,
        "ProductRelated_Duration":   prod_duration,
        "BounceRates":               round(bounce_rate, 5),
        "ExitRates":                 round(exit_rate,   5),
    }


def _single_predict_view(model, meta, limits, ratio, raw: pd.DataFrame | None = None):  # type: ignore[assignment]
    # raw is passed in from load_assets; fall back gracefully
    if raw is None:
        st.error("Dataset unavailable — cannot derive behavioral rates.", icon="❌")
        return

    st.markdown("""
    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.22);
        border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.1rem;">
    <strong style="color:#a5b4fc;">How dynamic scoring works:</strong>
    <span style="color:#94a3b8;font-size:0.88rem;">
      Describe the visitor's session journey using the controls below.
      Bounce &amp; exit rates are <em>automatically derived from observed dataset statistics</em>
      for that browsing depth — you cannot manually enter them, preventing
      inflated or cherry-picked scores.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Section A: Session Context (knowable from server/GA at session start) ──
    sec("1. Session Context  ·  Auto-detected from environment")
    st.caption("These values come from the visitor's browser, referral channel, and calendar — not their browsing behaviour.")

    ctx1, ctx2, ctx3 = st.columns(3)
    month   = ctx1.selectbox("Month of Visit", meta["observed_months"],
                              index=meta["observed_months"].index("Nov") if "Nov" in meta["observed_months"] else 0,
                              key="dyn_Month",
                              help="Calendar month of the session. Automatically from server timestamp.")
    visitor = ctx2.selectbox("Visitor Cohort",
                              meta["visitor_types"],
                              format_func=lambda x: x.replace("_", " "),
                              key="dyn_VisitorType",
                              help="New, Returning, or Other — from browser cookie / GA.")
    weekend = ctx3.selectbox("Session Timing",
                              [0, 1],
                              format_func=lambda x: "Weekend" if x else "Weekday",
                              key="dyn_Weekend",
                              help="Automatically derived from server date.")

    tech     = meta["technical_category_options"]
    tc1, tc2, tc3, tc4 = st.columns(4)
    os_code  = tc1.selectbox("OS",      tech["OperatingSystems"], key="dyn_OS",
                              help="Detected from User-Agent header.")
    br_code  = tc2.selectbox("Browser", tech["Browser"],          key="dyn_Browser",
                              help="Detected from User-Agent header.")
    reg_code = tc3.selectbox("Region",  tech["Region"],           key="dyn_Region",
                              help="Derived from visitor IP geolocation.")
    trf_code = tc4.selectbox("Traffic Source", tech["TrafficType"], key="dyn_Traffic",
                              help="Traffic channel code from UTM parameters / GA.")

    special  = ctx1.select_slider("Special Day Proximity",
                                  options=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                  value=0.0,
                                  key="dyn_SpecialDay",
                                  help="0 = normal day, 1 = special day (e.g. Valentine's). Comes from calendar.")

    # ── Section B: Session Journey (describes what the visitor DID) ──
    sec("2. Session Journey  ·  Describe what the visitor browsed")
    st.caption("Select the visitor's browsing pattern. Bounce & exit rates are computed automatically from real data statistics — they cannot be entered manually.")

    depth_options = [
        "Just arrived (1–2 pages)",
        "Light browsing (3–6 pages)",
        "Active browsing (7–15 pages)",
        "Deep research (16–30 pages)",
        "Intensive session (31+ pages)",
    ]

    j1, j2 = st.columns([1.4, 1.0])
    with j1:
        browsing_depth = st.select_slider(
            "Product Browsing Depth",
            options=depth_options,
            value="Active browsing (7–15 pages)",
            key="dyn_depth",
            help="How many product pages did the visitor view this session?",
        )
    with j2:
        avg_time = st.slider(
            "Avg. Time per Product Page (sec)",
            min_value=5,
            max_value=300,
            value=45,
            step=5,
            key="dyn_avg_time",
            help="Estimated average dwell time per product page. Longer = more engaged.",
        )

    j3, j4, j5 = st.columns(3)
    admin_pages = j3.number_input(
        "Admin Pages Visited",
        min_value=0, max_value=20, value=0, step=1,
        key="dyn_admin",
        help="Account, cart, checkout-related pages. Typically 0–3 for normal browsing.",
    )
    info_pages = j4.number_input(
        "Info / Help Pages Visited",
        min_value=0, max_value=10, value=0, step=1,
        key="dyn_info",
        help="FAQ, About, Policy pages. Typically 0–2.",
    )
    is_focused = j5.toggle(
        "Focused / Add-to-cart intent",
        value=False,
        key="dyn_focused",
        help="Enable if the visitor showed strong buying intent (e.g., added to cart, visited checkout). "
             "Reduces exit/bounce rates by ~45–55% to match observed buyer behaviour.",
    )

    # ── Live Feature Preview ──
    derived = _derive_session_features(
        browsing_depth, float(avg_time),
        int(admin_pages), int(info_pages),
        bool(is_focused), raw,
    )

    with st.expander("🔍 View auto-derived behavioral features", expanded=False):
        st.caption("These values are computed from your journey inputs and real dataset statistics — not entered by hand.")
        prev_cols = st.columns(4)
        prev_cols[0].metric("Product Pages", f"{derived['ProductRelated']}")
        prev_cols[1].metric("Product Duration", f"{derived['ProductRelated_Duration']:.0f}s")
        prev_cols[2].metric("Bounce Rate (derived)", f"{derived['BounceRates']:.4f}")
        prev_cols[3].metric("Exit Rate (derived)",   f"{derived['ExitRates']:.4f}")
        sub_cols = st.columns(4)
        sub_cols[0].metric("Admin Pages",    f"{derived['Administrative']}")
        sub_cols[1].metric("Admin Duration", f"{derived['Administrative_Duration']:.0f}s")
        sub_cols[2].metric("Info Pages",     f"{derived['Informational']}")
        sub_cols[3].metric("Info Duration",  f"{derived['Informational_Duration']:.0f}s")
        # compute bucket size for caption
        depth_map_local = {
            "Just arrived (1–2 pages)":       (1,  2),
            "Light browsing (3–6 pages)":      (3,  6),
            "Active browsing (7–15 pages)":    (7, 15),
            "Deep research (16–30 pages)":     (16, 30),
            "Intensive session (31+ pages)":   (31, 999),
        }
        b_lo, b_hi = depth_map_local[browsing_depth]
        bucket_size = int(((raw["ProductRelated"] >= b_lo) & (raw["ProductRelated"] <= b_hi)).sum())
        st.caption(f"Bounce & exit rates are medians from {bucket_size:,} real sessions with similar product browsing depth.")

    # ── Score Button ──
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🔮  Compute Purchase Intent Score", type="primary", use_container_width=True, key="dyn_score_btn"):
        row = {
            **derived,
            "SpecialDay":     special,
            "Month":          month,
            "VisitorType":    visitor,
            "Weekend":        weekend,
            "OperatingSystems": os_code,
            "Browser":          br_code,
            "Region":           reg_code,
            "TrafficType":      trf_code,
        }
        try:
            with st.spinner("Executing pipeline transformation and XGBoost scoring…"):
                clean = validate_frame(pd.DataFrame([row]), meta, limits)
                result = score_sessions(model, clean, meta, ratio).iloc[0].to_dict()
                st.session_state["last_result"] = result
                st.session_state["last_input"]  = row
        except InputValidationError as exc:
            st.error(str(exc), icon="⚠️")
        except Exception:
            LOGGER.exception("Prediction failed")
            st.error("Error executing model inference. Please verify inputs.", icon="❌")

    if "last_result" in st.session_state:
        _render_prediction_output(
            st.session_state["last_result"],
            st.session_state.get("last_input", {}),
            ratio, meta, limits,
        )


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
    b1, _ = st.columns([1.2, 2.8])
    b1.download_button("📥 Download Starter CSV Template", tmpl, "visitor_batch_template.csv", "text/csv", use_container_width=True)

    uploaded = st.file_uploader("Upload CSV sessions file", type=["csv"], key="batch_csv_file")
    if not uploaded:
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
        s4.metric("Flagged for Action", f"{int(output['intervention_flag'].sum()):,}")

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
        "feature importance diagnostics, and PageValues ablation analysis.",
        "BENCHMARKS & DIAGNOSTICS",
    )
    st.info("📊 **PR-AUC is Primary:** With only ~15.6% positive conversion prevalence, precision-recall AUC is the decisive metric.", icon="📊")

    sec("Held-Out Test Results (Leakage-Free Deployed Model)")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("PR-AUC", "0.376")
    s2.metric("ROC-AUC", "0.795")
    s3.metric("Precision (R=5)", "0.311")
    s4.metric("Recall (R=5)", "0.728")
    s5.metric("Flagged Share", "36.7%")

    sec("PageValues Ablation Study")
    ab1, ab2 = st.columns([1.3, 0.7])
    with ab1:
        fig = _grouped(
            ["PR-AUC", "ROC-AUC"],
            {"With PageValues (Leaky)": [0.7577, 0.9378], "Without PageValues (Deployed)": [0.3762, 0.7949]},
            ["#8b5cf6", "#475569"], "Metric Score", 350,
        )
        st.markdown(
            '<div class="ch-wrap"><h4>Held-Out Test Set Ablation</h4>'
            '<div class="ch-sub">Excluding PageValues prevents false lookahead inflation</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="pv_ab")
    with ab2:
        vcard("🛡️", "Leakage-Aware Engineering",
              "PageValues reflects revenue generated by pages viewed prior to purchase. In live production at session start, this value is unavailable. The deployed model uses only genuine session behavior.")
        st.markdown("")
        st.metric("Ablation Differential", "−0.382 PR-AUC", delta="-50.4% Signal (Honest Model)", delta_color="inverse")

    sec("Matched-Settings Cross-Validation (5-Fold)")
    comp = results.get("matched_model_comparison", [])
    if comp:
        fig = _grouped(
            ["Logistic Regression", "Random Forest", "XGBoost"],
            {
                "With PageValues": [r["mean_pr_auc"] for r in comp if r["feature_set"] == "with PageValues"],
                "Without PageValues": [r["mean_pr_auc"] for r in comp if r["feature_set"] == "without PageValues"],
            },
            ["#6366f1", "#475569"], "Mean CV PR-AUC", 360,
        )
        st.markdown(
            '<div class="ch-wrap"><h4>Matched Algorithm Comparison</h4>'
            '<div class="ch-sub">Evaluated under consistent preprocessing across folds</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, key="cv_mc")

    sec("Cost-Sensitive Threshold Optimization")
    cost_df = pd.DataFrame([
        {"Assumed Cost Ratio (R)": "R=3 (Conservative)", "Decision Threshold": 0.58, "Precision": "34.8%", "Recall": "63.1%", "Sessions Flagged": "28.4%", "Relative Cost vs Inaction": "76.4%"},
        {"Assumed Cost Ratio (R)": "R=5 (Balanced)", "Decision Threshold": 0.52, "Precision": "31.1%", "Recall": "72.8%", "Sessions Flagged": "36.7%", "Relative Cost vs Inaction": "59.5%"},
        {"Assumed Cost Ratio (R)": "R=10 (Aggressive)", "Decision Threshold": 0.33, "Precision": "24.7%", "Recall": "93.2%", "Sessions Flagged": "59.0%", "Relative Cost vs Inaction": "35.2%"},
    ])
    st.dataframe(cost_df, hide_index=True, use_container_width=True)
    st.caption("Thresholds calibrated on Out-Of-Fold predictions to minimize FP + R×FN business loss.")

    sec("Decile Lift & Cumulative Capture")
    l1, l2, l3 = st.columns(3)
    l1.metric("Top 10% Conversion Rate", "44.5%", "2.85× Base Rate")
    l2.metric("Top 30% Buyer Capture", "64.7%", "247 of 382 Buyers")
    l3.metric("Bottom 40% Buyers", "6.0%", "Safe Non-Intervention")

    fig_lift = _bar(["Top 10%", "Top 20%", "Top 30%", "Bottom 40%"], [44.5, 35.0, 21.6, 6.0], "#06b6d4", "Conversion %", 290, tfmt=".1f")
    st.markdown('<div class="ch-wrap"><h4>Decile Conversion Concentration</h4><div class="ch-sub">Demonstrates robust ranking ability</div></div>', unsafe_allow_html=True)
    st.plotly_chart(fig_lift, use_container_width=True, key="lift_chart")

    sec("Global Feature Importance (Test Set)")
    fi1, fi2 = st.columns(2)
    with fi1:
        shap_feats = ["Product Duration", "Month", "Exit Rate", "Browser", "Traffic Type", "Bounce Rate", "Admin Pages", "OS", "Product Pages", "Special Day"]
        shap_vals = [0.444, 0.428, 0.410, 0.207, 0.160, 0.111, 0.109, 0.108, 0.089, 0.071]
        fig_shap = _bar(shap_feats, shap_vals, "#8b5cf6", "Mean |SHAP| Value", 380, horiz=True)
        st.markdown('<div class="ch-wrap"><h4>SHAP Summary (Magnitude of Signal)</h4><div class="ch-sub">ProductRelated_Duration is primary driver</div></div>', unsafe_allow_html=True)
        st.plotly_chart(fig_shap, use_container_width=True, key="shap_chart")

    with fi2:
        perm_feats = ["Product Duration", "Exit Rate", "Month", "Product Pages", "Visitor Type", "Bounce Rate", "Admin Duration", "Traffic Type", "OS", "Admin Pages"]
        perm_vals = [0.0794, 0.0738, 0.0588, 0.0184, 0.0151, 0.0100, 0.0093, 0.0082, 0.0080, 0.0048]
        fig_perm = _bar(perm_feats, perm_vals, "#3b82f6", "PR-AUC Loss on Shuffle", 380, horiz=True)
        st.markdown('<div class="ch-wrap"><h4>Permutation Importance (Loss of Signal)</h4><div class="ch-sub">Exit Rate & Duration shuffle impact</div></div>', unsafe_allow_html=True)
        st.plotly_chart(fig_perm, use_container_width=True, key="perm_chart")

    sec("Diagnostic Curves")
    d_c1, d_c2 = st.columns(2)
    lc = ROOT / "reports" / "figures" / "learning_curve.png"
    cc = ROOT / "reports" / "figures" / "calibration_curve.png"
    with d_c1:
        st.markdown('<div class="ch-wrap"><h4>Sample Efficiency Learning Curve</h4><div class="ch-sub">PR-AUC progression across training subsets</div></div>', unsafe_allow_html=True)
        if lc.is_file():
            st.image(str(lc), use_container_width=True)
    with d_c2:
        st.markdown('<div class="ch-wrap"><h4>Reliability Calibration Diagnostic</h4><div class="ch-sub">Raw model serves as ranking score rather than pure probability</div></div>', unsafe_allow_html=True)
        if cc.is_file():
            st.image(str(cc), use_container_width=True)


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
        user = st.session_state.get("user")

        # ── Pre-build user HTML safely (no nested f-strings) ──
        if user:
            display_name = user.get("name") or user.get("email", "Analyst").split("@")[0]
            initials = display_name[:2].upper()
            provider = user.get("provider", "password")
            badge_label = {"google.com": "Google", "password": "Email", "demo": "Guest"}.get(provider, "Auth")
            user_html = (
                '<div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">'
                '<div style="width:28px;height:28px;border-radius:50%;background:#6366f1;'
                'display:flex;align-items:center;justify-content:center;'
                'font-size:11px;font-weight:700;color:#fff;">' + initials + '</div>'
                '<div>'
                '<div style="font-size:.80rem;font-weight:600;color:#e2e8f0;max-width:130px;'
                'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + display_name + '</div>'
                '<div style="font-size:.60rem;font-weight:600;padding:1px 6px;border-radius:99px;'
                'background:rgba(99,102,241,.12);color:#a5b4fc;border:1px solid rgba(99,102,241,.2);'
                'display:inline-block;">' + badge_label + '</div>'
                '</div></div>'
            )
        else:
            user_html = ""

        # ── Render navbar shell ──
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:space-between;'
            'padding:.55rem 1.3rem;background:var(--surface);border:1px solid var(--border);'
            'border-radius:14px;margin-bottom:.9rem;box-shadow:0 2px 12px rgba(0,0,0,.35);">'
            '<div style="display:flex;align-items:center;gap:9px;flex-shrink:0;">'
            '<div style="width:32px;height:32px;border-radius:8px;background:var(--primary);'
            'display:flex;align-items:center;justify-content:center;'
            'font-size:13px;font-weight:800;color:#fff;">VI</div>'
            '<div>'
            '<div style="font-size:.88rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;line-height:1.2;">Visitor Intelligence</div>'
            '<div style="font-size:.58rem;font-weight:600;color:#818cf8;text-transform:uppercase;letter-spacing:.07em;">Purchase Intent Engine</div>'
            '</div></div>'
            + user_html +
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Segmented nav + sign-out button in one row ──
        if "top_main_segmented_control" not in st.session_state or st.session_state["top_main_segmented_control"] not in nav_pages:
            st.session_state["top_main_segmented_control"] = current_page

        nav_col, usr_btn_col = st.columns([5.5, 1.2], vertical_alignment="center")
        with nav_col:
            st.segmented_control(
                "Navigation",
                options=nav_pages,
                format_func=lambda p: f"{nav_icons[p]} {p}",
                key="top_main_segmented_control",
                on_change=on_top_nav_change,
                label_visibility="collapsed",
            )
        with usr_btn_col:
            if user:
                st.button("Sign out", key="top_navbar_logout_btn", on_click=do_logout,
                          use_container_width=True)
            else:
                st.button("Sign in", key="top_navbar_signin_btn", on_click=nav_to,
                          args=("Login",), use_container_width=True, type="primary")

        st.markdown(
            '<div style="height:1px;background:var(--border);margin:.2rem 0 1rem;"></div>',
            unsafe_allow_html=True,
        )

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
    pg_predict(model, metadata, input_limits, ratio, results, raw=dataset)
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
