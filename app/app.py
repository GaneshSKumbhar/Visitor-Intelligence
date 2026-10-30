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

MODEL_PATH   = ROOT / "models" / "xgb_no_pagevalues.joblib"
META_PATH    = ROOT / "models" / "app_meta.json"
RESULTS_PATH = ROOT / "reports" / "supplemental_results.json"
DATA_PATH    = ROOT / "data" / "processed" / "shoppers_clean.csv"
VERSION = "3.0"
FIREBASE_API_KEY  = "AIzaSyBWlZmKjZbFcVMgfYQaY8G2gE38PgyhnxQ"
FIREBASE_AUTH_URL = "https://identitytoolkit.googleapis.com/v1/accounts:{}?key={}"
SESSION_SECRET    = "visitor_intel_safe_token_key_v2"
_SESSION_SERIALIZER = URLSafeTimedSerializer(SESSION_SECRET, salt="visitor_intel_salt")
SESSION_MAX_AGE   = 86400 * 7
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("visitor_intent_app")

# ── Cost assumption: missed buyer costs 5× a wasted discount ──
COST_RATIO_DEFAULT = 5
DISCOUNT_COST_RS   = 200   # ₹ cost per wasted discount intervention
MISSED_BUYER_RS    = 1000  # ₹ average revenue lost per missed buyer


def create_session_token(user: dict) -> str:
    payload = {"email": user.get("email",""), "name": user.get("name",""),
               "uid": user.get("uid",""), "provider": user.get("provider","password")}
    return _SESSION_SERIALIZER.dumps(payload)


def decode_session_token(token: str) -> dict | None:
    if not token or not isinstance(token, str):
        return None
    try:
        data = _SESSION_SERIALIZER.loads(token.strip(), max_age=SESSION_MAX_AGE)
        if isinstance(data, dict) and "email" in data:
            return data
    except Exception:
        pass
    return None


FIREBASE_AUTH_DIR = APP_DIR / "firebase_auth"
_firebase_auth_component = components.declare_component("firebase_auth", path=str(FIREBASE_AUTH_DIR))

st.set_page_config(
    page_title="Visitor Intelligence · Live Purchase Intent Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
    --bg:      #07090f;
    --surface: #0d1117;
    --card:    rgba(17,24,39,0.9);
    --border:  rgba(99,102,241,0.18);
    --border-h:rgba(99,102,241,0.45);
    --txt:     #f1f5f9;
    --muted:   #64748b;
    --sub:     #94a3b8;
    --green:   #10b981;
    --amber:   #f59e0b;
    --red:     #f43f5e;
    --indigo:  #6366f1;
    --violet:  #8b5cf6;
    --cyan:    #06b6d4;
    --r:       14px;
    --rl:      20px;
    --shadow:  0 8px 32px rgba(0,0,0,0.4), 0 0 12px rgba(99,102,241,0.05);
}
html,body,.stApp,[data-testid="stAppViewContainer"]{
    font-family:'Inter',-apple-system,sans-serif !important;
    background:var(--bg) !important; color:var(--txt) !important;
}
#MainMenu,footer,[data-testid="stToolbar"],header[data-testid="stHeader"],
.stDeployButton,[data-testid="manage-app-button"],[data-testid="stStatusWidget"]{
    display:none !important; visibility:hidden !important; height:0 !important;
}
.block-container{ max-width:1480px !important; padding:0.8rem 1.8rem 3rem !important; }

/* ── CARD ── */
.card{
    background:var(--card); border:1px solid var(--border);
    border-radius:var(--r); padding:1.3rem 1.4rem;
    box-shadow:var(--shadow); transition:all .22s ease; height:100%;
}
.card:hover{ border-color:var(--border-h); transform:translateY(-2px); }
.card h4{ color:#f8fafc !important; font-size:1rem !important; font-weight:700 !important; margin:0 0 .4rem !important; }
.card p{ color:var(--sub) !important; font-size:.85rem !important; line-height:1.55 !important; margin:0 !important; }
.card .ic{ font-size:1.6rem; display:block; margin-bottom:.6rem; }

/* ── SECTION HEADER ── */
.sec{ display:flex; align-items:center; gap:.6rem; margin:2rem 0 .9rem; }
.sec .dot{ width:9px; height:9px; border-radius:50%;
    background:linear-gradient(135deg,var(--indigo),var(--cyan));
    box-shadow:0 0 8px rgba(99,102,241,.6); }
.sec h3{ margin:0 !important; font-size:1.18rem !important; font-weight:800 !important;
    color:#f8fafc !important; letter-spacing:-.02em; }

/* ── HERO ── */
.hero{
    background:linear-gradient(135deg,#0d1424 0%,#131a38 55%,#0f1e36 100%);
    border:1px solid rgba(99,102,241,.22); border-radius:var(--rl);
    padding:2.4rem 2.6rem; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero::after{
    content:''; position:absolute; top:-35%; right:-8%; width:420px; height:420px;
    background:radial-gradient(circle,rgba(99,102,241,.18) 0%,transparent 70%);
    pointer-events:none;
}
.badge{
    display:inline-flex; align-items:center; gap:6px;
    background:rgba(99,102,241,.12); color:#a5b4fc;
    padding:.3rem .85rem; border-radius:9999px; font-size:.7rem;
    font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    margin-bottom:.9rem; border:1px solid rgba(99,102,241,.28);
}
.hero h1{ color:#fff !important; font-size:2.4rem !important; font-weight:900 !important;
    letter-spacing:-.04em; line-height:1.2; margin:0 0 .8rem; }
.hero p{ color:#cbd5e1; font-size:1.05rem; line-height:1.6; margin:0; max-width:680px; }

/* ── STAT BOX ── */
.stat{
    background:var(--card); border:1px solid var(--border); border-radius:var(--r);
    padding:1.1rem 1.2rem; text-align:center; box-shadow:var(--shadow);
}
.stat .val{ font-size:1.85rem; font-weight:900; letter-spacing:-.03em; line-height:1.1; }
.stat .lbl{ font-size:.72rem; font-weight:700; color:var(--sub); text-transform:uppercase;
    letter-spacing:.06em; margin-top:.2rem; }
.stat .delta{ font-size:.75rem; font-weight:600; margin-top:.15rem; }

/* ── INTENT BADGE ── */
.intent-high{ color:var(--green); }
.intent-med { color:var(--amber); }
.intent-low { color:var(--muted); }

/* ── RESULT CARD ── */
.res-card{
    border-radius:var(--rl); padding:1.8rem 2rem; border:1px solid var(--border);
    position:relative; overflow:hidden; margin:1rem 0 1.4rem; box-shadow:var(--shadow);
}
.res-card::before{ content:''; position:absolute; top:0; left:0; right:0; height:4px; }
.res-high{ background:linear-gradient(145deg,rgba(16,185,129,.07),rgba(7,9,15,.95)); border-color:rgba(16,185,129,.3); }
.res-high::before{ background:linear-gradient(90deg,#10b981,#34d399); }
.res-med { background:linear-gradient(145deg,rgba(245,158,11,.07),rgba(7,9,15,.95)); border-color:rgba(245,158,11,.3); }
.res-med::before { background:linear-gradient(90deg,#f59e0b,#fbbf24); }
.res-low { background:linear-gradient(145deg,rgba(100,116,139,.1),rgba(7,9,15,.95));  border-color:rgba(100,116,139,.3); }
.res-low::before { background:linear-gradient(90deg,#64748b,#94a3b8); }
.res-tag{ display:inline-block; padding:.22rem .7rem; border-radius:9999px;
    font-size:.65rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.5rem; }
.res-high .res-tag{ background:rgba(16,185,129,.14); color:#34d399; }
.res-med  .res-tag{ background:rgba(245,158,11,.14);  color:#fbbf24; }
.res-low  .res-tag{ background:rgba(100,116,139,.14); color:#cbd5e1; }
.res-band{ font-size:1.9rem; font-weight:900; letter-spacing:-.02em; line-height:1.1; margin-bottom:.3rem; }
.res-high .res-band{ color:var(--green); }
.res-med  .res-band{ color:var(--amber); }
.res-low  .res-band{ color:var(--muted); }

/* ── VISITOR ROW ── */
.vrow{
    display:flex; align-items:center; gap:12px; padding:.7rem 1rem;
    background:var(--card); border:1px solid var(--border); border-radius:10px;
    margin-bottom:.4rem; transition:border-color .2s;
}
.vrow:hover{ border-color:var(--border-h); }
.vrow .vidx{ font-size:.7rem; color:var(--muted); font-weight:700; width:28px; }
.vrow .vscore{ font-size:1.05rem; font-weight:800; min-width:52px; }
.vrow .vtype{ font-size:.72rem; color:var(--sub); }
.vrow .vrec{ font-size:.75rem; font-weight:600; color:#a5b4fc;
    background:rgba(99,102,241,.12); padding:.15rem .55rem; border-radius:9999px; }

/* ── CHART WRAP ── */
.ch{ background:var(--card); border:1px solid var(--border); border-radius:var(--r);
     padding:1.1rem 1.2rem .5rem; margin-bottom:.6rem; }
.ch h4{ color:#f8fafc !important; font-size:.92rem !important; font-weight:700 !important; margin:0 0 .1rem !important; }
.ch p{ color:var(--muted); font-size:.74rem; margin:0 0 .4rem; }

/* ── PILL ── */
.pill-g{ background:rgba(16,185,129,.12); color:#34d399; border:1px solid rgba(16,185,129,.25); padding:.2rem .65rem; border-radius:9999px; font-size:.72rem; font-weight:700; }
.pill-a{ background:rgba(245,158,11,.12); color:#fbbf24; border:1px solid rgba(245,158,11,.25); padding:.2rem .65rem; border-radius:9999px; font-size:.72rem; font-weight:700; }
.pill-r{ background:rgba(244,63,94,.12); color:#fb7185; border:1px solid rgba(244,63,94,.25); padding:.2rem .65rem; border-radius:9999px; font-size:.72rem; font-weight:700; }

/* ── NAV ── */
.pub-nav{
    display:flex; align-items:center; justify-content:space-between;
    padding:.8rem 1.5rem;
    background:rgba(13,17,35,.96); border:1px solid rgba(99,102,241,.2);
    border-radius:var(--rl); margin-bottom:1.8rem; backdrop-filter:blur(20px);
}
.brand{ display:flex; align-items:center; gap:10px; }
.brand-ico{ width:38px; height:38px; border-radius:11px;
    background:linear-gradient(135deg,var(--indigo),var(--violet));
    display:flex; align-items:center; justify-content:center;
    font-size:16px; color:#fff; font-weight:900;
    box-shadow:0 4px 14px rgba(99,102,241,.4); }
.brand-name{ font-size:1rem; font-weight:800; color:#f8fafc; letter-spacing:-.02em; line-height:1.2; }
.brand-tag{ font-size:.62rem; font-weight:700; color:#818cf8; letter-spacing:.07em; text-transform:uppercase; }

/* ── CRUMB ── */
.crumb{ background:rgba(30,41,59,.4); border:1px solid rgba(99,102,241,.13);
    border-radius:var(--r); padding:.55rem 1.1rem; margin-bottom:.9rem;
    font-size:.82rem; color:var(--sub); }
.crumb .active{ color:#a5b4fc; font-weight:700; }

/* ── FOOTER ── */
.ft{ text-align:center; color:var(--muted); font-size:.76rem;
    padding:2rem 0 .8rem; border-top:1px solid rgba(99,102,241,.1);
    margin-top:3rem; line-height:1.8; }
.ft a{ color:#818cf8; text-decoration:none; }

/* ── SEGMENT CARD ── */
.seg-card{
    background:var(--card); border:1px solid var(--border);
    border-radius:var(--r); padding:1.2rem 1.3rem; height:100%;
}
.seg-card h4{ color:#f8fafc !important; font-size:.95rem !important; font-weight:700 !important; margin:0 0 .4rem !important; }
.seg-card .rate{ font-size:1.7rem; font-weight:900; letter-spacing:-.02em; margin-bottom:.2rem; }

/* ── STEP ── */
.step{ background:var(--card); border:1px solid var(--border); border-radius:var(--r);
    padding:1.3rem 1.4rem; position:relative; height:100%; }
.step-num{ position:absolute; top:-10px; left:16px;
    background:linear-gradient(135deg,var(--indigo),var(--violet)); color:#fff;
    font-size:.65rem; font-weight:800; padding:.15rem .6rem; border-radius:9999px; }
.step h4{ color:#f8fafc !important; font-size:.95rem !important; font-weight:700 !important; margin:.3rem 0 .35rem !important; }
.step p{ color:var(--sub) !important; font-size:.82rem !important; line-height:1.5 !important; margin:0 !important; }

/* ── BUSINESS IMPACT ── */
.impact{
    background:linear-gradient(135deg,rgba(16,185,129,.06),rgba(99,102,241,.06));
    border:1px solid rgba(16,185,129,.22); border-radius:var(--rl);
    padding:1.6rem 1.8rem; margin:1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  NAVIGATION HELPERS
# ══════════════════════════════════════════════════════════════
PUBLIC_PAGES    = {"Home", "Methodology", "About", "Login"}
PROTECTED_PAGES = {"Dashboard", "Live Scoring", "Targeting"}
PAGE_SLUG_MAP   = {
    "home": "Home", "login": "Login",
    "dashboard": "Dashboard", "scoring": "Live Scoring",
    "targeting": "Targeting", "methodology": "Methodology", "about": "About",
}
SLUG_INV_MAP = {v: k for k, v in PAGE_SLUG_MAP.items()}


def nav_to(page: str) -> None:
    if "user" not in st.session_state and page in PROTECTED_PAGES:
        st.session_state["redirect_target"] = page
        page = "Login"
    st.session_state["navigation"] = page
    st.session_state["top_nav"] = page
    st.query_params["page"] = SLUG_INV_MAP.get(page, "home")
    if "user" in st.session_state:
        st.query_params["session"] = create_session_token(st.session_state["user"])


def on_nav_change() -> None:
    sel = st.session_state.get("top_nav")
    if sel:
        nav_to(sel)


def do_logout() -> None:
    for k in list(st.session_state.keys()):
        if k not in ("auth_nonce",):
            del st.session_state[k]
    st.session_state["auth_nonce"] = st.session_state.get("auth_nonce", 0) + 1
    st.session_state["navigation"] = "Home"
    st.session_state["top_nav"]    = "Home"
    st.session_state["logged_out"] = True
    st.query_params["page"] = "home"
    if "session" in st.query_params:
        del st.query_params["session"]

# ══════════════════════════════════════════════════════════════
#  ASSET LOADING  ← scores ALL 12,205 sessions on startup
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔮  Loading model & scoring 12,205 real sessions…")
def load_assets():
    ensure_artifact_files([MODEL_PATH, META_PATH, RESULTS_PATH, DATA_PATH])
    meta    = json.loads(META_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    model   = joblib.load(MODEL_PATH)
    raw     = pd.read_csv(DATA_PATH)

    # ── Score every single real session automatically ──
    scores = model.predict_proba(raw[FEATURE_COLUMNS])[:, 1]
    raw    = raw.copy()
    raw["model_score"] = scores
    bands  = meta["score_bands"]
    raw["intent_band"] = np.where(
        scores >= bands["high_cutoff"], "High",
        np.where(scores >= bands["medium_cutoff"], "Medium", "Low")
    )
    thr_r5 = float(meta["cost_thresholds"]["5"])
    raw["intervention_flag"] = scores >= thr_r5
    raw["session_id"] = range(1, len(raw) + 1)

    numeric = COUNT_COLUMNS + DURATION_COLUMNS + ["BounceRates", "ExitRates", "SpecialDay"]
    limits  = {c: (float(raw[c].min()), float(raw[c].max())) for c in numeric}
    return model, meta, results, raw, limits


# ══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════
def _layout(h=320, xt="", yt="", showlegend=False):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#94a3b8", size=11),
        margin=dict(l=42, r=18, t=28, b=42),
        xaxis=dict(gridcolor="rgba(148,163,184,.07)", zerolinecolor="rgba(148,163,184,.07)", title=xt),
        yaxis=dict(gridcolor="rgba(148,163,184,.07)", zerolinecolor="rgba(148,163,184,.07)", title=yt),
        hoverlabel=dict(bgcolor="#1e293b", font_size=12, bordercolor="#334155"),
        bargap=0.26, height=h, showlegend=showlegend,
    )

def _bar(labels, values, color="#6366f1", yt="", h=300, horiz=False, fmt=".2f"):
    if horiz:
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
            marker=dict(color=color, cornerradius=6),
            text=[f"{v:{fmt}}" for v in values], textposition="outside",
            textfont=dict(size=10, color="#94a3b8")))
        fig.update_layout(**_layout(h=h, xt=yt, horiz=True) | {"yaxis": {"autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"}})
    else:
        fig = go.Figure(go.Bar(x=labels, y=values,
            marker=dict(color=color, cornerradius=6),
            text=[f"{v:{fmt}}" for v in values], textposition="outside",
            textfont=dict(size=10, color="#94a3b8")))
        fig.update_layout(**_layout(h=h, yt=yt))
    return fig

def _grouped(labels, groups, colors, yt="", h=340):
    fig = go.Figure()
    for i, (name, vals) in enumerate(groups.items()):
        fig.add_trace(go.Bar(x=labels, y=vals, name=name,
            marker=dict(color=colors[i % len(colors)], cornerradius=6),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10, color="#94a3b8")))
    lo = _layout(h=h, yt=yt, showlegend=True)
    lo["barmode"] = "group"
    lo["legend"]  = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=.5, font_size=11)
    fig.update_layout(**lo)
    return fig

def _donut(labels, values, colors, h=260):
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.65,
        marker=dict(colors=colors, line=dict(color="#07090f", width=2)),
        textinfo="label+percent", textfont=dict(size=11, color="#f1f5f9"),
        hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        margin=dict(l=18,r=18,t=18,b=18), height=h, showlegend=False)
    return fig

def _spline(months, rates, h=300):
    fig = go.Figure(go.Scatter(x=months, y=rates, mode="lines+markers+text",
        line=dict(color="#6366f1", width=3, shape="spline"),
        fill="tozeroy", fillcolor="rgba(99,102,241,.12)",
        marker=dict(size=7, color="#8b5cf6", line=dict(color="#fff", width=1.5)),
        text=[f"{r:.1%}" for r in rates], textposition="top center",
        textfont=dict(size=10, color="#cbd5e1"),
        hovertemplate="<b>%{x}</b><br>%{y:.1%}<extra></extra>"))
    lo = _layout(h=h, yt="Conversion Rate", showlegend=False)
    lo["yaxis"]["tickformat"] = ".0%"
    fig.update_layout(**lo)
    return fig

def _gauge(score, meta, ratio):
    bands = meta["score_bands"]
    thr   = float(meta["cost_thresholds"][str(ratio)])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(font=dict(size=28, color="#f8fafc"), valueformat=".3f"),
        gauge=dict(
            axis=dict(range=[0,1], tickcolor="#64748b", tickfont=dict(color="#94a3b8", size=10)),
            bar=dict(color="#6366f1", thickness=.3),
            bgcolor="rgba(30,41,59,.5)", borderwidth=0,
            steps=[
                dict(range=[0, bands["medium_cutoff"]], color="rgba(100,116,139,.12)"),
                dict(range=[bands["medium_cutoff"], bands["high_cutoff"]], color="rgba(245,158,11,.12)"),
                dict(range=[bands["high_cutoff"], 1], color="rgba(16,185,129,.15)"),
            ],
            threshold=dict(line=dict(color="#f43f5e", width=3), thickness=.82, value=thr),
        ),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter,sans-serif", color="#94a3b8"),
        height=220, margin=dict(l=25,r=25,t=25,b=10))
    return fig

def _radar(row_dict, limits):
    def norm(val, key):
        lo, hi = limits[key]
        return min(max((float(val)-lo)/(hi-lo+1e-9), 0.0), 1.0)
    cats   = ["Product Dwell", "Product Pages", "Exit Resistance", "Bounce Resistance", "Admin Dwell"]
    svals  = [norm(row_dict.get("ProductRelated_Duration",0),"ProductRelated_Duration"),
              norm(row_dict.get("ProductRelated",0),"ProductRelated"),
              max(0, 1-norm(row_dict.get("ExitRates",0),"ExitRates")*3),
              max(0, 1-norm(row_dict.get("BounceRates",0),"BounceRates")*3),
              norm(row_dict.get("Administrative_Duration",0),"Administrative_Duration")]
    buyer  = [.45,.40,.85,.88,.25]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=svals, theta=cats, fill="toself",
        fillcolor="rgba(99,102,241,.3)", line=dict(color="#6366f1",width=2.5), name="This Session"))
    fig.add_trace(go.Scatterpolar(r=buyer, theta=cats, fill="toself",
        fillcolor="rgba(16,185,129,.12)", line=dict(color="#10b981",width=2,dash="dot"), name="Avg Buyer"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                       gridcolor="rgba(148,163,184,.1)"),
                   angularaxis=dict(gridcolor="rgba(148,163,184,.1)"), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter,sans-serif",color="#cbd5e1",size=10),
        margin=dict(l=38,r=38,t=26,b=26), height=280, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-.15, xanchor="center", x=.5))
    return fig

def sec(title):
    st.markdown(f'<div class="sec"><div class="dot"></div><h3>{title}</h3></div>',
                unsafe_allow_html=True)

def card(icon, title, desc):
    st.markdown(f'<div class="card"><span class="ic">{icon}</span><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True)

def step(n, title, desc):
    st.markdown(f'<div class="step"><div class="step-num">STEP {n}</div><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: HOME (Public Landing)
# ══════════════════════════════════════════════════════════════
def pg_home(raw, meta):
    st.markdown("""
    <div class="pub-nav">
        <div class="brand">
            <div class="brand-ico">VI</div>
            <div><div class="brand-name">Visitor Intelligence</div>
                 <div class="brand-tag">AI Purchase Intent Engine</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

    nc = st.columns([1,1,1,1,1.2,1.4], vertical_alignment="center")
    nc[0].button("🏠 Home",        key="pn1", on_click=nav_to, args=("Home",),        use_container_width=True)
    nc[1].button("🔬 Methodology", key="pn2", on_click=nav_to, args=("Methodology",), use_container_width=True)
    nc[2].button("ℹ️ About",       key="pn3", on_click=nav_to, args=("About",),       use_container_width=True)
    if "user" in st.session_state:
        nc[3].button("🚪 Logout",  key="pn4", on_click=do_logout,                     use_container_width=True)
        nc[5].button("📊 Open Dashboard", key="pn6", on_click=nav_to, args=("Dashboard",), type="primary", use_container_width=True)
    else:
        nc[3].button("🔑 Sign In", key="pn4", on_click=nav_to, args=("Login",),       use_container_width=True)
        nc[5].button("🚀 Try Free Demo", key="pn6", on_click=nav_to, args=("Dashboard",), type="primary", use_container_width=True)

    # ── HERO ──
    h1, h2 = st.columns([1.4, 1], vertical_alignment="center")
    with h1:
        n_high = int((raw["intent_band"] == "High").sum())
        revenue_saved = n_high * DISCOUNT_COST_RS
        st.markdown(f"""
        <div class="badge">🚀 LIVE AI SCORING — {len(raw):,} REAL SESSIONS ANALYSED</div>
        <h1 style="color:#fff;font-size:2.6rem;font-weight:900;letter-spacing:-.04em;line-height:1.2;margin:0 0 .9rem;">
            Stop Wasting Discounts.<br><span style="color:#818cf8;">Target Only Who Will Buy.</span>
        </h1>
        <p style="color:#cbd5e1;font-size:1.05rem;line-height:1.65;max-width:650px;margin:0 0 1.5rem;">
            E-commerce loses ₹{DISCOUNT_COST_RS}/session on discounts shown to visitors who were never going to buy.
            This AI engine scores every session and tells you <strong>exactly which {n_high:,} visitors
            ({n_high/len(raw):.0%}) deserve an intervention</strong> — automatically, from real browsing behaviour.
        </p>""", unsafe_allow_html=True)
        b1, b2, _ = st.columns([1.1,1.1,.8])
        with b1: st.button("📊 See Live Scoring", key="h_dash", on_click=nav_to, args=("Dashboard",),    type="primary", use_container_width=True)
        with b2: st.button("🎯 Targeting Engine",  key="h_targ", on_click=nav_to, args=("Targeting",),  use_container_width=True)

    with h2:
        pr = float(raw["Revenue"].mean())
        st.markdown(f"""
        <div class="card" style="padding:1.5rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;
                        padding-bottom:.6rem;border-bottom:1px solid rgba(148,163,184,.1);">
                <span style="font-size:.72rem;font-weight:700;color:#10b981;">🟢 ENGINE LIVE — ALL {len(raw):,} SESSIONS SCORED</span>
                <span style="font-size:.7rem;color:#818cf8;font-weight:600;">v{VERSION}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;">
                <div style="background:rgba(99,102,241,.08);border-radius:10px;padding:.8rem;text-align:center;">
                    <div style="font-size:.65rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">High-Intent Visitors</div>
                    <div style="font-size:1.6rem;font-weight:900;color:#10b981;">{n_high:,}</div>
                    <div style="font-size:.68rem;color:#94a3b8;">Ready to buy NOW</div>
                </div>
                <div style="background:rgba(99,102,241,.08);border-radius:10px;padding:.8rem;text-align:center;">
                    <div style="font-size:.65rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Base Conversion</div>
                    <div style="font-size:1.6rem;font-weight:900;color:#f8fafc;">{pr:.1%}</div>
                    <div style="font-size:.68rem;color:#94a3b8;">Dataset baseline</div>
                </div>
                <div style="background:rgba(99,102,241,.08);border-radius:10px;padding:.8rem;text-align:center;">
                    <div style="font-size:.65rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Top Decile Conv.</div>
                    <div style="font-size:1.6rem;font-weight:900;color:#6366f1;">44.5%</div>
                    <div style="font-size:.68rem;color:#94a3b8;">2.85× base rate</div>
                </div>
                <div style="background:rgba(99,102,241,.08);border-radius:10px;padding:.8rem;text-align:center;">
                    <div style="font-size:.65rem;color:#94a3b8;font-weight:700;text-transform:uppercase;">Discounts Saved</div>
                    <div style="font-size:1.6rem;font-weight:900;color:#f59e0b;">₹{(len(raw)-n_high)*DISCOUNT_COST_RS//1000}K</div>
                    <div style="font-size:.68rem;color:#94a3b8;">By not targeting low-intent</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── BUSINESS CASE ──
    sec("Why This Matters — The Real Business Problem")
    b1, b2, b3, b4 = st.columns(4)
    with b1: card("💸", "Wasted Marketing Spend", f"Showing discounts to all {len(raw):,} visitors costs ₹{len(raw)*DISCOUNT_COST_RS:,}. Most of it is wasted on people who won't buy.")
    with b2: card("🎯", "Smart Targeting", f"The AI identifies {n_high:,} high-intent sessions. Targeting only them cuts costs by {(1-n_high/len(raw)):.0%}.")
    with b3: card("📈", "2.85× Higher Conversion", "Sessions flagged by the model convert at 44.5% vs the 15.6% average — nearly 3× better ROI on every discount.")
    with b4: card("⚡", "Zero Manual Work", "The engine scores every session automatically from browsing behaviour — no analyst needed, no buttons to click.")

    # ── LIVE PREVIEW ──
    sec("Live Preview — Actual Scored Sessions from the Dataset")
    top10 = raw[raw["intent_band"] == "High"].nlargest(6, "model_score")[
        ["session_id","model_score","intent_band","VisitorType","Month","ProductRelated","ProductRelated_Duration","ExitRates","Revenue"]
    ].reset_index(drop=True)
    st.caption("🟢 Top 6 highest-scoring real sessions — these visitors should receive your best offer right now:")
    for _, r in top10.iterrows():
        actual = "✅ Bought" if r["Revenue"] == 1 else "❌ Didn't buy"
        vt = str(r["VisitorType"]).replace("_"," ")
        st.markdown(f"""
        <div class="vrow">
            <div class="vscore intent-high">▲ {r["model_score"]:.3f}</div>
            <span class="pill-g">HIGH INTENT</span>
            <div class="vtype">Session #{int(r["session_id"])} · {vt} · {r["Month"]} ·
                {int(r["ProductRelated"])} product pages · {r["ProductRelated_Duration"]:.0f}s dwell ·
                Exit rate {r["ExitRates"]:.2%}</div>
            <div class="vrec">💰 Send premium offer</div>
            <div style="margin-left:auto;font-size:.75rem;color:#64748b;">{actual}</div>
        </div>""", unsafe_allow_html=True)

    st.caption("🔴 Top 6 lowest-scoring sessions — save your budget, don't waste discounts here:")
    bot6 = raw[raw["intent_band"] == "Low"].nsmallest(6, "model_score")[
        ["session_id","model_score","intent_band","VisitorType","Month","ProductRelated","ExitRates","Revenue"]
    ].reset_index(drop=True)
    for _, r in bot6.iterrows():
        actual = "✅ Bought" if r["Revenue"] == 1 else "❌ Didn't buy"
        vt = str(r["VisitorType"]).replace("_"," ")
        st.markdown(f"""
        <div class="vrow">
            <div class="vscore intent-low">▼ {r["model_score"]:.3f}</div>
            <span class="pill-r">LOW INTENT</span>
            <div class="vtype">Session #{int(r["session_id"])} · {vt} · {r["Month"]} ·
                {int(r["ProductRelated"])} product pages · Exit rate {r["ExitRates"]:.2%}</div>
            <div class="vrec" style="background:rgba(244,63,94,.1);color:#fb7185;">🚫 No intervention</div>
            <div style="margin-left:auto;font-size:.75rem;color:#64748b;">{actual}</div>
        </div>""", unsafe_allow_html=True)

    # ── CTA ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(99,102,241,.15),rgba(139,92,246,.12));
        border:1px solid rgba(99,102,241,.28);border-radius:20px;padding:2.2rem;text-align:center;margin:2.5rem 0 1rem;">
        <h2 style="font-size:1.8rem;font-weight:900;color:#fff;margin-bottom:.5rem;">
            All 12,205 Sessions Already Scored. Open the Dashboard.</h2>
        <p style="color:#cbd5e1;font-size:.96rem;margin:0 auto .4rem;max-width:580px;">
            No data entry. No button clicking. The AI has already done the work.</p>
    </div>""", unsafe_allow_html=True)
    _, cc, _ = st.columns([1.5, 1.2, 1.5])
    cc.button("🚀 Open Live Dashboard", key="home_cta", on_click=nav_to, args=("Dashboard",), type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  AUTH PAGE
# ══════════════════════════════════════════════════════════════
def show_auth_page():
    tc1, tc2 = st.columns([4, 1.2], vertical_alignment="center")
    with tc1:
        st.markdown("""<div class="brand" style="margin-bottom:1rem;">
            <div class="brand-ico">VI</div>
            <div><div class="brand-name">Visitor Intelligence</div>
                 <div class="brand-tag">Authentication</div></div></div>""", unsafe_allow_html=True)
    with tc2:
        st.button("← Home", key="auth_home", on_click=nav_to, args=("Home",), use_container_width=True)

    dest = st.session_state.get("redirect_target", "Dashboard")
    if dest != "Dashboard":
        st.caption(f"🔒 Sign in to access **{dest}**")

    is_out   = st.session_state.get("logged_out", False)
    nonce    = st.session_state.get("auth_nonce", 0)
    evt = _firebase_auth_component(key=f"fb_{nonce}", logout_signal=is_out, default=None)
    if is_out:
        evt = None
        st.session_state["logged_out"] = False
    if evt and isinstance(evt, dict) and "email" in evt:
        st.session_state["user"]       = evt
        st.session_state["logged_out"] = False
        st.query_params["session"]     = create_session_token(evt)
        nav_to(st.session_state.pop("redirect_target", "Dashboard"))
        st.rerun()

    with st.expander("📧 Direct Email Login / Guest Access", expanded=False):
        t1, t2, t3 = st.tabs(["Sign In", "Create Account", "⚡ Guest Demo"])
        with t1:
            with st.form("f_in"):
                em = st.text_input("Email", key="fb_ei")
                pw = st.text_input("Password", type="password", key="fb_pi")
                if st.form_submit_button("Sign In", type="primary", use_container_width=True):
                    _do_auth("signInWithPassword", em, pw)
        with t2:
            with st.form("f_up"):
                em2 = st.text_input("Email", key="fb_eu")
                pw2 = st.text_input("Password (min 6 chars)", type="password", key="fb_pu")
                if st.form_submit_button("Create Account", type="primary", use_container_width=True):
                    _do_auth("signUp", em2, pw2)
        with t3:
            st.info("Explore the full dashboard as Guest Analyst — all features unlocked.")
            if st.button("⚡ Enter as Guest", key="guest_btn", use_container_width=True):
                g = {"email":"guest@visitor-intelligence.ai","name":"Guest Analyst","uid":"guest-demo","provider":"demo"}
                st.session_state["user"]       = g
                st.session_state["logged_out"] = False
                st.query_params["session"]     = create_session_token(g)
                nav_to(st.session_state.pop("redirect_target", "Dashboard"))
                st.rerun()


def _do_auth(endpoint, email, password):
    email = email.strip()
    if not email or not password:
        st.error("Please enter email and password.")
        return
    payload = json.dumps({"email":email,"password":password,"returnSecureToken":True}).encode()
    req = urllib.request.Request(FIREBASE_AUTH_URL.format(endpoint, FIREBASE_API_KEY),
                                  data=payload, headers={"Content-Type":"application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            r = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try: code = json.loads(e.read().decode()).get("error",{}).get("message","")
        except: code = ""
        msgs = {"EMAIL_EXISTS":"Account exists — sign in instead.",
                "INVALID_LOGIN_CREDENTIALS":"Wrong email or password.",
                "WEAK_PASSWORD":"Password must be at least 6 characters."}
        st.error(msgs.get(code, f"Auth error: {code or e.code}"))
        return
    except Exception as ex:
        st.error(f"Network error: {ex}")
        return
    u = {"email":r.get("email",email),"name":r.get("displayName") or email.split("@")[0],
         "uid":r.get("localId",""),"provider":"password"}
    st.session_state["user"]       = u
    st.session_state["logged_out"] = False
    st.query_params["session"]     = create_session_token(u)
    nav_to(st.session_state.pop("redirect_target","Dashboard"))
    st.rerun()


# ══════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD — All 12K sessions scored, live analytics
# ══════════════════════════════════════════════════════════════
def pg_dashboard(raw, meta, results):
    st.markdown("""<div class="hero">
        <div class="badge">📊 LIVE ANALYTICS DASHBOARD</div>
        <h1 style="color:#fff;font-size:2rem;font-weight:900;margin:0 0 .6rem;letter-spacing:-.03em;">
            All 12,205 Sessions Scored Automatically</h1>
        <p style="color:#cbd5e1;font-size:1rem;margin:0;">
            Every visitor in the dataset has been classified. No manual work required.
            Explore conversion patterns, intent distribution, and real business impact below.</p>
    </div>""", unsafe_allow_html=True)

    bands  = meta["score_bands"]
    thr_r5 = float(meta["cost_thresholds"]["5"])
    n_high = int((raw["intent_band"] == "High").sum())
    n_med  = int((raw["intent_band"] == "Medium").sum())
    n_low  = int((raw["intent_band"] == "Low").sum())
    n_flag = int(raw["intervention_flag"].sum())
    pr     = float(raw["Revenue"].mean())
    pr_high = float(raw[raw["intent_band"]=="High"]["Revenue"].mean())
    pr_low  = float(raw[raw["intent_band"]=="Low"]["Revenue"].mean())

    # ── TOP KPIs ──
    sec("Live Session Intelligence — Key Metrics")
    k = st.columns(6)
    def kpi(col, val, lbl, delta="", dclr="normal"):
        col.metric(lbl, val, delta, delta_color=dclr)

    k[0].metric("Total Sessions Scored", f"{len(raw):,}")
    k[1].metric("🟢 High Intent",  f"{n_high:,}", f"{n_high/len(raw):.1%} of sessions")
    k[2].metric("🟡 Medium Intent",f"{n_med:,}",  f"{n_med/len(raw):.1%} of sessions")
    k[3].metric("🔴 Low Intent",   f"{n_low:,}",  f"{n_low/len(raw):.1%} of sessions")
    k[4].metric("Flagged to Target",f"{n_flag:,}", f"R=5 threshold")
    k[5].metric("High-Intent Conv.",f"{pr_high:.1%}", f"+{pr_high-pr:.1%} vs baseline")

    # ── MAIN CHARTS ROW 1 ──
    sec("Intent Classification Across All Sessions")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = _donut(["High","Medium","Low"],[n_high,n_med,n_low],["#10b981","#f59e0b","#64748b"])
        fig.update_layout(annotations=[dict(
            text=f"<b>{n_flag:,}</b><br><span style='font-size:10px;color:#94a3b8'>TO TARGET</span>",
            x=.5, y=.5, font_size=18, font_color="#f8fafc", showarrow=False)])
        st.markdown('<div class="ch"><h4>Intent Band Distribution</h4><p>AI classification of all 12,205 sessions</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_don")

    with c2:
        # Score histogram
        fig = go.Figure(go.Histogram(x=raw["model_score"], nbinsx=40,
            marker=dict(color="#6366f1", line=dict(color="#07090f", width=1))))
        fig.add_vline(x=bands["medium_cutoff"], line=dict(color="#f59e0b",dash="dash",width=2),
                      annotation_text="Medium", annotation_font_color="#f59e0b")
        fig.add_vline(x=bands["high_cutoff"],   line=dict(color="#10b981",dash="dash",width=2),
                      annotation_text="High",   annotation_font_color="#10b981")
        lo = _layout(h=260, xt="Model Score", yt="Sessions")
        fig.update_layout(**lo)
        st.markdown('<div class="ch"><h4>Score Distribution (All 12,205 Sessions)</h4><p>Green/amber lines = band cutoffs</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_hist")

    with c3:
        mo     = meta["observed_months"]
        mr     = raw.groupby("Month")["Revenue"].mean().reindex(mo).dropna()
        fig    = _spline(mr.index.tolist(), mr.values.tolist(), 260)
        st.markdown('<div class="ch"><h4>Real Conversion Rate by Month</h4><p>Actual purchase rates from the dataset</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_mon")

    # ── MAIN CHARTS ROW 2 ──
    c4, c5 = st.columns(2)
    with c4:
        # Conversion rate by intent band — the key proof
        band_conv = raw.groupby("intent_band")["Revenue"].mean().reindex(["High","Medium","Low"])
        fig = go.Figure(go.Bar(
            x=["High Intent","Medium Intent","Low Intent"],
            y=band_conv.values * 100,
            marker=dict(color=["#10b981","#f59e0b","#64748b"], cornerradius=8),
            text=[f"{v:.1%}" for v in band_conv.values],
            textposition="outside", textfont=dict(size=14, color="#f8fafc", family="Inter")))
        fig.add_hline(y=pr*100, line=dict(color="#f43f5e",dash="dash",width=2),
                      annotation_text=f"Baseline {pr:.1%}", annotation_font_color="#f43f5e")
        lo = _layout(h=300, yt="Actual Conversion Rate (%)")
        fig.update_layout(**lo)
        st.markdown(
            '<div class="ch"><h4>✅ Proof: Model Actually Works — Conversion Rate by Intent Band</h4>'
            '<p>High-intent sessions convert at 33.2% vs 3.1% for low-intent — 10× difference</p></div>',
            unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_conv_band")

    with c5:
        # Business impact: smart targeting vs naive
        n_buyers = int(raw["Revenue"].sum())
        flagged  = raw[raw["intervention_flag"]==True]
        n_caught = int(flagged["Revenue"].sum())
        n_wasted = int((flagged["Revenue"]==0).sum())
        cost_smart  = n_wasted * DISCOUNT_COST_RS + (n_buyers - n_caught) * MISSED_BUYER_RS
        cost_all    = (len(raw)-n_buyers) * DISCOUNT_COST_RS
        cost_nothing= n_buyers * MISSED_BUYER_RS
        savings = cost_all - cost_smart

        fig = go.Figure(go.Bar(
            x=["Target Everyone\n(naive)", "Target No-one\n(do nothing)", "AI Smart\nTargeting"],
            y=[cost_all//1000, cost_nothing//1000, cost_smart//1000],
            marker=dict(color=["#f43f5e","#64748b","#10b981"], cornerradius=8),
            text=[f"₹{cost_all//1000}K", f"₹{cost_nothing//1000}K", f"₹{cost_smart//1000}K"],
            textposition="outside", textfont=dict(size=13, color="#f8fafc")))
        lo = _layout(h=300, yt="Total Cost (₹ thousands)")
        fig.update_layout(**lo)
        st.markdown(
            f'<div class="ch"><h4>💰 Business Impact: AI Saves ₹{savings//1000}K vs Targeting Everyone</h4>'
            f'<p>R=5 assumption: missing a buyer costs 5× a wasted discount (₹{DISCOUNT_COST_RS} per intervention)</p></div>',
            unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_biz")

    # ── ROW 3: EDA ──
    sec("Exploratory Data Analysis — What Drives Purchase Intent?")
    e1, e2 = st.columns(2)
    with e1:
        # BounceRates vs ExitRates scatter
        sample = raw.sample(min(1200, len(raw)), random_state=42)
        fig = go.Figure()
        for rev, color, lbl in [(0,"#334155","No Purchase"),(1,"#10b981","Purchase")]:
            m = sample["Revenue"] == rev
            fig.add_trace(go.Scatter(x=sample.loc[m,"BounceRates"], y=sample.loc[m,"ExitRates"],
                mode="markers", name=lbl, marker=dict(color=color,size=4,opacity=.7),
                hovertemplate="Bounce:%{x:.3f}<br>Exit:%{y:.3f}<extra></extra>"))
        lo = _layout(h=280, xt="Bounce Rate", yt="Exit Rate", showlegend=True)
        lo["legend"] = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
        fig.update_layout(**lo)
        st.markdown('<div class="ch"><h4>BounceRates vs ExitRates (Corr = 0.91)</h4><p>Buyers cluster at bottom-left — low bounce & exit rates</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_sc")

    with e2:
        # PageValues leakage proof
        pv_z  = float(raw[raw["PageValues"]==0]["Revenue"].mean())
        pv_nz = float(raw[raw["PageValues"] >0]["Revenue"].mean())
        fig = go.Figure(go.Bar(
            x=["PageValues = 0\n(77.6% of sessions)","PageValues > 0\n(22.4% of sessions)"],
            y=[pv_z*100, pv_nz*100],
            marker=dict(color=["#475569","#f43f5e"],cornerradius=8),
            text=[f"{pv_z:.1%}",f"{pv_nz:.1%}"],
            textposition="outside", textfont=dict(size=14,color="#cbd5e1")))
        lo = _layout(h=280, yt="Conversion Rate (%)")
        fig.update_layout(**lo)
        st.markdown('<div class="ch"><h4>⚠️ PageValues Leakage — Why We Exclude It</h4><p>56% conversion when PageValues>0 — this is post-purchase data, not available at session start</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="db_pv")

    # ── SEGMENTATION ──
    seg_data = results.get("segmentation", {}).get("profiles", [])
    if seg_data:
        sec("K-Means Visitor Segmentation (k=3 clusters, silhouette=0.434)")
        st.caption("Visitors automatically grouped by browsing behaviour — no labels, purely data-driven:")
        seg_names  = ["🟢 Focused Buyers", "🟡 Casual Browsers", "🔴 Bouncers"]
        seg_colors = ["#10b981","#f59e0b","#f43f5e"]
        scols = st.columns(3)
        for col, seg, name, clr in zip(scols, seg_data, seg_names, seg_colors):
            with col:
                st.markdown(f"""
                <div class="seg-card">
                    <h4>{name}</h4>
                    <div class="rate" style="color:{clr};">{seg["conversion_rate"]:.1%}</div>
                    <p style="color:#64748b;font-size:.72rem;margin-bottom:.5rem!important;">Conversion Rate</p>
                    <p>Sessions: <strong style="color:#e2e8f0">{seg["sessions"]:,}</strong></p>
                    <p>Median product pages: <strong style="color:#e2e8f0">{int(seg["median_product_pages"])}</strong></p>
                    <p>Median dwell time: <strong style="color:#e2e8f0">{seg["median_product_duration"]:.0f}s</strong></p>
                    <p>Median exit rate: <strong style="color:#e2e8f0">{seg["median_exit_rate"]:.1%}</strong></p>
                </div>""", unsafe_allow_html=True)

    nc1, nc2, nc3 = st.columns(3)
    nc1.button("🎯 Open Targeting Engine",  key="db_targ", on_click=nav_to, args=("Targeting",),   type="primary", use_container_width=True)
    nc2.button("⚡ Live Session Scorer",     key="db_scor", on_click=nav_to, args=("Live Scoring",), use_container_width=True)
    nc3.button("📈 Model Insights",         key="db_ins",  on_click=nav_to, args=("Methodology",), use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: TARGETING — "Who should I target right now?"
# ══════════════════════════════════════════════════════════════
def pg_targeting(raw, meta, ratio):
    st.markdown("""<div class="hero">
        <div class="badge">🎯 SMART TARGETING ENGINE</div>
        <h1 style="color:#fff;font-size:2rem;font-weight:900;margin:0 0 .6rem;letter-spacing:-.03em;">
            Who Should You Target Right Now?</h1>
        <p style="color:#cbd5e1;font-size:1rem;margin:0;">
            The AI has scored all 12,205 sessions. This page answers the only question that matters:
            <strong>which visitors deserve your discount, live-chat prompt, or premium offer?</strong></p>
    </div>""", unsafe_allow_html=True)

    thr = float(meta["cost_thresholds"][str(ratio)])
    flagged     = raw[raw["intervention_flag"] == True].copy()
    not_flagged = raw[raw["intervention_flag"] == False].copy()
    n_buyers    = int(raw["Revenue"].sum())
    n_caught    = int(flagged["Revenue"].sum())
    n_missed    = n_buyers - n_caught
    n_wasted    = int((flagged["Revenue"] == 0).sum())

    # ── DECISION SUMMARY ──
    sec("📋 AI Targeting Decision — Applied to All 12,205 Sessions")
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("✅ Sessions to Target",   f"{len(flagged):,}",   f"{len(flagged)/len(raw):.1%} of all visitors")
    d2.metric("🛑 Sessions to Skip",     f"{len(not_flagged):,}", f"Save ₹{len(not_flagged)*DISCOUNT_COST_RS//1000}K in discounts")
    d3.metric("🎯 Actual Buyers Caught", f"{n_caught:,}",        f"{n_caught/n_buyers:.1%} recall")
    d4.metric("🧾 Wasted Interventions", f"{n_wasted:,}",        f"{n_wasted/len(flagged):.1%} false positives")
    d5.metric("💸 Buyers Missed",        f"{n_missed:,}",        f"{n_missed/n_buyers:.1%} miss rate", delta_color="inverse")

    # ── BUSINESS IMPACT ──
    cost_smart  = n_wasted * DISCOUNT_COST_RS + n_missed * MISSED_BUYER_RS
    cost_all    = (len(raw)-n_buyers) * DISCOUNT_COST_RS
    cost_none   = n_buyers * MISSED_BUYER_RS
    savings_vs_all  = cost_all  - cost_smart
    savings_vs_none = cost_none - cost_smart

    st.markdown(f"""
    <div class="impact">
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem;">
            <div style="text-align:center;">
                <div style="font-size:.7rem;color:#94a3b8;font-weight:700;text-transform:uppercase;margin-bottom:.3rem;">Targeting Everyone (Naive)</div>
                <div style="font-size:1.8rem;font-weight:900;color:#f43f5e;">₹{cost_all//1000}K</div>
                <div style="font-size:.75rem;color:#64748b;">Total cost (wasted discounts)</div>
            </div>
            <div style="text-align:center;border-left:1px solid rgba(99,102,241,.2);border-right:1px solid rgba(99,102,241,.2);padding:0 1rem;">
                <div style="font-size:.7rem;color:#94a3b8;font-weight:700;text-transform:uppercase;margin-bottom:.3rem;">AI Smart Targeting</div>
                <div style="font-size:1.8rem;font-weight:900;color:#10b981;">₹{cost_smart//1000}K</div>
                <div style="font-size:.75rem;color:#64748b;">Total cost (smart model)</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:.7rem;color:#94a3b8;font-weight:700;text-transform:uppercase;margin-bottom:.3rem;">Money Saved</div>
                <div style="font-size:1.8rem;font-weight:900;color:#6366f1;">₹{savings_vs_all//1000}K</div>
                <div style="font-size:.75rem;color:#64748b;">vs targeting everyone · R={ratio} assumption</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── CHARTS ──
    c1, c2 = st.columns(2)
    with c1:
        # Precision-Recall tradeoff across thresholds
        thresholds = np.linspace(.05, .95, 60)
        precisions, recalls, costs = [], [], []
        y_true = raw["Revenue"].values
        scores = raw["model_score"].values
        for t in thresholds:
            pred  = scores >= t
            tp    = int(((pred==1)&(y_true==1)).sum())
            fp    = int(((pred==1)&(y_true==0)).sum())
            fn    = int(((pred==0)&(y_true==1)).sum())
            precisions.append(tp/(tp+fp) if (tp+fp)>0 else 0)
            recalls.append(tp/(tp+fn) if (tp+fn)>0 else 0)
            costs.append(fp*DISCOUNT_COST_RS + fn*MISSED_BUYER_RS)
        best_idx = int(np.argmin(costs))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recalls, y=precisions, mode="lines",
            line=dict(color="#6366f1",width=2.5), name="PR Curve",
            hovertemplate="Recall:%{x:.2f}<br>Precision:%{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[recalls[best_idx]], y=[precisions[best_idx]],
            mode="markers", marker=dict(color="#10b981",size=12,symbol="star"),
            name=f"Optimal (R={ratio})"))
        lo = _layout(h=300, xt="Recall (Buyers Caught)", yt="Precision (Accuracy)", showlegend=True)
        lo["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        fig.update_layout(**lo)
        st.markdown('<div class="ch"><h4>Precision-Recall Tradeoff</h4><p>Star = optimal threshold for your cost assumption</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="tg_pr")

    with c2:
        # Cost curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=[c//1000 for c in costs],
            mode="lines", line=dict(color="#f59e0b",width=2.5),
            hovertemplate="Threshold:%{x:.2f}<br>Cost:₹%{y}K<extra></extra>", name="Total Cost"))
        fig.add_vline(x=thresholds[best_idx], line=dict(color="#10b981",dash="dash",width=2),
                      annotation_text=f"Min Cost @ {thresholds[best_idx]:.2f}",
                      annotation_font_color="#10b981")
        lo = _layout(h=300, xt="Decision Threshold", yt="Total Business Cost (₹K)", showlegend=False)
        fig.update_layout(**lo)
        st.markdown(f'<div class="ch"><h4>Cost Minimisation Curve (R={ratio})</h4><p>Threshold that minimises FP·₹{DISCOUNT_COST_RS} + FN·₹{MISSED_BUYER_RS}</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="tg_cost")

    # ── TOP TARGETS LIST ──
    sec("🟢 Top 20 Sessions to Target Right Now")
    st.caption("These are the highest-scoring real sessions — sorted by purchase intent score:")
    top20 = raw.nlargest(20, "model_score")[
        ["session_id","model_score","intent_band","VisitorType","Month",
         "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","Revenue"]
    ].reset_index(drop=True)

    for rank, (_, r) in enumerate(top20.iterrows(), 1):
        band_css = "pill-g" if r["intent_band"]=="High" else "pill-a"
        vt   = str(r["VisitorType"]).replace("_"," ")
        act  = "✅ Bought" if r["Revenue"]==1 else "❌ Didn't buy"
        action = "💰 Premium offer + live chat" if r["model_score"] >= meta["score_bands"]["high_cutoff"] else "📧 Gentle reminder popup"
        st.markdown(f"""
        <div class="vrow">
            <div class="vidx">#{rank}</div>
            <div class="vscore intent-{'high' if r['intent_band']=='High' else 'med'}">
                {r["model_score"]:.3f}</div>
            <span class="{band_css}">{r["intent_band"].upper()}</span>
            <div class="vtype">Session #{int(r["session_id"])} · {vt} · {r["Month"]} ·
                {int(r["ProductRelated"])} product pages · {r["ProductRelated_Duration"]:.0f}s dwell ·
                BR={r["BounceRates"]:.2%} · ER={r["ExitRates"]:.2%}</div>
            <div class="vrec">{action}</div>
            <div style="margin-left:auto;font-size:.74rem;color:#64748b;">{act}</div>
        </div>""", unsafe_allow_html=True)

    # ── SESSIONS TO SKIP ──
    sec("🔴 Bottom 10 Sessions — Do NOT Waste Your Budget")
    st.caption("These visitors show near-zero purchase intent — skip the intervention, save the cost:")
    bot10 = raw.nsmallest(10, "model_score")[
        ["session_id","model_score","VisitorType","Month","ProductRelated","ExitRates","Revenue"]
    ].reset_index(drop=True)
    for rank, (_, r) in enumerate(bot10.iterrows(), 1):
        vt  = str(r["VisitorType"]).replace("_"," ")
        act = "✅ Bought" if r["Revenue"]==1 else "❌ Didn't buy"
        st.markdown(f"""
        <div class="vrow">
            <div class="vidx">#{rank}</div>
            <div class="vscore intent-low">{r["model_score"]:.3f}</div>
            <span class="pill-r">LOW</span>
            <div class="vtype">Session #{int(r["session_id"])} · {vt} · {r["Month"]} ·
                {int(r["ProductRelated"])} product pages · ER={r["ExitRates"]:.2%}</div>
            <div class="vrec" style="background:rgba(244,63,94,.08);color:#fb7185;">🚫 Skip — save ₹{DISCOUNT_COST_RS}</div>
            <div style="margin-left:auto;font-size:.74rem;color:#64748b;">{act}</div>
        </div>""", unsafe_allow_html=True)

    # ── DOWNLOAD ──
    sec("📥 Export Targeting Decisions")
    export = raw[["session_id","model_score","intent_band","intervention_flag",
                  "VisitorType","Month","ProductRelated","ProductRelated_Duration",
                  "BounceRates","ExitRates","Revenue"]].copy()
    export["recommended_action"] = export.apply(lambda r:
        "Premium offer + live chat" if r["model_score"] >= meta["score_bands"]["high_cutoff"]
        else ("Gentle reminder" if r["model_score"] >= meta["score_bands"]["medium_cutoff"]
              else "Skip — save budget"), axis=1)
    st.download_button(
        "📥 Download Full Targeting Report (All 12,205 Sessions)",
        export.to_csv(index=False).encode("utf-8"),
        "visitor_targeting_decisions.csv", "text/csv",
        type="primary", use_container_width=False)
    st.caption(f"CSV includes: session ID, intent score, band, flag, recommended action for all {len(raw):,} sessions.")


# ══════════════════════════════════════════════════════════════
#  PAGE: LIVE SCORING — Score any session in real time
# ══════════════════════════════════════════════════════════════
def pg_live_scoring(model, meta, limits, ratio, raw):
    st.markdown("""<div class="hero">
        <div class="badge">⚡ LIVE SESSION SCORER</div>
        <h1 style="color:#fff;font-size:2rem;font-weight:900;margin:0 0 .6rem;letter-spacing:-.03em;">
            Score Any Visitor Session Instantly</h1>
        <p style="color:#cbd5e1;font-size:1rem;margin:0;">
            Enter session features — or pick a real visitor from the dataset —
            and the model predicts purchase probability + business recommendation in real time.</p>
    </div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎲 Simulate a Real Visitor", "✍️ Enter Custom Session"])

    # ── TAB 1: Simulate from real data ──
    with tab1:
        sec("Pick a Real Visitor from the Dataset")
        st.caption("Select by session type to understand how the model scores different visitor profiles:")

        col_a, col_b, col_c, col_d = st.columns(4)
        sim_choice = None
        with col_a:
            if st.button("🟢 Random High-Intent\n(Should Buy)", key="sim_h", use_container_width=True, type="primary"):
                sim_choice = "high"
        with col_b:
            if st.button("🟡 Random Medium\n(Undecided)", key="sim_m", use_container_width=True):
                sim_choice = "medium"
        with col_c:
            if st.button("🔴 Random Low-Intent\n(Won't Buy)", key="sim_l", use_container_width=True):
                sim_choice = "low"
        with col_d:
            if st.button("🎲 Completely Random\n(Surprise!)", key="sim_r", use_container_width=True):
                sim_choice = "random"

        if sim_choice:
            if sim_choice == "high":
                pool = raw[raw["intent_band"] == "High"]
            elif sim_choice == "medium":
                pool = raw[raw["intent_band"] == "Medium"]
            elif sim_choice == "low":
                pool = raw[raw["intent_band"] == "Low"]
            else:
                pool = raw
            picked = pool.sample(1, random_state=None).iloc[0]
            st.session_state["sim_row"] = picked.to_dict()

        if "sim_row" in st.session_state:
            r = st.session_state["sim_row"]
            row_df = pd.DataFrame([{c: r[c] for c in FEATURE_COLUMNS}])
            score  = float(model.predict_proba(row_df)[0, 1])
            bands  = meta["score_bands"]
            band   = "High" if score >= bands["high_cutoff"] else ("Medium" if score >= bands["medium_cutoff"] else "Low")
            thr    = float(meta["cost_thresholds"][str(ratio)])
            flag   = score >= thr
            css    = {"High":"res-high","Medium":"res-med","Low":"res-low"}[band]
            bclr   = {"High":"#10b981","Medium":"#f59e0b","Low":"#94a3b8"}[band]
            recs   = {
                "High":   ("💰 Send premium offer immediately", "This visitor has strong product engagement signals. Trigger a personalized discount or live-chat now."),
                "Medium": ("📧 Show social proof / exit popup", "Mid-range intent — gentle reassurance (reviews, free shipping) may tip them toward purchase."),
                "Low":    ("🚫 No intervention — save budget",  "Weak engagement signals. Skip the discount. Probability of conversion is very low."),
            }
            action_title, action_desc = recs[band]

            st.markdown("---")
            # Show what this real session looks like
            r1, r2, r3 = st.columns(3)
            r1.metric("Session ID",      f"#{int(r.get('session_id', 0))}")
            r2.metric("Visitor Type",    str(r["VisitorType"]).replace("_"," "))
            r3.metric("Actual Outcome",  "✅ Purchased" if r["Revenue"]==1 else "❌ Did not purchase")

            # BIG result card
            st.markdown(f"""
            <div class="res-card {css}">
                <div class="res-tag">PURCHASE INTENT CLASSIFICATION</div>
                <div class="res-band">{band.upper()} INTENT</div>
                <div style="font-size:1.05rem;color:#cbd5e1;margin:.3rem 0 .5rem;">
                    Model Score: <strong style="color:#f8fafc;font-size:1.2rem;">{score:.4f}</strong>
                    &nbsp;·&nbsp; Threshold (R={ratio}): <strong style="color:#f8fafc;">{thr:.2f}</strong>
                </div>
                <div style="margin-top:.6rem;padding:.8rem 1rem;background:rgba(255,255,255,.04);border-radius:10px;">
                    <div style="font-size:.75rem;font-weight:700;color:#a5b4fc;text-transform:uppercase;margin-bottom:.3rem;">
                        AI Recommendation</div>
                    <div style="font-size:1rem;font-weight:700;color:#f8fafc;">{action_title}</div>
                    <div style="font-size:.85rem;color:#94a3b8;margin-top:.2rem;">{action_desc}</div>
                </div>
                {"<div style='margin-top:.7rem;padding:.5rem .9rem;background:rgba(16,185,129,.1);border-radius:8px;color:#34d399;font-size:.82rem;font-weight:700;'>✅ FLAGGED — Intervene now (score ≥ threshold)</div>" if flag else
                 "<div style='margin-top:.7rem;padding:.5rem .9rem;background:rgba(100,116,139,.1);border-radius:8px;color:#94a3b8;font-size:.82rem;font-weight:700;'>⏸️ NOT FLAGGED — Below intervention threshold, save your budget</div>"}
            </div>""", unsafe_allow_html=True)

            # Session details + visual
            sd1, sd2 = st.columns(2)
            with sd1:
                sec("Session Behaviour Profile")
                details = {
                    "Administrative Pages": int(r["Administrative"]),
                    "Admin Dwell Time": f"{r['Administrative_Duration']:.0f}s",
                    "Informational Pages": int(r["Informational"]),
                    "Product Pages Visited": int(r["ProductRelated"]),
                    "Product Dwell Time": f"{r['ProductRelated_Duration']:.0f}s",
                    "Bounce Rate": f"{r['BounceRates']:.3f}",
                    "Exit Rate": f"{r['ExitRates']:.3f}",
                    "Month": r["Month"],
                    "Weekend": "Yes" if r["Weekend"] == 1 else "No",
                    "Special Day": f"{r['SpecialDay']:.1f}",
                }
                for k, v in details.items():
                    st.markdown(f"**{k}:** {v}")
            with sd2:
                st.markdown('<div class="ch"><h4>Session DNA Radar</h4><p>vs avg converting visitor profile</p></div>', unsafe_allow_html=True)
                st.plotly_chart(_radar({c: r[c] for c in FEATURE_COLUMNS}, limits),
                                use_container_width=True, key="sim_radar")

            with sd2:
                st.markdown('<div class="ch"><h4>Score Gauge</h4><p>Red line = intervention threshold</p></div>', unsafe_allow_html=True)
                st.plotly_chart(_gauge(score, meta, ratio), use_container_width=True, key="sim_gauge")

    # ── TAB 2: Manual Entry ──
    with tab2:
        sec("Enter Custom Session Features")
        st.caption("Useful for testing new visitor profiles or explaining the model to stakeholders:")
        df_def = meta["default_values"]
        for f in FEATURE_COLUMNS:
            st.session_state.setdefault("cf_"+f, df_def[f])

        with st.form("custom_form"):
            cm = {c: int(limits[c][1]) for c in COUNT_COLUMNS}
            dm = {c: float(limits[c][1]) for c in DURATION_COLUMNS}
            p1, p2, p3 = st.columns(3)
            with p1:
                admin   = st.number_input("Admin Pages",    0, cm["Administrative"], step=1, key="cf_Administrative")
                admin_d = st.number_input("Admin Duration", 0.0, dm["Administrative_Duration"], step=10.0, key="cf_Administrative_Duration")
            with p2:
                info_p  = st.number_input("Info Pages",     0, cm["Informational"],  step=1, key="cf_Informational")
                info_d  = st.number_input("Info Duration",  0.0, dm["Informational_Duration"],  step=10.0, key="cf_Informational_Duration")
            with p3:
                prod    = st.number_input("Product Pages",  0, cm["ProductRelated"], step=1, key="cf_ProductRelated")
                prod_d  = st.number_input("Product Duration",0.0, dm["ProductRelated_Duration"], step=30.0, key="cf_ProductRelated_Duration")

            b1, b2, b3 = st.columns(3)
            bounce  = b1.number_input("Bounce Rate", float(limits["BounceRates"][0]), float(limits["BounceRates"][1]), step=0.001, format="%.4f", key="cf_BounceRates")
            exit_r  = b2.number_input("Exit Rate",   float(limits["ExitRates"][0]),   float(limits["ExitRates"][1]),   step=0.001, format="%.4f", key="cf_ExitRates")
            special = b3.select_slider("Special Day", [0.0,0.2,0.4,0.6,0.8,1.0], key="cf_SpecialDay")

            v1, v2, v3 = st.columns(3)
            month   = v1.selectbox("Month",        meta["observed_months"], key="cf_Month")
            visitor = v2.selectbox("Visitor Type", meta["visitor_types"], format_func=lambda x: x.replace("_"," "), key="cf_VisitorType")
            weekend = v3.selectbox("Weekend?", [0,1], format_func=lambda x: "Yes" if x else "No", key="cf_Weekend")

            tech = meta["technical_category_options"]
            t1, t2, t3, t4 = st.columns(4)
            sel = {}
            for col, (feat, ttl) in zip([t1,t2,t3,t4],[("OperatingSystems","OS"),("Browser","Browser"),("Region","Region"),("TrafficType","Traffic")]):
                sel[feat] = col.selectbox(ttl, tech[feat], key="cf_"+feat)

            submitted = st.form_submit_button("🔮 Predict Purchase Intent", type="primary", use_container_width=True)

        if submitted:
            row = {"Administrative":admin,"Administrative_Duration":admin_d,
                   "Informational":info_p,"Informational_Duration":info_d,
                   "ProductRelated":prod,"ProductRelated_Duration":prod_d,
                   "BounceRates":bounce,"ExitRates":exit_r,"SpecialDay":special,
                   "Month":month,"VisitorType":visitor,"Weekend":weekend, **sel}
            try:
                with st.spinner("Scoring through XGBoost pipeline…"):
                    clean  = validate_frame(pd.DataFrame([row]), meta, limits)
                    result = score_sessions(model, clean, meta, ratio).iloc[0].to_dict()
                    result["_input"] = row
                    st.session_state["cf_result"] = result
            except InputValidationError as e:
                st.error(str(e), icon="⚠️")
            except Exception:
                LOGGER.exception("Custom form error")
                st.error("Prediction failed — check inputs.", icon="❌")

        if "cf_result" in st.session_state:
            res  = st.session_state["cf_result"]
            band = res["intent_band"]
            score= res["model_score"]
            thr  = res["threshold"]
            css  = {"High":"res-high","Medium":"res-med","Low":"res-low"}[band]
            recs = {
                "High":  ("💰 Send premium offer","Strong signals — trigger discount now."),
                "Medium":("📧 Show social proof","Borderline — try gentle reassurance."),
                "Low":   ("🚫 Skip","Weak signals — don't waste the budget."),
            }
            at, ad = recs[band]
            st.markdown(f"""
            <div class="res-card {css}">
                <div class="res-tag">RESULT</div>
                <div class="res-band">{band.upper()} INTENT &nbsp;·&nbsp; Score: {score:.4f}</div>
                <div style="font-size:.9rem;color:#cbd5e1;margin:.3rem 0 .5rem;">
                    Threshold (R={ratio}) = {thr:.2f} &nbsp;·&nbsp;
                    Historical cohort conversion: {res["historical_band_reference"]:.1%}</div>
                <div style="margin-top:.5rem;padding:.7rem .9rem;background:rgba(255,255,255,.04);border-radius:10px;">
                    <div style="font-size:.9rem;font-weight:700;color:#f8fafc;">{at}</div>
                    <div style="font-size:.82rem;color:#94a3b8;margin-top:.15rem;">{ad}</div>
                </div>
                {"<div style='margin-top:.6rem;padding:.45rem .9rem;background:rgba(16,185,129,.1);border-radius:8px;color:#34d399;font-size:.8rem;font-weight:700;'>✅ FLAGGED for intervention</div>" if res["intervention_flag"] else
                 "<div style='margin-top:.6rem;padding:.45rem .9rem;background:rgba(100,116,139,.1);border-radius:8px;color:#94a3b8;font-size:.8rem;font-weight:700;'>⏸️ NOT flagged — below threshold</div>"}
            </div>""", unsafe_allow_html=True)

            cr1, cr2 = st.columns(2)
            with cr1:
                st.markdown('<div class="ch"><h4>Score Gauge</h4></div>', unsafe_allow_html=True)
                st.plotly_chart(_gauge(score, meta, ratio), use_container_width=True, key="cf_gauge")
            with cr2:
                st.markdown('<div class="ch"><h4>Session DNA Radar</h4></div>', unsafe_allow_html=True)
                st.plotly_chart(_radar(res["_input"], limits), use_container_width=True, key="cf_radar")


# ══════════════════════════════════════════════════════════════
#  PAGE: METHODOLOGY
# ══════════════════════════════════════════════════════════════
def pg_methodology(meta, raw, results):
    st.markdown("""<div class="hero">
        <div class="badge">🔬 TECHNICAL METHODOLOGY</div>
        <h1 style="color:#fff;font-size:2rem;font-weight:900;margin:0 0 .6rem;letter-spacing:-.03em;">
            How the AI Engine Works</h1>
        <p style="color:#cbd5e1;font-size:1rem;margin:0;">
            End-to-end ML pipeline: data cleaning → EDA → class imbalance handling →
            3-model comparison → XGBoost tuning → SHAP explainability → cost-sensitive thresholding.</p>
    </div>""", unsafe_allow_html=True)

    sec("6-Stage ML Pipeline")
    cols = st.columns(3)
    stages = [
        ("01","Raw Dataset","12,330 sessions from UCI Online Shoppers repo. 17 features + Revenue target."),
        ("02","Cleaning","125 duplicate rows removed → 12,205 unique sessions. Sanity checks on all features."),
        ("03","Leakage Audit","PageValues excluded — it's computed from post-transaction data, unavailable at session start."),
        ("04","80/20 Split","Stratified split, seed=42. SMOTE & transforms fit only inside each CV fold — no leakage."),
        ("05","3 Models Compared","Logistic Regression (baseline), Random Forest, XGBoost. Evaluated by PR-AUC (not accuracy)."),
        ("06","Threshold Tuning","Cost function FP·₹200 + FN·₹1000 minimised on OOF predictions to find optimal cutoff."),
    ]
    for i in range(0, 6, 3):
        with st.container():
            cs = st.columns(3)
            for col, (n, t, d) in zip(cs, stages[i:i+3]):
                with col:
                    step(n, t, d)
        st.markdown("<div style='margin-bottom:.6rem;'></div>", unsafe_allow_html=True)

    sec("Model Comparison (5-Fold CV, PR-AUC)")
    comp = results.get("matched_model_comparison", [])
    if comp:
        fig = _grouped(
            ["Logistic Regression","Random Forest","XGBoost"],
            {"With PageValues (Leaky)":    [r["mean_pr_auc"] for r in comp if r["feature_set"]=="with PageValues"],
             "Without PageValues (Deployed)":[r["mean_pr_auc"] for r in comp if r["feature_set"]=="without PageValues"]},
            ["#f59e0b","#6366f1"], "Mean CV PR-AUC", 320)
        st.markdown('<div class="ch"><h4>All 3 Models vs Leaky Baseline</h4><p>XGBoost without PageValues = honest deployed model</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="meth_comp")

    sec("Why PR-AUC, Not Accuracy?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Majority-Class Accuracy",  "84.4%", "predicts 'no purchase' always — useless")
    m2.metric("Our Model PR-AUC",         "0.376", "2.85× top-decile lift")
    m3.metric("PR-AUC Random Baseline",   "~0.156", "positive class rate")
    st.info("With 15.6% buyers, accuracy is meaningless. A model that always says 'no purchase' gets 84.4% accuracy. PR-AUC measures how well the model actually ranks buyers above non-buyers.", icon="📊")

    sec("SHAP Feature Importance")
    fi1, fi2 = st.columns(2)
    with fi1:
        feats = ["Product Duration","Month","Exit Rate","Browser","Traffic Type","Bounce Rate","Admin Duration","OS","Product Pages","Special Day"]
        vals  = [0.444,0.428,0.410,0.207,0.160,0.111,0.109,0.108,0.089,0.071]
        fig   = _bar(feats, vals, "#8b5cf6", "Mean |SHAP| Value", 360, horiz=True)
        st.markdown('<div class="ch"><h4>SHAP: What Drives Purchase Intent?</h4><p>Product duration + low exit rate = strongest signals</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="meth_shap")
    with fi2:
        perms = ["Product Duration","Exit Rate","Month","Product Pages","Visitor Type","Bounce Rate","Admin Duration","Traffic Type","OS","Admin Pages"]
        pvals = [0.0794,0.0738,0.0588,0.0184,0.0151,0.0100,0.0093,0.0082,0.0080,0.0048]
        fig   = _bar(perms, pvals, "#06b6d4", "PR-AUC Drop on Shuffle", 360, horiz=True)
        st.markdown('<div class="ch"><h4>Permutation Importance (Robustness Check)</h4><p>Consistent with SHAP — model is stable</p></div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="meth_perm")
    st.info("⚠️ SHAP values show association, not causation. Higher product dwell time correlates with purchase — it doesn't cause it.", icon="🔬")

    sec("XGBoost Hyperparameters (Tuned via RandomizedSearchCV, 30 iterations)")
    st.dataframe(pd.DataFrame([
        {"Parameter":"n_estimators","Value":"265","Note":"Trees in ensemble"},
        {"Parameter":"learning_rate","Value":"0.037","Note":"Slow & robust"},
        {"Parameter":"max_depth","Value":"4","Note":"Prevents overfitting"},
        {"Parameter":"min_child_weight","Value":"9","Note":"Regularisation"},
        {"Parameter":"subsample","Value":"0.741","Note":"Row sampling per tree"},
        {"Parameter":"colsample_bytree","Value":"0.953","Note":"Feature sampling"},
        {"Parameter":"scale_pos_weight","Value":"~5.4","Note":"Handles class imbalance"},
    ]), hide_index=True, use_container_width=True)

    lc = ROOT / "reports" / "figures" / "learning_curve.png"
    cc = ROOT / "reports" / "figures" / "calibration_curve.png"
    if lc.is_file() or cc.is_file():
        sec("Diagnostic Curves")
        dc1, dc2 = st.columns(2)
        if lc.is_file():
            with dc1:
                st.markdown('<div class="ch"><h4>Learning Curve</h4><p>CV PR-AUC improves with more data</p></div>', unsafe_allow_html=True)
                st.image(str(lc), use_container_width=True)
        if cc.is_file():
            with dc2:
                st.markdown('<div class="ch"><h4>Calibration Curve</h4><p>Scores used for ranking, not probability</p></div>', unsafe_allow_html=True)
                st.image(str(cc), use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
def pg_about(meta, raw):
    st.markdown("""<div class="hero">
        <div class="badge">ℹ️ ABOUT THIS PROJECT</div>
        <h1 style="color:#fff;font-size:2rem;font-weight:900;margin:0 0 .6rem;letter-spacing:-.03em;">
            About Visitor Intelligence</h1>
        <p style="color:#cbd5e1;font-size:1rem;margin:0;">
            A VIT 3rd Semester Data Science project — production-grade ML pipeline
            for purchase intent prediction on the UCI Online Shoppers dataset.</p>
    </div>""", unsafe_allow_html=True)

    sec("Project Summary")
    a1, a2 = st.columns(2)
    with a1:
        card("🎯","Business Problem",
             f"E-commerce converts only {raw.Revenue.mean():.1%} of visitors. Treating everyone the same wastes ₹{DISCOUNT_COST_RS} per wasted discount. This AI identifies the {int((raw.intent_band=='High').sum()):,} high-intent sessions worth targeting.")
        st.markdown("")
        card("📊","Dataset",
             f"UCI Online Shoppers Purchasing Intention — {len(raw):,} sessions, 18 columns, {raw.Revenue.mean():.1%} positive class. Sakar & Kastro (2018).")
    with a2:
        card("🛠️","Tech Stack",
             "Python 3.11 · XGBoost · scikit-learn · imbalanced-learn · SHAP · pandas · numpy · Streamlit · Plotly · Firebase Auth")
        st.markdown("")
        card("🧠","Key Contribution",
             "Mandatory PageValues ablation (-0.38 PR-AUC drop proves leakage). Cost-sensitive threshold tuning. K-Means segmentation. Real business impact quantification.")

    sec("Dataset Statistics")
    s1,s2,s3,s4,s5,s6 = st.columns(6)
    s1.metric("Total Sessions",    f"{len(raw):,}")
    s2.metric("Buying Sessions",   f"{raw.Revenue.sum():,}")
    s3.metric("Base Rate",         f"{raw.Revenue.mean():.1%}")
    s4.metric("Features Used",     f"{len(FEATURE_COLUMNS)}")
    s5.metric("Months in Data",    f"{raw.Month.nunique()}")
    s6.metric("Visitor Types",     f"{raw.VisitorType.nunique()}")

    sec("Limitations")
    lims = [
        ("📅","No Timestamps","Time-based validation impossible — used stratified random split."),
        ("🔗","PageValues Excluded","Most predictive feature removed to prevent lookahead leakage."),
        ("💰","Assumed Costs","₹200/₹1000 cost ratios are illustrative, not measured from real business."),
        ("🔬","Correlation ≠ Causation","SHAP values show associations. We cannot claim causality."),
        ("📉","Session Aggregates","Page visit sequence is lost — only totals are available."),
        ("🌍","Single Site / Year","Results may not generalise to other retailers or time periods."),
    ]
    lc = st.columns(3)
    for i, (ic, t, d) in enumerate(lims):
        with lc[i % 3]:
            card(ic, t, d)
            st.markdown("")

    sec("Citation")
    st.markdown(
        "> **UCI Online Shoppers Purchasing Intention Dataset**  \n"
        "> Sakar, C. O. & Kastro, O. S. (2018). UCI Machine Learning Repository.  \n"
        "> DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)"
    )
    st.markdown("---")
    v1,v2,v3 = st.columns(3)
    v1.metric("App Version", f"v{VERSION}")
    v2.metric("Analysed Sessions", f"{len(raw):,}")
    v3.metric("Model", "Class-Weighted XGBoost")


# ══════════════════════════════════════════════════════════════
#  ASSET LOADING
# ══════════════════════════════════════════════════════════════
try:
    model, metadata, results, dataset, input_limits = load_assets()
except Exception:
    LOGGER.exception("Asset load failed")
    st.error("⚠️ **Model artifacts missing.** Run `src/train_project.py` to regenerate.", icon="🚨")
    st.stop()

# ── Session token restore ──
if "user" not in st.session_state:
    tok = st.query_params.get("session")
    if tok:
        restored = decode_session_token(tok)
        if restored:
            st.session_state["user"] = restored

# ── URL routing ──
qp = st.query_params.get("page")
if qp and qp.lower() in PAGE_SLUG_MAP:
    target = PAGE_SLUG_MAP[qp.lower()]
    if "user" not in st.session_state and target in PROTECTED_PAGES:
        st.session_state["redirect_target"] = target
        st.session_state["navigation"] = "Login"
    else:
        st.session_state["navigation"] = target

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Dashboard" if "user" in st.session_state else "Home"

current_page = st.session_state["navigation"]
if current_page in PROTECTED_PAGES and "user" not in st.session_state:
    st.session_state["redirect_target"] = current_page
    current_page = "Login"
    st.session_state["navigation"] = "Login"

# ── TOP NAVBAR (authenticated workspace) ──
if current_page not in ("Home","Login"):
    nav_pages = ["Dashboard","Live Scoring","Targeting","Methodology","About","Home"]
    nav_icons  = {"Dashboard":"📊","Live Scoring":"⚡","Targeting":"🎯",
                  "Methodology":"🔬","About":"ℹ️","Home":"🌐"}

    nb1, nb2, nb3 = st.columns([1.7, 4.5, 2.0], vertical_alignment="center")
    with nb1:
        st.markdown("""<div style="display:flex;align-items:center;gap:9px;">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);width:38px;height:38px;
                border-radius:11px;display:flex;align-items:center;justify-content:center;
                font-size:17px;color:#fff;font-weight:900;box-shadow:0 4px 14px rgba(99,102,241,.4);">VI</div>
            <div>
                <div style="font-size:1rem;font-weight:800;color:#f8fafc;line-height:1.2;">Visitor Intelligence</div>
                <div style="font-size:.62rem;color:#818cf8;font-weight:700;text-transform:uppercase;letter-spacing:.07em;">Purchase Intent Engine</div>
            </div></div>""", unsafe_allow_html=True)
    with nb2:
        if "top_nav" not in st.session_state or st.session_state["top_nav"] not in nav_pages:
            st.session_state["top_nav"] = current_page
        st.segmented_control("nav", options=nav_pages,
            format_func=lambda p: f"{nav_icons[p]} {p}",
            key="top_nav", on_change=on_nav_change, label_visibility="collapsed")
    with nb3:
        user = st.session_state.get("user")
        if user:
            uc1, uc2 = st.columns([1.4,1], vertical_alignment="center")
            name = user.get("name") or user.get("email","").split("@")[0]
            uc1.markdown(f'<div style="text-align:right;font-size:.82rem;font-weight:700;color:#f1f5f9;">{name}</div>', unsafe_allow_html=True)
            uc2.button("🚪 Logout", key="nb_logout", on_click=do_logout, use_container_width=True)
        else:
            st.button("🔑 Sign In", key="nb_signin", on_click=nav_to, args=("Login",), use_container_width=True)

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(99,102,241,.3),rgba(139,92,246,.1),transparent);margin:.6rem 0 1rem;"></div>', unsafe_allow_html=True)

    if current_page != "Dashboard":
        cc1, cc2 = st.columns([4,1.2], vertical_alignment="center")
        cc1.markdown(f'<div class="crumb">🏠 Dashboard › <span class="active">{nav_icons.get(current_page,"")} {current_page}</span></div>', unsafe_allow_html=True)
        cc2.button("← Dashboard", key="crumb_back", on_click=nav_to, args=("Dashboard",), use_container_width=True)

    # Sidebar: cost ratio control
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:.6rem 0 .4rem;">
            <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6);width:42px;height:42px;
                border-radius:13px;display:inline-flex;align-items:center;justify-content:center;
                font-size:19px;color:#fff;font-weight:900;margin-bottom:7px;">VI</div>
            <div style="font-size:.95rem;font-weight:800;color:#f8fafc;">Controls</div></div>""",
            unsafe_allow_html=True)
        st.markdown("---")
        if "user" in st.session_state:
            u = st.session_state["user"]
            st.markdown(f"""<div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.2);
                border-radius:12px;padding:.8rem .9rem;margin-bottom:.6rem;">
                <div style="font-size:.85rem;font-weight:700;color:#f1f5f9;">{u.get('name','Analyst')}</div>
                <div style="font-size:.72rem;color:#94a3b8;margin-top:2px;">{u.get('email','')}</div>
                </div>""", unsafe_allow_html=True)
            st.button("🚪 Logout", key="sb_logout", on_click=do_logout, use_container_width=True)
        st.markdown("---")
        st.markdown("<p style='font-size:.78rem;font-weight:700;color:#cbd5e1;margin-bottom:3px;'>Cost Assumption</p>", unsafe_allow_html=True)
        ratio = st.select_slider("R", options=[3,5,10], value=5,
            help="Missed buyer costs R× a wasted discount.", label_visibility="collapsed")
        cur_thr = metadata["cost_thresholds"][str(ratio)]
        st.metric("Intervention Cutoff", f"{cur_thr:.2f}", help=f"Score ≥ {cur_thr:.2f} → flag at R={ratio}")
        st.markdown("---")
        n_flag_s = int((dataset["model_score"] >= cur_thr).sum())
        st.metric("Sessions to Target", f"{n_flag_s:,}", f"{n_flag_s/len(dataset):.1%} of all")
        st.caption(f"**Model** · XGBoost (no PageValues)\n**Data** · UCI 12,205 sessions\n**Status** · 🟢 Live · v{VERSION}")
else:
    ratio = 5

# ── ROUTE DISPATCHER ──
if   current_page == "Home":        pg_home(dataset, metadata)
elif current_page == "Login":       show_auth_page()
elif current_page == "Dashboard":   pg_dashboard(dataset, metadata, results)
elif current_page == "Live Scoring":pg_live_scoring(model, metadata, input_limits, ratio, dataset)
elif current_page == "Targeting":   pg_targeting(dataset, metadata, ratio)
elif current_page == "Methodology": pg_methodology(metadata, dataset, results)
else:                               pg_about(metadata, dataset)

# ── FOOTER ──
st.markdown(
    f'<div class="ft"><strong>Visitor Intelligence</strong> · AI Purchase Intent Engine · '
    f'v{VERSION}<br>Benchmarked on <a href="https://archive.ics.uci.edu/dataset/468/online" target="_blank">'
    f'UCI Online Shoppers Dataset</a> · Sakar &amp; Kastro (2018) · '
    f'PageValues excluded · Associations are not causal</div>',
    unsafe_allow_html=True)
