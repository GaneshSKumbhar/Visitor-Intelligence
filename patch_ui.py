"""
patch_ui.py  —  Surgical UI patch for Visitor Intelligence app.py
Replaces: CSS block, top-navbar HTML, public home nav, smp-card CSS.
Run once from project root: python patch_ui.py
"""
from pathlib import Path

APP = Path(__file__).parent / "app" / "app.py"
src = APP.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────
# 1.  NEW CSS BLOCK
# ─────────────────────────────────────────────────────────────
OLD_CSS_START = '# ─── Design System & CSS Tokens ───\nst.markdown("""\n<style>'
OLD_CSS_END   = '</style>\n""", unsafe_allow_html=True)'

NEW_CSS = '''# ─── Design System & CSS Tokens (v3) ───
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
</style>
""", unsafe_allow_html=True)'''

# ─────────────────────────────────────────────────────────────
# 2.  NEW TOP NAVBAR for authenticated pages
# ─────────────────────────────────────────────────────────────
OLD_NAV = '''    with st.container():
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

        st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(99,102,241,0.3) 0%,rgba(139,92,246,0.1) 60%,transparent 100%);margin:0.7rem 0 1.2rem;"></div>', unsafe_allow_html=True)'''

NEW_NAV = '''    with st.container():
        # ── Brand + segmented nav + user block ──
        col_brand, col_nav, col_user = st.columns([1.6, 4.6, 2.2], vertical_alignment="center")

        with col_brand:
            st.markdown("""
            <div class="nb-brand">
                <div class="nb-logo">VI</div>
                <div>
                    <div class="nb-title">Visitor Intelligence</div>
                    <div class="nb-sub">Purchase Intent Engine</div>
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
                display_name = user.get("name") or user.get("email", "Analyst").split("@")[0]
                initials = display_name[:2].upper()
                provider = user.get("provider", "password")
                badge_map = {"google.com": "Google", "password": "Email", "demo": "Guest"}
                badge = badge_map.get(provider, "Auth")
                u_info_col, u_btn_col = st.columns([1.5, 1.0], vertical_alignment="center")
                with u_info_col:
                    st.markdown(f"""
                    <div class="nb-user">
                        <div class="nb-avatar">{initials}</div>
                        <div>
                            <div class="nb-uname">{display_name}</div>
                            <div class="nb-badge">{badge}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with u_btn_col:
                    st.button("Sign out", key="top_navbar_logout_btn", on_click=do_logout, use_container_width=True)
            else:
                st.button("Sign in", key="top_navbar_signin_btn", on_click=nav_to, args=("Login",), use_container_width=True)

        st.markdown('<div style="height:1px;background:var(--border);margin:.5rem 0 1rem;"></div>', unsafe_allow_html=True)'''

# ─────────────────────────────────────────────────────────────
# 3.  NEW PUBLIC HOME NAV
# ─────────────────────────────────────────────────────────────
OLD_HOME_NAV = '''    # Public Navbar
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

    st.markdown("<div style='margin-bottom: 1.8rem;'></div>", unsafe_allow_html=True)'''

NEW_HOME_NAV = '''    # ── Public Navbar (single clean row) ──
    nb_brand_col, nb_links_col, nb_cta_col = st.columns([1.8, 3.5, 2.4], vertical_alignment="center")
    with nb_brand_col:
        st.markdown("""
        <div class="pub-nav" style="margin-bottom:0;border:none;background:transparent;padding:.2rem 0;box-shadow:none;">
            <div class="brand-group">
                <div class="brand-icon">VI</div>
                <div>
                    <div class="brand-title">Visitor Intelligence</div>
                    <div class="brand-sub">Purchase Intent Engine</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with nb_links_col:
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.button("Home",        key="pnav_home",  on_click=nav_to, args=("Home",),        use_container_width=True)
        lc2.button("Methodology", key="pnav_meth",  on_click=nav_to, args=("Methodology",), use_container_width=True)
        lc3.button("Insights",    key="pnav_ins",   on_click=nav_to, args=("Insights",),    use_container_width=True)
        lc4.button("About",       key="pnav_about", on_click=nav_to, args=("About",),       use_container_width=True)
    with nb_cta_col:
        rc1, rc2 = st.columns(2)
        if "user" in st.session_state:
            rc1.button("Dashboard", key="pnav_dash",   on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)
            rc2.button("Sign out",  key="pnav_logout", on_click=do_logout,                  use_container_width=True)
        else:
            rc1.button("Sign in",    key="pnav_login",    on_click=nav_to, args=("Login",),    use_container_width=True)
            rc2.button("Get started",key="pnav_getstart", on_click=nav_to, args=("Overview",), type="primary", use_container_width=True)
    st.markdown("<div style='height:1px;background:var(--border);margin:.6rem 0 1.6rem;'></div>", unsafe_allow_html=True)'''

# ─────────────────────────────────────────────────────────────
# Apply patches
# ─────────────────────────────────────────────────────────────
def patch(text, old, new, label):
    if old in text:
        print(f"  [OK] Patched: {label}")
        return text.replace(old, new, 1)
    else:
        print(f"  [SKIP] Not found (already patched or changed): {label}")
        return text

src = patch(src, OLD_CSS_START + src[src.index(OLD_CSS_START)+len(OLD_CSS_START):src.index(OLD_CSS_END)+len(OLD_CSS_END)],
            NEW_CSS, "CSS block") if OLD_CSS_START in src and OLD_CSS_END in src else src
src = patch(src, OLD_NAV,      NEW_NAV,      "Top navbar")
src = patch(src, OLD_HOME_NAV, NEW_HOME_NAV, "Public home nav")

APP.write_text(src, encoding="utf-8")
print(f"\nDone — {APP} updated ({len(src):,} bytes)")

