"""
patch_navbar.py  —  Replace top navbar + public nav with a professional design.
Run: python patch_navbar.py
"""
from pathlib import Path

APP = Path(__file__).parent / "app" / "app.py"
src = APP.read_text(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Add extra navbar CSS right before the closing </style> tag in the CSS block
# ──────────────────────────────────────────────────────────────────────────────
EXTRA_CSS = """
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
"""

# Inject just before the closing </style>
if "PROFESSIONAL NAVBAR OVERRIDE" not in src:
    src = src.replace("</style>\n\"\"\"", EXTRA_CSS + "\n</style>\n\"\"\"", 1)
    print("  [OK] Extra navbar CSS injected")
else:
    print("  [SKIP] Extra navbar CSS already present")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  Replace the authenticated top navbar block
# ──────────────────────────────────────────────────────────────────────────────
OLD_NAV = '''    with st.container():
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

NEW_NAV = '''    with st.container():
        user = st.session_state.get("user")
        display_name = ""
        initials = "VI"
        badge_label = ""
        if user:
            display_name = user.get("name") or user.get("email", "Analyst").split("@")[0]
            initials = (display_name[:2]).upper()
            provider = user.get("provider", "password")
            badge_label = {"google.com": "Google", "password": "Email", "demo": "Guest"}.get(provider, "Auth")

        # ── Render the HTML shell of the navbar ──
        st.markdown(f"""
        <div style="
            display:flex; align-items:center; justify-content:space-between;
            padding:.55rem 1.3rem;
            background:var(--surface);
            border:1px solid var(--border);
            border-radius:14px;
            margin-bottom:.9rem;
            box-shadow:0 2px 12px rgba(0,0,0,.35);
        ">
            <!-- Brand -->
            <div style="display:flex;align-items:center;gap:9px;flex-shrink:0;">
                <div style="width:32px;height:32px;border-radius:8px;background:var(--primary);
                    display:flex;align-items:center;justify-content:center;
                    font-size:13px;font-weight:800;color:#fff;letter-spacing:-.02em;">VI</div>
                <div>
                    <div style="font-size:.88rem;font-weight:700;color:#f1f5f9;letter-spacing:-.02em;line-height:1.2;">Visitor Intelligence</div>
                    <div style="font-size:.58rem;font-weight:600;color:#818cf8;text-transform:uppercase;letter-spacing:.07em;">Purchase Intent Engine</div>
                </div>
            </div>
            <!-- Right: user info (buttons rendered via Streamlit columns below) -->
            <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
                {"" if not user else f\'\'\'
                <div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    display:flex;align-items:center;justify-content:center;
                    font-size:11px;font-weight:700;color:#fff;">{initials}</div>
                <div>
                    <div style="font-size:.80rem;font-weight:600;color:#e2e8f0;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{display_name}</div>
                    <div style="font-size:.60rem;font-weight:600;padding:1px 6px;border-radius:99px;
                        background:rgba(99,102,241,.12);color:#a5b4fc;border:1px solid rgba(99,102,241,.2);display:inline-block;">{badge_label}</div>
                </div>
                \'\'\'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Segmented nav control + logout/signin in one row ──
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

        st.markdown('<div style="height:1px;background:var(--border);margin:.2rem 0 1rem;"></div>',
                    unsafe_allow_html=True)'''

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Replace public homepage nav
# ──────────────────────────────────────────────────────────────────────────────
OLD_PUB = '''    # ── Public Navbar (single clean row) ──
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

NEW_PUB = '''    # ── Professional Public Navbar ──
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
                unsafe_allow_html=True)'''

# Apply patches
def patch(text, old, new, label):
    if old in text:
        print(f"  [OK] {label}")
        return text.replace(old, new, 1)
    print(f"  [SKIP — not found] {label}")
    return text

src = patch(src, OLD_NAV, NEW_NAV, "Authenticated top navbar")
src = patch(src, OLD_PUB, NEW_PUB, "Public homepage nav")

APP.write_text(src, encoding="utf-8")
print(f"\nDone — {APP.name} updated ({len(src):,} bytes)")

