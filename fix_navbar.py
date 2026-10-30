"""
fix_navbar.py  —  Fix raw HTML showing as text in the top navbar.
The issue: nested f-string with escaped quotes broke interpolation.
Solution: pre-build the user HTML as a plain string, then inject it.
"""
from pathlib import Path
import re

APP = Path(__file__).parent / "app" / "app.py"
src = APP.read_text(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Replace the broken authenticated navbar block
# ──────────────────────────────────────────────────────────────────────────────
OLD = '''    with st.container():
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

NEW = '''    with st.container():
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
        )'''

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    print("  [OK] Navbar fixed")
else:
    print("  [ERROR] Could not find the old navbar block — check manually")

APP.write_text(src, encoding="utf-8")
print(f"Done — {APP.name} updated ({len(src):,} bytes)")

