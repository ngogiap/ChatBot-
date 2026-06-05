import streamlit as st
import requests
import os
from dotenv import load_dotenv
from streamlit_oauth import OAuth2Component

from database import (
    init_db, create_admin,
    create_user, create_google_user,
    login, login_google, login_local,
    get_user
)
from admin import admin_page
from userchat import chat_page
from style import apply_styles

load_dotenv()

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI         = "http://localhost:8501"
AUTHORIZE_URL        = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL            = "https://oauth2.googleapis.com/token"
USERINFO_URL         = "https://www.googleapis.com/oauth2/v3/userinfo"

st.set_page_config(
    page_title="Chatbot Sinh viên GTVT",
    page_icon="🎓",
    layout="wide"
)

apply_styles()

if "user" not in st.session_state:
    st.session_state.user = None


# ─── Google OAuth ─────────────────────────────────────────────────────────────

def handle_google_login():
    oauth2 = OAuth2Component(
        GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
        AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, TOKEN_URL
    )
    result = oauth2.authorize_button(
        "🔵 Tiếp tục với Google",
        redirect_uri=REDIRECT_URI,
        scope="openid email profile",
        use_container_width=True,
        pkce="S256"
    )
    if not result or "token" not in result:
        return None

    token    = result["token"]["access_token"]
    userinfo = requests.get(
        USERINFO_URL,
        headers={"Authorization": f"Bearer {token}"}
    ).json()

    email     = userinfo.get("email")
    full_name = userinfo.get("name", "")

    if not email:
        st.error("Không lấy được email từ Google.")
        return None

    existing = get_user(email)
    if existing:
        return login_google(email)
    else:
        st.session_state.google_pending = {"email": email, "full_name": full_name}
        return None


# ─── Form đặt mật khẩu sau Google login lần đầu ──────────────────────────────

def set_password_page():
    pending = st.session_state.google_pending
    email   = pending["email"]

    st.markdown(f"""
    <div style="text-align:center;padding:2rem 0 1rem">
        <h2>👋 Xin chào!</h2>
        <p style="color:#64748b">
            Đăng nhập Google thành công với <b>{email}</b><br>
            Bạn có muốn đặt mật khẩu để đăng nhập bằng email/mật khẩu sau này không?
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_set_password"):
        pw1 = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu (tối thiểu 6 ký tự)...")
        pw2 = st.text_input("Xác nhận", type="password", placeholder="Nhập lại mật khẩu...")
        c1, c2 = st.columns(2)
        with c1:
            submitted = st.form_submit_button("✅ Đặt mật khẩu", use_container_width=True, type="primary")
        with c2:
            skipped = st.form_submit_button("⏭️ Bỏ qua", use_container_width=True)

    if submitted:
        if not pw1 or not pw2:
            st.error("Vui lòng nhập mật khẩu.")
        elif pw1 != pw2:
            st.error("Mật khẩu không khớp.")
        elif len(pw1) < 6:
            st.error("Mật khẩu phải ít nhất 6 ký tự.")
        else:
            create_user(
                username=email, password=pw1,
                role="user", full_name=pending["full_name"],
                auth_type="both"
            )
            st.session_state.user           = get_user(email)
            st.session_state.google_pending = None
            st.rerun()

    if skipped:
        create_google_user(email, pending["full_name"])
        st.session_state.user           = get_user(email)
        st.session_state.google_pending = None
        st.rerun()


# ─── Sidebar login/logout ─────────────────────────────────────────────────────

def render_sidebar_auth():
    """
    Hiện thông tin user hoặc form đăng nhập trong sidebar.
    Chỉ render 1 lần mỗi script run để tránh duplicate key.
    """
    if st.session_state.get("_auth_rendered"):
        return
    st.session_state["_auth_rendered"] = True

    user = st.session_state.user

    if user:
        # Đã đăng nhập
        st.sidebar.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:10px;padding:12px 14px;margin-bottom:12px">
            <div style="color:var(--text-primary);font-weight:600;font-size:0.92rem">
                {'🔴' if user.get('role')=='admin' else '🟢'} {user.get('full_name') or user['username']}
            </div>
            <div style="color:var(--text-muted);font-size:0.75rem;margin-top:2px">
                @{user['username']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.sidebar.button("🚪 Đăng xuất", use_container_width=True, key="btn_logout_sidebar"):
            st.session_state.user     = None
            st.session_state.messages = []
            st.session_state.current_conv_id = None
            st.rerun()
    else:
        # Chưa đăng nhập — hiện form nhỏ gọn trong sidebar
        with st.sidebar.expander("🔐 Đăng nhập / Đăng ký", expanded=False):
            tab_login, tab_reg, tab_gg = st.tabs(["Đăng nhập", "Đăng ký", "Google"])

            with tab_login:
                u = st.text_input("Username", key="sl_user")
                p = st.text_input("Mật khẩu", type="password",autocomplete="off", key="sl_pass")
                if st.button("Đăng nhập", key="sl_btn", use_container_width=True, type="primary"):
                    if u and p:
                        result = login_local(u, p) or login(u, p)
                        if result:
                            st.session_state.user = dict(result)
                            st.rerun()
                        else:
                            st.error("Sai tài khoản hoặc mật khẩu.")
                    else:
                        st.warning("Nhập đầy đủ thông tin.")

            with tab_reg:
                nu = st.text_input("Username", key="sr_user")
                nf = st.text_input("Họ tên",   key="sr_name")
                np = st.text_input("Mật khẩu", type="password", autocomplete="off", key="sr_pass")
                if st.button("Tạo tài khoản", key="sr_btn", use_container_width=True, type="primary"):
                    if nu and np:
                        ok = create_user(nu, np, role="user", full_name=nf)
                        if ok:
                            st.success("✅ Tạo thành công!")
                        else:
                            st.error("Username đã tồn tại.")
                    else:
                        st.warning("Nhập đầy đủ thông tin.")

            with tab_gg:
                if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
                    result = handle_google_login()
                    if result:
                        st.session_state.user = result
                        st.rerun()
                else:
                    st.warning("Chưa cấu hình Google OAuth.")


# ─── Routing ─────────────────────────────────────────────────────────────────

# Reset flag mỗi lần script chạy lại
st.session_state["_auth_rendered"] = False

init_db()
create_admin()

# Đang đặt mật khẩu sau Google login
if st.session_state.get("google_pending"):
    apply_styles()
    set_password_page()

# Admin → trang quản trị
elif st.session_state.user and st.session_state.user.get("role") == "admin":
    admin_page()

# Tất cả trường hợp còn lại → trang chat (guest hoặc member)
else:
    render_sidebar_auth()
    chat_page()