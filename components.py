"""
components.py — Tất cả UI components dùng chung cho admin và user
"""

import streamlit as st


# ─── ADMIN COMPONENTS ─────────────────────────────────────────────────────────

def sidebar_user_card(full_name, username, role="user"):
    """Card thông tin user trong sidebar."""
    role_icon  = "🔴" if role == "admin" else "🟢"
    role_label = "Administrator" if role == "admin" else "User"
    st.markdown(f"""
    <div class="sidebar-user">
        <div class="sidebar-user-name">{role_icon} {full_name or username}</div>
        <div class="sidebar-user-role">@{username} · {role_label}</div>
    </div>
    """, unsafe_allow_html=True)


def admin_header(title, subtitle, icon="⚙️"):
    """Header của mỗi trang admin."""
    st.markdown(f"""
    <div class="admin-header">
        <div class="admin-header-icon">{icon}</div>
        <div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_card(label, value):
    """Card hiển thị số liệu thống kê."""
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-card-label">{label}</div>
        <div class="stat-card-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def collection_card(name, desc, n_docs, n_chunks):
    """Card hiển thị thông tin kho tài liệu."""
    st.markdown(f"""
    <div class="collection-card">
        <div class="collection-card-title">📚 {name}</div>
        <div class="collection-card-desc">{desc or 'Không có mô tả'}</div>
        <div class="collection-card-meta">
            <div class="meta-pill"><b>{n_docs}</b> tài liệu</div>
            <div class="meta-pill"><b>{n_chunks}</b> chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def doc_card(name, n_chunks):
    """Card hiển thị thông tin tài liệu."""
    st.markdown(f"""
    <div class="doc-card">
        <div class="doc-icon">📄</div>
        <div class="doc-info">
            <div class="doc-name">{name}</div>
            <div class="doc-meta">{n_chunks} chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def breadcrumb(items):
    """
    Breadcrumb điều hướng.
    items = [("Kho tri thức", False), ("Sổ tay SV", False), ("file.pdf", True)]
    Item cuối là active.
    """
    parts = []
    for i, (label, is_active) in enumerate(items):
        if is_active:
            parts.append(f'<span class="breadcrumb-active">{label}</span>')
        else:
            parts.append(f'<span>{label}</span>')
        if i < len(items) - 1:
            parts.append('<span class="breadcrumb-sep">›</span>')

    st.markdown(
        f'<div class="breadcrumb">{"".join(parts)}</div>',
        unsafe_allow_html=True
    )


def section_title(text):
    """Tiêu đề section nhỏ kiểu uppercase."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ─── CHAT COMPONENTS ──────────────────────────────────────────────────────────

def user_message(text):
    """Bubble tin nhắn của user."""
    st.markdown(f"""
    <div class="user-row">
        <div class="user-bubble">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def bot_message(text):
    """Bubble tin nhắn của bot."""
    st.markdown(f"""
    <div class="bot-row">
        <div class="bot-bubble">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def source_card(file_name, page, url, section=""):
    """Card hiển thị nguồn tham khảo."""
    section_html = (
        f"<br><small style='color:#64748b'>📑 {section}</small>"
        if section else ""
    )
    st.markdown(f"""
    <div class="source-card">
        📄 <b style='color:#e2e8f0'>{file_name}</b>{section_html}<br>
        <small style='color:#64748b;font-family:DM Mono'>Trang {page}</small><br>
        <a href="{url}" target="_blank">🔗 Mở PDF trang {page}</a>
    </div>
    """, unsafe_allow_html=True)

def source_card_with_toggle(file_name, page, url, section="", excerpt="", key="src"):
    
    # Tạo key unique nếu không truyền
    if not key.startswith("src_"):
        key = f"src_{key}"
    
    is_open = st.session_state.get(key, False)
    
    col1, col2 = st.columns([10, 1])
    
    with col1:
        st.markdown(f"""
        <div class="source-line">
            📄 <b>{file_name}</b> 
            · Tr.{page} 
            <a href="{url}" target="_blank" class="source-open-link">🔗 Mở</a>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if excerpt is not None:
            toggle_icon = "🙈" if is_open else "👁️"
            if st.button(toggle_icon, key=f"btn_{key}", help="Xem trích dẫn"):
                st.session_state[key] = not is_open
                st.rerun()
    
    # Excerpt khi mở toggle
    if is_open and excerpt:
        st.markdown(f"""
        <div class="source-excerpt-compact">
            {excerpt}
        </div>
        """, unsafe_allow_html=True)