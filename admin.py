import streamlit as st
import pandas as pd
import shutil
import os

from style import apply_styles
from components import (
    sidebar_user_card, admin_header, stat_card,
    collection_card, doc_card, breadcrumb, section_title
)
from embedding import (
    add_document, delete_document, delete_chunk, update_chunk,
    add_chunk_manual, insert_chunk, list_documents, list_chunks,
    preview_chunks, get_doc_config, get_collection_stats,
    delete_collection_data, CHUNK_METHODS, EMBEDDING_MODELS
)
from database import (
    get_all_users, create_user, update_user,
    change_password, delete_user, get_chat_logs,
    create_collection, get_all_collections,get_public_collections,set_collection_public,
    update_collection, delete_collection, save_document_info, get_document_info
)
from chain import reset_retriever_cache, reset_llm_cache, load_llm
from config import load_config, save_config,get_api_key, DEFAULT_CONFIG


# ─── Entry point ─────────────────────────────────────────────────────────────

def admin_page():
    apply_styles()

    user     = st.session_state.user
    username = user["username"]

    # Init state
    if "admin_page"            not in st.session_state: st.session_state.admin_page            = "knowledge"
    if "selected_collection"   not in st.session_state: st.session_state.selected_collection   = None
    if "selected_document"     not in st.session_state: st.session_state.selected_document      = None

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        sidebar_user_card(user.get("full_name"), username, user.get("role"))

        menu = {
            "knowledge": ("📚", "Kho tri thức"),
            "config":    ("⚙️", "Cấu hình"),
            "users":     ("👥", "Người dùng"),
            "logs":      ("📊", "Chat Logs"),
        }

        for key, (icon, label) in menu.items():
            is_active = (st.session_state.admin_page == key)
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.admin_page          = key
                st.session_state.selected_collection = None
                st.session_state.selected_document   = None
                st.rerun()

        st.divider()
        if st.button("🚪  Đăng xuất", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    # ── Render page ──────────────────────────────────────────────────
    pages = {
        "knowledge": _page_knowledge,
        "config":    _page_config,
        "users":     _page_users,
        "logs":      _page_logs,
    }
    pages.get(st.session_state.admin_page, _page_knowledge)()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — KHO TRI THỨC
# ══════════════════════════════════════════════════════════════════════════════

def _page_knowledge():
    col      = st.session_state.selected_collection
    doc_name = st.session_state.selected_document

    # Breadcrumb + nút quay lại
    if doc_name:
        bc_col, back_col = st.columns([4, 1])
        with bc_col:
            breadcrumb([
                ("📚 Kho tri thức", False),
                (col["name"],       False),
                (f"📄 {doc_name}",  True)
            ])
        with back_col:
            if st.button("← Quay lại", use_container_width=True):
                st.session_state.selected_document = None
                st.rerun()

    elif col:
        bc_col, back_col = st.columns([4, 1])
        with bc_col:
            breadcrumb([
                ("📚 Kho tri thức", False),
                (col["name"],       True)
            ])
        with back_col:
            if st.button("← Quay lại", use_container_width=True):
                st.session_state.selected_collection = None
                st.rerun()

    # ── Cấp 1: Danh sách kho ─────────────────────────────────────────
    if not col:
        _view_collections()

    # ── Cấp 2: Danh sách tài liệu ────────────────────────────────────
    elif col and not doc_name:
        _view_documents(col)

    # ── Cấp 3: Chunks ────────────────────────────────────────────────
    elif doc_name:
        _view_chunks(col, doc_name)


def _view_collections():
    admin_header("Kho tri thức", "Quản lý các kho tài liệu và nội dung", "📚")

    # Cảnh báo nếu chưa có kho public
    if not get_public_collections():
        st.warning(
            "⚠️ Chưa có kho nào được đặt **Công khai** — "
            "Guest vào trang chủ sẽ không chat được. "
            "Hãy bật toggle 🔓 cho ít nhất 1 kho bên dưới."
        )

    # Thống kê
    collections  = get_all_collections()
    total_docs   = sum(get_collection_stats(c["chroma_name"])["total_documents"] for c in collections)
    total_chunks = sum(get_collection_stats(c["chroma_name"])["total_chunks"]    for c in collections)

    c1, c2, c3 = st.columns(3)
    with c1: stat_card("Tổng kho",      len(collections))
    with c2: stat_card("Tổng tài liệu", total_docs)
    with c3: stat_card("Tổng chunks",   total_chunks)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Form tạo kho mới
    with st.expander("➕ Tạo kho mới"):
        with st.form("form_create_collection", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1: kho_name   = st.text_input("Tên kho",        placeholder="VD: Sổ tay SV")
            with col2: kho_chroma = st.text_input("Tên ChromaDB",   placeholder="VD: so_tay_sv")
            kho_desc = st.text_area("Mô tả", height=70)
            if st.form_submit_button("Tạo kho", use_container_width=True, type="primary"):
                if not kho_name or not kho_chroma:
                    st.error("Vui lòng điền đầy đủ thông tin.")
                elif create_collection(kho_name, kho_chroma, kho_desc):
                    st.success(f"✅ Đã tạo kho '{kho_name}'!")
                    st.rerun()
                else:
                    st.error("Tên kho đã tồn tại.")

    if not collections:
        st.info("Chưa có kho nào. Hãy tạo kho mới ở trên.")
        return

    section_title("DANH SÁCH KHO")
    for c in collections:
        stats = get_collection_stats(c["chroma_name"])
        collection_card(c["name"], c["description"], stats["total_documents"], stats["total_chunks"])

        # Toggle public/private
        is_pub  = bool(c.get("is_public", 0))
        tog_col, lab_col = st.columns([1, 4])
        with tog_col:
            new_pub = st.toggle("", value=is_pub, key=f"pub_{c['chroma_name']}")
        with lab_col:
            st.markdown(
                f"<div style='padding-top:6px;color:var(--text-secondary);font-size:0.83rem'>"
                f"{'🔓 Công khai — Guest có thể xem' if new_pub else '🔒 Riêng tư — Chỉ thành viên'}"
                f"</div>",
                unsafe_allow_html=True
            )
        if new_pub != is_pub:
            set_collection_public(c["chroma_name"], new_pub)
            st.rerun()

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            if st.button("📂 Mở kho", key=f"open_{c['chroma_name']}",
                         use_container_width=True, type="primary"):
                st.session_state.selected_collection = c
                st.rerun()
        with c2:
            if st.button("✏️ Sửa", key=f"edit_{c['chroma_name']}", use_container_width=True):
                key = f"editing_{c['chroma_name']}"
                st.session_state[key] = not st.session_state.get(key, False)
                st.rerun()
        with c3:
            if st.button("🗑️ Xóa", key=f"del_{c['chroma_name']}", use_container_width=True):
                delete_collection_data(c["chroma_name"])
                delete_collection(c["chroma_name"])
                reset_retriever_cache(c["chroma_name"])
                st.warning(f"Đã xóa kho '{c['name']}'")
                st.rerun()

        if st.session_state.get(f"editing_{c['chroma_name']}"):
            with st.form(f"form_edit_{c['chroma_name']}"):
                e_name = st.text_input("Tên kho", c["name"])
                e_desc = st.text_area("Mô tả",    c["description"], height=70)
                if st.form_submit_button("💾 Lưu", type="primary"):
                    update_collection(c["chroma_name"], e_name, e_desc)
                    st.session_state[f"editing_{c['chroma_name']}"] = False
                    st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def _view_documents(col):
    col_name = col["chroma_name"]
    admin_header(col["name"], col.get("description") or "Quản lý tài liệu trong kho", "📂")
    with st.expander("⬆️ Upload tài liệu mới"):
        uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])
        doc_title     = st.text_input("Tiêu đề", placeholder="VD: Sổ tay sinh viên 2024")
        doc_desc      = st.text_area("Mô tả", placeholder="VD: Tài liệu hướng dẫn dành cho sinh viên...", height=80)
        st.caption("⚙️ Cấu hình chunk và embedding lấy từ trang **Cấu hình hệ thống**.")

        if uploaded_file:
            os.makedirs("data", exist_ok=True)
            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getvalue())

            if st.button("⚡ Lưu & Embed", type="primary"):
                with st.spinner(f"Đang xử lý {uploaded_file.name}..."):
                    n = add_document(uploaded_file.name, col_name)
                    os.makedirs("static", exist_ok=True)
                    shutil.copy2(f"data/{uploaded_file.name}", f"static/{uploaded_file.name}")
                    # Lưu tiêu đề và mô tả vào DB
                    save_document_info(
                        uploaded_file.name, col_name,
                        title=doc_title or uploaded_file.name,
                        description=doc_desc
                    )
                    reset_retriever_cache(col_name)
                st.success(f"✅ Đã thêm {n} chunks từ '{doc_title or uploaded_file.name}'!")
                st.rerun()
 

    docs = list_documents(col_name)
    if not docs:
        st.info("Kho chưa có tài liệu nào.")
        return

    section_title(f"{len(docs)} TÀI LIỆU")

    for doc in docs:
        chunks   = list_chunks(col_name, doc)
        cfg      = get_doc_config(col_name, doc)
        doc_info = get_document_info(doc, col_name)
        title    = doc_info.get("title") or doc
        desc     = doc_info.get("description", "")

        c_card, c1, c2, c3 = st.columns([7, 1, 1, 1])

        with c_card:
            st.markdown(f"""
            <div class="doc-card">
                <div class="doc-icon">📄</div>
                <div class="doc-info">
                    <div class="doc-name">{title}</div>
                    <div class="doc-meta">{len(chunks)} chunks
                    {f' · {desc[:50]}...' if desc and len(desc) > 50 else f' · {desc}' if desc else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c1:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button("👁️", key=f"open_doc_{doc}",
                         use_container_width=True, type="primary", help="Xem chunks"):
                st.session_state.selected_document = doc
                st.rerun()

        with c2:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button("✏️", key=f"edit_doc_{doc}", use_container_width=True, help="Sửa"):
                st.session_state[f"editing_doc_{doc}"] = True
                st.rerun()

        with c3:
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"del_doc_{doc}", use_container_width=True, help="Xóa tài liệu"):
                delete_document(doc, col_name)
                reset_retriever_cache(col_name)
                st.rerun()

        if st.session_state.get(f"editing_doc_{doc}"):
            with st.form(f"form_edit_doc_{doc}"):
                st.markdown(f"**Đang sửa: {title}**")
                new_title = st.text_input("Tiêu đề", value=title)
                new_desc = st.text_area("Mô tả", value=desc, height=100)

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.form_submit_button("💾 Lưu thay đổi", type="primary"):
                        save_document_info(doc, col_name, title=new_title, description=new_desc)
                        st.success("✅ Đã cập nhật thông tin tài liệu!")
                        st.session_state[f"editing_doc_{doc}"] = False
                        st.rerun()
                with col_btn2:
                    if st.form_submit_button("Hủy"):
                        st.session_state[f"editing_doc_{doc}"] = False
                        st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def _view_chunks(col, doc_name):
    col_name = col["chroma_name"]
    admin_header(doc_name, f"Kho: {col['name']}", "🧩")

    cfg = get_doc_config(col_name, doc_name)
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Phương pháp", cfg.get("method", "—"))
    with c2: stat_card("Chunk size",  cfg.get("chunk_size", "—"))
    with c3: stat_card("Overlap",     cfg.get("chunk_overlap", "—"))
    with c4: stat_card("Model",       cfg.get("embed_model", "—").split("-")[0])

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    search = st.text_input("🔍 Tìm trong nội dung:", placeholder="Nhập từ khóa...")

    with st.expander("➕ Thêm chunk thủ công"):
        manual_page = st.number_input("Số trang:", min_value=0, value=0)
        manual_text = st.text_area("Nội dung chunk:", height=100)
        if st.button("💾 Thêm chunk", type="primary"):
            if manual_text.strip():
                cid = add_chunk_manual(doc_name, col_name, manual_text.strip(), manual_page)
                reset_retriever_cache(col_name)
                st.success(f"✅ Đã thêm: `{cid}`")
                st.rerun()
            else:
                st.warning("Nội dung không được để trống.")

    chunks = list_chunks(col_name, doc_name)
    if search:
        chunks = [c for c in chunks if search.lower() in c["content"].lower()]

    section_title(f"{len(chunks)} CHUNKS")

    for idx, chunk in enumerate(chunks):
        meta  = chunk["metadata"]
        label = (
            f"#{meta.get('chunk_index', idx)}  "
            f"Trang {meta.get('page_number', 'N/A')}  "
            f"· {chunk['content'][:70]}..."
        )

        with st.expander(label):
            edited = st.text_area(
                "Nội dung:",
                value=chunk["content"],
                height=140,
                key=f"ta_{chunk['id']}"
            )

            # Nút lưu / xóa
            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Lưu & Re-embed",
                             key=f"save_{chunk['id']}", type="primary"):
                    if update_chunk(chunk["id"], edited, col_name):
                        reset_retriever_cache(col_name)
                        st.success("✅ Đã lưu!")
                    else:
                        st.error("Lỗi khi lưu.")
            with c2:
                if st.button("🗑️ Xóa", key=f"del_{chunk['id']}"):
                    delete_chunk(chunk["id"], col_name)
                    reset_retriever_cache(col_name)
                    st.rerun()

            st.divider()

            # Nút chèn chunk mới
            st.caption("➕ Chèn chunk mới:")
            ci1, ci2 = st.columns(2)
            with ci1:
                if st.button("⬆️ Thêm trước chunk này",
                             key=f"ins_before_{chunk['id']}",
                             use_container_width=True):
                    st.session_state[f"inserting_{chunk['id']}"] = "before"
                    st.rerun()
            with ci2:
                if st.button("⬇️ Thêm sau chunk này",
                             key=f"ins_after_{chunk['id']}",
                             use_container_width=True):
                    st.session_state[f"inserting_{chunk['id']}"] = "after"
                    st.rerun()

            st.caption(f"ID: `{chunk['id']}`")

        # Form chèn chunk — hiện ngay bên dưới chunk đang chọn
        insert_pos = st.session_state.get(f"inserting_{chunk['id']}")
        if insert_pos:
            pos_label = "trước" if insert_pos == "before" else "sau"
            st.markdown(f"""
            <div style="background:var(--bg-card);border:1px solid var(--accent);
                        border-left:3px solid var(--accent);
                        border-radius:8px;padding:10px 14px;margin:2px 0 4px 0">
                <span style="color:var(--accent);font-size:0.8rem;font-weight:600">
                    ➕ Chèn chunk {pos_label} chunk #{meta.get('chunk_index', idx)}
                </span>
            </div>
            """, unsafe_allow_html=True)

            new_content = st.text_area(
                "Nội dung chunk mới:",
                height=120,
                key=f"new_content_{chunk['id']}",
                placeholder="Nhập nội dung chunk mới..."
            )

            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Chèn", key=f"confirm_ins_{chunk['id']}",
                             type="primary", use_container_width=True):
                    if new_content.strip():
                        with st.spinner("Đang chèn chunk..."):
                            insert_chunk(
                                doc_name, col_name,
                                new_content.strip(),
                                insert_at=idx,
                                position=insert_pos
                            )
                            reset_retriever_cache(col_name)
                        st.session_state[f"inserting_{chunk['id']}"] = None
                        st.success("✅ Đã chèn!")
                        st.rerun()
                    else:
                        st.warning("Nội dung không được để trống.")
            with b2:
                if st.button("Hủy", key=f"cancel_ins_{chunk['id']}",
                             use_container_width=True):
                    st.session_state[f"inserting_{chunk['id']}"] = None
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CẤU HÌNH
# ══════════════════════════════════════════════════════════════════════════════

def _page_config():
    admin_header("Cấu hình hệ thống", "Điều chỉnh LLM, Chunk và API Key", "⚙️")

    cfg = load_config()

    # ====================== CHỌN PROVIDER ======================
    section_title("🤖 LLM Provider")

    provider = st.selectbox(
        "Chọn nhà cung cấp mô hình:",
        ["ollama", "gemini", "groq"],
        index=["ollama", "gemini", "groq"].index(cfg.get("provider", "ollama")),
        format_func=lambda x: {
            "ollama": "🖥️ Ollama (Local)",
            "gemini": "☁️ Gemini (Google)",
            "groq": "⚡ Groq (Siêu nhanh)"
        }.get(x, x)
    )

    # ====================== CHỌN MODEL ======================
    section_title("📋 Chọn Model")

    # Model mặc định
    default_models = {
        "ollama": ["llama3", "llama3.2", "mistral", "gemma2", "qwen2.5"],
        "gemini": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-pro"],
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it", "qwen-qwq-32b"]
    }
    # Lấy danh sách model tùy chỉnh đã lưu
    custom_models = cfg.get("custom_models", {})
    current_custom = custom_models.get(provider, [])
    # Kết hợp model mặc định + tùy chỉnh (không trùng)
    all_models = default_models.get(provider, []) + [m for m in current_custom if m not in default_models.get(provider, [])]

    model = st.selectbox(
        f"{provider.capitalize()} Model:",
        all_models,
        index=0
    )
    
    # ====================== THÊM MODEL TÙY CHỈNH ======================
    with st.expander(f"➕ Thêm model {provider.upper()} tùy chỉnh", expanded=False):
        new_model = st.text_input("Nhập tên model", )
        
        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("Thêm model", type="primary", use_container_width=True):
                new_model = new_model.strip()
                if new_model and new_model not in all_models:
                    if provider not in custom_models:
                        custom_models[provider] = []
                    custom_models[provider].append(new_model)
                    
                    new_cfg = cfg.copy()
                    new_cfg["custom_models"] = custom_models
                    save_config(new_cfg)
                    st.success(f"✅ Đã thêm model: **{new_model}**")
                    st.rerun()
                else:
                    st.warning("Model đã tồn tại hoặc tên trống.")

        with col_clear:
            if st.button("Xóa tất cả model tùy chỉnh", use_container_width=True):
                if st.checkbox("Xác nhận xóa hết?", key=f"del_{provider}"):
                    new_cfg = cfg.copy()
                    if provider in new_cfg.get("custom_models", {}):
                        del new_cfg["custom_models"][provider]
                    save_config(new_cfg)
                    st.success(f"Đã xóa model tùy chỉnh của {provider}")
                    st.rerun()
    temperature = st.slider(
        "Temperature (Độ sáng tạo)", 
        0.0, 1.0, 
        float(cfg.get("temperature", 0.01)), 
        0.01,
        help="Thấp = trả lời chính xác hơn | Cao = sáng tạo hơn"
    )

    # ====================== API KEY (chỉ hiện với Gemini) ======================
    api_key = ""
    if provider in ["gemini", "groq"]:
        section_title("🔑 API Key")
        if provider == "gemini":
            api_key = st.text_input(
                "Gemini API Key",
                value=get_api_key("gemini"),
                type="password",
                help="Lấy tại: https://aistudio.google.com/app/apikey"
            )
        else:  # groq
            api_key = st.text_input(
                "Groq API Key",
                value=get_api_key("groq"),
                type="password",
                help="Lấy tại: https://console.groq.com/keys"
            )
        st.caption("🔐 Key được lưu an toàn trong file `.env`")

    # ====================== CẤU HÌNH CHUNK ======================
    section_title("📋 Cấu hình Chunk & Embedding")

    c1, c2 = st.columns(2)
    with c1:
        chunk_method = st.selectbox(
            "Phương pháp chunk:",
            ["by_size", "by_section", "by_sentence"],
            index=["by_size", "by_section", "by_sentence"].index(cfg.get("chunk_method", "by_size"))
        )
        embed_model = st.selectbox(
            "Embedding model:",
            ["nomic-embed-text", "mxbai-embed-large", "all-minilm"],
            index=0
        )

    with c2:
        chunk_size = st.slider("Chunk size", 200, 2000, int(cfg.get("chunk_size", 1000)), 50)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, int(cfg.get("chunk_overlap", 200)), 25)

    # ====================== NÚT HÀNH ĐỘNG ======================
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        if st.button("💾 Lưu cấu hình", type="primary", use_container_width=True):
            new_config = {
                "provider": provider,
                "model": model,
                "temperature": temperature,
                "chunk_method": chunk_method,
                "embed_model": embed_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "custom_models": custom_models
            }
            save_config(new_config)
            
            if api_key and provider in ["gemini", "groq"]:
                key_name = "GEMINI_API_KEY" if provider == "gemini" else "GROQ_API_KEY"
                
                try:
                    # Đọc toàn bộ nội dung .env cũ
                    env_content = {}
                    if os.path.exists(".env"):
                        with open(".env", "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#") and "=" in line:
                                    k, v = line.split("=", 1)
                                    env_content[k.strip()] = v.strip()

                    # Cập nhật key mới
                    env_content[key_name] = api_key

                    # Ghi lại toàn bộ file
                    with open(".env", "w", encoding="utf-8") as f:
                        for k, v in env_content.items():
                            f.write(f"{k}={v}\n")

                    st.success(f"✅ Đã lưu {provider.upper()} API Key!")
                
                except Exception as e:
                    st.error(f"Không thể lưu .env: {e}")

            reset_llm_cache()
            st.success("✅ Đã lưu cấu hình hệ thống!")
            st.rerun()

    with col2:
        if st.button("🔄 Test Connection", use_container_width=True):
            with st.spinner("Đang kiểm tra kết nối..."):
                try:
                    # Tạm thời set key để test
                    if provider == "gemini" and api_key:
                        os.environ["GEMINI_API_KEY"] = api_key
                    
                    llm = load_llm()
                    response = llm.invoke("Trả lời ngắn gọn bằng tiếng Việt: Xin chào, bạn khỏe không?")
                    st.success(f"✅ Kết nối thành công!\n\n**Model:** {provider} - {model}\n**Trả lời:** {response.content}")
                except Exception as e:
                    st.error(f"❌ Kết nối thất bại: {str(e)}")

    with col3:
        if st.button("🔄 Reset Cache", use_container_width=True):
            reset_llm_cache()
            st.success("Đã reset cache LLM")
            st.rerun()
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — NGƯỜI DÙNG
# ══════════════════════════════════════════════════════════════════════════════

def _page_users():
    admin_header("Người dùng", "Quản lý tài khoản và phân quyền", "👥")

    col_list, col_create = st.columns([3, 2])

    with col_create:
        section_title("TẠO TÀI KHOẢN")
        with st.form("form_create_user", clear_on_submit=True):
            nu = st.text_input("Username")
            nf = st.text_input("Họ tên")
            np = st.text_input("Mật khẩu", type="password")
            nr = st.selectbox("Role", ["user", "admin"])
            if st.form_submit_button("Tạo tài khoản", use_container_width=True, type="primary"):
                if not all([nu, nf, np]):
                    st.error("Điền đầy đủ thông tin.")
                elif create_user(nu, np, role=nr, full_name=nf):
                    st.success(f"✅ Tạo '{nu}' thành công!")
                    st.rerun()
                else:
                    st.error("Username đã tồn tại.")

    with col_list:
        section_title("DANH SÁCH TÀI KHOẢN")
        current_user = st.session_state.user
        users        = get_all_users()
        for u in users:
            is_self   = (u["username"] == current_user["username"])
            role_icon = "🔴" if u["role"] == "admin" else "🟢"
            with st.expander(
                f"{role_icon} **{u['full_name'] or u['username']}** "
                f"(@{u['username']}) {'· [BẠN]' if is_self else ''}"
            ):
                with st.form(f"form_edit_{u['username']}"):
                    ef = st.text_input("Họ tên", u.get("full_name", ""))
                    er = st.selectbox("Role", ["user", "admin"], index=0 if u["role"] == "user" else 1)
                    ea = st.checkbox("Kích hoạt", bool(u.get("is_active", 1)))
                    ep = st.text_input("Đổi mật khẩu (bỏ trống = giữ)", type="password")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("💾 Lưu", type="primary"):
                            update_user(u["username"], ef, er, int(ea))
                            if ep: change_password(u["username"], ep)
                            st.success("Đã cập nhật!")
                            st.rerun()
                    with c2:
                        if not is_self:
                            if st.form_submit_button("🗑️ Xóa"):
                                delete_user(u["username"])
                                st.rerun()
                st.caption(
                    f"Tạo: {(u.get('created_at') or '')[:16]}  |  "
                    f"Login cuối: {(u.get('last_login') or 'Chưa')[:16]}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CHAT LOGS
# ══════════════════════════════════════════════════════════════════════════════

def _page_logs():
    admin_header("Chat Logs", "Lịch sử hỏi đáp của toàn bộ người dùng", "📊")

    users_list      = ["Tất cả"] + [u["username"] for u in get_all_users()]
    filter_user     = st.selectbox("Lọc theo user:", users_list)
    username_filter = None if filter_user == "Tất cả" else filter_user

    logs = get_chat_logs(username=username_filter)
    if not logs:
        st.info("Chưa có log nào.")
        return

    section_title(f"{len(logs)} LOGS")
    df = pd.DataFrame(logs)
    df["created_at"] = df["created_at"].str[:16]
    st.dataframe(
        df[["created_at", "username", "question", "answer", "sources"]],
        use_container_width=True,
        height=480,
        column_config={
            "created_at": "Thời gian",
            "username":   "User",
            "question":   "Câu hỏi",
            "answer":     "Trả lời",
            "sources":    "Nguồn",
        }
    )
    st.download_button(
        "⬇️ Tải CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="chat_logs.csv",
        mime="text/csv"
    )