import streamlit as st
import urllib.parse
from chain import build_qa_chain
from style import apply_styles
from components import user_message, bot_message, source_card_with_toggle
from database import (
    save_chat_log, get_all_collections, get_public_collections,
    create_conversation, get_conversations, get_conversation,
    update_conversation_title, delete_conversation,
    add_message, get_messages, get_document_info
)

# @st.cache_resource
def load_chain(collection_name):
    return build_qa_chain(collection_name)

def has_valid_answer(answer: str) -> bool:
    """Kiểm tra LLM có tìm được thông tin không."""
    not_found_phrases = [
        "không tìm thấy",
        "không có thông tin",
        "không tìm được",
        "không có trong tài liệu",
        "không tìm thấy thông tin",
    ]
    answer_lower = answer.lower()
    return not any(phrase in answer_lower for phrase in not_found_phrases)

def chat_page():
    apply_styles()

    user     = st.session_state.get("user")
    username = user["username"] if user else None
    is_guest = user is None

    # ─── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:

        # Auth block — login hoặc thông tin user
        st.divider()

        # Lấy danh sách kho theo quyền
        if is_guest:
            collections = get_public_collections()
            if not collections:
                st.markdown("""
                <div style="background:var(--bg-card);border:1px solid var(--border);
                            border-radius:10px;padding:16px;text-align:center">
                    <div style="font-size:1.5rem;margin-bottom:8px">🔒</div>
                    <div style="color:var(--text-primary);font-weight:600;margin-bottom:4px">
                        Chưa có kho công khai
                    </div>
                    <div style="color:var(--text-muted);font-size:0.82rem">
                        Vui lòng đăng nhập để truy cập tài liệu
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.stop()
            st.caption("🔓 Bạn đang xem kho công khai")
        else:
            collections = get_all_collections()
            if not collections:
                st.warning("Chưa có kho tài liệu nào.")
                st.stop()

        # Selectbox chọn kho
        col_names       = {c["name"]: c for c in collections}
        selected        = st.selectbox(
            "📚 Chọn kho:",
            list(col_names.keys()),
            format_func=lambda x: (
                f"🔓 {x}" if col_names[x].get("is_public") else f"🔒 {x}"
            ),
        )
        collection_name = col_names[selected]["chroma_name"]

        # Reset khi đổi kho
        if st.session_state.get("current_collection") != collection_name:
            st.session_state.current_collection = collection_name
            st.session_state.current_conv_id    = None
            st.session_state.messages           = []

        st.divider()

        # Lịch sử chat — chỉ hiện khi đã đăng nhập
        if not is_guest:
            if st.button("✏️ Cuộc hội thoại mới", use_container_width=True, type="primary"):
                st.session_state.current_conv_id = None
                st.session_state.messages        = []
                st.rerun()

            st.markdown("**Lịch sử**")
            conversations = get_conversations(username, collection_name)

            if not conversations:
                st.caption("Chưa có cuộc hội thoại nào.")
            else:
                for conv in conversations:
                    is_active = (st.session_state.get("current_conv_id") == conv["id"])
                    title     = conv["title"]
                    if len(title) > 28:
                        title = title[:28] + "..."

                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            title,
                            key=f"conv_{conv['id']}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary"
                        ):
                            st.session_state.current_conv_id = conv["id"]
                            msgs = get_messages(conv["id"])
                            st.session_state.messages = [
                                {"role": m["role"], "content": m["content"]}
                                for m in msgs
                            ]
                            st.session_state[f"menu_{conv['id']}"] = False
                            st.rerun()

                    with col2:
                        if st.button("···", key=f"menu_btn_{conv['id']}"):
                            key = f"menu_{conv['id']}"
                            st.session_state[key] = not st.session_state.get(key, False)
                            st.rerun()

                    if st.session_state.get(f"menu_{conv['id']}", False):
                        if st.button("✏️  Đổi tên", key=f"rename_{conv['id']}", use_container_width=True):
                            st.session_state[f"renaming_{conv['id']}"] = True
                            st.session_state[f"menu_{conv['id']}"]     = False
                            st.rerun()
                        if st.button("🗑️  Xóa", key=f"del_conv_{conv['id']}", use_container_width=True):
                            delete_conversation(conv["id"])
                            if st.session_state.get("current_conv_id") == conv["id"]:
                                st.session_state.current_conv_id = None
                                st.session_state.messages        = []
                            st.rerun()
                        if st.button("✕  Đóng", key=f"close_menu_{conv['id']}", use_container_width=True):
                            st.session_state[f"menu_{conv['id']}"] = False
                            st.rerun()

                    if st.session_state.get(f"renaming_{conv['id']}"):
                        new_title = st.text_input(
                            "Tên mới:", value=conv["title"],
                            key=f"title_input_{conv['id']}"
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("💾 Lưu", key=f"save_title_{conv['id']}", type="primary"):
                                if new_title.strip():
                                    update_conversation_title(conv["id"], new_title.strip())
                                st.session_state[f"renaming_{conv['id']}"] = False
                                st.rerun()
                        with c2:
                            if st.button("Hủy", key=f"cancel_title_{conv['id']}"):
                                st.session_state[f"renaming_{conv['id']}"] = False
                                st.rerun()
        else:
            # Guest — gợi ý đăng nhập để dùng thêm tính năng
            st.markdown("""
            <div style="background:var(--bg-card);border:1px solid var(--border);
                        border-radius:8px;padding:12px;font-size:0.8rem;color:var(--text-muted)">
                🔒 Đăng nhập để lưu lịch sử chat và xem thêm kho tài liệu
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        show_sources = st.checkbox("Hiển thị nguồn", value=True)
        show_excerpt = st.checkbox("Hiển thị đoạn trích", value=True)

    # ─── Main chat area ───────────────────────────────────────────────
    st.title("🎓 Chatbot Sinh viên GTVT")
    st.caption("Giải đáp những thắc mắc của sinh viên Đại học Giao thông vận tải")

    qa_chain = load_chain(collection_name)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_conv_id" not in st.session_state:
        st.session_state.current_conv_id = None

    # Câu hỏi gợi ý
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Tôi có thể giúp gì cho bạn?")
        col1, col2 = st.columns(2)
        if col1.button("📚 Điều kiện xét cấp học bổng", use_container_width=True):
            st.session_state.suggest = "điều kiện để sinh viên được xét cấp học bổng"
        if col2.button("📅 Chương trình đào tạo ngoại ngữ", use_container_width=True):
            st.session_state.suggest = "chương trình đào tạo ngoại ngữ"
        col3, col4 = st.columns(2)
        if col3.button("🎓 Điều kiện nhận đồ án tốt nghiệp", use_container_width=True):
            st.session_state.suggest = "điều kiện nhận đồ án tốt nghiệp"
        if col4.button("🏫 Giới thiệu trường GTVT", use_container_width=True):
            st.session_state.suggest = "giới thiệu về trường đại học giao thông vận tải"

    # Hiển thị lịch sử
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            user_message(msg["content"])
        else:
            bot_message(msg["content"])

            # Hiển thị nguồn đã lưu
            if show_sources and msg.get("sources") and has_valid_answer(msg["content"]):
                st.markdown("---")
                st.markdown("### 📚 Nguồn tham khảo")
                for i, doc in enumerate(msg["sources"]):
                    file_name   = doc.metadata.get("document_name", "")
                    page        = doc.metadata.get("page_number", doc.metadata.get("page", 0))
                    page_number = page + 1 if isinstance(page, int) else page
                    section     = doc.metadata.get("section_title", "")
                    encoded     = urllib.parse.quote(file_name)
                    pdf_url     = f"/app/static/{encoded}#page={page_number}"
                    doc_info         = get_document_info(file_name, collection_name)
                    display_name     = doc_info.get("title") or file_name
                    source_card_with_toggle(display_name, page_number, pdf_url, section, doc.page_content if show_excerpt else None, key=f"src_{id(msg)}_{i}")

    # Input
    question = st.chat_input("Nhập câu hỏi của bạn...")
    if "suggest" in st.session_state:
        question = st.session_state.pop("suggest")

    if question:
        # Tạo conversation (chỉ khi đã đăng nhập)
        if not is_guest and not st.session_state.current_conv_id:
            conv_id = create_conversation(username, collection_name, title=question[:50])
            st.session_state.current_conv_id = conv_id

        conv_id = st.session_state.get("current_conv_id")

        st.session_state.messages.append({"role": "user", "content": question})
        if not is_guest and conv_id:
            add_message(conv_id, "user", question)
        user_message(question)

        with st.spinner("Đang tra cứu tài liệu..."):
            result  = qa_chain(question)
            answer  = result["answer"]
            sources = result["source_documents"]

        bot_message(answer)

        # Lưu log (chỉ khi đã đăng nhập)
        sources_str = ", ".join(
            f"{d.metadata.get('document_name','')} tr.{d.metadata.get('page_number','')}"
            for d in sources
        )
        if not is_guest:
            if conv_id:
                add_message(conv_id, "assistant", answer, sources_str)
            save_chat_log(username, question, answer, sources_str)

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources
        })
        st.rerun()