import streamlit as st
from chain import build_qa_chain

st.set_page_config(
    page_title="Chatbot Sinh viên GTVT",
    page_icon="🎓",
    layout="centered"
)
st.title("🎓 Chatbot Sinh viên GTVT")
st.caption("Giải đáp những thắc mắc của sinh viên Đại học Giao thông vận tải")

# Load chain (cache để không load lại mỗi lần gõ)
@st.cache_resource
def load_chain():
    return build_qa_chain()

qa_chain = load_chain()
# Lưu lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
# Ô nhập câu hỏi
if question := st.chat_input("Nhập câu hỏi của bạn..."):
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)
        
     # Bot trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu tài liệu..."):
            answer = qa_chain.invoke(question)
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })