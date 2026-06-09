from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config import load_config,get_api_key
import streamlit as st

VECTOR_DB_PATH = "vectorstores/chroma"

_llm_cache       = None
_retriever_cache = {}


def load_llm():
    """Load LLM theo provider trong config (hỗ trợ Ollama và Gemini)"""
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    cfg = load_config()
    provider = cfg.get("provider", "ollama").lower()
    temperature = float(cfg.get("temperature", 0.01))
    model = cfg.get("model")

    try:
        if provider == "ollama":
            _llm_cache = ChatOllama(
                model=model or "llama3",
                temperature=temperature
            )
            print(f"✅ Đã load Ollama model: {model}")

        elif provider == "gemini":
            api_key = get_api_key("gemini")
            if not api_key:
                st.error("❌ Chưa có Gemini API Key. Vui lòng thêm vào file `.env`")
                raise ValueError("Gemini API Key is missing")

            _llm_cache = ChatGoogleGenerativeAI(
                model=model or "gemini-2.5-flash",
                temperature=temperature,
                google_api_key=api_key,
            )
            print(f"✅ Đã load Gemini model: {model}")
            
        elif provider == "groq":
            api_key = get_api_key("groq")
            if not api_key:
                raise ValueError("Groq API Key chưa được cấu hình!")
            _llm_cache = ChatGroq(
                model=model or "llama3-70b-8192",
                temperature=temperature,
                groq_api_key=api_key
            )
            print(f"✅ Đã load Grog model: {model}")

        else:
            st.warning(f"Provider '{provider}' chưa được hỗ trợ. Tự động chuyển sang Ollama.")
            _llm_cache = ChatOllama(model="llama3", temperature=0.01)

    except Exception as e:
        st.error(f"Lỗi khi load LLM: {str(e)}")
        # Fallback về Ollama
        _llm_cache = ChatOllama(model="llama3", temperature=0.01)

    return _llm_cache

def reset_llm_cache():
    global _llm_cache
    _llm_cache = None


def load_retriever(collection_name):
    cfg       = load_config()
    embedding = OllamaEmbeddings(model=cfg["embed_model"])

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        collection_name=collection_name,
        embedding_function=embedding
    )

    retriever_chroma = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    raw      = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(raw["documents"], raw["metadatas"])
    ]

    if not all_docs:
        return retriever_chroma

    retriever_bm25   = BM25Retriever.from_documents(all_docs)
    retriever_bm25.k = 4

    return EnsembleRetriever(
        retrievers=[retriever_chroma, retriever_bm25],
        weights=[0.7, 0.3]
    )


def reset_retriever_cache(collection_name=None):
    global _retriever_cache
    if collection_name:
        _retriever_cache.pop(collection_name, None)
    else:
        _retriever_cache = {}


def format_docs(docs):
    return "\n\n".join(
        f"[Trang {d.metadata.get('page_number', d.metadata.get('page', 'N/A'))}] {d.page_content}"
        for d in docs
    )


def build_qa_chain(collection_name):
    llm       = load_llm()
    retriever = load_retriever(collection_name)

    qa_prompt = ChatPromptTemplate.from_template("""
Bạn là trợ lý AI chuyên trả lời dựa trên tài liệu được cung cấp.

Quy tắc bắt buộc:
- CHỈ dùng thông tin trong các đoạn trích bên dưới để trả lời.
- KHÔNG hỏi lại người dùng, KHÔNG yêu cầu thêm thông tin.
- Nếu câu hỏi ngắn hoặc chung chung, hãy tóm tắt thông tin liên quan từ tài liệu.
- Nếu không có thông tin liên quan, trả lời: "Không tìm thấy thông tin trong tài liệu."
- Luôn trả lời trực tiếp, ngắn gọn, đúng trọng tâm bằng tiếng Việt.

Các đoạn trích:
{context}

Câu hỏi: {question}

Trả lời bằng tiếng Việt:
""")

    def chain_fn(question):
        print(f"🔍 Tìm kiếm trong kho '{collection_name}'...")
        docs = retriever.invoke(question)
        print(f"✅ Tìm được {len(docs)} đoạn")

        context = format_docs(docs)
        prompt  = qa_prompt.invoke({"context": context, "question": question})

        print("🤖 Đang gọi LLM...")
        answer = llm.invoke(prompt)
        print("✅ LLM trả lời xong")

        return {
            "answer":           answer.content,
            "source_documents": docs
        }

    return chain_fn