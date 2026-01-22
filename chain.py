from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.schema import Document
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#load llm
def load_llm():
    llm=ChatOllama(
        model="llama3",
        temperature=0.01
    )
    return llm

#load Hybrid Retriever
def load_retriever():
    # Load Chroma
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="vectorstores/chroma_db",
        embedding_function=embedding
    )

    # 1. Chroma retriever (semantic)
    retriever_chroma = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 2. BM25 retriever (keyword) — TOÀN BỘ dữ liệu
    raw = vectorstore.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(raw["documents"], raw["metadatas"])
    ]

    retriever_bm25 = BM25Retriever.from_documents(all_docs)
    retriever_bm25.k = 4

    # 3. Ensemble retriever (7:3)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[retriever_chroma, retriever_bm25],
        weights=[0.7, 0.3]
    )
    return hybrid_retriever
# docs=hybrid_retriever.invoke(" 4. Mức phí đóng, nộp khi sinh viên vào KTX")
# for i, d in enumerate(docs):
#     print(f"\n--- CHUNK {i+1} ---")
#     print(d.page_content) 

#Format context
def format_docs(docs):
    return "\n\n".join(
        f"[Trang {d.metadata.get('page', 'N/A')}] {d.page_content}"
        for d in docs
    )
# QA chain
def build_qa_chain():
    llm=load_llm()
    retriever=load_retriever()
    qa_prompt = ChatPromptTemplate.from_template("""
    Bạn là trợ lý AI chuyên trả lời dựa trên văn bản sổ tay sinh viên.

    CHỈ sử dụng thông tin trong các đoạn sau để trả lời.
    Nếu không tìm thấy thông tin phù hợp, hãy trả lời:
    "Không tìm thấy thông tin trong tài liệu."

    Câu hỏi:
    {question}

    Các đoạn trích:
    {context}

    Trả lời đầy đủ, ngắn gọn, đúng trọng tâm, không suy diễn.
    """)


    qa_chain = (
        {
            "context": retriever | format_docs ,
            "question": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain
    # question = "Hướng dẫn đăng ký học"
    # answer = qa_chain.invoke(question)

    # print(answer)

# if __name__ == "__main__":
#     qa_chain = build_qa_chain()
#     question = "Hướng dẫn đăng ký học"
#     answer = qa_chain.invoke(question)
#     print(answer)

