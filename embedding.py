from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

#khai bao bien
data_path="data"
vectorDb_path="vectorstores/chroma_db"

#Tao db
def create_db_from_files():
    #load du lieu
    loader=DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs=loader.load()
    #chua nho van ban
    text_spitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_spitter.split_documents(docs)
    print(f"Tổng số document gốc: {len(docs)}")
    print(f"Tổng số chunk sau khi chia: {len(chunks)}")
    
    # for i, chunk in enumerate(chunks):
    #     print(f"\n===== CHUNK {i+1} =====")
    #     print(chunk.page_content)
    # if chunks:  # kiểm tra xem có chunk nào không
    #     print("===== CHUNK ĐẦU TIÊN =====")
    #     print(chunks[1].page_content)
    # print( docs[4].page_content)

    # return chunks

    #embedding
    embedding=OllamaEmbeddings(model="nomic-embed-text")

    #dua vao chroma
    db=Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=vectorDb_path
    )
    db.persist()
    print("✅ Đã lưu ChromaDB")


    
create_db_from_files()
