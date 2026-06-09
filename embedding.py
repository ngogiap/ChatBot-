from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import re
import easyocr 
from pdf2image import convert_from_path
import cv2
import numpy as np
from config import load_config   

DATA_PATH      = "data"
VECTOR_DB_PATH = "vectorstores/chroma"

POPPLER_PATH = r"C:\poppler\poppler-26.02.0\Library\bin"
EMBEDDING_MODELS = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]

_ocr_instance = None
 
def get_ocr():
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = easyocr.Reader(['vi', 'en'], gpu=False)
    return _ocr_instance

def _safe_collection(name: str) -> str:
    import unicodedata
    # Bỏ dấu tiếng Việt
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Thay ký tự không hợp lệ bằng "_"
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    # Xóa "_.-" thừa ở đầu/cuối
    name = name.strip("_.-")
    # Đảm bảo bắt đầu bằng chữ/số
    if not name or not name[0].isalnum():
        name = "col_" + name
    # Đảm bảo kết thúc bằng chữ/số
    if not name[-1].isalnum():
        name = name.rstrip("_.-") or "col"
    # Đảm bảo đủ 3 ký tự
    while len(name) < 3:
        name += "_0"
    return name[:512]

# ─── Kết nối ChromaDB — thêm collection_name ─────────────────────────────────

def get_vector_db(collection_name="default", embed_model="nomic-embed-text"):
    """
    Kết nối tới ChromaDB với collection chỉ định.
    Mỗi kho tài liệu là 1 collection riêng.
    """
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    return Chroma(
        persist_directory=VECTOR_DB_PATH,
        collection_name=_safe_collection(collection_name),     
        embedding_function=OllamaEmbeddings(model=embed_model)
    )


def _safe_id(filename):
    """Chuẩn hóa tên file thành ID an toàn."""
    return re.sub(r"[^\w]", "_", filename)


# ─── Các phương pháp chia chunk ──────────────────────────────────────────────

def _chunk_by_size(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(docs)

def _chunk_by_section(docs, chunk_size, chunk_overlap):
    full_text = "\n".join(d.page_content for d in docs)
    base_meta = docs[0].metadata if docs else {}
    pattern   = r"(?=^\s*(?:Chương\s+\d+|CHƯƠNG\s+\d+|Phần\s+\d+|\d+\.\s+\S))"
    parts     = re.split(pattern, full_text, flags=re.MULTILINE)
    result    = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n")
        title = lines[0].strip()
        body  = "\n".join(lines[1:]).strip()
        if len(body) > chunk_size:
            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            for sub in sub_splitter.split_text(body):
                result.append(Document(
                    page_content=sub,
                    metadata={**base_meta, "section_title": title}
                ))
        else:
            result.append(Document(
                page_content=body or part,
                metadata={**base_meta, "section_title": title}
            ))
    return result if result else _chunk_by_size(docs, chunk_size, chunk_overlap)

def _chunk_by_sentence(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[".", "!", "?", "\n"]
    )
    return splitter.split_documents(docs)

CHUNK_METHODS = {
    "by_size":     _chunk_by_size,
    "by_section":  _chunk_by_section,
    "by_sentence": _chunk_by_sentence,
}
# xử lý ảnh
def preprocess_image(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
 
    # CLAHE tăng tương phản cục bộ — tốt hơn equalizeHist cho văn bản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
 
    # Khử nhiễu nhẹ
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
 
    # Binarize Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
    # EasyOCR nhận ảnh grayscale hoặc RGB đều được
    return binary
# ─── Thêm tài liệu — thêm collection_name ────────────────────────────────────

def add_document(filename, collection_name, method=None, chunk_size=None, chunk_overlap=None, embed_model=None):
    cfg          = load_config()
    method       = method       or cfg["chunk_method"]
    chunk_size   = chunk_size   or cfg["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else cfg["chunk_overlap"]
    embed_model  = embed_model  or cfg["embed_model"]
    """Thêm tài liệu vào kho (collection) chỉ định."""
    file_path = os.path.join(DATA_PATH, filename)

    if not os.path.exists(file_path):
        print("File không tồn tại")
        return 0

    # Xóa bản cũ nếu đã tồn tại trong kho này
    if filename in list_documents(collection_name):
        delete_document(filename, collection_name)

    loader = PyPDFLoader(file_path)
    docs   = loader.load()

    # ─── Phát hiện PDF scan → PaddleOCR ─────────────────────────────────────
    full_text = "\n".join(d.page_content.strip() for d in docs).strip()
 
    if len(full_text) < 50:
        print("📸 PDF scan detected → EasyOCR")
 
        ocr_docs = []
        ocr      = get_ocr()
 
        images = convert_from_path(
            file_path,
            dpi=300,
            poppler_path=POPPLER_PATH,   
        )
 
        for i, img in enumerate(images):
            try:
                img_np    = np.array(img)               
                processed = preprocess_image(img_np)    
 
                lines = ocr.readtext(processed, detail=0, paragraph=True)
 
                page_text = "\n".join(lines).strip()
 
                if len(page_text) < 20:
                    print(f"⚠️  Trang {i+1} rỗng sau OCR")
                    continue
 
                ocr_docs.append(Document(
                    page_content=page_text,
                    metadata={"source": filename, "page": i + 1}
                ))
                print(f"✅ OCR trang {i+1}: {len(page_text)} ký tự")
 
            except Exception as e:
                print(f"❌ OCR lỗi trang {i+1}: {e}")
 
        if not ocr_docs:
            print("❌ Không OCR được trang nào — kiểm tra lại file PDF")
            return 0
 
        docs = ocr_docs  

    chunk_fn  = CHUNK_METHODS.get(method, _chunk_by_size)
    chunks    = chunk_fn(docs, chunk_size, chunk_overlap)
    db        = get_vector_db(collection_name, embed_model)
    safe_name = _safe_id(filename)
    all_chunks = []
    all_ids    = []

    for i, chunk in enumerate(chunks):
        actual_page = chunk.metadata.get("page", 0)
        chunk.metadata.update({
            "document_name":   filename,
            "collection_name": collection_name,
            "page_number":     actual_page,
            "chunk_index":     i,
            "chunk_method":    method,
            "chunk_size":      chunk_size,
            "chunk_overlap":   chunk_overlap,
            "embed_model":     embed_model,
        })
        all_chunks.append(chunk)
        all_ids.append(f"{safe_name}_chunk_{i}")

    db.add_documents(documents=all_chunks, ids=all_ids)
    print(f"Đã thêm {len(all_chunks)} chunks từ '{filename}' vào kho '{collection_name}'")
    return len(all_chunks)


# ─── Xóa tài liệu / chunk — thêm collection_name ─────────────────────────────

def delete_document(filename, collection_name):
    """Xóa toàn bộ chunks của 1 file trong kho chỉ định."""
    db      = get_vector_db(collection_name)
    results = db._collection.get(where={"document_name": filename})
    ids     = results.get("ids", [])
    if ids:
        db.delete(ids=ids)
    print(f"Đã xóa '{filename}' khỏi kho '{collection_name}'")
    return len(ids)


def delete_chunk(chunk_id, collection_name):
    """Xóa 1 chunk theo ID trong kho chỉ định."""
    db = get_vector_db(collection_name)
    db.delete(ids=[chunk_id])
    print(f"Đã xóa chunk '{chunk_id}'")


def delete_collection_data(collection_name):
    """Xóa toàn bộ dữ liệu của 1 kho trong ChromaDB."""
    db      = get_vector_db(collection_name)
    results = db._collection.get()
    ids     = results.get("ids", [])
    if ids:
        db.delete(ids=ids)
    print(f"Đã xóa toàn bộ dữ liệu kho '{collection_name}'")


# ─── Sửa chunk / thêm chunk thủ công ─────────────────────────────────────────

def update_chunk(chunk_id, new_content, collection_name):
    """Sửa nội dung chunk và re-embed."""
    db      = get_vector_db(collection_name)
    results = db._collection.get(ids=[chunk_id], include=["metadatas"])
    if not results["ids"]:
        return False

    meta        = results["metadatas"][0]
    embed_model = meta.get("embed_model", "nomic-embed-text")

    db.delete(ids=[chunk_id])
    db2 = get_vector_db(collection_name, embed_model)
    db2.add_texts(texts=[new_content], metadatas=[meta], ids=[chunk_id])
    return True


def add_chunk_manual(filename, collection_name, content, page=0):
    """Thêm chunk thủ công vào tài liệu trong kho."""
    db      = get_vector_db(collection_name)
    results = db._collection.get(where={"document_name": filename})
    next_i  = len(results["ids"])

    chunk_id = f"{_safe_id(filename)}_chunk_{next_i}"
    db.add_texts(
        texts=[content],
        metadatas=[{
            "document_name":   filename,
            "collection_name": collection_name,
            "page_number":     page,
            "chunk_index":     next_i,
            "chunk_method":    "manual",
        }],
        ids=[chunk_id]
    )
    return chunk_id

def insert_chunk(filename, collection_name, content, insert_at, position="after"):
    """
    Chèn chunk mới vào trước hoặc sau chunk tại vị trí insert_at.
    position: "before" hoặc "after"
    """
    db      = get_vector_db(collection_name)
    results = db._collection.get(
        where={"document_name": filename},
        include=["documents", "metadatas"]
    )
 
    # Sắp xếp theo chunk_index
    chunks = sorted(
        zip(results["ids"], results["documents"], results["metadatas"]),
        key=lambda x: x[2].get("chunk_index", 0)
    )
 
    # Xác định vị trí chèn
    insert_pos = insert_at + 1 if position == "after" else insert_at
 
    # Tạo chunk mới
    ref_meta = chunks[insert_at][2].copy() if chunks else {}
    new_meta = {
        **ref_meta,
        "document_name":   filename,
        "collection_name": collection_name,
        "page_number":     ref_meta.get("page_number", 0),
        "chunk_method":    "manual",
    }
 
    # Chèn vào đúng vị trí
    chunks.insert(insert_pos, (None, content, new_meta))
 
    # Xóa toàn bộ chunks cũ
    old_ids = [c[0] for c in chunks if c[0] is not None]
    if old_ids:
        db.delete(ids=old_ids)
 
    # Thêm lại với chunk_index được cập nhật đúng thứ tự
    safe_name  = _safe_id(filename)
    new_texts  = []
    new_metas  = []
    new_ids    = []
 
    for i, (_, doc_text, meta) in enumerate(chunks):
        new_texts.append(doc_text)
        new_metas.append({**meta, "chunk_index": i})
        new_ids.append(f"{safe_name}_chunk_{i}")
 
    db2 = get_vector_db(collection_name)
    db2.add_texts(texts=new_texts, metadatas=new_metas, ids=new_ids)
    return len(new_texts)

# ─── Xem danh sách — thêm collection_name ────────────────────────────────────

def list_documents(collection_name):
    """Danh sách file đã index trong kho chỉ định."""
    db      = get_vector_db(collection_name)
    results = db._collection.get(include=["metadatas"])
    return sorted({
        meta["document_name"]
        for meta in results["metadatas"]
        if meta and "document_name" in meta
    })


def list_chunks(collection_name, filename=None):
    """Danh sách chunks trong kho, có thể lọc theo file."""
    db     = get_vector_db(collection_name)
    kwargs = {}
    if filename:
        kwargs["where"] = {"document_name": filename}

    results = db._collection.get(**kwargs, include=["documents", "metadatas"])

    chunks = [
        {
            "id":       results["ids"][i],
            "content":  results["documents"][i],
            "metadata": results["metadatas"][i]
        }
        for i in range(len(results["ids"]))
    ]
    chunks.sort(key=lambda x: (
        x["metadata"].get("page_number", 0),
        x["metadata"].get("chunk_index", 0)
    ))
    return chunks


def preview_chunks(filename, method="by_size", chunk_size=1000, chunk_overlap=200):
    """Xem trước chunks — không lưu vào DB."""
    file_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(file_path):
        return []
    docs     = PyPDFLoader(file_path).load()
    chunk_fn = CHUNK_METHODS.get(method, _chunk_by_size)
    chunks   = chunk_fn(docs, chunk_size, chunk_overlap)
    return [
        {
            "index":   i,
            "content": c.page_content,
            "length":  len(c.page_content),
            "page":    c.metadata.get("page", 0),
            "section": c.metadata.get("section_title", "")
        }
        for i, c in enumerate(chunks)
    ]


def get_doc_config(collection_name, filename):
    """Lấy cấu hình chunk của 1 file trong kho."""
    chunks = list_chunks(collection_name, filename)
    if not chunks:
        return {}
    meta = chunks[0]["metadata"]
    return {
        "method":        meta.get("chunk_method",  "by_size"),
        "chunk_size":    meta.get("chunk_size",    1000),
        "chunk_overlap": meta.get("chunk_overlap", 200),
        "embed_model":   meta.get("embed_model",   "nomic-embed-text"),
    }


def get_collection_stats(collection_name):
    """Thống kê tổng quan của 1 kho."""
    db      = get_vector_db(collection_name)
    results = db._collection.get(include=["metadatas"])
    docs    = list_documents(collection_name)
    return {
        "total_documents": len(docs),
        "total_chunks":    len(results["ids"]),
        "documents":       docs
    }