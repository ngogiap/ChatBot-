import sqlite3
import bcrypt
import os
from datetime import datetime

DB_PATH = "data/app.db"


# ═══════════════════════════════════════════════════════════════════════════════
# KẾT NỐI DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def get_connection():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ═══════════════════════════════════════════════════════════════════════════════
# KHỞI TẠO DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def init_db():
    conn = get_connection()
    cur  = conn.cursor()

    # Bảng 1: Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT UNIQUE,
        password   TEXT,
        role       TEXT,
        full_name  TEXT DEFAULT '',
        is_active  INTEGER DEFAULT 1,
        auth_type  TEXT DEFAULT 'local',
        created_at TEXT,
        last_login TEXT
    )
    """)

    # Migrate: thêm cột auth_type nếu DB cũ chưa có
    try:
        cur.execute("ALTER TABLE users ADD COLUMN auth_type TEXT DEFAULT 'local'")
    except:
        pass

    # Bảng 2: Kho tài liệu
    cur.execute("""
    CREATE TABLE IF NOT EXISTS collections (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT UNIQUE,
        chroma_name TEXT UNIQUE,
        description TEXT DEFAULT '',
        is_public   INTEGER DEFAULT 0,
        created_at  TEXT
    )
    """)

    # Migrate: thêm cột is_public nếu DB cũ chưa có
    try:
        cur.execute("ALTER TABLE collections ADD COLUMN is_public INTEGER DEFAULT 0")
    except:
        pass

    # Bảng 3: Cuộc hội thoại
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        username     TEXT NOT NULL,
        collection   TEXT NOT NULL,
        title        TEXT DEFAULT 'Cuộc hội thoại mới',
        created_at   TEXT,
        updated_at   TEXT
    )
    """)

    # Bảng 4: Tin nhắn trong cuộc hội thoại
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation_messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role            TEXT NOT NULL,
        content         TEXT NOT NULL,
        sources         TEXT DEFAULT '',
        created_at      TEXT,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id)
    )
    """)

    # Bảng 5: Chat logs (lưu toàn bộ lịch sử để admin xem)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        username   TEXT,
        question   TEXT,
        answer     TEXT,
        sources    TEXT DEFAULT '',
        created_at TEXT
    )
    """)
    # Bangr6: Tạo bảng documents
    conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        filename        TEXT NOT NULL,
        collection_name TEXT NOT NULL,
        title           TEXT DEFAULT '',
        description     TEXT DEFAULT '',
        created_at      TEXT,
        UNIQUE(filename, collection_name)
    )
    """)

    conn.commit()
    conn.close()

    # Tạo admin mặc định nếu chưa có
    if not get_user("admin"):
        create_user("admin", "admin123", role="admin", full_name="Administrator")
        print("Đã tạo admin mặc định: admin / admin123")


# ═══════════════════════════════════════════════════════════════════════════════
# USERS
# ═══════════════════════════════════════════════════════════════════════════════

def create_user(username, password, role="user", full_name="", auth_type="local"):
    conn   = get_connection()
    cur    = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() if password else ""
    try:
        cur.execute(
            "INSERT INTO users (username, password, role, full_name, auth_type, created_at) VALUES (?,?,?,?,?,?)",
            (username, hashed, role, full_name, auth_type, datetime.now().isoformat())
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def create_google_user(email, full_name):
    """Tạo tài khoản mới từ Google OAuth — chưa có mật khẩu."""
    return create_user(
        username  = email,
        password  = "",
        role      = "user",
        full_name = full_name,
        auth_type = "google"
    )


def set_password_for_google_user(username, new_password):
    """
    Cho phép user Google đặt mật khẩu để đăng nhập local.
    Sau khi đặt, auth_type = 'both' (dùng được cả 2 cách).
    """
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    conn   = get_connection()
    conn.execute(
        "UPDATE users SET password=?, auth_type='both' WHERE username=?",
        (hashed, username)
    )
    conn.commit()
    conn.close()


def login_google(email):
    """Đăng nhập bằng Google — chỉ cần email khớp."""
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM users WHERE username=? AND is_active=1 AND auth_type IN ('google','both')",
        (email,)
    ).fetchone()
    conn.close()

    if row:
        conn2 = get_connection()
        conn2.execute(
            "UPDATE users SET last_login=? WHERE username=?",
            (datetime.now().isoformat(), email)
        )
        conn2.commit()
        conn2.close()
        return dict(row)
    return None


def login_local(username, password):
    """Đăng nhập bằng username/password — hỗ trợ cả local và both."""
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM users WHERE username=? AND is_active=1 AND auth_type IN ('local','both')",
        (username,)
    ).fetchone()
    conn.close()

    if row and row["password"] and bcrypt.checkpw(password.encode(), row["password"].encode()):
        conn2 = get_connection()
        conn2.execute(
            "UPDATE users SET last_login=? WHERE username=?",
            (datetime.now().isoformat(), username)
        )
        conn2.commit()
        conn2.close()
        return dict(row)
    return None


def login(username, password):
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,))
    user = cur.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
        conn2 = get_connection()
        conn2.execute(
            "UPDATE users SET last_login=? WHERE username=?",
            (datetime.now().isoformat(), username)
        )
        conn2.commit()
        conn2.close()
        return dict(user)
    return None


def create_admin():
    if not get_user("admin"):
        create_user("admin", "admin123", role="admin", full_name="Administrator")


def get_user(username):
    conn = get_connection()
    row  = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_users():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, full_name, role, is_active, created_at, last_login FROM users"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_user(username, full_name, role, is_active):
    conn = get_connection()
    conn.execute(
        "UPDATE users SET full_name=?, role=?, is_active=? WHERE username=?",
        (full_name, role, is_active, username)
    )
    conn.commit()
    conn.close()


def change_password(username, new_password):
    hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    conn   = get_connection()
    conn.execute("UPDATE users SET password=? WHERE username=?", (hashed, username))
    conn.commit()
    conn.close()


def delete_user(username):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE username=?", (username,))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# COLLECTIONS (KHO TÀI LIỆU)
# ═══════════════════════════════════════════════════════════════════════════════

def create_collection(name, chroma_name, description="", is_public=0):
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO collections (name, chroma_name, description, is_public, created_at) VALUES (?,?,?,?,?)",
            (name, chroma_name, description, is_public, datetime.now().isoformat())
        )
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def get_public_collections():
    """Lấy danh sách kho public — dành cho guest."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM collections WHERE is_public=1 ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_collection_public(chroma_name, is_public):
    """Bật/tắt public cho kho."""
    conn = get_connection()
    conn.execute(
        "UPDATE collections SET is_public=? WHERE chroma_name=?",
        (int(is_public), chroma_name)
    )
    conn.commit()
    conn.close()


def get_all_collections():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM collections ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_collection(chroma_name):
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM collections WHERE chroma_name=?", (chroma_name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_collection(chroma_name, name, description):
    conn = get_connection()
    conn.execute(
        "UPDATE collections SET name=?, description=? WHERE chroma_name=?",
        (name, description, chroma_name)
    )
    conn.commit()
    conn.close()


def delete_collection(chroma_name):
    conn = get_connection()
    conn.execute("DELETE FROM collections WHERE chroma_name=?", (chroma_name,))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONS (CUỘC HỘI THOẠI)
# ═══════════════════════════════════════════════════════════════════════════════

def create_conversation(username, collection, title="Cuộc hội thoại mới"):
    """Tạo cuộc hội thoại mới, trả về id."""
    conn = get_connection()
    cur  = conn.cursor()
    now  = datetime.now().isoformat()
    cur.execute(
        "INSERT INTO conversations (username, collection, title, created_at, updated_at) VALUES (?,?,?,?,?)",
        (username, collection, title, now, now)
    )
    conn.commit()
    conv_id = cur.lastrowid
    conn.close()
    return conv_id


def get_conversations(username, collection=None):
    """Lấy danh sách cuộc hội thoại của user, mới nhất lên trên."""
    conn = get_connection()
    if collection:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE username=? AND collection=? ORDER BY updated_at DESC",
            (username, collection)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM conversations WHERE username=? ORDER BY updated_at DESC",
            (username,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_conversation(conv_id):
    conn = get_connection()
    row  = conn.execute("SELECT * FROM conversations WHERE id=?", (conv_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_conversation_title(conv_id, title):
    conn = get_connection()
    conn.execute(
        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
        (title, datetime.now().isoformat(), conv_id)
    )
    conn.commit()
    conn.close()


def delete_conversation(conv_id):
    """Xóa cuộc hội thoại và toàn bộ tin nhắn."""
    conn = get_connection()
    conn.execute("DELETE FROM conversation_messages WHERE conversation_id=?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATION MESSAGES (TIN NHẮN)
# ═══════════════════════════════════════════════════════════════════════════════

def add_message(conv_id, role, content, sources=""):
    """Thêm tin nhắn vào cuộc hội thoại."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO conversation_messages (conversation_id, role, content, sources, created_at) VALUES (?,?,?,?,?)",
        (conv_id, role, content, sources, datetime.now().isoformat())
    )
    conn.execute(
        "UPDATE conversations SET updated_at=? WHERE id=?",
        (datetime.now().isoformat(), conv_id)
    )
    conn.commit()
    conn.close()


def get_messages(conv_id):
    """Lấy toàn bộ tin nhắn của 1 cuộc hội thoại."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM conversation_messages WHERE conversation_id=? ORDER BY created_at ASC",
        (conv_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ═══════════════════════════════════════════════════════════════════════════════
# CHAT LOGS (ADMIN XEM)
# ═══════════════════════════════════════════════════════════════════════════════

def save_chat_log(username, question, answer, sources=""):
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_logs (username, question, answer, sources, created_at) VALUES (?,?,?,?,?)",
        (username, question, answer, sources, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_chat_logs(username=None):
    conn = get_connection()
    if username:
        rows = conn.execute(
            "SELECT * FROM chat_logs WHERE username=? ORDER BY created_at DESC LIMIT 200",
            (username,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chat_logs ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ═══════════════════════════════════════════════════════════════════════════════
# ─── Documents (Tiêu đề + Mô tả) ─────────────────────────────────────────────

def save_document_info(filename, collection_name, title="", description=""):
    """Lưu tiêu đề và mô tả cho tài liệu."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO documents (filename, collection_name, title, description, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(filename, collection_name) DO UPDATE SET
            title=excluded.title,
            description=excluded.description
    """, (filename, collection_name, title, description, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_document_info(filename, collection_name):
    """Lấy tiêu đề và mô tả của tài liệu."""
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM documents WHERE filename=? AND collection_name=?",
        (filename, collection_name)
    ).fetchone()
    conn.close()
    return dict(row) if row else {"title": "", "description": ""}