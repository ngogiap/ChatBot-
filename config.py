import json
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = "config.json"

# Giá trị mặc định nếu chưa có config.json
DEFAULT_CONFIG = {
    "chunk_size":    1000,
    "chunk_overlap": 200,
    "chunk_method":  "by_size",
    "embed_model":   "nomic-embed-text",
    "provider": "ollama",
    "llm_model":     "llama3",
    "temperature":   0.01
}


def load_config():
    """Đọc cấu hình từ file, nếu chưa có thì dùng mặc định."""
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Đảm bảo không thiếu key nào so với mặc định
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value

    return config


def save_config(config):
    """Ghi cấu hình xuống file."""
    safe_config = {k: v for k, v in config.items() if not k.endswith("_api_key")}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def get_api_key(provider: str) -> str:
    """
    Lấy API Key theo thứ tự ưu tiên:
    1. Từ file .env (an toàn nhất)
    2. Từ config.json (fallback)
    """
    if not provider:
        return ""
    
    provider = provider.lower().strip()
    env_name = f"{provider.upper()}_API_KEY"
    
    # Ưu tiên 1: Đọc từ .env
    key = os.getenv(env_name)
    if key and key.strip():
        return key.strip()
    
    # Ưu tiên 2: Đọc từ config.json
    cfg = load_config()
    return cfg.get(f"{provider}_api_key", "").strip()

def get_provider_config():
    """Lấy toàn bộ config + API key (dùng trong trang config)"""
    cfg = load_config()
    cfg["gemini_api_key"] = get_api_key("gemini")
    cfg["groq_api_key"] = get_api_key("groq")
    return cfg

def get(key):
    """Lấy 1 giá trị cấu hình theo key."""
    return load_config().get(key, DEFAULT_CONFIG.get(key))