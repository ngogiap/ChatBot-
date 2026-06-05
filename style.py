import streamlit as st

# ─── CSS ─────────────────────────────────────────────────────────────────────

ADMIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* Dark mode (mặc định) */
:root {
    --bg-main:        #1a1a1a;
    --bg-card:        #222222;
    --bg-sidebar:     #1e1e1e;
    --bg-input:       #2a2a2a;
    --border:         #333333;
    --border-hover:   #555555;
    --accent:         #6366f1;
    --accent-dim:     rgba(99,102,241,0.15);
    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted:     #6b7280;
    --font-main:      'DM Sans', sans-serif;
    --font-mono:      'DM Mono', monospace;
}

/* Light mode */
@media (prefers-color-scheme: light) {
    :root {
        --bg-main:        #f5f5f7;
        --bg-card:        #ffffff;
        --bg-sidebar:     #f0f0f2;
        --bg-input:       #f8f8fa;
        --border:         #e2e2e6;
        --border-hover:   #c0c0c8;
        --accent:         #4f46e5;
        --accent-dim:     rgba(79,70,229,0.1);
        --text-primary:   #111827;
        --text-secondary: #374151;
        --text-muted:     #6b7280;
        --font-main:      'DM Sans', sans-serif;
        --font-mono:      'DM Mono', monospace;
    }
}

* { font-family: var(--font-main); }

/* ── BACKGROUND ── */
.main                              { background: var(--bg-main) !important; }
.block-container                   { padding-top: 1.5rem !important; max-width: 960px; }
section[data-testid="stSidebar"]   { background: var(--bg-sidebar) !important; border-right: 1px solid var(--border); }

/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu, footer { visibility: hidden; }

/* ── SIDEBAR USER CARD ── */
.sidebar-user {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
}
.sidebar-user-name  { color: var(--text-primary); font-weight: 600; font-size: 0.95rem; }
.sidebar-user-role  { color: var(--text-muted);   font-size: 0.75rem; margin-top: 3px; }

/* ── ADMIN HEADER ── */
.admin-header {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.admin-header-icon  { font-size: 2rem; line-height: 1; }
.admin-header h1    { color: var(--text-primary) !important; font-size: 1.4rem !important; font-weight: 600 !important; margin: 0 !important; padding: 0 !important; }
.admin-header p     { color: var(--text-muted)   !important; font-size: 0.82rem !important; margin: 4px 0 0 0 !important; }

/* ── STAT CARD ── */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
}
.stat-card:hover      { border-color: var(--border-hover); }
.stat-card-label      { color: var(--text-muted); font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.stat-card-value      { color: var(--text-primary); font-size: 1.8rem; font-weight: 600; font-family: var(--font-mono); line-height: 1; }

/* ── COLLECTION CARD ── */
.collection-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.collection-card:hover          { border-color: var(--border-hover); }
.collection-card-title          { color: var(--text-primary); font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; }
.collection-card-desc           { color: var(--text-muted);   font-size: 0.8rem;  margin-bottom: 10px; }
.collection-card-meta           { display: flex; gap: 8px; flex-wrap: wrap; }
.meta-pill {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
}
.meta-pill b { color: var(--accent); }

/* ── DOC CARD ── */
.doc-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: border-color 0.15s;
}
.doc-card:hover     { border-color: var(--border-hover); }
.doc-icon           { font-size: 1.3rem; flex-shrink: 0; }
.doc-info           { flex: 1; }
.doc-name           { color: var(--text-primary); font-size: 0.88rem; font-weight: 500; }
.doc-meta           { color: var(--text-muted);   font-size: 0.72rem; font-family: var(--font-mono); margin-top: 2px; }

/* ── BREADCRUMB ── */
.breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
    font-size: 0.82rem;
    padding: 8px 14px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
}
.breadcrumb-sep     { color: var(--border-hover); }
.breadcrumb-active  { color: var(--text-primary); font-weight: 500; }

/* ── SECTION TITLE ── */
.section-title {
    color: var(--text-muted);
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 20px 0 10px 0;
}

/* ── SOURCE CARD ── */
.source-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: border-color 0.15s;
}
.source-card:hover  { border-color: var(--border-hover); }
.source-card a      { color: #818cf8; text-decoration: none; font-size: 0.82rem; }
.source-card a:hover{ text-decoration: underline; }

/* ── CHAT BUBBLES ── */
.user-row   { display: flex; justify-content: flex-end; margin: 10px 0; }
.user-bubble {
    background: var(--accent);
    color: white;
    padding: 10px 14px;
    border-radius: 16px 16px 4px 16px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.bot-row    { display: flex; align-items: flex-start; gap: 8px; margin: 10px 0; }
.bot-bubble {
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 10px 14px;
    border-radius: 4px 16px 16px 16px;
    max-width: 700px;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid var(--border);
}

/* ── BUTTONS ── */
.stButton > button                      { border-radius: 7px !important; font-weight: 500 !important; font-size: 0.85rem !important; transition: all 0.15s !important; }
.stButton > button[kind="primary"]      { background: var(--accent) !important; border: none !important; }
.stButton > button[kind="secondary"]    { background: var(--bg-card) !important; border: 1px solid var(--border) !important; color: var(--text-secondary) !important; }
.stButton > button:hover                { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── INPUTS ── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea,
.stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 7px !important;
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-dim) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"]   { background: var(--bg-card) !important; border-radius: 8px !important; padding: 3px !important; border: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"]        { border-radius: 6px !important; color: var(--text-muted) !important; font-size: 0.83rem !important; }
.stTabs [aria-selected="true"]      { background: var(--bg-input) !important; color: var(--text-primary) !important; }

/* ── EXPANDER ── */
.streamlit-expanderHeader   { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 7px !important; color: var(--text-secondary) !important; font-size: 0.83rem !important; }
.streamlit-expanderContent  { background: var(--bg-main)  !important; border: 1px solid var(--border) !important; border-top: none !important; }

/* ── METRIC ── */
[data-testid="stMetricValue"]   { font-family: var(--font-mono) !important; color: var(--text-primary) !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"]   { color: var(--text-muted) !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }

/* ── SOURCE CARD COMPACT ── */
.source-card-compact {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    transition: all 0.15s;
}
.source-card-compact:hover {
    border-color: var(--border-hover);
}

.source-meta {
    margin-top: 6px;
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-muted);
}

.source-meta span {
    font-family: var(--font-mono);
}

.source-link {
    color: #818cf8 !important;
    text-decoration: none;
    font-size: 0.82rem;
}
.source-link:hover {
    text-decoration: underline;
}
/* ── SOURCE LINE COMPACT ── */
.source-line {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 14px;
    margin: 4px 0;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.15s;
}
.source-line:hover {
    border-color: var(--border-hover);
    background: var(--bg-input);
}

.source-open-link {
    color: #818cf8 !important;
    text-decoration: none;
    margin-left: 4px;
}
.source-open-link:hover {
    text-decoration: underline;
}

/* Excerpt khi mở */
.source-excerpt-compact {
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 2px 0 10px 0;
    color: var(--text-secondary);
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    max-width: 860px !important;
    margin: 0 auto !important;
}

div[class*="stChatFloatingInputContainer"] {
    max-width: 860px !important;
    margin: 0 auto !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
}
/* ── DIVIDER / SCROLLBAR ── */
hr { border-color: var(--border) !important; }
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-main); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""


def apply_styles():
    st.markdown(ADMIN_CSS, unsafe_allow_html=True)

    # Detect Streamlit theme và override CSS variables
    # Streamlit thêm class .st-emotion-cache-... khác nhau cho light/dark
    # Dùng JS để detect và apply đúng theme
    st.markdown("""
    <script>
    (function() {
        function applyTheme() {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark'
                        || window.matchMedia('(prefers-color-scheme: dark)').matches;

            const root = document.documentElement;
            if (!isDark) {
                root.style.setProperty('--bg-main',        '#f5f5f7');
                root.style.setProperty('--bg-card',        '#ffffff');
                root.style.setProperty('--bg-sidebar',     '#f0f0f2');
                root.style.setProperty('--bg-input',       '#f8f8fa');
                root.style.setProperty('--border',         '#e2e2e6');
                root.style.setProperty('--border-hover',   '#c0c0c8');
                root.style.setProperty('--accent',         '#4f46e5');
                root.style.setProperty('--accent-dim',     'rgba(79,70,229,0.1)');
                root.style.setProperty('--text-primary',   '#111827');
                root.style.setProperty('--text-secondary', '#374151');
                root.style.setProperty('--text-muted',     '#6b7280');
            }
        }
        applyTheme();
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyTheme);
    })();
    </script>
    """, unsafe_allow_html=True)