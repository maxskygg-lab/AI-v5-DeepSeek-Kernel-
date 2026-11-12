import importlib

import streamlit as st

# ================= 1. 环境检查与导入 =================
_REQUIRED_MODULES = (
    "langchain_community",
    "fitz",
    "langchain_openai",
    "langchain_community.embeddings",
    "langchain_pinecone",
    "rank_bm25",
    "sentence_transformers",
)
for _module in _REQUIRED_MODULES:
    try:
        importlib.import_module(_module)
    except ImportError as e:
        st.error(
            f"🚑 环境缺失库 -> {e.name}. 请运行: pip install langchain-openai sentence-transformers pymupdf langchain-pinecone pinecone-client rank_bm25"
        )
        st.stop()

from services.config import SS_API_KEY, USER_API_KEY
from services.log import get_logger, setup_logging
from services.notes import retry_unsynced_notes, sync_notes_from_cloud
from ui.layout import setup_page
from ui.notes_tab import render_notes_tab
from ui.read_tab import render_read_tab
from ui.search_tab import render_search_tab
from ui.sidebar import render_sidebar

setup_logging()
logger = get_logger("app")
setup_page()

# ================= 2. 状态初始化 =================
defaults = {
    "search_results": [],
    "search_generator": None,
    "citations_loaded": False,
    "citations_global_cache": {},
    "suggested_query": "",
    "focus_paper_id": None,
    "contributions_cache": {},
    "score_cache": {},
    "chat_history": [],
    "topics": {"default_topic": {"files": [], "chunks": [], "db": None}},
    "active_topic": "default_topic",
    "selected_scope": "🌐 对比所有论文",
    "notes": [],
    "pending_note": None,
    "notes_synced": False,  # 本次会话是否已从云端拉取过笔记
    "notes_unsynced": [],  # 尚未成功写入云端、待重试的笔记
    "notes_last_error": None,
    "preload_done_ids": set(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- 启动时同步云端笔记（每个会话一次；失败不阻塞应用启动） ---
if not st.session_state.get("notes_synced"):
    try:
        with st.spinner("正在同步云端笔记..."):
            sync_notes_from_cloud()
        retry_unsynced_notes()
    except Exception as e:
        # 标记已尝试，避免每次交互都重复联网；失败后可点「同步云端笔记」手动重试
        logger.warning("startup_notes_sync_failed error=%s", e)
        st.session_state["notes_synced"] = True
        st.session_state["notes_last_error"] = str(e)

# ================= 3. 侧边栏 =================
user_api_key, ss_api_key = render_sidebar(USER_API_KEY, SS_API_KEY)

# ================= 4. 主界面 =================
tab_main, tab_read, tab_notes = st.tabs(["🔍 学术检索 & 图谱", "📖 研读空间", "📌 我的笔记"])

with tab_main:
    render_search_tab(user_api_key, ss_api_key)

with tab_read:
    render_read_tab(user_api_key, ss_api_key)

with tab_notes:
    render_notes_tab()
