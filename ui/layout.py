"""页面级配置：主题 CSS 与标题。"""

import streamlit as st

_CSS = """
<style>
    .stButton>button { width:100%; border-radius:8px; }
    .abstract-box {
        background:#f0f2f6; padding:12px; border-radius:8px;
        border-left:5px solid #4CAF50; font-size:.9em;
        line-height:1.6; margin-bottom:6px;
    }
    .contribution-box {
        background:linear-gradient(90deg,#fffbeb,#fef3c7);
        border-left:4px solid #f59e0b; padding:7px 12px;
        border-radius:6px; font-size:.85em; color:#78350f;
        margin-bottom:8px; font-weight:500;
    }
    .cite-badge {
        background:#ff4b4b; color:white; padding:2px 7px;
        border-radius:12px; font-size:.78em; font-weight:bold;
    }
    .cite-loading { color:#94a3b8; font-size:.78em; font-style:italic; }
    .topic-badge {
        display:inline-block; background:#6366f1; color:white;
        padding:2px 10px; border-radius:20px; font-size:.78em;
        font-weight:600; margin-right:4px;
    }
    .note-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 18px; margin-bottom:10px;
    }
    .note-tag {
        display:inline-block; background:#e0e7ff; color:#3730a3;
        padding:1px 8px; border-radius:12px; font-size:.75em;
        margin-right:4px; font-weight:500;
    }
    .chat-panel {
        height:520px; overflow-y:auto; border:1px solid #e2e8f0;
        border-radius:10px; padding:12px; background:#fafafa; margin-bottom:10px;
    }
    .chat-user { background:#dbeafe; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:.9em; }
    .chat-bot  { background:#f0fdf4; border-radius:8px; padding:8px 12px; margin:6px 0; font-size:.9em; }
    .chat-notice { color:#6366f1; font-size:.82em; font-style:italic; margin:4px 0; }
    .section-divider {
        font-size:.72em; text-transform:uppercase; letter-spacing:2px;
        color:#94a3b8; margin:18px 0 8px;
    }
    .perf-badge {
        display:inline-block; background:#dcfce7; color:#166534;
        border:1px solid #86efac; padding:2px 8px; border-radius:12px;
        font-size:.75em; font-weight:600; margin-left:6px;
    }
    .tracker-card {
        background:#f8fafc; border:1px solid #e2e8f0;
        border-radius:10px; padding:14px 16px; margin-bottom:14px;
    }
    .tracker-new-badge {
        display:inline-block; background:#f59e0b; color:#fff;
        padding:2px 9px; border-radius:12px; font-size:.75em; font-weight:700; margin-left:6px;
    }
    .new-paper-card {
        background:#fffbeb; border:1px solid #fde68a;
        border-left:4px solid #f59e0b; border-radius:8px;
        padding:12px 16px; margin:8px 0; font-size:.88em; line-height:1.65;
    }
</style>
"""


def setup_page():
    st.set_page_config(page_title="AI 深度研读助手", layout="wide", page_icon="🎓")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("📖 AI 深度研读助手 v5 (DeepSeek Kernel)")
