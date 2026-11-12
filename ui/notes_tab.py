"""Tab 3：笔记库。"""

import streamlit as st

from services.log import get_logger
from services.notes import (
    clear_notes_in_cloud,
    delete_notes_in_cloud,
    retry_unsynced_notes,
    sync_notes_from_cloud,
)

logger = get_logger("ui.notes")


def render_notes_tab():
    c_title, c_sync = st.columns([4, 1])
    with c_title:
        st.subheader("📌 我的笔记库")
    with c_sync:
        st.write("")
        if st.button("🔄 同步云端笔记", help="重新从云端拉取并合并本工作区的笔记"):
            try:
                with st.spinner("同步中..."):
                    n_cloud = sync_notes_from_cloud()
                    retry_unsynced_notes()
                st.toast(f"已同步 {n_cloud} 条云端笔记", icon="✅")
            except Exception as e:
                logger.warning("notes_manual_sync_failed error=%s", e)
                st.session_state["notes_last_error"] = str(e)
                st.error(f"云端同步失败：{e}")
    if st.session_state.get("notes_last_error"):
        st.caption(
            f"⚠️ 最近一次云端同步失败：{st.session_state['notes_last_error']}。"
            "笔记内容仍保留在当前页面，可点「同步云端笔记」重试。"
        )

    if not st.session_state.notes:
        st.info(
            "还没有笔记。在「📖 研读空间」提问后点「📌 存笔记」即可；笔记会同步到云端，刷新或换设备后自动恢复。"
        )
    else:
        all_tags = sorted(set(tag for n in st.session_state.notes for tag in n["tags"]))
        all_topics = sorted(set(n["topic"] for n in st.session_state.notes))
        f1, f2, f3 = st.columns([2, 2, 2])
        with f1:
            filter_tag = st.selectbox("按标签", ["全部"] + all_tags)
        with f2:
            filter_topic = st.selectbox("按主题", ["全部"] + all_topics)
        with f3:
            search_note = st.text_input("关键词", placeholder="搜索笔记")

        filtered = st.session_state.notes.copy()
        if filter_tag != "全部":
            filtered = [n for n in filtered if filter_tag in n["tags"]]
        if filter_topic != "全部":
            filtered = [n for n in filtered if n["topic"] == filter_topic]
        if search_note:
            filtered = [
                n
                for n in filtered
                if search_note.lower() in (n["content"] + n.get("question", "")).lower()
            ]

        st.caption(f"共 {len(filtered)} 条")
        for note in reversed(filtered):
            with st.container(border=True):
                h1, h2, h3 = st.columns([3, 2, 1])
            with h1:
                st.markdown(
                    f'<span class="topic-badge">🗂️ {note["topic"]}</span>', unsafe_allow_html=True
                )
                for tag in note["tags"]:
                    st.markdown(f'<span class="note-tag">#{tag}</span>', unsafe_allow_html=True)
            with h2:
                st.caption(f"🕐 {note['ts']}")
            with h3:
                if st.button("🗑️", key=f"delnote_{note['id']}"):
                    # 先删云端，成功后再删本地；失败则保留，避免刷新后又“复活”
                    if delete_notes_in_cloud([note["id"]]):
                        st.session_state.notes = [
                            n for n in st.session_state.notes if n["id"] != note["id"]
                        ]
                    else:
                        st.toast("云端删除失败，笔记已保留，请稍后重试", icon="⚠️")
                    st.rerun()
            if note.get("question"):
                st.markdown(f"**❓ {note['question']}**")
            st.markdown(note["content"])  # 完整内容，不截断

        st.markdown("---")
        if st.button("🗑️ 清空所有笔记", type="secondary"):
            if not clear_notes_in_cloud():
                st.warning("云端清空失败，已停止本地清空以免数据不一致，请稍后重试。")
            else:
                st.session_state.notes = []
            st.rerun()

