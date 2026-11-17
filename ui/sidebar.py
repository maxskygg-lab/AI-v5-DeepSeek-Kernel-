"""左侧控制台：主题管理、PDF 上传。"""

import os
import tempfile

import streamlit as st

from services.log import get_logger
from services.pdf import process_and_add_to_topic
from services.topics import save_topics_to_cloud
from services.vectorstore import (
    active_topic_data,
    make_namespace,
    migrate_legacy_namespaces,
    rebuild_topic_index,
    restore_topic_from_pinecone,
)

logger = get_logger("ui.sidebar")


def _persist_topics():
    if not save_topics_to_cloud():
        st.warning("主题列表云端保存失败，刷新后可能丢失；请检查 PINECONE_API_KEY 或网络。")


def render_sidebar(user_api_key, ss_api_key):
    with st.sidebar:
        st.header("🎛️ 控制台")
        st.success("🚀 高速调研模式已激活")
        if not ss_api_key:
            st.caption("⚠️ 未配置 SS_API_KEY，引用数与关联图谱可能受限流影响")

        cache_sz = len(st.session_state.citations_global_cache)
        if cache_sz:
            st.info(f"⚡ 引用数缓存：{cache_sz} 篇")

        st.markdown("---")
        st.subheader("🗂️ 研究主题")
        tnames = list(st.session_state.topics.keys())
        aidx = (
            tnames.index(st.session_state.active_topic)
            if st.session_state.active_topic in tnames
            else 0
        )
        chosen = st.selectbox("当前主题", tnames, index=aidx)
        if chosen != st.session_state.active_topic:
            st.session_state.active_topic = chosen
            st.session_state.selected_scope = "🌐 对比所有论文"
            _persist_topics()
            st.rerun()

        cn, ca = st.columns([3, 1])
        with cn:
            new_tn = st.text_input("新建主题", placeholder="输入名称", label_visibility="collapsed")
        with ca:
            if st.button("➕") and new_tn.strip():
                nm = new_tn.strip()
                if nm not in st.session_state.topics:
                    st.session_state.topics[nm] = {"files": [], "chunks": [], "db": None}
                    st.session_state.active_topic = nm
                    _persist_topics()
                    st.rerun()

        if len(st.session_state.topics) > 1:
            if st.button(f"🗑️ 删除「{st.session_state.active_topic}」"):
                del st.session_state.topics[st.session_state.active_topic]
                st.session_state.active_topic = list(st.session_state.topics.keys())[0]
                _persist_topics()
                st.rerun()

        ts = active_topic_data()
        # 启动时从云端自动恢复一次本主题的已入库论文（重开页面可继续研读）
        _rk = f"_restored_{st.session_state.active_topic}"
        if not ts["files"] and not st.session_state.get(_rk):
            st.session_state[_rk] = True
            with st.spinner("正在从云端恢复已入库论文..."):
                restore_topic_from_pinecone(st.session_state.active_topic)
                ts = active_topic_data()
        if not ts["files"]:
            if st.button("🔄 从云端恢复本主题", help="重开页面后拉取已入库论文"):
                with st.spinner("恢复中..."):
                    restore_topic_from_pinecone(st.session_state.active_topic, verbose=True)
                st.rerun()
            if st.button(
                "🛠 迁移旧数据到当前主题",
                help="把散落在旧 namespace 的论文去重后搬入本主题（一次性维护）",
            ):
                with st.spinner("正在迁移旧 namespace 数据..."):
                    _n = migrate_legacy_namespaces(st.session_state.active_topic)
                if _n:
                    st.success(f"迁移完成：写入 {_n} 个唯一切片")
                with st.spinner("正在恢复..."):
                    restore_topic_from_pinecone(st.session_state.active_topic, verbose=True)
                st.rerun()
        if ts["files"]:
            st.markdown(f"**已入库（{len(ts['files'])}篇）**")
            for f in list(ts["files"]):
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.text(f"📄 {f[:16]}..." if len(f) > 18 else f"📄 {f}")
                with c2:
                    if st.button("🗑️", key=f"del_{f}"):
                        ts["files"].remove(f)
                        ts["chunks"] = [
                            c for c in ts["chunks"] if c.metadata.get("source_paper") != f
                        ]
                        if ts["db"]:
                            try:
                                ts["db"].delete(filter={"source_paper": f})
                            except Exception as e:
                                logger.warning(
                                    "paper_delete_failed topic=%s file=%s error=%s",
                                    st.session_state.active_topic,
                                    f,
                                    e,
                                )
                        rebuild_topic_index(st.session_state.active_topic, user_api_key)
                        st.rerun()
            if st.button("🗑️ 清空主题", type="primary"):
                if ts["db"]:
                    try:
                        safe_namespace = make_namespace(st.session_state.active_topic)
                        ts["db"].delete(delete_all=True, namespace=safe_namespace)
                    except Exception as e:
                        logger.warning(
                            "topic_clear_failed topic=%s error=%s",
                            st.session_state.active_topic,
                            e,
                        )
                ts["files"], ts["chunks"], ts["db"] = [], [], None
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")
        st.subheader("📥 上传 PDF")
        uploaded_file = st.file_uploader("拖入 PDF", type="pdf")
        if uploaded_file and st.button("确认加载"):
            with st.spinner("解析中..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                if process_and_add_to_topic(path, uploaded_file.name, user_api_key):
                    _persist_topics()
                os.remove(path)
                st.rerun()

        return user_api_key, ss_api_key
