"""主题列表云端持久化：刷新/换设备后仍能看到已创建的研究主题。"""

import hashlib
import json

import streamlit as st

from .config import PINECONE_API_KEY
from .embeddings import get_embeddings_model
from .log import get_logger
from .vectorstore import _pinecone_index, get_workspace_id

logger = get_logger("topics")

REGISTRY_ID = "topic-registry"


def _topics_namespace():
    return "topics-" + hashlib.sha1(f"{get_workspace_id()}:topics".encode()).hexdigest()[:32]


def _metadata_of(vector):
    if vector is None:
        return {}
    if isinstance(vector, dict):
        return vector.get("metadata") or {}
    return getattr(vector, "metadata", None) or {}


def load_topics_from_cloud():
    """从云端恢复主题名称和上次选中的主题；失败不阻塞启动。"""
    if not PINECONE_API_KEY:
        return 0
    try:
        idx = _pinecone_index()
        resp = idx.fetch(ids=[REGISTRY_ID], namespace=_topics_namespace())
        vectors = getattr(resp, "vectors", None)
        if vectors is None and isinstance(resp, dict):
            vectors = resp.get("vectors")
        vector = (vectors or {}).get(REGISTRY_ID)
        names = json.loads(_metadata_of(vector).get("topics") or "[]")
        if not isinstance(names, list):
            return 0

        added = 0
        for name in names:
            name = str(name).strip()
            if not name:
                continue
            if name not in st.session_state.topics:
                st.session_state.topics[name] = {"files": [], "chunks": [], "db": None}
                added += 1

        active = str(_metadata_of(vector).get("active_topic") or "").strip()
        if active and active in st.session_state.topics:
            st.session_state.active_topic = active
        logger.info(
            "topics_loaded count=%d added=%d active=%s",
            len(names),
            added,
            st.session_state.active_topic,
        )
        return added
    except Exception as e:
        logger.warning("topics_load_failed error=%s", e)
        return 0


def save_topics_to_cloud():
    """把当前主题列表和选中主题写入云端；失败只记录，不影响界面。"""
    if not PINECONE_API_KEY:
        return False
    try:
        vec = st.session_state.get("_topic_registry_vec")
        if not vec:
            vec = get_embeddings_model().embed_query("topic registry")
            st.session_state["_topic_registry_vec"] = vec

        md = {
            "type": "topic_registry",
            "topics": json.dumps(list(st.session_state.topics.keys()), ensure_ascii=False),
            "active_topic": str(st.session_state.get("active_topic") or "default_topic"),
        }
        idx = _pinecone_index()
        try:
            idx.upsert(
                vectors=[{"id": REGISTRY_ID, "values": vec, "metadata": md}],
                namespace=_topics_namespace(),
            )
        except Exception as e2:
            logger.debug("topics_upsert_legacy_payload error=%s", e2)
            idx.upsert(vectors=[(REGISTRY_ID, vec, md)], namespace=_topics_namespace())
        logger.debug(
            "topics_saved count=%d active=%s", len(st.session_state.topics), md["active_topic"]
        )
        return True
    except Exception as e:
        logger.warning("topics_save_failed error=%s", e)
        return False
