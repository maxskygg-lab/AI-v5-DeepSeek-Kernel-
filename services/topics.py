"""主题列表云端持久化：刷新/换设备后仍能看到已创建的研究主题。"""

import hashlib
import json

import streamlit as st

from .config import PINECONE_API_KEY
from .embeddings import get_embeddings_model
from .log import get_logger
from .vectorstore import _pinecone_index, _vector_id, get_workspace_id

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


def _set_sync(action, ok, count=0, error=None):
    st.session_state["topics_sync"] = {
        "action": action,
        "ok": bool(ok),
        "count": int(count),
        "error": str(error or ""),
    }


def _add_topics(names):
    added = 0
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        if name not in st.session_state.topics:
            st.session_state.topics[name] = {"files": [], "chunks": [], "db": None}
            added += 1
    return added


def _registry(idx, namespace):
    resp = idx.fetch(ids=[REGISTRY_ID], namespace=namespace)
    vectors = getattr(resp, "vectors", None)
    if vectors is None and isinstance(resp, dict):
        vectors = resp.get("vectors")
    if isinstance(vectors, list):
        for item in vectors:
            md = _metadata_of(item)
            if md.get("type") == "topic_registry":
                return md
        return {}
    return _metadata_of((vectors or {}).get(REGISTRY_ID))


def load_topics_from_cloud():
    """从云端恢复主题名称和上次选中的主题；失败不阻塞启动。"""
    if not PINECONE_API_KEY:
        _set_sync("load", False, error="未配置 PINECONE_API_KEY")
        return 0
    try:
        idx = _pinecone_index()
        md = _registry(idx, _topics_namespace())
        names = json.loads(md.get("topics") or "[]")
        if not isinstance(names, list):
            names = []
        added = _add_topics(names)
        active = str(md.get("active_topic") or "").strip()
        if active and active in st.session_state.topics:
            st.session_state.active_topic = active
        _set_sync("load", True, count=len(names))
        logger.info(
            "topics_loaded count=%d added=%d active=%s",
            len(names),
            added,
            st.session_state.active_topic,
        )
        return added
    except Exception as e:
        _set_sync("load", False, error=e)
        logger.warning("topics_load_failed error=%s", e)
        return 0


def _existing_names(idx, namespace):
    try:
        md = _registry(idx, namespace)
        names = json.loads(md.get("topics") or "[]")
        return [str(n).strip() for n in names if str(n).strip()]
    except Exception as e:
        logger.debug("topics_existing_read_failed error=%s", e)
        return []


def save_topics_to_cloud(removed=None):
    """把当前主题列表和选中主题写入云端；与云端已有列表合并，避免旧会话覆盖新主题。"""
    if not PINECONE_API_KEY:
        _set_sync("save", False, error="未配置 PINECONE_API_KEY")
        return False
    try:
        idx = _pinecone_index()
        namespace = _topics_namespace()
        names = _existing_names(idx, namespace)
        for name in st.session_state.topics.keys():
            if name not in names:
                names.append(name)
        if removed:
            removed = {str(n) for n in removed}
            names = [n for n in names if n not in removed]
        if not names:
            names = ["default_topic"]
        active = str(st.session_state.get("active_topic") or "default_topic")
        if active not in names:
            names.append(active)

        vec = st.session_state.get("_topic_registry_vec")
        if not vec:
            vec = get_embeddings_model().embed_query("topic registry")
            st.session_state["_topic_registry_vec"] = vec

        md = {
            "type": "topic_registry",
            "topics": json.dumps(names, ensure_ascii=False),
            "active_topic": active,
        }
        try:
            idx.upsert(
                vectors=[{"id": REGISTRY_ID, "values": vec, "metadata": md}],
                namespace=namespace,
            )
        except Exception as e2:
            logger.debug("topics_upsert_legacy_payload error=%s", e2)
            idx.upsert(vectors=[(REGISTRY_ID, vec, md)], namespace=namespace)
        _set_sync("save", True, count=len(names))
        logger.info("topics_saved count=%d active=%s ns=%s", len(names), active, namespace)
        return True
    except Exception as e:
        _set_sync("save", False, error=e)
        logger.warning("topics_save_failed error=%s", e)
        return False


def _sample_namespace_ids(idx, namespace, limit=3):
    try:
        resp = idx.list_paginated(namespace=namespace, limit=limit)
        vectors = getattr(resp, "vectors", None) or getattr(resp, "ids", None) or []
        ids = [_vector_id(v) for v in vectors][:limit]
        if ids:
            return ids
    except Exception as e:
        logger.debug("topics_sample_paginated_failed ns=%s error=%s", namespace, e)
    try:
        for batch in idx.list(namespace=namespace):
            items = batch if isinstance(batch, (list, tuple)) else [batch]
            return [_vector_id(v) for v in items][:limit]
    except Exception as e:
        logger.debug("topics_sample_list_failed ns=%s error=%s", namespace, e)
    return []


def discover_topics_from_cloud():
    """扫描各 namespace 的 topic 元数据，找回未出现在注册表中的主题。"""
    if not PINECONE_API_KEY:
        _set_sync("discover", False, error="未配置 PINECONE_API_KEY")
        return []
    try:
        idx = _pinecone_index()
        stats = idx.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None)
        if namespaces is None and isinstance(stats, dict):
            namespaces = stats.get("namespaces")

        found = set()
        for ns in namespaces or {}:
            name = ns if isinstance(ns, str) else getattr(ns, "name", "")
            if not name or name.startswith("notes-") or name.startswith("topics-"):
                continue
            ids = _sample_namespace_ids(idx, name)
            if not ids:
                continue
            resp = idx.fetch(ids=ids, namespace=name)
            vectors = getattr(resp, "vectors", None)
            if vectors is None and isinstance(resp, dict):
                vectors = resp.get("vectors")
            items = vectors.values() if isinstance(vectors, dict) else (vectors or [])
            for item in items:
                topic = str(_metadata_of(item).get("topic") or "").strip()
                if topic:
                    found.add(topic)

        _add_topics(found)
        if found:
            save_topics_to_cloud()
        _set_sync("discover", True, count=len(found))
        logger.info("topics_discovered count=%d names=%s", len(found), sorted(found))
        return sorted(found)
    except Exception as e:
        _set_sync("discover", False, error=e)
        logger.warning("topics_discover_failed error=%s", e)
        return []
