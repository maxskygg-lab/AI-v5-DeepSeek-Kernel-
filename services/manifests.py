"""论文页级清单的云端持久化：刷新页面后仍保留总页数、空页与低质量页信息。"""

import hashlib
import json
import time

import streamlit as st

from .embeddings import get_embeddings_model
from .log import get_logger
from .vectorstore import (
    _fetch_vectors,
    _list_namespace_ids,
    _pinecone_index,
    get_workspace_id,
)

logger = get_logger("manifests")

_PAGE_FIELDS = ("text_pages", "empty_pages", "short_pages", "garbled_pages")


def _manifests_namespace():
    return "manifests-" + hashlib.sha1(f"{get_workspace_id()}:manifests".encode()).hexdigest()[:32]


def manifest_vector_id(topic_name, file_name):
    return hashlib.sha1(f"{topic_name}::{file_name}".encode("utf-8")).hexdigest()


def _manifest_to_metadata(topic_name, file_name, manifest):
    md = {
        "type": "paper_manifest",
        "topic": str(topic_name),
        "file": str(file_name),
        "total_pages": int(manifest.get("total_pages") or 0),
        "chunks": int(manifest.get("chunks") or 0),
        "dropped_chunks": int(manifest.get("dropped_chunks") or 0),
        "chars": int(manifest.get("chars") or 0),
    }
    for field in _PAGE_FIELDS:
        md[field] = json.dumps(list(manifest.get(field) or []))
    return md


def _metadata_to_manifest(md):
    manifest = {
        "file": md.get("file") or "",
        "total_pages": int(md.get("total_pages") or 0),
        "chunks": int(md.get("chunks") or 0),
        "dropped_chunks": int(md.get("dropped_chunks") or 0),
        "chars": int(md.get("chars") or 0),
    }
    for field in _PAGE_FIELDS:
        try:
            manifest[field] = [int(p) for p in json.loads(md.get(field) or "[]")]
        except (TypeError, ValueError):
            manifest[field] = []
    return manifest


def save_manifest_to_cloud(topic_name, file_name, manifest):
    """写入单篇论文的页级清单；失败只记日志，不打断入库主流程。"""
    started = time.perf_counter()
    ns = ""
    try:
        ns = _manifests_namespace()
        vec = get_embeddings_model().embed_query(f"{topic_name}::{file_name}")
        vid = manifest_vector_id(topic_name, file_name)
        md = _manifest_to_metadata(topic_name, file_name, manifest)

        idx = _pinecone_index()
        try:
            idx.upsert(vectors=[{"id": vid, "values": vec, "metadata": md}], namespace=ns)
        except Exception as e:
            logger.debug("manifest_upsert_legacy_payload file=%s error=%s", file_name, e)
            idx.upsert(vectors=[(vid, vec, md)], namespace=ns)

        logger.debug(
            "manifest_saved ns=%s topic=%s file=%s elapsed_ms=%.0f",
            ns,
            topic_name,
            file_name,
            (time.perf_counter() - started) * 1000,
        )
        return True
    except Exception as e:
        logger.warning(
            "manifest_save_failed ns=%s topic=%s file=%s error=%s elapsed_ms=%.0f",
            ns,
            topic_name,
            file_name,
            e,
            (time.perf_counter() - started) * 1000,
        )
        return False


def load_manifests_from_cloud():
    """启动时把云端页级清单合并进 session_state，返回加载数量。失败不阻塞启动。"""
    started = time.perf_counter()
    try:
        ns = _manifests_namespace()
        idx = _pinecone_index()
        ids = _list_namespace_ids(idx, ns)
        if not ids:
            return 0

        manifests = st.session_state.setdefault("paper_manifests", {})
        loaded = 0
        for v in _fetch_vectors(idx, ids, ns).values():
            md = dict(getattr(v, "metadata", None) or {})
            if md.get("type") != "paper_manifest":
                continue
            topic_name = str(md.get("topic") or "")
            file_name = str(md.get("file") or "")
            if not topic_name or not file_name:
                continue
            manifests.setdefault(f"{topic_name}::{file_name}", _metadata_to_manifest(md))
            loaded += 1

        logger.debug(
            "manifests_loaded ns=%s count=%d elapsed_ms=%.0f",
            ns,
            loaded,
            (time.perf_counter() - started) * 1000,
        )
        return loaded
    except Exception as e:
        logger.warning("manifests_load_failed error=%s", e)
        return 0
