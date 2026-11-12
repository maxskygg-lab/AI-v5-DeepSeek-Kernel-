"""笔记云端持久化：独立 namespace、失败重试、云端恢复。"""

import hashlib
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

logger = get_logger("notes")


def _notes_namespace():
    """笔记使用独立 namespace：按工作区隔离、跨主题共享（主题记录在笔记自身的 topic 字段）。"""
    return "notes-" + hashlib.sha1(f"{get_workspace_id()}:notes".encode()).hexdigest()[:32]


def _retry_cloud(fn, times=3, delay=0.6):
    """网络抖动重试：0.6s → 1.2s 指数退避，缓解瞬时失败。"""
    last = None
    for i in range(times):
        try:
            return fn()
        except Exception as e:
            last = e
            logger.debug("cloud_retry attempt=%d/%d error=%s", i + 1, times, e)
            if i < times - 1:
                time.sleep(delay * (2**i))
    raise last


def _note_to_metadata(note):
    return {
        "type": "note",
        "note_id": str(note.get("id") or ""),
        "content": note.get("content") or "",
        "question": note.get("question") or "",
        "tags": ",".join(str(t) for t in (note.get("tags") or [])),
        "topic": note.get("topic") or "",
        "ts": note.get("ts") or "",
    }


def _metadata_to_note(md):
    tags = md.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    return {
        "id": str(md.get("note_id") or ""),
        "content": md.get("content") or "",
        "question": md.get("question") or "",
        "tags": [str(t) for t in tags],
        "topic": md.get("topic") or "",
        "ts": md.get("ts") or "",
    }


def save_note_to_cloud(note):
    """把单条笔记写入云端。失败不抛异常，登记到待重试队列，保证内容不丢。"""
    nid = str(note.get("id") or "")
    text = f"{note.get('question') or ''}\n{note.get('content') or ''}".strip() or "空笔记"
    started = time.perf_counter()
    ns = ""
    try:
        vec = get_embeddings_model().embed_query(text[:4000])
        ns = _notes_namespace()
        md = _note_to_metadata(note)

        def _do():
            idx = _pinecone_index()
            try:
                idx.upsert(vectors=[{"id": nid, "values": vec, "metadata": md}], namespace=ns)
            except Exception as e2:
                logger.debug("note_upsert_legacy_payload note_id=%s error=%s", nid, e2)
                idx.upsert(vectors=[(nid, vec, md)], namespace=ns)

        _retry_cloud(_do)
        st.session_state["notes_unsynced"] = [
            n for n in st.session_state.get("notes_unsynced", []) if str(n.get("id")) != nid
        ]
        st.session_state["notes_last_error"] = None
        logger.debug(
            "note_saved ns=%s note_id=%s elapsed_ms=%.0f",
            ns,
            nid,
            (time.perf_counter() - started) * 1000,
        )
        return True
    except Exception as e:
        st.session_state["notes_last_error"] = str(e)
        logger.warning(
            "note_save_failed ns=%s note_id=%s error=%s elapsed_ms=%.0f",
            ns,
            nid,
            e,
            (time.perf_counter() - started) * 1000,
        )
        pending_ids = {str(n.get("id")) for n in st.session_state.get("notes_unsynced", [])}
        if nid not in pending_ids:
            st.session_state.setdefault("notes_unsynced", []).append(note)
        return False


def delete_notes_in_cloud(note_ids):
    """按 id 删除云端笔记；失败仅记录，不打断界面操作。"""
    ids = [str(i) for i in note_ids if i]
    if not ids:
        return True
    ns = _notes_namespace()
    started = time.perf_counter()
    try:
        _retry_cloud(lambda: _pinecone_index().delete(ids=ids, namespace=ns))
        st.session_state["notes_last_error"] = None
        logger.debug(
            "notes_deleted ns=%s count=%d elapsed_ms=%.0f",
            ns,
            len(ids),
            (time.perf_counter() - started) * 1000,
        )
        return True
    except Exception as e:
        st.session_state["notes_last_error"] = str(e)
        logger.warning("note_delete_failed ns=%s count=%d error=%s", ns, len(ids), e)
        return False


def clear_notes_in_cloud():
    """清空云端笔记 namespace（先列出 id 再批量删除）。"""
    ns = _notes_namespace()
    started = time.perf_counter()
    try:
        idx = _pinecone_index()
        ids = _list_namespace_ids(idx, ns)
        for start in range(0, len(ids), 500):
            batch = ids[start : start + 500]
            _retry_cloud(lambda: idx.delete(ids=batch, namespace=ns))
        st.session_state["notes_last_error"] = None
        logger.info(
            "notes_cleared ns=%s count=%d elapsed_ms=%.0f",
            ns,
            len(ids),
            (time.perf_counter() - started) * 1000,
        )
        return True
    except Exception as e:
        st.session_state["notes_last_error"] = str(e)
        logger.warning("notes_clear_failed ns=%s error=%s", ns, e)
        return False


def load_notes_from_cloud():
    """读取云端全部笔记，返回按时间升序的列表。"""
    ns = _notes_namespace()
    started = time.perf_counter()
    idx = _pinecone_index()
    ids = _list_namespace_ids(idx, ns)
    notes = []
    for v in _fetch_vectors(idx, ids, ns).values():
        md = dict(getattr(v, "metadata", None) or {})
        if md.get("type") != "note":
            continue
        n = _metadata_to_note(md)
        if n["id"] and n["content"]:
            notes.append(n)
    notes.sort(key=lambda n: n.get("ts") or "")
    logger.debug(
        "notes_loaded ns=%s count=%d elapsed_ms=%.0f",
        ns,
        len(notes),
        (time.perf_counter() - started) * 1000,
    )
    return notes


def sync_notes_from_cloud():
    """云端 → 本地合并：同 id 以云端为准，本地尚未上传的笔记保留。失败时抛异常由调用方提示。"""
    cloud = load_notes_from_cloud()
    by_id = {str(n.get("id")): n for n in st.session_state.notes}
    for n in cloud:
        by_id[str(n["id"])] = n
    st.session_state.notes = sorted(by_id.values(), key=lambda n: n.get("ts") or "")
    st.session_state["notes_synced"] = True
    logger.info("notes_synced cloud=%d total=%d", len(cloud), len(st.session_state.notes))
    return len(cloud)


def retry_unsynced_notes():
    """把上次写入失败的笔记补传云端（网络恢复后自动补漏）。"""
    pending = list(st.session_state.get("notes_unsynced", []))
    if pending:
        logger.debug("notes_retry_pending count=%d", len(pending))
    for n in pending:
        save_note_to_cloud(n)
