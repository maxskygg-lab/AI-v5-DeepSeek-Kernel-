"""向量库与主题空间：稳定的工作区标识、云端恢复、旧数据迁移。"""

import hashlib
import os
import time

import streamlit as st

from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from .embeddings import get_embeddings_model
from .log import get_logger

logger = get_logger("vectorstore")


def get_workspace_id():
    """稳定的工作区标识（跨会话不变），用于 namespace 隔离。
    优先级：secrets 的 WORKSPACE_ID > 登录邮箱 > 固定兜底。绝不使用随机值，
    否则每次刷新页面 namespace 都会变，导致上传的数据“消失”。"""
    if "wm_workspace_id" in st.session_state:
        return st.session_state["wm_workspace_id"]
    wid = None
    # 1) 显式配置（最稳定，推荐在 secrets 里设置 WORKSPACE_ID）
    try:
        if "WORKSPACE_ID" in st.secrets and str(st.secrets["WORKSPACE_ID"]).strip():
            wid = hashlib.sha1(str(st.secrets["WORKSPACE_ID"]).strip().encode()).hexdigest()[:16]
    except Exception as e:
        logger.debug("workspace_secret_unavailable error=%s", e)
    # 2) 登录邮箱（启用鉴权时可按用户隔离）
    if not wid:
        for attr in ("user", "experimental_user"):
            try:
                email = getattr(getattr(st, attr), "email", None)
                if email:
                    wid = hashlib.sha1(email.encode()).hexdigest()[:16]
                    break
            except Exception as e:
                logger.debug("workspace_email_unavailable attr=%s error=%s", attr, e)
    # 3) 固定兜底（保证跨会话稳定，不使用随机 uuid）
    if not wid:
        wid = "default"
    st.session_state["wm_workspace_id"] = wid
    return wid


def make_namespace(topic_name):
    return hashlib.sha1(f"{get_workspace_id()}:{topic_name}".encode("utf-8")).hexdigest()


@st.cache_resource
def get_vectorstore(namespace):
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=get_embeddings_model(), namespace=namespace
    )


def _vector_id(v):
    """不同 pinecone SDK 版本返回的向量对象形状不同，统一取出字符串 id。"""
    if isinstance(v, str):
        return v
    if hasattr(v, "id"):
        return v.id
    return str(v)


def _ns_vector_count(v):
    """兼容 describe_index_stats().namespaces 的两种返回形状：dict / NamespaceDescription 对象。"""
    if isinstance(v, dict):
        return int(v.get("vector_count", 0) or 0)
    return int(getattr(v, "vector_count", 0) or 0)


def _int_meta(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _list_namespace_ids(idx, ns):
    """版本兼容地列出 namespace 下全部向量 id（优先新版 list_paginated）。"""
    ids = []
    try:
        # 新版 SDK (>=5.x)：list_paginated 返回分页响应
        token = None
        while True:
            resp = (
                idx.list_paginated(namespace=ns, limit=1000)
                if token is None
                else idx.list_paginated(namespace=ns, limit=1000, pagination_token=token)
            )
            vecs = getattr(resp, "vectors", None) or getattr(resp, "ids", None) or []
            for v in vecs:
                ids.append(_vector_id(v))
            token = getattr(getattr(resp, "pagination", None), "next", None)
            if not token or len(ids) >= 20000:
                break
        return ids
    except Exception as e:
        logger.debug("list_paginated_unavailable ns=%s error=%s", ns, e)
    # 旧版 SDK (<5.x)：list() 直接迭代
    ids = []
    for batch in idx.list(namespace=ns):
        if isinstance(batch, (list, tuple)):
            for item in batch:
                ids.append(_vector_id(item))
        else:
            ids.append(_vector_id(batch))
        if len(ids) >= 20000:
            break
    return ids


def _fetch_vectors(idx, ids, ns, max_url_chars=3000):
    """批量取向量。Pinecone 的 fetch 走 GET，id 全部拼在 URL 上，
    500 个 UUID 约 20KB，会超过网关上限直接返回 414，故按 URL 长度分组。"""
    vectors = {}
    batch, size = [], 0
    for vid in ids:
        cost = len(str(vid)) + 5  # "&ids=" + id
        if batch and size + cost > max_url_chars:
            got = idx.fetch(ids=batch, namespace=ns)
            vectors.update(getattr(got, "vectors", None) or {})
            batch, size = [], 0
        batch.append(vid)
        size += cost
    if batch:
        got = idx.fetch(ids=batch, namespace=ns)
        vectors.update(getattr(got, "vectors", None) or {})
    return vectors


def restore_topic_from_pinecone(topic_name, verbose=False):
    """从 Pinecone 云端恢复某主题已入库的论文清单与切片，支持重开页面继续研读。"""
    t = st.session_state.topics.get(topic_name)
    if not t:
        logger.debug("restore_skipped topic=%s reason=no_topic", topic_name)
        return
    if t.get("chunks") and t.get("files"):
        return
    ns = make_namespace(topic_name)
    api_key = PINECONE_API_KEY or ""
    started = time.perf_counter()
    try:
        from pinecone import Pinecone
        from langchain_core.documents import Document

        os.environ.setdefault("PINECONE_API_KEY", api_key)
        idx = Pinecone(api_key=api_key).Index(PINECONE_INDEX_NAME)
        ids = _list_namespace_ids(idx, ns)

        if not ids:
            logger.info("restore_empty topic=%s ns=%s", topic_name, ns)
            if verbose:
                exist = {}
                try:
                    stats = idx.describe_index_stats()
                    ns_map = getattr(stats, "namespaces", None) or {}
                    exist = {k: _ns_vector_count(v) for k, v in ns_map.items()}
                except Exception as e:
                    exist = {"__err__": str(e)}
                st.info(
                    f"云端主题「{topic_name}」暂无数据。\n\n"
                    f"本应用应查 namespace = `{ns}`\n\n"
                    f"索引现有 namespace / 记录数 = `{exist}`"
                )
            t["db"] = get_vectorstore(ns)
            return

        docs, files = [], []
        vectors = _fetch_vectors(idx, ids, ns)
        for v in vectors.values():
            md = dict(getattr(v, "metadata", None) or {})
            text = md.pop("text", "") or ""
            if not text:
                continue
            docs.append(Document(page_content=text, metadata=md))
            sp = md.get("source_paper")
            if sp and sp not in files:
                files.append(sp)

        if docs:
            # 云端返回顺序不稳定，按论文、页码、切片序号恢复文档顺序
            docs.sort(
                key=lambda d: (
                    str(d.metadata.get("source_paper", "")),
                    _int_meta(d.metadata.get("page")),
                    _int_meta(d.metadata.get("chunk_index")),
                )
            )
            t["chunks"], t["files"] = docs, files
            t["db"] = get_vectorstore(ns)
            st.success(f"已从云端恢复 {len(files)} 篇论文、{len(docs)} 个切片。")
            logger.info(
                "restore_ok topic=%s ns=%s files=%d chunks=%d elapsed_ms=%.0f",
                topic_name,
                ns,
                len(files),
                len(docs),
                (time.perf_counter() - started) * 1000,
            )
        else:
            st.warning(f"云端找到 {len(ids)} 条向量，但未能还原正文（metadata 缺少 text 字段）。")
            t["db"] = get_vectorstore(ns)
            logger.warning("restore_no_text topic=%s ns=%s vectors=%d", topic_name, ns, len(ids))
    except Exception as e:
        st.warning(f"从 Pinecone 恢复失败：{e}")
        logger.warning(
            "restore_failed topic=%s ns=%s error=%s elapsed_ms=%.0f",
            topic_name,
            ns,
            e,
            (time.perf_counter() - started) * 1000,
        )
        try:
            t["db"] = get_vectorstore(ns)
        except Exception as e2:
            logger.debug("restore_db_init_failed ns=%s error=%s", ns, e2)


def migrate_legacy_namespaces(topic_name):
    """把散落在旧 namespace 的向量，按正文去重后搬到当前稳定 namespace（不动旧数据）。"""
    started = time.perf_counter()
    ns = make_namespace(topic_name)
    api_key = PINECONE_API_KEY or ""
    os.environ.setdefault("PINECONE_API_KEY", api_key)
    from pinecone import Pinecone

    idx = Pinecone(api_key=api_key).Index(PINECONE_INDEX_NAME)
    stats = idx.describe_index_stats()
    ns_map = getattr(stats, "namespaces", None) or {}
    src = [k for k, v in ns_map.items() if k != ns and _ns_vector_count(v) > 0]
    legacy_ns = {topic_name.encode("utf-8").hex(), "ns_" + topic_name.encode("utf-8").hex()}

    unique = {}
    for s in src:
        try:
            ids = _list_namespace_ids(idx, s)
            vecs = _fetch_vectors(idx, ids, s)
            for _vid, v in vecs.items():
                md = dict(getattr(v, "metadata", None) or {})
                text = (md.get("text") or "").strip()
                if not text:
                    continue
                # 只迁移属于当前主题的数据，避免把其它主题搬进来
                if md.get("topic") != topic_name and s not in legacy_ns:
                    continue
                h = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if h in unique:
                    continue
                sp = md.get("source_paper") or "迁移论文"
                new_md = {"source_paper": sp, "topic": topic_name, "text": text}
                if isinstance(md.get("page"), (int, float)):
                    new_md["page"] = int(md["page"])
                if isinstance(md.get("chunk_index"), (int, float)):
                    new_md["chunk_index"] = int(md["chunk_index"])
                vals = getattr(v, "values", None)
                if not vals:
                    continue
                unique[h] = {"id": "mig_" + h[:32], "values": list(vals), "metadata": new_md}
        except Exception as e:
            logger.warning("migrate_namespace_skipped ns=%s error=%s", s, e)
            st.warning(f"跳过旧 namespace {s[:8]}…：{e}")

    if not unique:
        logger.info(
            "migrate_noop topic=%s ns=%s elapsed_ms=%.0f",
            topic_name,
            ns,
            (time.perf_counter() - started) * 1000,
        )
        return 0
    items = list(unique.values())
    for i in range(0, len(items), 100):
        batch = items[i : i + 100]
        try:
            idx.upsert(vectors=batch, namespace=ns)
        except Exception:
            idx.upsert(vectors=[(b["id"], b["values"], b["metadata"]) for b in batch], namespace=ns)
    logger.info(
        "migrate_ok topic=%s ns=%s migrated=%d elapsed_ms=%.0f",
        topic_name,
        ns,
        len(items),
        (time.perf_counter() - started) * 1000,
    )
    return len(items)


def active_topic_data():
    return st.session_state.topics[st.session_state.active_topic]


def _pinecone_index():
    """统一获取 Pinecone Index 句柄（与主题数据共用同一索引）。"""
    from pinecone import Pinecone

    os.environ.setdefault("PINECONE_API_KEY", PINECONE_API_KEY or "")
    return Pinecone(api_key=PINECONE_API_KEY or "").Index(PINECONE_INDEX_NAME)


def rebuild_topic_index(topic_name, api_key):
    t = st.session_state.topics[topic_name]
    if not t["chunks"]:
        t["db"] = None
        return

    if PINECONE_API_KEY:
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    safe_namespace = make_namespace(topic_name)
    t["db"] = get_vectorstore(safe_namespace)
