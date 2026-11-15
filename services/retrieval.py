"""混合检索：BM25 + Dense → RRF 融合 → Cross-Encoder 重排。"""

import hashlib
import re
import time

import numpy as np
import streamlit as st

from .embeddings import get_cross_encoder
from .log import get_logger

logger = get_logger("retrieval")

_REF_HEADING_RE = re.compile(r"(?im)^\s*(references|bibliography|参考文献)\s*$")
_CITATION_RE = re.compile(
    r"\[\d{1,3}\]|\(\d{4}\)|et al\.|arxiv:\s*\d{4}\.\d{4,5}|doi:\s*10\.", re.I
)
_SECTION_WEIGHTS = (
    ("conclusion", 6),
    ("limitation", 5),
    ("future work", 3),
    ("result", 4),
    ("experiment", 3),
    ("evaluation", 3),
    ("ablation", 3),
    ("abstract", 3),
    ("method", 2),
    ("approach", 2),
    ("references", -20),
    ("bibliography", -20),
    ("appendix", -2),
)


def _as_int(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _page_of(doc):
    return _as_int(doc.metadata.get("page"), 0)


def _chunk_index_of(doc):
    return _as_int(doc.metadata.get("chunk_index"), 0)


def _order_key(doc):
    return (_page_of(doc), _chunk_index_of(doc), doc.page_content[:40])


def _section_score(doc):
    text = doc.page_content.lower()
    return sum(weight for keyword, weight in _SECTION_WEIGHTS if keyword in text)


def _is_reference_chunk(doc, max_page):
    """识别参考文献页，避免把上下文全喂成引用列表。"""
    text = doc.page_content
    if _REF_HEADING_RE.search(text[:400]):
        return True
    if max_page and _page_of(doc) >= max_page * 0.6:
        if len(_CITATION_RE.findall(text)) >= 8:
            return True
    return False


def select_representative_chunks(chunks, max_chunks=15, exclude_references=True):
    """按页均匀抽样，优先覆盖摘要、方法、结果与结论。"""
    if not chunks:
        return []

    unique = {}
    for chunk in chunks:
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        unique.setdefault(digest, chunk)
    items = list(unique.values())
    max_page = max((_page_of(c) for c in items), default=0)

    if exclude_references:
        filtered = [c for c in items if not _is_reference_chunk(c, max_page)]
        if filtered:
            items = filtered

    items.sort(key=_order_key)
    if len(items) <= max_chunks:
        return items

    selected, chosen = [], set()

    def add(chunk):
        digest = hashlib.sha1(chunk.page_content.encode("utf-8")).hexdigest()
        if digest not in chosen and len(selected) < max_chunks:
            selected.append(chunk)
            chosen.add(digest)

    add(items[0])  # 标题/摘要
    tail_start = max(0, int(len(items) * 0.75))
    tail = sorted(items[tail_start:], key=_section_score, reverse=True)
    if tail:
        add(tail[0])  # 结论/讨论优先

    slots = max_chunks - len(selected)
    if slots > 0:
        step = len(items) / slots
        for i in range(slots):
            add(items[min(len(items) - 1, int((i + 0.5) * step))])

    if len(selected) < max_chunks:
        for chunk in sorted(items, key=_section_score, reverse=True):
            add(chunk)

    selected.sort(key=_order_key)
    return selected


def get_bm25(target_chunks, cache_key):
    if "_bm25_cache" not in st.session_state:
        st.session_state["_bm25_cache"] = {}
    store = st.session_state["_bm25_cache"]
    sig = hashlib.sha1(
        "|".join(
            [
                str(len(target_chunks)),
                target_chunks[0].page_content[:80] if target_chunks else "",
                target_chunks[-1].page_content[-80:] if target_chunks else "",
            ]
        ).encode()
    ).hexdigest()
    hit = store.get(cache_key)
    if hit and hit[0] == sig:
        return hit[1]
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([list(c.page_content.lower()) for c in target_chunks])
    store[cache_key] = (sig, bm25)
    return bm25


def rrf_fuse(bm25_contents, dense_contents, rrf_k=60, limit=50):
    """Reciprocal Rank Fusion：消除 BM25 与向量检索的分数纲量差异。"""
    scores = {}
    for rank, content in enumerate(bm25_contents):
        scores[content] = scores.get(content, 0.0) + 1.0 / (rank + 1 + rrf_k)
    for rank, content in enumerate(dense_contents):
        scores[content] = scores.get(content, 0.0) + 1.0 / (rank + 1 + rrf_k)
    return [text for text, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]]


def hybrid_retrieve(query, target_chunks, db, search_kwargs, cache_key, cross_paper=False):
    """执行 BM25 + 向量召回 + RRF 融合 + Cross-Encoder 重排。

    cross_paper=True 时按论文分组取段，保证结论覆盖所有已入库论文。
    """
    started = time.perf_counter()
    bm25 = get_bm25(target_chunks, cache_key)
    bm25_scores = bm25.get_scores(list(query.lower()))
    top_bm25_idx = np.argsort(bm25_scores)[::-1][:50]

    dense_kwargs = search_kwargs.copy()
    dense_kwargs["k"] = 50
    dense_kwargs.pop("fetch_k", None)
    dense_kwargs.pop("lambda_mult", None)
    dense_docs = db.similarity_search(query, **dense_kwargs)

    candidate_texts = rrf_fuse(
        [target_chunks[idx].page_content for idx in top_bm25_idx],
        [d.page_content for d in dense_docs],
    )
    text_to_doc = {c.page_content: c for c in target_chunks}
    candidate_docs = [text_to_doc[txt] for txt in candidate_texts if txt in text_to_doc]

    cross_encoder = get_cross_encoder()
    cross_scores = np.asarray(
        cross_encoder.predict([[query, d.page_content] for d in candidate_docs]), dtype=float
    )
    max_page = max((_page_of(d) for d in candidate_docs), default=0)
    if cross_paper:
        # 参考文献页在对比场景几乎没有信息量，降权而不是直接丢弃
        for i, doc in enumerate(candidate_docs):
            if _is_reference_chunk(doc, max_page):
                cross_scores[i] -= 10.0

    top_cross_indices = np.argsort(cross_scores)[::-1]
    docs, chosen = [], set()
    paper_count = 0

    if cross_paper:
        paper_count = len({d.metadata.get("source_paper", "未知") for d in candidate_docs})
        per_paper = 6 if paper_count <= 2 else 4
        max_docs = min(20, max(paper_count, per_paper * paper_count))
        quotas = {}

        # 第一轮：每篇论文先取 1 段，保证所有论文都被覆盖
        for i in top_cross_indices:
            if len(docs) >= max_docs:
                break
            d = candidate_docs[i]
            src = d.metadata.get("source_paper", "未知")
            if src in quotas:
                continue
            docs.append(d)
            quotas[src] = 1
            chosen.add(i)

        # 第二轮：按每篇配额补齐，让每篇都有摘要/方法/结果/结论多段
        for i in top_cross_indices:
            if len(docs) >= max_docs:
                break
            if i in chosen:
                continue
            d = candidate_docs[i]
            src = d.metadata.get("source_paper", "未知")
            if quotas.get(src, 0) >= per_paper:
                continue
            if _is_reference_chunk(d, max_page):
                continue
            docs.append(d)
            quotas[src] = quotas.get(src, 0) + 1
            chosen.add(i)

        docs.sort(key=lambda d: (str(d.metadata.get("source_paper", "")), _order_key(d)))
    else:
        docs = [candidate_docs[i] for i in top_cross_indices[:5]]
    logger.debug(
        "hybrid_retrieve cross_paper=%s papers=%d chunks=%d candidates=%d selected=%d "
        "elapsed_ms=%.0f",
        cross_paper,
        paper_count,
        len(target_chunks),
        len(candidate_docs),
        len(docs),
        (time.perf_counter() - started) * 1000,
    )
    return docs
