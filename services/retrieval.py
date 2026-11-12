"""混合检索：BM25 + Dense → RRF 融合 → Cross-Encoder 重排。"""

import hashlib
import time

import numpy as np
import streamlit as st

from .embeddings import get_cross_encoder
from .log import get_logger

logger = get_logger("retrieval")


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
    cross_scores = cross_encoder.predict([[query, d.page_content] for d in candidate_docs])
    top_cross_indices = np.argsort(cross_scores)[::-1]
    docs, chosen = [], set()

    if cross_paper:
        seen_papers = set()
        # 第一轮：每篇论文只保留最高分的一个块，保证跨论文覆盖
        for i in top_cross_indices:
            if len(docs) >= 5:
                break
            d = candidate_docs[i]
            src = d.metadata.get("source_paper", "未知")
            if src in seen_papers:
                continue
            seen_papers.add(src)
            docs.append(d)
            chosen.add(i)
        # 第二轮：候选覆盖的论文不足 5 篇时，用剩余高分块补齐
        for i in top_cross_indices:
            if len(docs) >= 5:
                break
            if i not in chosen:
                docs.append(candidate_docs[i])
                chosen.add(i)
    else:
        docs = [candidate_docs[i] for i in top_cross_indices[:5]]
    logger.debug(
        "hybrid_retrieve cross_paper=%s chunks=%d candidates=%d selected=%d elapsed_ms=%.0f",
        cross_paper,
        len(target_chunks),
        len(candidate_docs),
        len(docs),
        (time.perf_counter() - started) * 1000,
    )
    return docs
