"""Semantic Scholar 请求层：引用数、论文详情与关系图谱。"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import streamlit as st

from .log import get_logger
from .utils import get_pure_arxiv_id

logger = get_logger("semantic_scholar")


def _ss_headers(ss_key):
    return {"x-api-key": ss_key} if ss_key else {}


def _ss_error_text(status):
    """把 Semantic Scholar 的状态码翻译成可读原因。403 只代表 Key 无效或已失效。"""
    if status == 403:
        return "API Key 无效或已失效（HTTP 403）"
    if status == 429:
        return "Semantic Scholar 限流（HTTP 429），请稍后重试"
    if status >= 500:
        return f"Semantic Scholar 服务端异常（HTTP {status}）"
    return f"HTTP {status}"


def _ss_get_json(url, ss_key="", timeout=12):
    r = requests.get(url, headers=_ss_headers(ss_key), timeout=timeout)
    if r.status_code != 200:
        raise requests.HTTPError(_ss_error_text(r.status_code), response=r)
    return r.json()


@st.cache_data(ttl=1800)
def fetch_citations_batch_cached(arxiv_ids_tuple: tuple, ss_key=None) -> dict:
    clean_ids = [f"ArXiv:{get_pure_arxiv_id(arxiv_id)}" for arxiv_id in arxiv_ids_tuple]
    url = "https://api.semanticscholar.org/graph/v1/paper/batch"
    r = requests.post(
        url,
        headers=_ss_headers(ss_key or ""),
        params={"fields": "citationCount,externalIds"},
        json={"ids": clean_ids},
        timeout=15,
    )
    # 失败抛异常而不是返回 {}，否则空结果会被缓存 30 分钟
    if r.status_code != 200:
        raise requests.HTTPError(_ss_error_text(r.status_code), response=r)
    out = {}
    for item in r.json():
        if item and item.get("externalIds"):
            aid = item["externalIds"].get("ArXiv", "")
            if aid:
                out[aid] = item.get("citationCount", 0)
    return out


def fetch_one_citation(args):
    arxiv_id, ss_key = args
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{get_pure_arxiv_id(arxiv_id)}?fields=citationCount"
        headers = _ss_headers(ss_key)
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200:
            return arxiv_id, r.json().get("citationCount", 0)
        logger.debug("citation_fetch_http arxiv_id=%s status=%d", arxiv_id, r.status_code)
    except Exception as e:
        logger.debug("citation_fetch_failed arxiv_id=%s error=%s", arxiv_id, e)
    return arxiv_id, 0


def fetch_citations_parallel(results, ss_key=None):
    args_list = [(item["obj"].entry_id, ss_key) for item in results]
    out = {}
    with ThreadPoolExecutor(max_workers=8 if ss_key else 3) as pool:
        for future in as_completed({pool.submit(fetch_one_citation, a): a[0] for a in args_list}):
            aid, count = future.result()
            out[aid] = count
    return out


def smart_fetch_citations(results, ss_key=None):
    cache = st.session_state.citations_global_cache
    missing = [item for item in results if get_pure_arxiv_id(item["obj"].entry_id) not in cache]
    hits = len(results) - len(missing)
    if hits:
        st.caption(f"⚡ {hits} 篇命中缓存，{len(missing)} 篇需请求")
    if missing:
        ids = tuple(item["obj"].entry_id for item in missing)
        try:
            new_data = fetch_citations_batch_cached(ids, ss_key)
        except Exception as e:
            logger.warning("citation_batch_failed count=%d error=%s", len(missing), e)
            new_data = {}
        if not new_data:
            new_data = {
                get_pure_arxiv_id(k): v
                for k, v in fetch_citations_parallel(missing, ss_key).items()
            }
        cache.update(new_data)
    return {
        item["obj"].entry_id: cache.get(get_pure_arxiv_id(item["obj"].entry_id), 0)
        for item in results
    }


def preload_top_graphs(results, ss_key=None, top_n=3):
    done = st.session_state.preload_done_ids
    to_do = [
        item
        for item in sorted(results, key=lambda x: x.get("citations") or 0, reverse=True)[:top_n]
        if item["obj"].entry_id not in done
    ]
    if not to_do:
        return
    ph = st.empty()
    ph.caption(f"🔄 后台预加载 Top {len(to_do)} 图谱…")
    for item in to_do:
        try:
            fetch_graph_data(item["obj"].entry_id, ss_key=ss_key)
        except requests.RequestException as e:
            # 预加载失败不影响主流程，用户点开时会再次请求
            logger.debug("graph_preload_failed arxiv_id=%s error=%s", item["obj"].entry_id, e)
        done.add(item["obj"].entry_id)
        time.sleep(0.3)
    ph.caption("✅ 图谱预加载完成")


@st.cache_data(ttl=3600)
def fetch_graph_data(arxiv_id, ss_key=None):
    """拉取论文的引用 / 被引图谱。

    失败时抛异常而不是返回 None：cache_data 不缓存异常，
    否则一次限流就会让“无图”状态持续一小时。
    """
    clean_id = get_pure_arxiv_id(arxiv_id)
    fields = (
        "paperId,title,year,citationCount,abstract,"
        "references.paperId,references.title,references.citationCount,"
        "references.year,references.abstract,references.externalIds,"
        "citations.paperId,citations.title,citations.citationCount,"
        "citations.year,citations.abstract,citations.externalIds"
    )
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields={fields}"
    for attempt in range(3):
        try:
            return _ss_get_json(url, ss_key or "", timeout=12)
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            logger.debug(
                "graph_fetch_retry arxiv_id=%s attempt=%d status=%s error=%s",
                clean_id,
                attempt + 1,
                status,
                e,
            )
            # 403 / 404 是鉴权失败或资源不存在，重试无意义
            if status in (403, 404) or attempt == 2:
                raise
            time.sleep((attempt + 1) * 1.5)


@st.cache_data(ttl=3600)
def fetch_ss_paper_details_by_title(title, ss_key=None):
    """按标题查论文元数据（年份 / 作者 / 引用数 / TLDR / 期刊）。
    查不到时返回 None：详情缺失只是少几个徽章，不影响问答流程。"""
    clean_title = title.replace(".pdf", "").strip()
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={clean_title}&limit=1"
        "&fields=title,year,authors,citationCount,influentialCitationCount,tldr,venue"
    )
    try:
        data = _ss_get_json(url, ss_key or "", timeout=8).get("data", [])
    except requests.RequestException as e:
        logger.debug("paper_details_unavailable title=%s error=%s", clean_title[:60], e)
        return None
    return data[0] if data else None
