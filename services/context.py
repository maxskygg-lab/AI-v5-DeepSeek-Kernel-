"""按论文构建结构化摘要，作为对比/综述与跨论文问答的上下文层。"""

import hashlib
import re

import streamlit as st

from .llm import get_deepseek_llm
from .log import get_logger
from .retrieval import select_representative_chunks
from .semantic_scholar import fetch_ss_paper_details_by_title

logger = get_logger("context")

DIGEST_CHUNKS = 24


def _page(chunk):
    try:
        return int(chunk.metadata.get("page", 0) or 0) + 1
    except (TypeError, ValueError):
        return 1


def _chunk_index(chunk):
    try:
        return int(chunk.metadata.get("chunk_index", 0) or 0)
    except (TypeError, ValueError):
        return 0


_CITE_PATTERNS = (
    re.compile(r"P\s*(\d{1,3})", re.I),
    re.compile(r"第\s*(\d{1,3})\s*页"),
    re.compile(r"page\s*(\d{1,3})", re.I),
)


def _pages_in_text(text):
    pages = []
    for pattern in _CITE_PATTERNS:
        for raw in pattern.findall(text):
            page = int(raw)
            if page not in pages:
                pages.append(page)
    return pages


_WORD_RE = re.compile(r"[a-z][a-z0-9\-\.]*", re.I)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _tokens(text):
    text = (text or "").lower()
    tokens = set(_WORD_RE.findall(text))
    for block in _CJK_RE.findall(text):
        if len(block) == 1:
            tokens.add(block)
        else:
            tokens.update(block[i : i + 2] for i in range(len(block) - 1))
    return tokens


def _citation_queries(digest):
    """返回 {页码: 摘要中该页的引用上下文}，关键证据优先。"""
    queries = {}
    key_section = ""
    match = re.search(r"#{2,4}\s*关键证据(.*)$", digest, re.S)
    if match:
        key_section = match.group(1)
    for text in (key_section, digest):
        for line in text.splitlines():
            pages = _pages_in_text(line)
            if not pages:
                continue
            for page in pages:
                queries.setdefault(page, line.strip())
    return queries


def _score_chunk(query_tokens, chunk):
    if not query_tokens:
        return 0
    return len(query_tokens & _tokens(chunk.page_content))


def extract_cited_pages(digest, limit=12):
    """从摘要中提取引用页码，优先解析“关键证据”段。"""
    if not digest:
        return []
    key_section = ""
    match = re.search(r"#{2,4}\s*关键证据(.*)$", digest, re.S)
    if match:
        key_section = match.group(1)
    pages = []
    for text in (key_section, digest):
        for page in _pages_in_text(text):
            if page not in pages:
                pages.append(page)
        if len(pages) >= limit:
            break
    return pages[:limit]


def collect_cited_chunks(paper, chunks, digest, max_chunks=8, max_per_page=2):
    """按摘要引用页码取回原文，保证页码引用有原文可核对。"""
    paper_chunks = [c for c in chunks if c.metadata.get("source_paper") == paper]
    cited_pages = extract_cited_pages(digest, limit=max_chunks)
    if not cited_pages:
        return select_representative_chunks(paper_chunks, max_chunks=min(6, max_chunks))

    by_page = {}
    for chunk in paper_chunks:
        page = _page(chunk)
        if page in cited_pages:
            by_page.setdefault(page, []).append(chunk)

    cites = _citation_queries(digest)

    def ranked(page):
        query_tokens = _tokens(cites.get(page, ""))
        return sorted(
            by_page.get(page, []),
            key=lambda c: (-_score_chunk(query_tokens, c), _chunk_index(c)),
        )

    result = []
    # 先保证每个引用页至少一段，再补同页的第二段
    for page in cited_pages:
        page_chunks = ranked(page)
        if page_chunks and len(result) < max_chunks:
            result.append(page_chunks[0])
    for page in cited_pages:
        page_chunks = ranked(page)
        for chunk in page_chunks[1:max_per_page]:
            if len(result) >= max_chunks:
                break
            result.append(chunk)
        if len(result) >= max_chunks:
            break
    if not result:
        return select_representative_chunks(paper_chunks, max_chunks=min(6, max_chunks))
    result.sort(key=lambda c: (_page(c), _chunk_index(c)))
    return result[:max_chunks]


def cited_page_coverage(chunks, digest):
    """返回 (引用页数, 有原文的页数, 缺失页列表)。"""
    cited = set(extract_cited_pages(digest))
    available = {_page(c) for c in chunks}
    missing = sorted(cited - available)
    return len(cited), len(cited & available), missing


def strip_overlap(previous, current, min_overlap=80, max_overlap=400):
    """去掉相邻切片之间因 chunk_overlap 产生的重复文本。"""
    if not previous:
        return current
    limit = min(len(previous), len(current), max_overlap)
    for size in range(limit, min_overlap - 1, -1):
        if previous[-size:] == current[:size]:
            return current[size:]
    return current


def _paper_meta(paper, ss_key):
    title = paper.replace(".pdf", "").strip()
    meta = {"title": title, "authors": "", "year": "", "venue": "", "tldr": ""}
    if not ss_key:
        return meta
    try:
        data = fetch_ss_paper_details_by_title(paper, ss_key, timeout=4)
    except Exception as e:
        logger.debug("paper_meta_failed paper=%s error=%s", paper, e)
        return meta
    if not data:
        return meta

    authors = [a.get("name", "") for a in (data.get("authors") or []) if a.get("name")]
    if authors:
        meta["authors"] = ", ".join(authors[:5]) + (" 等" if len(authors) > 5 else "")
    meta["title"] = data.get("title") or title
    meta["year"] = str(data.get("year") or "")
    meta["venue"] = data.get("venue") or ""
    tldr = data.get("tldr") or {}
    meta["tldr"] = tldr.get("text", "") if isinstance(tldr, dict) else ""
    return meta


def build_digest_prompt(paper, chunks, meta):
    """把代表性片段拼成结构化摘要提示词（纯函数，便于测试）。"""
    meta_line = "；".join(
        part
        for part in [
            f"标题：{meta.get('title', paper)}",
            f"作者：{meta.get('authors')}" if meta.get("authors") else "",
            f"年份：{meta.get('year')}" if meta.get("year") else "",
            f"发表处：{meta.get('venue')}" if meta.get("venue") else "",
        ]
        if part
    )
    lines = [
        "你是一名严谨的学术论文信息抽取助手。请只根据下面提供的论文片段，输出结构化摘要。",
        "要求：",
        "1. 只使用片段中明确出现的信息，不得臆测或补全；确实缺失的字段写“资料未覆盖”。",
        "2. 每条结论后附页码，格式 [P12]；不要编造页码。",
        "3. 每个字段用 1-3 句话，保持紧凑。",
        "4. 请特别完整地提取基准数据集、SOTA 指标与局限性；片段中有数字时保留数字。",
        "",
        "输出格式：",
        "## 基本信息",
        "## 研究问题",
        "## 核心方法",
        "## 数据集 / 实验设置",
        "## 主要结果与指标",
        "## 创新点",
        "## 局限性",
        "## 关键证据（3-5 条，含页码与短引文）",
        "",
        f"【论文】{paper}",
        f"【元信息】{meta_line}",
        "",
        "【原文片段】",
    ]
    previous = ""
    for chunk in chunks:
        text = chunk.page_content.replace("\n", " ").strip()
        text = strip_overlap(previous, text)
        previous = chunk.page_content.replace("\n", " ").strip()
        label = f"P{_page(chunk)}"
        chunk_no = chunk.metadata.get("chunk_index")
        if chunk_no is not None:
            label += f" | 片段{int(chunk_no) + 1}"
        lines.append(f"[{label}] {text}")
    return "\n".join(lines)


def get_paper_digest(paper, chunks, api_key, ss_key="", with_meta=True):
    """生成并缓存单篇论文摘要；失败时退回代表性原文摘录。"""
    selected = select_representative_chunks(chunks, max_chunks=DIGEST_CHUNKS)
    signature = hashlib.sha1(
        "|".join(
            [
                paper,
                str(len(chunks)),
                selected[0].page_content[:80] if selected else "",
                selected[-1].page_content[-80:] if selected else "",
                "meta" if with_meta else "plain",
            ]
        ).encode("utf-8")
    ).hexdigest()
    cache = st.session_state.setdefault("paper_digests", {})
    cached = cache.get(paper)
    if cached and cached.get("signature") == signature:
        return cached

    try:
        meta = (
            _paper_meta(paper, ss_key)
            if with_meta
            else {"title": paper.replace(".pdf", "")}
        )
    except Exception as e:
        logger.debug("paper_meta_parse_failed paper=%s error=%s", paper, e)
        meta = {"title": paper.replace(".pdf", "")}
    pages = sorted({_page(c) for c in selected})
    if not selected:
        result = {
            "digest": "（没有可用片段，可能是扫描版或解析失败）",
            "meta": meta,
            "pages": [],
            "chunk_count": 0,
            "signature": signature,
        }
        cache[paper] = result
        return result

    try:
        llm = get_deepseek_llm(api_key, temperature=0.0)
        response = llm.invoke(build_digest_prompt(paper, selected, meta))
        digest = response.content.strip()
    except Exception as e:
        logger.warning("paper_digest_failed paper=%s error=%s", paper, e)
        digest = "（摘要生成失败，以下为代表性原文摘录）\n" + "\n".join(
            f"[P{_page(c)}] {c.page_content[:300]}" for c in selected[:8]
        )

    result = {
        "digest": digest,
        "meta": meta,
        "pages": pages,
        "chunk_count": len(selected),
        "signature": signature,
    }
    cache[paper] = result
    logger.info(
        "paper_digest_ok paper=%s chunks=%d pages=%d", paper, len(selected), len(pages)
    )
    return result


def get_paper_digests(papers, chunks, api_key, ss_key="", with_meta=True):
    """批量生成摘要，供矩阵、综述和跨论文问答共用。"""
    result = {}
    for paper in papers:
        paper_chunks = [c for c in chunks if c.metadata.get("source_paper") == paper]
        result[paper] = get_paper_digest(paper, paper_chunks, api_key, ss_key, with_meta)
    return result


def format_digests(digests):
    """把摘要字典拼成 LLM 上下文。"""
    parts = []
    for paper, info in digests.items():
        meta = info.get("meta", {})
        header = meta.get("title") or paper
        parts.append(f"【论文：{header}】\n{info.get('digest', '')}")
    return "\n\n".join(parts)
