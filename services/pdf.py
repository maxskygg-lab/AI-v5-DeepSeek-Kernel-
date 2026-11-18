"""PDF 下载、解析与主题入库。"""

import os
import tempfile
import time

import requests
import streamlit as st

from .config import PINECONE_API_KEY
from .log import get_logger
from .manifests import save_manifest_to_cloud
from .utils import get_pure_arxiv_id
from .vectorstore import get_vectorstore, make_namespace

logger = get_logger("pdf")

# 解析产物至少要有这些字符才认为提取成功
MIN_CHUNK_CHARS = 20

# 页级体检阈值：过短或乱码比例过高都会进入提示
MANIFEST_SHORT_CHARS = 80
MANIFEST_GARBLED_RATIO = 0.05


def _text_stats(text):
    """返回 (长度, 有效字符数, 替换字符数)，用于识别空页/乱码页。"""
    text = text or ""
    stripped = text.strip()
    good = sum(1 for ch in stripped if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    bad = stripped.count("\ufffd")
    return len(stripped), good, bad


def format_pages(pages, limit=10):
    """把页码列表压成一行，避免空页很多时提示过长。"""
    pages = list(pages)
    if not pages:
        return "无"
    text = "、".join(f"P{p}" for p in pages[:limit])
    if len(pages) > limit:
        text += f" 等 {len(pages)} 页"
    return text


def build_pdf_manifest(file_path, file_name, docs, total_pages=None):
    """生成页级清单：总页数、有文本页、空页、过短页、疑似乱码页。

    total_pages 允许注入，便于测试；生产环境从 PyMuPDF 读取真实页数。
    """
    if total_pages is None:
        try:
            import fitz

            with fitz.open(file_path) as pdf:
                total_pages = pdf.page_count
        except Exception:
            total_pages = max((_page_of_doc(d, i) + 1 for i, d in enumerate(docs)), default=0)

    text_pages = set()
    short_pages = set()
    garbled_pages = set()
    for index, doc in enumerate(docs):
        page = _page_of_doc(doc, index) + 1
        text = doc.page_content or ""
        length, _good, bad = _text_stats(text)
        if length == 0:
            continue
        text_pages.add(page)
        if length < MANIFEST_SHORT_CHARS:
            short_pages.add(page)
        if bad and bad / max(1, length) > MANIFEST_GARBLED_RATIO:
            garbled_pages.add(page)

    all_pages = set(range(1, int(total_pages or 0) + 1))
    return {
        "file": file_name,
        "total_pages": int(total_pages or 0),
        "text_pages": sorted(text_pages),
        "empty_pages": sorted(all_pages - text_pages),
        "short_pages": sorted(short_pages),
        "garbled_pages": sorted(garbled_pages),
        "chunks": 0,
        "dropped_chunks": 0,
        "chars": sum(len((d.page_content or "").strip()) for d in docs),
    }


def reconstruct_manifest_from_chunks(file_name, chunks):
    """旧数据没有 manifest 时，从已入库 chunk 反推一个下界清单。"""
    paper_chunks = [c for c in chunks if c.metadata.get("source_paper") == file_name]
    pages = sorted({_page_of_doc(c, 0) + 1 for c in paper_chunks})
    return {
        "file": file_name,
        "total_pages": max(pages, default=0),
        "text_pages": pages,
        "empty_pages": [],
        "short_pages": [],
        "garbled_pages": [],
        "chunks": len(paper_chunks),
        "dropped_chunks": 0,
        "chars": sum(len(c.page_content) for c in paper_chunks),
        "reconstructed": True,
    }


def get_paper_manifest(topic_name, file_name):
    return st.session_state.get("paper_manifests", {}).get(f"{topic_name}::{file_name}")


def manifest_key(topic_name, file_name):
    return f"{topic_name}::{file_name}"


def download_arxiv_pdf_direct(arxiv_id):
    started = time.perf_counter()
    clean_id = get_pure_arxiv_id(arxiv_id)
    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    pdf_path = os.path.join(tempfile.gettempdir(), f"{clean_id}.pdf")
    r = requests.get(pdf_url, timeout=15)
    with open(pdf_path, "wb") as f:
        f.write(r.content)
    logger.debug(
        "pdf_downloaded arxiv_id=%s bytes=%d elapsed_ms=%.0f",
        clean_id,
        len(r.content),
        (time.perf_counter() - started) * 1000,
    )
    return pdf_path


def has_extractable_text(docs):
    """判断解析结果里是否存在非空正文。"""
    return bool(docs) and any(d.page_content.strip() for d in docs)


def _extract_text_with_fitz(file_path, file_name, topic_name):
    """PyPDF 提取不到文字时回退到 PyMuPDF，兼容扫描版/特殊编码 PDF。"""
    import fitz
    from langchain_core.documents import Document

    docs = []
    with fitz.open(file_path) as pdf:
        for i, page in enumerate(pdf):
            txt = page.get_text()
            if txt and txt.strip():
                docs.append(
                    Document(
                        page_content=txt,
                        metadata={"source_paper": file_name, "topic": topic_name, "page": i},
                    )
                )
    return docs


def _page_of_doc(doc, default=0):
    try:
        return int(doc.metadata.get("page", default) or default)
    except (TypeError, ValueError):
        return default


def _merge_fitz_fallback(docs, fitz_docs, min_chars=200):
    """用 PyMuPDF 的页面文本补齐 PyPDF 提取过短的页面。"""
    if not fitz_docs:
        return docs
    by_page = {_page_of_doc(d): d for d in fitz_docs}
    merged = []
    for index, doc in enumerate(docs):
        page = _page_of_doc(doc, index)
        alt = by_page.get(page)
        if (
            alt
            and len(doc.page_content.strip()) < min_chars
            and len(alt.page_content.strip()) > len(doc.page_content.strip())
        ):
            merged.append(alt)
        else:
            merged.append(doc)

    known_pages = {_page_of_doc(d, i) for i, d in enumerate(merged)}
    for alt in fitz_docs:
        if _page_of_doc(alt) not in known_pages:
            merged.append(alt)
    merged.sort(key=lambda d: _page_of_doc(d))
    return merged


def load_pdf_documents(file_path, file_name, topic_name):
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_paper"] = file_name
        doc.metadata["topic"] = topic_name

    total_pages = len(docs)
    nonempty_pages = sum(1 for doc in docs if doc.page_content.strip())
    # pypdf 只抽出少量页面时（如大部分页面解析失败），用 PyMuPDF 补齐
    if total_pages and nonempty_pages / total_pages < 0.6:
        logger.info(
            "pdf_text_coverage_low file=%s pages=%d nonempty=%d",
            file_name,
            total_pages,
            nonempty_pages,
        )
        docs = _merge_fitz_fallback(docs, _extract_text_with_fitz(file_path, file_name, topic_name))

    # 若 pypdf 完全未能提取文字（如扫描版 PDF），回退到 PyMuPDF
    if not has_extractable_text(docs):
        docs = _extract_text_with_fitz(file_path, file_name, topic_name)
    return docs


def split_documents_with_stats(docs):
    """切块并返回丢弃数量，便于页级体检说明“有没有静默丢内容”。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=200, separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    chunks = []
    dropped = 0
    for index, chunk in enumerate(splitter.split_documents(docs)):
        if len(chunk.page_content.strip()) <= MIN_CHUNK_CHARS:
            dropped += 1
            continue
        chunk.metadata["chunk_index"] = index
        chunks.append(chunk)
    return chunks, dropped


def split_documents(docs):
    chunks, _ = split_documents_with_stats(docs)
    return chunks


def process_and_add_to_topic(file_path, file_name, api_key, topic_name=None):
    started = time.perf_counter()
    topic_name = topic_name or st.session_state.active_topic
    t = st.session_state.topics[topic_name]
    try:
        docs = load_pdf_documents(file_path, file_name, topic_name)
        manifest = build_pdf_manifest(file_path, file_name, docs)
        chunks, dropped = split_documents_with_stats(docs)
        manifest["chunks"] = len(chunks)
        manifest["dropped_chunks"] = dropped
        st.session_state.setdefault("paper_manifests", {})[
            manifest_key(topic_name, file_name)
        ] = manifest

        if not chunks:
            logger.warning(
                "pdf_empty_text topic=%s file=%s pages=%d empty_pages=%d elapsed_ms=%.0f",
                topic_name,
                file_name,
                manifest["total_pages"],
                len(manifest["empty_pages"]),
                (time.perf_counter() - started) * 1000,
            )
            st.error(
                f"🚫 《{file_name}》未能提取到任何文字（可能是扫描版 PDF 或文件已损坏），已跳过入库，未加入可检索内容。"
                f"体检结果：共 {manifest['total_pages']} 页，空页 {format_pages(manifest['empty_pages'])}。"
            )
            return False
        t["chunks"].extend(chunks)

        if manifest["empty_pages"]:
            st.warning(
                f"《{file_name}》共 {manifest['total_pages']} 页，"
                f"其中 {format_pages(manifest['empty_pages'])} 没有可提取文字（可能是图片/扫描页）。"
                "这些页已记录在资料体检中，需要内容时可做 OCR。"
            )

        batch = 10
        if not PINECONE_API_KEY:
            logger.warning("paper_ingest_no_pinecone_key topic=%s file=%s", topic_name, file_name)
            st.error("🚑 请先在 Streamlit Secrets 中配置 PINECONE_API_KEY")
            return False
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

        if t["db"] is None:
            # 初始化连接云端索引，按主题划分 namespace（转码为 ASCII 兼容格式）
            safe_namespace = make_namespace(topic_name)
            t["db"] = get_vectorstore(safe_namespace)

        # 直接向 Pinecone 批量添加文档向量
        for i in range(0, len(chunks), batch):
            t["db"].add_documents(chunks[i : i + batch])
            time.sleep(0.1)

        if file_name not in t["files"]:
            t["files"].append(file_name)
        st.session_state.chat_history.append(
            {"role": "system_notice", "content": f"📚 《{file_name}》已加入主题「{topic_name}」。"}
        )
        save_manifest_to_cloud(topic_name, file_name, manifest)
        logger.info(
            "paper_ingested topic=%s file=%s chunks=%d elapsed_ms=%.0f",
            topic_name,
            file_name,
            len(chunks),
            (time.perf_counter() - started) * 1000,
        )
        return True
    except Exception as e:
        logger.exception("paper_ingest_failed topic=%s file=%s error=%s", topic_name, file_name, e)
        st.error(f"处理失败: {e}")
        return False
