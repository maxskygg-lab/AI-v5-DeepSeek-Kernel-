import numpy as np

from services import retrieval


def test_rrf_prefers_document_hit_by_both_rankers():
    fused = retrieval.rrf_fuse(["a", "b"], ["b", "c"])

    assert fused[0] == "b"
    assert set(fused) == {"a", "b", "c"}


def test_rrf_respects_limit():
    bm25 = [f"b{i}" for i in range(20)]
    dense = [f"d{i}" for i in range(20)]

    assert len(retrieval.rrf_fuse(bm25, dense, limit=5)) == 5


class _Doc:
    def __init__(self, text, paper, page=0, chunk_index=0):
        self.page_content = text
        self.metadata = {
            "source_paper": paper,
            "page": page,
            "chunk_index": chunk_index,
        }


def test_hybrid_retrieve_covers_multiple_papers(monkeypatch):
    docs = [_Doc(f"d{i}", f"paper{i // 2}") for i in range(6)]

    class _BM25:
        def get_scores(self, tokens):
            return np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])

    class _DB:
        def similarity_search(self, query, **kwargs):
            return docs

    class _CrossEncoder:
        def predict(self, pairs):
            return np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])

    monkeypatch.setattr(retrieval, "get_bm25", lambda *a, **k: _BM25())
    monkeypatch.setattr(retrieval, "get_cross_encoder", lambda: _CrossEncoder())

    result = retrieval.hybrid_retrieve(
        "query", docs, _DB(), {"k": 15}, "cache", cross_paper=True
    )

    sources = {d.metadata["source_paper"] for d in result}
    assert len(result) == 6
    assert sources == {"paper0", "paper1", "paper2"}


def test_select_representative_chunks_spreads_pages_and_skips_references():
    docs = [
        _Doc(
            f"page {i} introduces method and reports results and conclusion",
            "paper",
            page=i,
            chunk_index=i,
        )
        for i in range(30)
    ]
    docs[29] = _Doc(
        "References\n[1] A. Author. 2020.\n[2] B. Author. 2021.\n[3] C. Author. 2022.",
        "paper",
        page=29,
        chunk_index=29,
    )

    selected = retrieval.select_representative_chunks(docs, max_chunks=8)
    pages = [d.metadata["page"] for d in selected]

    assert len(selected) == 8
    assert 0 in pages
    assert 29 not in pages
    assert max(pages) >= 20
    assert len(set(pages)) >= 5


def test_select_representative_chunks_returns_all_when_small():
    docs = [_Doc(f"text {i}", "paper", page=i, chunk_index=i) for i in range(3)]

    assert len(retrieval.select_representative_chunks(docs, max_chunks=8)) == 3
