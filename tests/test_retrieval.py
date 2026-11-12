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
    def __init__(self, text, paper):
        self.page_content = text
        self.metadata = {"source_paper": paper}


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
    assert len(result) == 5
    assert sources == {"paper0", "paper1", "paper2"}
