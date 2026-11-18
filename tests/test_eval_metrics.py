import types

from eval import metrics


def _doc(paper, page):
    return types.SimpleNamespace(
        page_content="text", metadata={"source_paper": paper, "page": page}
    )


def test_doc_key_converts_zero_based_page():
    assert metrics.doc_key(_doc("p.pdf", 0)) == ("p.pdf", 1)
    assert metrics.doc_key(_doc("p.pdf", 4)) == ("p.pdf", 5)


def test_page_recall_counts_expected_pages():
    expected = metrics.expected_keys([{"paper": "p.pdf", "pages": [1, 3]}])
    docs = [_doc("p.pdf", 0), _doc("p.pdf", 1), _doc("p.pdf", 2)]

    assert metrics.page_recall(docs, expected) == 1.0
    assert metrics.page_recall(docs[:1], expected) == 0.5
    assert metrics.page_recall(docs, set()) is None


def test_mrr_uses_first_hit_rank():
    docs = [_doc("p.pdf", 9), _doc("p.pdf", 1)]
    expected = {("p.pdf", 2)}

    assert metrics.mrr(docs, expected) == 0.5
    assert metrics.mrr([_doc("q.pdf", 1)], expected) == 0.0


def test_citation_hit_requires_matching_paper_and_page():
    expected = {("p.pdf", 3)}

    assert metrics.citation_hit([_doc("p.pdf", 2)], expected) is True
    assert metrics.citation_hit([_doc("q.pdf", 2)], expected) is False
