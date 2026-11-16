import types

from services import pdf


class _Doc:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {}


def test_has_extractable_text():
    assert not pdf.has_extractable_text([])
    assert not pdf.has_extractable_text([_Doc("   ")])
    assert pdf.has_extractable_text([_Doc("正文")])


def test_merge_fitz_fallback_replaces_short_pages():
    class _Page:
        def __init__(self, text, page):
            self.page_content = text
            self.metadata = {"page": page}

    pypdf_docs = [_Page("", 0), _Page("short", 1), _Page("full text " * 50, 2)]
    fitz_docs = [_Page("recovered page zero " * 20, 0), _Page("recovered page one " * 20, 1)]

    merged = pdf._merge_fitz_fallback(pypdf_docs, fitz_docs)

    assert "recovered page zero" in merged[0].page_content
    assert "recovered page one" in merged[1].page_content
    assert merged[2] is pypdf_docs[2]


def test_build_pdf_manifest_marks_empty_short_and_garbled_pages():
    docs = [_Doc("A" * 100), _Doc("   "), _Doc("bad\ufffd\ufffd" * 20), _Doc("short")]
    for page, doc in enumerate(docs):
        doc.metadata["page"] = page

    manifest = pdf.build_pdf_manifest("x.pdf", "x.pdf", docs, total_pages=5)

    assert manifest["total_pages"] == 5
    assert manifest["text_pages"] == [1, 3, 4]
    assert manifest["empty_pages"] == [2, 5]
    assert manifest["garbled_pages"] == [3]
    assert manifest["short_pages"] == [4]


def test_reconstruct_manifest_from_chunks():
    chunks = [_Doc("a"), _Doc("b")]
    chunks[0].metadata = {"page": 0, "source_paper": "paper"}
    chunks[1].metadata = {"page": 2, "source_paper": "paper"}

    manifest = pdf.reconstruct_manifest_from_chunks("paper", chunks)

    assert manifest["reconstructed"] is True
    assert manifest["text_pages"] == [1, 3]
    assert manifest["chunks"] == 2


def test_process_rejects_pdf_without_text(fake_state, monkeypatch):
    fake_state["active_topic"] = "topic"
    fake_state["topics"] = {"topic": {"files": [], "chunks": [], "db": None}}
    errors = []
    fake_st = types.SimpleNamespace(
        session_state=fake_state,
        error=lambda msg: errors.append(msg),
        warning=lambda msg: None,
        success=lambda msg: None,
    )
    monkeypatch.setattr(pdf, "st", fake_st)
    monkeypatch.setattr(pdf, "load_pdf_documents", lambda *a, **k: [])
    monkeypatch.setattr(pdf, "split_documents_with_stats", lambda docs: ([], 0))
    monkeypatch.setattr(
        pdf,
        "build_pdf_manifest",
        lambda *a, **k: {
            "total_pages": 0,
            "text_pages": [],
            "empty_pages": [],
            "short_pages": [],
            "garbled_pages": [],
            "chunks": 0,
            "dropped_chunks": 0,
            "chars": 0,
        },
    )

    assert (
        pdf.process_and_add_to_topic("empty.pdf", "empty.pdf", "key", topic_name="topic") is False
    )
    assert len(errors) == 1
    assert "未能提取到任何文字" in errors[0]
    assert fake_state["topics"]["topic"]["files"] == []
