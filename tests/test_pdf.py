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
    monkeypatch.setattr(pdf, "split_documents", lambda docs: [])

    assert pdf.process_and_add_to_topic("empty.pdf", "empty.pdf", "key", topic_name="topic") is False
    assert len(errors) == 1
    assert "未能提取到任何文字" in errors[0]
    assert fake_state["topics"]["topic"]["files"] == []
