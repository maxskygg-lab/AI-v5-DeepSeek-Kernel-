import types

from services import context


class _Chunk:
    def __init__(self, text, page=0, chunk_index=None):
        self.page_content = text
        self.metadata = {"page": page}
        if chunk_index is not None:
            self.metadata["chunk_index"] = chunk_index


def test_strip_overlap_removes_duplicate_boundary():
    previous = "A" * 200
    current = "A" * 120 + "B" * 80

    assert context.strip_overlap(previous, current) == "B" * 80


def test_build_digest_prompt_contains_meta_and_pages():
    chunks = [
        _Chunk("first part " * 10, page=0, chunk_index=0),
        _Chunk("second part " * 10, page=1, chunk_index=1),
    ]
    meta = {"title": "Test Paper", "authors": "Alice, Bob", "year": "2024", "venue": "ICML"}

    prompt = context.build_digest_prompt("test.pdf", chunks, meta)

    assert "Test Paper" in prompt
    assert "Alice, Bob" in prompt
    assert "[P1 | 片段1]" in prompt
    assert "[P2 | 片段2]" in prompt
    assert "资料未覆盖" in prompt


def test_format_digests_uses_meta_title():
    text = context.format_digests(
        {"paper.pdf": {"meta": {"title": "Readable Title"}, "digest": "Digest body"}}
    )

    assert "Readable Title" in text
    assert "Digest body" in text


def test_get_paper_digest_caches_result(fake_state, monkeypatch):
    class _FakeLLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, prompt):
            self.calls += 1
            return types.SimpleNamespace(content="digest")

    fake_llm = _FakeLLM()
    monkeypatch.setattr(context, "get_deepseek_llm", lambda *a, **k: fake_llm)

    chunk = _Chunk("some paper text", page=0)
    first = context.get_paper_digest("paper.pdf", [chunk], "key", with_meta=False)
    second = context.get_paper_digest("paper.pdf", [chunk], "key", with_meta=False)

    assert first["digest"] == "digest"
    assert second["digest"] == "digest"
    assert fake_llm.calls == 1
