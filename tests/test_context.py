import types

from services import context


class _Chunk:
    def __init__(self, text, page=0, chunk_index=None, paper="paper"):
        self.page_content = text
        self.metadata = {"page": page, "source_paper": paper}
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


def test_extract_cited_pages_prefers_key_evidence():
    digest = (
        "## 主要结果\n核心指标见 P30 和 P65。\n"
        "## 关键证据（3-5 条）\n"
        "- [P12] 结果一\n"
        "- [P43] 结果二\n"
    )

    pages = context.extract_cited_pages(digest, limit=4)

    assert pages[:2] == [12, 43]
    assert 30 in pages


def test_collect_cited_chunks_returns_cited_pages():
    chunks = [
        _Chunk("page one", page=0, chunk_index=0),
        _Chunk("page two a", page=1, chunk_index=0),
        _Chunk("page two b", page=1, chunk_index=1),
        _Chunk("page three", page=2, chunk_index=0),
    ]
    digest = "## 关键证据\n[P2] 证据一\n[P3] 证据二"

    cited = context.collect_cited_chunks("paper", chunks, digest, max_chunks=2)

    assert [c.metadata["page"] for c in cited] == [1, 2]
    assert len(cited) == 2


def test_collect_cited_chunks_prefers_lexical_match():
    chunks = [
        _Chunk("无关内容，只讨论背景", page=0, chunk_index=0),
        _Chunk("GPT-4 在 GSM-8K 上达到 87.1%", page=0, chunk_index=1),
    ]
    digest = "## 关键证据\n[P1] GSM-8K 达到 87.1%"

    cited = context.collect_cited_chunks("paper", chunks, digest, max_chunks=1)

    assert "GSM-8K" in cited[0].page_content


def test_cited_page_coverage_reports_missing_pages():
    chunks = [_Chunk("page one", page=0), _Chunk("page two", page=1)]
    digest = "## 关键证据\n[P1] 证据一\n[P9] 证据二"

    total, matched, missing = context.cited_page_coverage(chunks, digest)

    assert total == 2
    assert matched == 1
    assert missing == [9]


def test_is_completeness_question():
    assert context.is_completeness_question("资料完整吗")
    assert context.is_completeness_question("这些论文的资料齐全吗")
    assert not context.is_completeness_question("GPT-4 在 GSM-8K 上的成绩是多少")


def test_expand_to_pages_merges_same_page_and_dedupes():
    chunks = [
        _Chunk("page one part a", page=0, chunk_index=0),
        _Chunk("page one part b", page=0, chunk_index=1),
        _Chunk("page two", page=1, chunk_index=0),
    ]
    docs = [
        _Chunk("hit page one", page=0),
        _Chunk("hit page two", page=1),
        _Chunk("hit page one again", page=0),
    ]

    pages = context.expand_to_pages(docs, chunks, max_pages=5)

    assert len(pages) == 2
    assert pages[0].metadata["page"] == 0
    assert pages[0].metadata["parent"] is True
    assert "page one part a" in pages[0].page_content
    assert "page one part b" in pages[0].page_content
    assert pages[1].page_content == "page two"


def test_expand_to_pages_falls_back_to_fragment():
    pages = context.expand_to_pages([_Chunk("only fragment", page=3)], [], max_pages=1)

    assert len(pages) == 1
    assert pages[0].page_content == "only fragment"


def test_format_manifest_report_includes_empty_pages_and_citations():
    chunks = [_Chunk("text", page=0, paper="p")]
    manifests = {
        "t::p": {
            "total_pages": 2,
            "text_pages": [1],
            "empty_pages": [2],
            "short_pages": [],
            "garbled_pages": [],
            "chunks": 1,
        }
    }
    digests = {"p": {"digest": "## 关键证据\n[P1] 结果\n- 数据集：论文未报告"}}

    text = context.format_manifest_report(["p"], chunks, digests, manifests, "t")

    assert "总页数 2" in text
    assert "空页：P2" in text
    assert "摘要引用页：1/1" in text
    assert "论文未报告" in text


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
