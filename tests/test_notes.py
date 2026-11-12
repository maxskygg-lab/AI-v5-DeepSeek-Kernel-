from services import notes


def test_sync_notes_keeps_local_only_and_prefers_cloud(fake_state, monkeypatch):
    fake_state["notes"] = [
        {"id": "1", "content": "本地旧内容", "tags": [], "topic": "t", "ts": "2026-01-01 10:00"},
        {"id": "2", "content": "仅本地", "tags": [], "topic": "t", "ts": "2026-01-02 10:00"},
    ]
    cloud = [
        {"id": "1", "content": "云端新内容", "tags": ["x"], "topic": "t", "ts": "2026-01-03 10:00"},
        {"id": "3", "content": "仅云端", "tags": [], "topic": "t", "ts": "2026-01-04 10:00"},
    ]
    monkeypatch.setattr(notes, "load_notes_from_cloud", lambda: cloud)

    assert notes.sync_notes_from_cloud() == 2

    by_id = {n["id"]: n for n in fake_state["notes"]}
    assert set(by_id) == {"1", "2", "3"}
    assert by_id["1"]["content"] == "云端新内容"
    assert [n["ts"] for n in fake_state["notes"]] == [
        "2026-01-02 10:00",
        "2026-01-03 10:00",
        "2026-01-04 10:00",
    ]


class _Embeddings:
    def embed_query(self, text):
        return [0.1, 0.2]


def _note():
    return {
        "id": "n1",
        "content": "内容",
        "question": "问题",
        "tags": ["t"],
        "topic": "x",
        "ts": "2026-01-01 10:00",
    }


def test_save_note_enqueues_when_cloud_unavailable(fake_state, monkeypatch):
    monkeypatch.setattr(notes, "get_embeddings_model", lambda: _Embeddings())
    monkeypatch.setattr(notes, "_retry_cloud", lambda fn, **kwargs: fn())

    def _offline():
        raise RuntimeError("offline")

    monkeypatch.setattr(notes, "_pinecone_index", _offline)

    note = _note()
    assert notes.save_note_to_cloud(note) is False
    assert fake_state["notes_unsynced"] == [note]
    assert "offline" in fake_state["notes_last_error"]


def test_save_note_clears_pending_on_success(fake_state, monkeypatch):
    monkeypatch.setattr(notes, "get_embeddings_model", lambda: _Embeddings())
    monkeypatch.setattr(notes, "_retry_cloud", lambda fn, **kwargs: fn())

    class _Index:
        def upsert(self, vectors, namespace):
            return None

    monkeypatch.setattr(notes, "_pinecone_index", lambda: _Index())

    note = _note()
    fake_state["notes_unsynced"] = [note]
    assert notes.save_note_to_cloud(note) is True
    assert fake_state["notes_unsynced"] == []
    assert fake_state["notes_last_error"] is None
