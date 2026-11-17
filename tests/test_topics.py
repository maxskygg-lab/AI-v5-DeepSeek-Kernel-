import json
import types

from services import topics


class _Index:
    def __init__(self, metadata=None):
        self.metadata = metadata
        self.upserts = []

    def fetch(self, ids, namespace):
        if self.metadata is None:
            return types.SimpleNamespace(vectors={})
        return types.SimpleNamespace(
            vectors={ids[0]: types.SimpleNamespace(metadata=self.metadata)}
        )

    def upsert(self, vectors, namespace):
        self.upserts.append((vectors, namespace))


def test_load_topics_from_cloud_restores_names_and_active_topic(fake_state, monkeypatch):
    fake_state["topics"] = {"default_topic": {"files": [], "chunks": [], "db": None}}
    fake_state["active_topic"] = "default_topic"
    idx = _Index(
        {
            "topics": json.dumps(["default_topic", "my_topic"], ensure_ascii=False),
            "active_topic": "my_topic",
        }
    )
    monkeypatch.setattr(topics, "PINECONE_API_KEY", "key")
    monkeypatch.setattr(topics, "_pinecone_index", lambda: idx)
    monkeypatch.setattr(topics, "get_workspace_id", lambda: "ws")

    added = topics.load_topics_from_cloud()

    assert added == 1
    assert "my_topic" in fake_state["topics"]
    assert fake_state["active_topic"] == "my_topic"


def test_save_topics_to_cloud_writes_registry(fake_state, monkeypatch):
    fake_state["topics"] = {
        "default_topic": {"files": [], "chunks": [], "db": None},
        "my_topic": {"files": [], "chunks": [], "db": None},
    }
    fake_state["active_topic"] = "my_topic"
    idx = _Index()
    monkeypatch.setattr(topics, "PINECONE_API_KEY", "key")
    monkeypatch.setattr(topics, "_pinecone_index", lambda: idx)
    monkeypatch.setattr(topics, "get_workspace_id", lambda: "ws")
    monkeypatch.setattr(
        topics,
        "get_embeddings_model",
        lambda: types.SimpleNamespace(embed_query=lambda text: [0.1, 0.2, 0.3]),
    )

    assert topics.save_topics_to_cloud() is True
    vectors, _namespace = idx.upserts[0]
    md = vectors[0]["metadata"]
    assert json.loads(md["topics"]) == ["default_topic", "my_topic"]
    assert md["active_topic"] == "my_topic"


def test_save_topics_to_cloud_handles_failure(fake_state, monkeypatch):
    fake_state["topics"] = {"default_topic": {"files": [], "chunks": [], "db": None}}
    fake_state["active_topic"] = "default_topic"
    monkeypatch.setattr(topics, "PINECONE_API_KEY", "key")
    monkeypatch.setattr(topics, "get_workspace_id", lambda: "ws")
    monkeypatch.setattr(
        topics,
        "get_embeddings_model",
        lambda: types.SimpleNamespace(embed_query=lambda text: [0.1]),
    )

    def _boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(topics, "_pinecone_index", _boom)

    assert topics.save_topics_to_cloud() is False
