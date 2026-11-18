import json
import types

from services import manifests


class _Index:
    def __init__(self):
        self.upserts = []

    def upsert(self, vectors, namespace):
        self.upserts.append((vectors, namespace))


def test_save_manifest_to_cloud_writes_page_fields(fake_state, monkeypatch):
    idx = _Index()
    monkeypatch.setattr(manifests, "get_workspace_id", lambda: "ws")
    monkeypatch.setattr(manifests, "_pinecone_index", lambda: idx)
    monkeypatch.setattr(
        manifests,
        "get_embeddings_model",
        lambda: types.SimpleNamespace(embed_query=lambda text: [0.1, 0.2]),
    )

    manifest = {
        "file": "paper.pdf",
        "total_pages": 6,
        "text_pages": [1, 2, 4],
        "empty_pages": [3],
        "short_pages": [2],
        "garbled_pages": [],
        "chunks": 12,
        "dropped_chunks": 1,
        "chars": 9000,
    }
    assert manifests.save_manifest_to_cloud("topic", "paper.pdf", manifest) is True

    vectors, namespace = idx.upserts[0]
    md = vectors[0]["metadata"]
    assert namespace.startswith("manifests-")
    assert md["type"] == "paper_manifest"
    assert md["topic"] == "topic"
    assert md["file"] == "paper.pdf"
    assert md["total_pages"] == 6
    assert json.loads(md["empty_pages"]) == [3]
    assert json.loads(md["text_pages"]) == [1, 2, 4]


def test_load_manifests_from_cloud_restores_session_state(fake_state, monkeypatch):
    fake_state["paper_manifests"] = {}
    monkeypatch.setattr(manifests, "get_workspace_id", lambda: "ws")
    monkeypatch.setattr(manifests, "_pinecone_index", lambda: _Index())
    monkeypatch.setattr(manifests, "_list_namespace_ids", lambda idx, ns: ["v1", "v2"])
    monkeypatch.setattr(
        manifests,
        "_fetch_vectors",
        lambda idx, ids, ns: {
            "v1": types.SimpleNamespace(
                metadata={
                    "type": "paper_manifest",
                    "topic": "topic",
                    "file": "paper.pdf",
                    "total_pages": "10",
                    "empty_pages": "[3, 4]",
                    "text_pages": "[1, 2]",
                    "short_pages": "[]",
                    "garbled_pages": "[]",
                    "chunks": "5",
                    "dropped_chunks": "0",
                    "chars": "800",
                }
            ),
            "v2": types.SimpleNamespace(metadata={"type": "other"}),
        },
    )

    loaded = manifests.load_manifests_from_cloud()

    assert loaded == 1
    manifest = fake_state["paper_manifests"]["topic::paper.pdf"]
    assert manifest["total_pages"] == 10
    assert manifest["empty_pages"] == [3, 4]
    assert manifest["chunks"] == 5


def test_save_manifest_to_cloud_handles_failure(fake_state, monkeypatch):
    monkeypatch.setattr(manifests, "get_workspace_id", lambda: "ws")
    monkeypatch.setattr(
        manifests,
        "get_embeddings_model",
        lambda: types.SimpleNamespace(embed_query=lambda text: [0.1]),
    )

    def _boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(manifests, "_pinecone_index", _boom)

    assert manifests.save_manifest_to_cloud("topic", "paper.pdf", {}) is False


def test_load_manifests_from_cloud_handles_failure(fake_state, monkeypatch):
    fake_state["paper_manifests"] = {}
    monkeypatch.setattr(manifests, "get_workspace_id", lambda: "ws")

    def _boom():
        raise RuntimeError("offline")

    monkeypatch.setattr(manifests, "_pinecone_index", _boom)

    assert manifests.load_manifests_from_cloud() == 0
    assert fake_state["paper_manifests"] == {}
