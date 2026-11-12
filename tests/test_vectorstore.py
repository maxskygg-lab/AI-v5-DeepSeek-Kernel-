from services import vectorstore


def test_namespace_stable_and_topic_scoped(fake_state):
    ns_a1 = vectorstore.make_namespace("topic-a")
    ns_a2 = vectorstore.make_namespace("topic-a")
    ns_b = vectorstore.make_namespace("topic-b")

    assert ns_a1 == ns_a2
    assert ns_a1 != ns_b
    assert len(ns_a1) == 40
    assert fake_state["wm_workspace_id"] == "default"


class _FakeIndex:
    def __init__(self):
        self.calls = []

    def fetch(self, ids, namespace):
        self.calls.append(list(ids))

        class _Resp:
            vectors = {vid: {"id": vid} for vid in ids}

        return _Resp()


def test_fetch_vectors_batches_all_ids_without_loss():
    ids = [f"{i:04d}-" + "x" * 20 for i in range(2500)]
    idx = _FakeIndex()

    got = vectorstore._fetch_vectors(idx, ids, "ns", max_url_chars=3000)

    assert len(got) == 2500
    assert set(got) == set(ids)
    assert len(idx.calls) > 1
    assert sum(len(call) for call in idx.calls) == 2500
    for call in idx.calls:
        assert sum(len(vid) + 5 for vid in call) <= 3000
    all_ids = [vid for call in idx.calls for vid in call]
    assert len(all_ids) == len(set(all_ids)) == 2500
