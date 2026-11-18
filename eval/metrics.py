"""检索评测指标：页级召回、MRR、证据命中。

约定：文档 metadata 里的 page 是 0 基（PDF 第 1 页为 0），
评测数据集里的页码是给人看的 1 基页码。
"""


def doc_key(doc):
    md = getattr(doc, "metadata", None) or {}
    try:
        page = int(md.get("page") or 0)
    except (TypeError, ValueError):
        page = 0
    return str(md.get("source_paper") or ""), page + 1


def expected_keys(evidence):
    keys = set()
    for item in evidence or []:
        paper = str(item.get("paper") or "")
        for page in item.get("pages") or []:
            try:
                keys.add((paper, int(page)))
            except (TypeError, ValueError):
                continue
    return keys


def page_recall(docs, expected, k=None):
    """命中页数 / 期望页数；没有期望页时返回 None。"""
    if not expected:
        return None
    subset = docs[:k] if k else docs
    got = {doc_key(d) for d in subset}
    return len(got & expected) / len(expected)


def mrr(docs, expected):
    """首个命中片段的名次倒数；未命中为 0。"""
    for rank, doc in enumerate(docs, 1):
        if doc_key(doc) in expected:
            return 1.0 / rank
    return 0.0


def citation_hit(docs, expected):
    """证据池里是否至少覆盖一个期望页。"""
    return bool({doc_key(d) for d in docs} & expected)


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0
