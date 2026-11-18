"""检索评测入口：读取 JSONL 题集，跑真实混合检索并输出页级指标。

题集每行一个 JSON：
    {"id": "q1", "topic": "default_topic", "paper": "可选，限定单篇论文",
     "question": "问题原文",
     "expected": [{"paper": "论文文件名.pdf", "pages": [3, 4]}]}

用法（在仓库根目录）：
    python eval/run_eval.py --dataset eval/questions.jsonl --topic default_topic

运行前需要配置 PINECONE_API_KEY（环境变量或 .streamlit/secrets.toml），
并且本地工作区标识要与线上一致，否则 namespace 对不上。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from eval.metrics import average, citation_hit, expected_keys, mrr, page_recall  # noqa: E402
from services.context import expand_to_pages  # noqa: E402
from services.retrieval import hybrid_retrieve  # noqa: E402
from services.vectorstore import (  # noqa: E402
    make_namespace,
    restore_topic_from_pinecone,
)


def load_dataset(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"题集第 {line_no} 行不是合法 JSON：{e}")
    return items


def prepare_topic(topic):
    if "topics" not in st.session_state:
        st.session_state["topics"] = {}
    st.session_state["topics"][topic] = {"files": [], "chunks": [], "db": None}
    st.session_state["active_topic"] = topic
    restore_topic_from_pinecone(topic)
    return st.session_state["topics"][topic]


def retrieve_for(item, topic_data, k):
    paper = str(item.get("paper") or "").strip()
    if paper:
        target_chunks = [
            c for c in topic_data["chunks"] if c.metadata.get("source_paper") == paper
        ]
        search_kwargs = {"k": 20, "filter": {"source_paper": paper}}
    else:
        target_chunks = topic_data["chunks"]
        search_kwargs = {"k": 15, "fetch_k": 50, "lambda_mult": 0.6}

    docs = hybrid_retrieve(
        item["question"],
        target_chunks,
        topic_data["db"],
        search_kwargs,
        cache_key=f"eval:{item.get('topic')}:{paper}",
        cross_paper=not paper,
    )
    pages = expand_to_pages(
        docs,
        target_chunks,
        max_pages=8,
        max_chars_per_page=4000,
        max_total_chars=30000,
    )
    return docs[:k], pages


def evaluate(dataset, topic, k, out_path=None):
    topic_data = prepare_topic(topic)
    print(f"topic={topic} namespace={make_namespace(topic)}")
    print(f"papers={len(topic_data['files'])} chunks={len(topic_data['chunks'])}")
    if not topic_data["chunks"]:
        raise SystemExit("该主题没有可检索内容：请确认主题名与工作区标识是否与线上一致。")

    rows = []
    for item in dataset:
        expected = expected_keys(item.get("expected"))
        if not expected:
            print(f"跳过 {item.get('id')}：缺少 expected 证据页")
            continue
        docs, pages = retrieve_for(item, topic_data, k)
        retrieved_pages = {
            (d.metadata.get("source_paper"), d.metadata.get("page")) for d in pages
        }
        row = {
            "id": item.get("id") or "",
            "paper": item.get("paper") or "",
            "recall_at_k": page_recall(docs, expected),
            "mrr": mrr(docs, expected),
            "page_recall": page_recall(pages, expected),
            "citation_hit": citation_hit(pages, expected),
            "retrieved_pages": len(retrieved_pages),
        }
        rows.append(row)
        print(
            f"{row['id']:<16} recall@{k}={row['recall_at_k']:.2f} mrr={row['mrr']:.2f} "
            f"page_recall={row['page_recall']:.2f} citation_hit={row['citation_hit']}"
        )

    if not rows:
        raise SystemExit("题集里没有可评测的题目。")

    summary = {
        "questions": len(rows),
        f"recall@{k}": average([r["recall_at_k"] for r in rows]),
        "mrr": average([r["mrr"] for r in rows]),
        "page_recall": average([r["page_recall"] for r in rows]),
        "citation_hit": average([1.0 if r["citation_hit"] else 0.0 for r in rows]),
    }
    print("\n== 汇总 ==")
    for name, value in summary.items():
        print(f"{name}: {value:.3f}" if isinstance(value, float) else f"{name}: {value}")

    if out_path:
        payload = {"topic": topic, "k": k, "summary": summary, "rows": rows}
        Path(out_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n明细已写入 {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="混合检索的页级评测")
    parser.add_argument("--dataset", default="eval/questions.jsonl")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    evaluate(dataset, args.topic, args.k, args.out or None)


if __name__ == "__main__":
    main()
