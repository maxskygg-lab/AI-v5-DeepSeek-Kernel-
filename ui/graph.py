"""文献关联图谱渲染。"""

import math

from streamlit_agraph import Config, Edge, Node, agraph


def render_connected_graph(data, min_cite_filter=0):
    if not data:
        return None, {}
    nodes, edges, details = [], [], {}
    cur_year = 2026

    def color(year, rel):
        if not year or year == "Unknown":
            return "#94a3b8"
        age = max(0, cur_year - int(year))
        if rel == "seed":
            return "#FF4B4B"
        if rel == "cite":
            return "#059669" if age < 2 else "#10b981" if age < 5 else "#6ee7b7"
        return "#2563eb" if age < 2 else "#3b82f6" if age < 5 else "#93c5fd"

    seed = data.get("paperId", "root")
    details[seed] = {
        "title": data.get("title", "Seed Paper"),
        "abstract": data.get("abstract") or "无摘要",
        "year": data.get("year", "Unknown"),
        "cites": data.get("citationCount", 0),
        "url": f"https://www.semanticscholar.org/paper/{seed}",
        "arxiv_id": None,
    }
    nodes.append(Node(id=seed, label="THIS PAPER", size=35, color=color(data.get("year"), "seed")))
    seen = {seed}

    combined = []
    for p in data.get("references", [])[:20]:
        p["rel_type"] = "ref"
        combined.append(p)
    for p in data.get("citations", [])[:20]:
        p["rel_type"] = "cite"
        combined.append(p)

    for item in combined:
        pid = item.get("paperId")
        cites = item.get("citationCount", 0) or 0
        if not pid or pid in seen or cites < min_cite_filter:
            continue
        seen.add(pid)
        title = item.get("title", "Unknown")
        year = item.get("year")
        ext = item.get("externalIds") or {}
        arxiv_id = ext.get("ArXiv")
        details[pid] = {
            "title": title,
            "abstract": item.get("abstract") or "暂无摘要",
            "year": year,
            "cites": cites,
            "url": f"https://www.semanticscholar.org/paper/{pid}",
            "arxiv_id": arxiv_id,
        }
        sz = 15 + math.log(cites + 1) * 3.5
        nodes.append(
            Node(id=pid, label=f"{title[:20]}…", size=sz, color=color(year, item["rel_type"]))
        )
        if item["rel_type"] == "cite":
            edges.append(Edge(source=pid, target=seed, color="#d1d5db", width=1, dashed=True))
        else:
            edges.append(Edge(source=seed, target=pid, color="#94a3b8", width=1.5))

    cfg = Config(
        width="100%",
        height=560,
        directed=True,
        physics=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7D154",
        d3={"alphaTarget": 0.05, "gravity": -250, "linkLength": 150, "linkStrength": 0.1},
    )
    clicked = agraph(nodes=nodes, edges=edges, config=cfg)
    return clicked, details
