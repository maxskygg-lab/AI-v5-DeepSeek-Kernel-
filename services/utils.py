"""通用工具：arXiv ID 归一化、检索结果导出。"""

import io
import re

import pandas as pd
import streamlit as st


def get_pure_arxiv_id(url_or_id):
    m = re.search(r"(\d{4}\.\d{4,5})", url_or_id)
    return m.group(1) if m else url_or_id.split("/")[-1].split("v")[0]


def convert_to_excel(results):
    data = []
    for item in results:
        res = item["obj"]
        contrib = st.session_state.contributions_cache.get(res.title[:60], "未生成")
        score = st.session_state.score_cache.get(res.title[:60], "未打分")
        data.append(
            {
                "标题": res.title,
                "作者": ", ".join([a.name for a in res.authors]),
                "年份": res.published.year,
                "引用数": item.get("citations", 0),
                "核心贡献 (AI)": contrib,
                "综合评分 (AI)": score,
                "真实语义相似度": round(item.get("sim_score", 0), 2),
                "链接": res.entry_id,
                "摘要": res.summary.replace("\n", " "),
            }
        )

    df = pd.DataFrame(data)
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="检索结果")
        workbook = writer.book
        worksheet = writer.sheets["检索结果"]

        header_fmt = workbook.add_format(
            {
                "bold": True,
                "bg_color": "#D7E4BC",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        cell_fmt = workbook.add_format({"border": 1, "valign": "top", "text_wrap": True})
        num_fmt = workbook.add_format({"border": 1, "align": "center", "valign": "vcenter"})

        worksheet.set_column("A:A", 40, cell_fmt)
        worksheet.set_column("B:B", 20, cell_fmt)
        worksheet.set_column("C:D", 10, num_fmt)
        worksheet.set_column("E:E", 50, cell_fmt)
        worksheet.set_column("F:F", 30, cell_fmt)
        worksheet.set_column("G:G", 15, num_fmt)
        worksheet.set_column("H:H", 30, cell_fmt)
        worksheet.set_column("I:I", 60, cell_fmt)

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)

    return output.getvalue()
