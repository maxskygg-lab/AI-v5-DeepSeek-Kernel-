"""Tab 1：学术检索与文献图谱。"""

import itertools
import math
import os
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import arxiv
import numpy as np
import requests
import streamlit as st

from services.embeddings import get_embeddings_model
from services.llm import get_one_line_contribution, get_paper_score
from services.log import get_logger
from services.pdf import download_arxiv_pdf_direct, process_and_add_to_topic
from services.semantic_scholar import (
    fetch_graph_data,
    preload_top_graphs,
    smart_fetch_citations,
)
from services.utils import convert_to_excel, get_pure_arxiv_id
from ui.graph import render_connected_graph

logger = get_logger("ui.search")


def render_search_tab(user_api_key, ss_api_key):
    st.markdown('<div class="section-divider">🌍 学术检索</div>', unsafe_allow_html=True)

    sq1, sq2 = st.columns([4, 2])
    with sq1:
        search_query = st.text_input(
            "关键词",
            value=st.session_state.suggested_query,
            placeholder="输入关键词，例如: education robot",
            label_visibility="collapsed",
        )
    with sq2:
        sort_mode = st.selectbox(
            "排序",
            ["🔥 相关性", "🌟 综合(相关+质量)", "💎 质量优先", "📅 最新", "📈 引用量"],
            label_visibility="collapsed",
        )

    rel_w = 1.0
    qual_w = 0.0
    if sort_mode in ["🌟 综合(相关+质量)", "💎 质量优先"]:
        # 根据选择的模式，给予不同的初始滑块位置（综合默认60%，质量优先默认20%）
        default_rel = 60 if "综合" in sort_mode else 20
        rel_weight_pct = st.slider(
            "⚖️ 自定义权重配比 (任意调节，左拉看重质量，右拉看重相关性)",
            min_value=0,
            max_value=100,
            value=default_rel,
            step=1,
            format="相关性 %d%%",
        )
        rel_w = rel_weight_pct / 100.0
        qual_w = 1.0 - rel_w
        st.caption(
            f"💡 当前计算规则：总分 = (真实语义分 × **{rel_w:.2f}**) + (AI质量分 × **{qual_w:.2f}**)"
        )

    with st.expander("⚙️ 高级筛选 (学科/期刊)"):
        adv1, adv2 = st.columns(2)
        with adv1:
            category_options = {
                "所有学科": "",
                "计算机科学 (Computer Science)": "cs.*",
                "物理学 (Physics)": "(physics.* OR astro-ph.* OR cond-mat.* OR gr-qc.* OR hep-ex.* OR hep-lat.* OR hep-ph.* OR hep-th.* OR nucl-ex.* OR nucl-th.* OR quant-ph.*)",
                "数学 (Mathematics)": "math.*",
                "统计学 (Statistics)": "stat.*",
                "电气工程与系统科学 (EESS)": "eess.*",
                "定量生物学 (Quantitative Biology)": "q-bio.*",
                "定量金融 (Quantitative Finance)": "q-fin.*",
                "经济学 (Economics)": "econ.*",
            }
            selected_category = st.selectbox("学科分类过滤", list(category_options.keys()))
        with adv2:
            journal_query = st.text_input(
                "期刊/杂志/会议名称 (选填)", placeholder="例如: Nature, IEEE,ACM,CVPR,NIPS"
            )

    if st.button("🚀 检索", use_container_width=True) and search_query:
        with st.spinner("正在向 ArXiv 请求数据..."):
            try:
                asort = arxiv.SortCriterion.Relevance
                if "最新" in sort_mode:
                    asort = arxiv.SortCriterion.SubmittedDate

                refined = search_query
                if " " in search_query and "AND" not in search_query and '"' not in search_query:
                    # 取消了双引号的强制短语匹配，改用 all 字段的 AND 组合，只要论文里包含这些词就统统找出来
                    refined = " AND ".join([f"all:{w}" for w in search_query.split()])
                else:
                    refined = f"({refined})"

                if category_options[selected_category]:
                    refined += f" AND cat:{category_options[selected_category]}"

                if journal_query.strip():
                    val = journal_query.strip()
                    refined += f' AND (jr:"{val}" OR co:"{val}")'

                # --- 新增辅助提示：在界面上显示真实发送给接口的查询语句 ---
                st.caption(f"🔍 检索指令预览: `{refined}`")

                # --- 429 防崩溃重试机制 ---
                max_retries = 3
                raw = []
                for attempt in range(max_retries):
                    try:
                        raw_gen = arxiv.Client().results(
                            arxiv.Search(query=refined, max_results=2000, sort_by=asort)
                        )
                        st.session_state.search_generator = raw_gen
                        raw = list(itertools.islice(raw_gen, 100))
                        break
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            time.sleep(3)
                            continue
                        else:
                            raise e

                # --- 新增辅助提示：针对零结果给出清晰引导 ---
                if not raw:
                    st.warning(
                        "⚠️ 未找到匹配论文。建议：1. 缩减关键词 2. 清空‘期刊名称’筛选框 3. 检查学科分类是否选错。"
                    )

                st.session_state.search_results = [{"obj": r, "citations": None} for r in raw]

                if st.session_state.search_results:
                    with st.spinner("正在利用本地向量模型进行底层语义匹配打分..."):
                        try:
                            embeddings_model = get_embeddings_model()
                            # 1. 向量化用户搜索词
                            query_vec = np.array(embeddings_model.embed_query(search_query))
                            # 2. 批量提取摘要并向量化
                            abstracts = [
                                item["obj"].summary for item in st.session_state.search_results
                            ]
                            doc_vecs = np.array(embeddings_model.embed_documents(abstracts))
                            # 3. 计算余弦相似度公式: (A·B) / (||A|| * ||B||)
                            norms_doc = np.linalg.norm(doc_vecs, axis=1)
                            norm_query = np.linalg.norm(query_vec)
                            similarities = np.dot(doc_vecs, query_vec) / (norms_doc * norm_query)
                            # 4. 将真实分数转为百分制并写回字典
                            for idx, item in enumerate(st.session_state.search_results):
                                item["sim_score"] = float(similarities[idx]) * 100
                        except Exception as e:
                            logger.warning("semantic_score_failed error=%s", e)
                            st.warning(f"本地语义计算异常，将降级使用基础分: {e}")
                            for item in st.session_state.search_results:
                                item["sim_score"] = 50.0  # 失败兜底分

                st.session_state.citations_loaded = False
                st.session_state.contributions_cache = {}
                st.session_state.focus_paper_id = None
            except Exception as e:
                logger.warning("arxiv_search_failed error=%s", e)
                st.error(f"检索失败: {e}")

        if st.session_state.search_results:
            with st.spinner("同步引用数..."):
                id2c = smart_fetch_citations(st.session_state.search_results, ss_key=ss_api_key)
                for item in st.session_state.search_results:
                    item["citations"] = id2c.get(item["obj"].entry_id, 0)

                if "引用量" in sort_mode:
                    st.session_state.search_results.sort(
                        key=lambda x: x["citations"] or 0, reverse=True
                    )
                elif "质量优先" in sort_mode or "综合" in sort_mode:
                    current_year = datetime.now().year
                    for idx, item in enumerate(st.session_state.search_results):
                        rel_score = item.get("sim_score", 50.0)

                        cites = item["citations"] or 0
                        cite_score = (math.log10(cites + 1) / 3.0) * 100

                        pub_year = item["obj"].published.year
                        age = max(0, current_year - pub_year)

                        # 针对不同模式计算时效补偿
                        if "质量优先" in sort_mode:
                            time_bonus = 0
                            if age == 0:
                                time_bonus = 40
                            elif age == 1:
                                time_bonus = 20
                            elif age == 2:
                                time_bonus = 10
                        else:
                            time_bonus = max(0, 30 - age * 10)

                        quality_score = min(100.0, cite_score + time_bonus)

                        # 融合真实语义分与质量分
                        item["total_score"] = (rel_score * rel_w) + (quality_score * qual_w)

                    st.session_state.search_results.sort(
                        key=lambda x: x.get("total_score", 0), reverse=True
                    )

                st.session_state.citations_loaded = True

            st.success(f"✅ 完成，找到 {len(st.session_state.search_results)} 篇")
            preload_top_graphs(st.session_state.search_results, ss_key=ss_api_key, top_n=3)

    # ── 图谱区：聚焦论文在列表中时就地渲染（贴在论文卡片下方），否则回退到列表上方 ──
    def render_graph_section():
        st.markdown('<div class="section-divider">📊 文献关联图谱</div>', unsafe_allow_html=True)
        min_cf = st.slider("最低引用数过滤", 0, 200, 5, step=1, key="graph_cite_filter")
        graph_err = None
        graph_status = None
        with st.spinner("加载图谱…"):
            try:
                g_data = fetch_graph_data(st.session_state.focus_paper_id, ss_key=ss_api_key)
            except requests.RequestException as e:
                g_data = None
                graph_err = str(e)
                graph_status = getattr(getattr(e, "response", None), "status_code", None)

        if not g_data:
            st.warning(f"⚠️ 暂时无法获取图谱：{graph_err or '无数据'}")
            if graph_status == 403:
                st.info("403 表示 SS_API_KEY 已失效，请重新申请后填入 Secrets。")
        else:
            gc_graph, gc_info = st.columns([1.6, 1])
            with gc_graph:
                clicked_id, all_details = render_connected_graph(g_data, min_cite_filter=min_cf)
                sid = g_data.get("paperId", "root")
                if sid in all_details:
                    all_details[sid]["arxiv_id"] = get_pure_arxiv_id(
                        st.session_state.focus_paper_id
                    )
                if not (clicked_id and clicked_id in all_details):
                    st.caption("👆 点击节点 → 右侧看完整详情 | 🔴 当前  🟢 引用本文  🔵 本文引用")

            with gc_info:
                if clicked_id and clicked_id in all_details:
                    info = all_details[clicked_id]
                    st.markdown(
                        f"""
                        <div style="background:#f8fafc;border:1px solid #e2e8f0;
                                    border-left:4px solid #6366f1;border-radius:10px;padding:14px 16px;">
                            <div style="font-size:.93em;font-weight:700;color:#1e293b;
                                        margin-bottom:10px;line-height:1.4;">
                                📑 {info['title']}
                            </div>
                            <div style="display:flex;gap:12px;margin-bottom:12px;flex-wrap:wrap;">
                                <span style="background:#e0e7ff;color:#3730a3;padding:2px 10px;
                                             border-radius:12px;font-size:.8em;font-weight:600;">
                                    📅 {info['year'] or '年份未知'}
                                </span>
                                <span style="background:#fee2e2;color:#991b1b;padding:2px 10px;
                                             border-radius:12px;font-size:.8em;font-weight:600;">
                                    🔥 {info['cites']} 引用
                                </span>
                            </div>
                            <div style="font-size:.82em;color:#475569;
                                        max-height:320px;overflow-y:auto;line-height:1.65;">
                                {info['abstract']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("")
                    target_topic = st.selectbox(
                        "加入主题",
                        list(st.session_state.topics.keys()),
                        index=list(st.session_state.topics.keys()).index(
                            st.session_state.active_topic
                        ),
                        key="graph_topic_sel",
                        label_visibility="collapsed",
                    )
                    arxiv_id = info.get("arxiv_id")
                    ga, gb, gc = st.columns(3)
                    with ga:
                        if arxiv_id and st.button(
                            "⬇️ 入库", type="primary", use_container_width=True, key="ginfo_dl"
                        ):
                            with st.spinner("下载中..."):
                                try:
                                    pdf_path = download_arxiv_pdf_direct(arxiv_id)
                                    if process_and_add_to_topic(
                                        pdf_path,
                                        info["title"],
                                        user_api_key,
                                        topic_name=target_topic,
                                    ):
                                        st.success("✅ 入库成功！")
                                        st.balloons()
                                except Exception as e:
                                    st.error(str(e))
                        elif not arxiv_id:
                            st.caption("暂无全文")
                    with gb:
                        st.link_button("🌐 SS", info["url"], use_container_width=True)
                    with gc:
                        if info.get("arxiv_id") and st.button(
                            "🕸️ 展开", use_container_width=True, key="ginfo_expand"
                        ):
                            st.session_state.focus_paper_id = info["arxiv_id"]
                            st.rerun()
                else:
                    st.markdown(
                        """<div style="background:#f1f5f9;border:1px dashed #cbd5e1;border-radius:10px;
                                      padding:40px 16px;text-align:center;color:#94a3b8;font-size:.88em;
                                      min-height:260px;display:flex;align-items:center;justify-content:center;">
                            ← 点击左侧节点<br>查看完整详情</div>""",
                        unsafe_allow_html=True,
                    )

    # 聚焦的论文不在当前检索列表里（例如点了图谱中的引用节点）时，图谱仍显示在列表上方
    _focus_id = st.session_state.focus_paper_id
    if _focus_id and not any(
        item["obj"].entry_id == _focus_id for item in st.session_state.search_results
    ):
        render_graph_section()

    # ── 检索结果列表 ──
    if st.session_state.search_results:
        st.markdown(
            f'<div class="section-divider">📋 检索结果（已加载 {len(st.session_state.search_results)} 篇）'
            f'<span class="perf-badge">⚡ 缓存 {len(st.session_state.citations_global_cache)} 篇</span></div>',
            unsafe_allow_html=True,
        )

        col_excel, col_score, col_pdf = st.columns(3)

        with col_excel:
            st.write("📊 **导出数据**")
            st.download_button(
                label="📥 下载 Excel 表格",
                data=convert_to_excel(st.session_state.search_results),
                file_name=f"ArXiv_Search_{datetime.now().strftime('%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col_score:
            score_num = st.number_input(
                "⭐ **打分数量 (篇)**",
                min_value=1,
                max_value=max(1, len(st.session_state.search_results)),
                value=min(20, len(st.session_state.search_results)),
                step=1,
                key="batch_score_n",
            )
            if st.button(f"🚀 开始打分 (前 {score_num} 篇)", use_container_width=True):
                with st.spinner(
                    f"🚀 正在后台并行请求 DeepSeek 给前 {score_num} 篇打分，请耐心等待..."
                ):
                    to_process = st.session_state.search_results[:score_num]
                    pending = [
                        p
                        for p in to_process
                        if p["obj"].title[:60] not in st.session_state.score_cache
                    ]
                    if pending:
                        with ThreadPoolExecutor(max_workers=5) as pool:
                            futures = {
                                pool.submit(
                                    get_paper_score,
                                    p["obj"].entry_id,
                                    p["obj"].title,
                                    p["obj"].summary,
                                    user_api_key,
                                    ss_api_key,
                                ): p["obj"].title[:60]
                                for p in pending
                            }
                            for future in as_completed(futures):
                                key = futures[future]
                                try:
                                    st.session_state.score_cache[key] = future.result()
                                except Exception as e:
                                    logger.debug("paper_score_task_failed title=%s error=%s", key, e)
                                    st.session_state.score_cache[key] = "（打分失败）"
                st.success(f"✅ 前 {score_num} 篇论文已打分完毕！")
                time.sleep(2)
                st.rerun()

        with col_pdf:
            st.write("📦 **下载原文范围 (篇)**")
            c_dl_1, c_dl_2 = st.columns(2)
            with c_dl_1:
                dl_start = st.number_input(
                    "从第",
                    min_value=1,
                    max_value=max(1, len(st.session_state.search_results)),
                    value=1,
                    step=1,
                    key="batch_dl_start",
                )
            with c_dl_2:
                dl_end = st.number_input(
                    "到第",
                    min_value=dl_start,
                    max_value=max(dl_start, len(st.session_state.search_results)),
                    value=max(dl_start, min(dl_start + 9, len(st.session_state.search_results))),
                    step=1,
                    key="batch_dl_end",
                )

            if st.button(f"🔄 打包 ZIP ({dl_start}-{dl_end} 篇)", use_container_width=True):
                # ==========================
                # ==========================
                import zipfile

                # 使用硬盘临时文件替代内存缓冲
                temp_zip_path = os.path.join(
                    tempfile.gettempdir(), f"arxiv_batch_{uuid.uuid4().hex[:8]}.zip"
                )
                to_dl = st.session_state.search_results[dl_start - 1 : dl_end]
                dl_total = len(to_dl)

                # 使用状态文本与进度条，保证网页实时与后台通讯
                status_text = st.empty()
                progress_bar = st.progress(0)

                with zipfile.ZipFile(temp_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for idx, item in enumerate(to_dl):
                        res = item["obj"]
                        status_text.text(
                            f"📥 正在提取并压缩 ({idx+1}/{dl_total}): {res.title[:25]}..."
                        )
                        try:
                            # 增加 1.5 秒安全休眠，防高并发触发 ArXiv 的防 DDoS 封锁
                            time.sleep(1.5)
                            p_path = download_arxiv_pdf_direct(res.entry_id)
                            safe_title = re.sub(r'[\\/*?:"<>|]', "", res.title)[:50]
                            # 文件名加上原始排名序号，方便对应
                            filename = f"{dl_start + idx}_{safe_title}.pdf"
                            zf.write(p_path, arcname=filename)
                            os.remove(p_path)
                        except Exception as e:
                            logger.debug("pdf_pack_skip title=%s error=%s", res.title[:60], e)
                        # 每次循环更新进度，维持前端存活
                        progress_bar.progress((idx + 1) / dl_total)

                status_text.text("✅ 打包完成！正在生成最终下载链接...")

                # 保留文件路径，而不是将几百兆数据塞进内存变量
                st.session_state.ready_zip_path = temp_zip_path
                st.session_state.ready_zip_name = (
                    f"ArXiv_PDFs_{dl_start}to{dl_end}_{datetime.now().strftime('%m%d_%H%M')}.zip"
                )
                # ==========================
                # ==========================

            # 如果缓存里有打包好的文件路径，打开文件流供按钮下载，零内存占用！
            if st.session_state.get("ready_zip_path") and os.path.exists(
                st.session_state.get("ready_zip_path")
            ):
                with open(st.session_state.ready_zip_path, "rb") as f:
                    st.download_button(
                        label="✅ 点击下载压缩包",
                        data=f,
                        file_name=st.session_state.ready_zip_name,
                        mime="application/zip",
                        use_container_width=True,
                        type="primary",
                    )

        st.markdown("---")

        for i, item in enumerate(st.session_state.search_results):
            res = item["obj"]
            cites = item["citations"]
            cite_html = (
                f"<span class='cite-badge'>{cites}</span>"
                if cites is not None
                else "<span class='cite-loading'>加载中…</span>"
            )
            is_focused = st.session_state.focus_paper_id == res.entry_id
            with st.expander(f"#{i+1} {res.title} ({res.published.year})", expanded=is_focused):
                sim_display = f" | 🎯 匹配度: {round(item.get('sim_score', 0), 1)}%"
                st.markdown(
                    f"**{', '.join([a.name for a in res.authors])}** | "
                    f"{res.published.strftime('%Y-%m-%d')} | 引用：{cite_html}{sim_display}",
                    unsafe_allow_html=True,
                )
                ck = res.title[:60]
                cc, cg, cs = st.columns([4, 1, 1])
                with cc:
                    box_content = ""
                    if ck in st.session_state.contributions_cache:
                        box_content += (
                            f"💡 <b>贡献:</b> {st.session_state.contributions_cache[ck]}<br>"
                        )
                    else:
                        box_content += (
                            "💡 <span style='color:#aaa;'>点击右侧 ✨ 生成核心贡献摘要</span><br>"
                        )
                    if ck in st.session_state.score_cache:
                        box_content += f"⭐ <b>评分:</b> {st.session_state.score_cache[ck]}"
                    else:
                        box_content += (
                            "⭐ <span style='color:#aaa;'>点击右侧 ⭐ 获取SS综合打分</span>"
                        )
                    st.markdown(
                        f'<div class="contribution-box">{box_content}</div>', unsafe_allow_html=True
                    )
                with cg:
                    if st.button("✨ 摘要", key=f"contrib_{i}"):
                        with st.spinner("分析..."):
                            get_one_line_contribution(res.summary, res.title, user_api_key)
                        st.rerun()
                with cs:
                    if st.button("⭐ 打分", key=f"score_{i}"):
                        with st.spinner("获取SS数据并打分..."):
                            score = get_paper_score(
                                res.entry_id, res.title, res.summary, user_api_key, ss_api_key
                            )
                            st.session_state.score_cache[ck] = score
                        st.rerun()
                st.markdown(
                    f'<div class="abstract-box"><b>摘要：</b>{res.summary.replace(chr(10)," ")}</div>',
                    unsafe_allow_html=True,
                )
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.markdown(f"[🔗 ArXiv]({res.entry_id})")
                with b2:
                    if st.button("⬇️ 下载入库", key=f"dl_{i}"):
                        with st.spinner("下载解析..."):
                            try:
                                pdf_path = download_arxiv_pdf_direct(res.entry_id)
                                process_and_add_to_topic(pdf_path, res.title, user_api_key)
                                st.success("入库成功！")
                            except Exception as e:
                                st.error(str(e))
                with b3:
                    lbl = (
                        "🕸️ 图谱 ⚡"
                        if res.entry_id in st.session_state.preload_done_ids
                        else "🕸️ 图谱"
                    )
                    if st.button(lbl, key=f"graph_{i}"):
                        st.session_state.focus_paper_id = res.entry_id
                        st.rerun()

                # 图谱直接渲染在这篇论文的卡片里，避免“论文在下面、图在上面”
                if is_focused:
                    render_graph_section()

    if st.session_state.search_generator:
        st.markdown("---")
        if st.button("🔽 加载更多 100 篇...", use_container_width=True):
            with st.spinner("正在拉取新论文摘要并计算语义分数..."):
                more_raw = list(itertools.islice(st.session_state.search_generator, 100))
                if more_raw:
                    new_results = [{"obj": r, "citations": None} for r in more_raw]

                    try:
                        embeddings_model = get_embeddings_model()
                        query_vec = np.array(embeddings_model.embed_query(search_query))
                        abstracts = [item["obj"].summary for item in new_results]
                        doc_vecs = np.array(embeddings_model.embed_documents(abstracts))
                        norms_doc = np.linalg.norm(doc_vecs, axis=1)
                        norm_query = np.linalg.norm(query_vec)
                        similarities = np.dot(doc_vecs, query_vec) / (norms_doc * norm_query)
                        for idx, item in enumerate(new_results):
                            item["sim_score"] = float(similarities[idx]) * 100
                    except Exception as e:
                        logger.warning("load_more_semantic_score_failed error=%s", e)
                        for item in new_results:
                            item["sim_score"] = 50.0

                    id2c = smart_fetch_citations(new_results, ss_key=ss_api_key)
                    for item in new_results:
                        item["citations"] = id2c.get(item["obj"].entry_id, 0)
                    st.session_state.search_results.extend(new_results)

                    if "引用量" in sort_mode:
                        st.session_state.search_results.sort(
                            key=lambda x: x["citations"] or 0, reverse=True
                        )
                    elif "质量优先" in sort_mode or "综合" in sort_mode:
                        import math

                        current_year = datetime.now().year
                        for idx, item in enumerate(st.session_state.search_results):
                            rel_score = item.get("sim_score", 50.0)

                            cites = item["citations"] or 0
                            cite_score = (math.log10(cites + 1) / 3.0) * 100

                            pub_year = item["obj"].published.year
                            age = max(0, current_year - pub_year)

                            if "质量优先" in sort_mode:
                                time_bonus = 0
                                if age == 0:
                                    time_bonus = 40
                                elif age == 1:
                                    time_bonus = 20
                                elif age == 2:
                                    time_bonus = 10
                            else:
                                time_bonus = max(0, 30 - age * 10)

                            quality_score = min(100.0, cite_score + time_bonus)
                            item["total_score"] = (rel_score * rel_w) + (quality_score * qual_w)

                        st.session_state.search_results.sort(
                            key=lambda x: x.get("total_score", 0), reverse=True
                        )
                    st.rerun()
                else:
                    st.info("✨ 到底啦。")

