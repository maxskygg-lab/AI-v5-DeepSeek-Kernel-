"""Tab 2：研读空间（单篇精读 / 多篇对比 / 自由问答）。"""

import uuid
from datetime import datetime

import streamlit as st

from services.context import (
    cited_page_coverage,
    collect_cited_chunks,
    expand_to_pages,
    format_digests,
    format_manifest_report,
    get_paper_digests,
    is_completeness_question,
)
from services.llm import get_deepseek_llm
from services.log import get_logger
from services.notes import save_note_to_cloud
from services.pdf import reconstruct_manifest_from_chunks
from services.retrieval import hybrid_retrieve
from services.semantic_scholar import fetch_ss_paper_details_by_title
from services.vectorstore import active_topic_data

logger = get_logger("ui.read")


def render_read_tab(user_api_key, ss_api_key):
    # --- 1. 顶部控制栏：精准控制 Scope ---
    t = active_topic_data()

    # 布局：左侧选模式，右侧状态显示
    c_ctrl, c_info = st.columns([2, 3])
    with c_ctrl:
        # 核心修改：明确的范围选择，不再默认全库检索
        scope_options = ["🌐 全库综合 (对比/综述)"] + t["files"]
        selected_scope = st.selectbox(
            "📚 阅读范围 (Scope)",
            scope_options,
            index=0,
            help="选择“全库”进行跨论文对比；选择“单篇”将只检索该论文内容，更省 Token 且更精准。",
        )
    with c_info:
        # 显示当前 Token 使用情况预估或入库状态
        if selected_scope == "🌐 全库综合 (对比/综述)":
            st.caption(f"🚀 当前模式：检索所有 {len(t['files'])} 篇论文")
        else:
            st.caption("🎯 当前模式：**专注研读** (已屏蔽其他论文干扰)")

    st.divider()

    # ── 资料体检：页级完整性（确定性数据，不经过 LLM） ──
    if t["files"]:
        with st.expander("🩺 资料体检（页级完整性）", expanded=False):
            manifests = st.session_state.get("paper_manifests", {})
            for paper in t["files"]:
                manifest = manifests.get(f"{st.session_state.active_topic}::{paper}")
                if not manifest:
                    manifest = reconstruct_manifest_from_chunks(paper, t["chunks"])
                if manifest.get("reconstructed"):
                    pages = manifest.get("text_pages") or []
                    span = f"P{pages[0]}–P{pages[-1]}" if pages else "无"
                    page_set = set(pages)
                    gaps = (
                        [p for p in range(pages[0], pages[-1] + 1) if p not in page_set]
                        if pages
                        else []
                    )
                    gap_text = (
                        "未发现中间缺页"
                        if not gaps
                        else "中间缺页 " + "、".join(f"P{p}" for p in gaps)
                    )
                    st.caption(
                        f"《{paper}》：按已入库 chunk 推断覆盖 {span}，{gap_text}；"
                        f"{manifest.get('chunks', 0)} 个 chunk。总页数需重新上传确认。"
                    )
                else:
                    low_quality = len(manifest.get("short_pages") or []) + len(
                        manifest.get("garbled_pages") or []
                    )
                    st.caption(
                        f"《{paper}》：总页数 {manifest.get('total_pages', 0)}，"
                        f"有文本 {len(manifest.get('text_pages') or [])} 页，"
                        f"空页 {len(manifest.get('empty_pages') or [])}，"
                        f"低质量 {low_quality}，{manifest.get('chunks', 0)} 个 chunk"
                    )

    # ── 多文献对比矩阵与溯源综述 ──
    if selected_scope == "🌐 全库综合 (对比/综述)" and len(t["files"]) > 0:
        with st.expander("📊 多文献结构化对比与综述生成 (Hybrid RAG 赋能)", expanded=False):
            st.markdown(
                "支持勾选多篇论文，系统将自动提取各项特征并渲染结构化矩阵，同时生成**带真实原文引证与段落脚标**的文献综述。"
            )
            matrix_papers = st.multiselect(
                "📌 勾选需对比的文献（建议 2-5 篇，避免超量）",
                t["files"],
                default=t["files"][:3] if len(t["files"]) >= 2 else t["files"],
            )

            if st.button("🚀 抽取文献生成矩阵与综述", use_container_width=True) and matrix_papers:
                with st.spinner(
                    "🧠 正在提取文献片段，构建对比矩阵及追溯综述初稿（这可能需要一些时间，请耐心等待）..."
                ):
                    try:
                        # 步骤 1：为每篇论文生成结构化摘要（基于全文代表性片段，缓存复用）
                        with st.spinner("正在生成每篇论文的结构化摘要（首次较慢，之后会缓存）..."):
                            digests = get_paper_digests(
                                matrix_papers,
                                t["chunks"],
                                user_api_key,
                                ss_api_key,
                                with_meta=True,
                            )
                        coverage = []
                        for p in matrix_papers:
                            info = digests[p]
                            pages = info.get("pages") or []
                            if pages:
                                coverage.append(
                                    f"{p}：{info.get('chunk_count', 0)} 段（P{pages[0]}–P{pages[-1]}）"
                                )
                                if info.get("chunk_count", 0) < 5:
                                    st.warning(
                                        f"《{p}》仅提取到 {info.get('chunk_count', 0)} 个片段"
                                        f"（P{pages[0]}–P{pages[-1]}），矩阵可能不完整；"
                                        "建议重新上传文本版 PDF 或先做 OCR。"
                                    )
                            else:
                                coverage.append(f"{p}：0 段")
                                st.warning(
                                    f"《{p}》没有可用片段，无法生成完整矩阵；"
                                    "请确认论文已入库，或重新上传文本版 PDF。"
                                )
                        st.caption("📎 本次抽取覆盖：" + "；".join(coverage))
                        digest_text = format_digests(digests)

                        # 按摘要引用页码回取原文，保证综述里的页码有原文可核对
                        cited_lines = []
                        for p in matrix_papers:
                            digest = digests[p].get("digest", "")
                            cited = collect_cited_chunks(
                                p,
                                t["chunks"],
                                digest,
                                max_chunks=10 if len(matrix_papers) <= 3 else 6,
                            )
                            cited = expand_to_pages(
                                cited,
                                t["chunks"],
                                max_pages=8,
                                max_chars_per_page=3500,
                                max_total_chars=18000,
                            )
                            if cited:
                                cited_lines.append(f"【{p}】")
                                for d in cited:
                                    page = int(d.metadata.get("page", 0) or 0) + 1
                                    text = d.page_content.replace("\n", " ").strip()
                                    cited_lines.append(f"[P{page}] {text}")
                            _, _, missing = cited_page_coverage(
                                [c for c in t["chunks"] if c.metadata.get("source_paper") == p],
                                digest,
                            )
                            if missing:
                                st.warning(
                                    f"《{p}》摘要引用了 P{'、P'.join(map(str, missing))}，"
                                    "但原文索引中没有这些页，无法提供原文核对。"
                                )
                        cited_text = "\n".join(cited_lines)

                        # 步骤 2：调用大模型进行信息整合与综述
                        llm = get_deepseek_llm(user_api_key, temperature=0.1)
                        matrix_prompt = (
                            "你是一位资深的AI顶会审稿人与学术研究员。请仔细阅读以下每篇论文的结构化摘要"
                            "（由全文代表性片段抽取，含页码证据），完成以下两项任务：\n\n"
                            "### 任务一：结构化对比矩阵 (Markdown Table)\n"
                            "请严格提取各论文的核心特征，输出一个对比表格，**表头必须且只能包含以下五列："
                            "论文名称、创新点、基准数据集、SOTA 指标、局限性**。"
                            "优先使用结构化摘要中的字段；若摘要明确写“论文未报告”，矩阵填“论文未报告”；"
                            "只有摘要和引用页证据都缺失时才填“资料未覆盖”。"
                            "不要因为摘要片段没有就判定整篇论文没有，也不要输出额外的资料覆盖审计。\n\n"
                            "### 任务二：带溯源的学术综述初稿\n"
                            "基于上述摘要与关键证据，撰写一段连贯的学术综述（约400-500字），"
                            "梳理这些文献的研究方法演进与优劣势。\n"
                            "**【致命要求】**：综述中的每一条结论，必须使用摘要中真实出现的页码进行引用溯源。"
                            "格式例如：`（《某文献名》[P页码]）`。绝对禁止编造任何未在资料中出现的虚假指标或事实。\n\n"
                            f"### 论文结构化摘要（仅以此为准）：\n{digest_text}\n\n"
                            f"### 摘要引用页原文（用于核对页码与指标）：\n"
                            f"{cited_text or '（无）'}"
                        )

                        res_matrix = llm.invoke(matrix_prompt)
                        st.markdown(res_matrix.content)

                        # 步骤 3：自动存入笔记系统（同时写入云端，刷新后可恢复）
                        new_note = {
                            "id": str(uuid.uuid4())[:8],
                            "content": res_matrix.content,
                            "question": f"多文献结构化对比与综述生成：{', '.join(matrix_papers)}",
                            "tags": ["结构化对比矩阵", "溯源文献综述", "混合检索"],
                            "topic": st.session_state.active_topic,
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        }
                        st.session_state.notes.append(new_note)
                        if save_note_to_cloud(new_note):
                            st.success(
                                "✅ 矩阵与综述生成完毕！内容已自动保存至「📌 我的笔记」（云端已同步）。"
                            )
                        else:
                            st.warning(
                                "⚠️ 内容已保存在当前页面，但云端同步失败，将自动重试；也可在「📌 我的笔记」页手动同步。"
                            )
                    except Exception as e:
                        logger.exception(
                            "matrix_generation_failed topic=%s error=%s",
                            st.session_state.active_topic,
                            e,
                        )
                        st.error(f"生成失败，请检查网络或日志：{e}")
        st.divider()

    # ── 聊天历史回显 ──
    if not st.session_state.chat_history:
        # 欢迎引导语
        with st.chat_message("assistant"):
            st.markdown(
                f"👋 我是你的 DeepSeek 研读助手。当前主题库中有 **{len(t['files'])}** 篇论文。"
            )
            if t["files"]:
                st.markdown(
                    "你可以问我：\n- *这篇论文的核心方法是什么？* (建议选择单篇范围)\n- *对比 Transformer 和 CNN 在这些论文中的观点差异* (建议选择全库范围)"
                )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- 3. 聊天输入与处理 ---
    if prompt := st.chat_input("输入问题..."):
        # 0. 检查是否有库
        if not t["db"]:
            st.error("请先在左侧上传 PDF 或从检索页下载论文入库！")
            st.stop()

        # 1. 用户消息上屏
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. AI 生成回答
        with st.chat_message("assistant"):
            # 占位符用于流式输出
            message_placeholder = st.empty()
            full_response = ""

            try:
                completeness = is_completeness_question(prompt)
                # --- 核心优化 A：精准检索策略 ---
                search_kwargs = {}
                filter_rule = None

                # 如果选择了特定论文，构建 metadata 过滤器
                if selected_scope != "🌐 全库综合 (对比/综述)":
                    filter_rule = {"source_paper": selected_scope}
                    search_kwargs = {"k": 20, "filter": filter_rule}
                    status_text = f"🔍 正在生成论文摘要并深度扫描论文《{selected_scope}》..."
                else:
                    search_kwargs = {"k": 15, "fetch_k": 50, "lambda_mult": 0.6}
                    status_text = f"🔍 正在生成论文摘要并在全库 {len(t['files'])} 篇论文中检索..."

                with st.spinner(status_text):
                    try:
                        # 步骤 1：圈定检索大池子
                        if filter_rule:
                            target_chunks = [
                                c
                                for c in t["chunks"]
                                if c.metadata.get("source_paper") == selected_scope
                            ]
                            ss_data = fetch_ss_paper_details_by_title(selected_scope, ss_api_key)
                        else:
                            target_chunks = t["chunks"]
                            ss_data = None

                        if not target_chunks:
                            docs = []
                        else:
                            docs = hybrid_retrieve(
                                prompt,
                                target_chunks,
                                t["db"],
                                search_kwargs,
                                f"{st.session_state.active_topic}:{selected_scope}",
                                cross_paper=selected_scope == "🌐 全库综合 (对比/综述)",
                            )
                    except Exception as hybrid_err:
                        # 降级预案：如因缺失库或其他原因导致混合管线崩溃，静默退回原始向量检索以防卡死
                        logger.warning(
                            "hybrid_retrieve_fallback scope=%s error=%s", selected_scope, hybrid_err
                        )
                        st.warning(
                            f"由于缺少部分依赖库(如 rank_bm25 / sentence_transformers)，混合检索退回基础向量检索模式。({hybrid_err})"
                        )
                        if filter_rule:
                            docs = t["db"].similarity_search(prompt, **search_kwargs)
                            ss_data = fetch_ss_paper_details_by_title(selected_scope, ss_api_key)
                        else:
                            docs = t["db"].max_marginal_relevance_search(prompt, **search_kwargs)
                            ss_data = None

                    # 步骤 2：生成每篇论文的结构化摘要（缓存复用）
                    digest_papers = (
                        [selected_scope]
                        if selected_scope != "🌐 全库综合 (对比/综述)"
                        else t["files"]
                    )
                    try:
                        digests = get_paper_digests(
                            digest_papers,
                            t["chunks"],
                            user_api_key,
                            ss_api_key,
                            with_meta=True,
                        )
                    except Exception as digest_err:
                        logger.warning("digest_generation_fallback error=%s", digest_err)
                        digests = {}

                    # 步骤 3：按摘要引用页码回取原文，避免“摘要说有、证据里没有”
                    cited_docs = []
                    if digests:
                        for p in digest_papers:
                            digest = digests.get(p, {}).get("digest", "")
                            cited_docs.extend(
                                collect_cited_chunks(
                                    p,
                                    t["chunks"],
                                    digest,
                                    max_chunks=12 if len(digest_papers) <= 2 else 8,
                                )
                            )
                            _, _, missing = cited_page_coverage(
                                [c for c in t["chunks"] if c.metadata.get("source_paper") == p],
                                digest,
                            )
                            if missing:
                                st.warning(
                                    f"《{p}》摘要引用了 P{'、P'.join(map(str, missing))}，"
                                    "但原文索引中没有这些页，无法提供原文核对。"
                                )

                # --- 核心优化 B：构建更智能的 Prompt ---
                if not docs and not cited_docs and not completeness:
                    full_response = "⚠️ 未在文档中利用混合管线检索到相关信息，请尝试更换关键词。"
                    message_placeholder.markdown(full_response)
                else:
                    digest_text = format_digests(digests)
                    # 把检索碎片扩展成整页，摘要引用页与检索页合并去重，统一成一个证据池
                    evidence_docs = (
                        []
                        if completeness
                        else expand_to_pages(
                            cited_docs + docs,
                            t["chunks"],
                            max_pages=8 if selected_scope != "🌐 全库综合 (对比/综述)" else 12,
                            max_chars_per_page=4000,
                            max_total_chars=30000,
                        )
                    )
                    manifest_text = (
                        format_manifest_report(
                            digest_papers,
                            t["chunks"],
                            digests,
                            st.session_state.get("paper_manifests", {}),
                            st.session_state.active_topic,
                        )
                        if completeness
                        else ""
                    )

                    context_text = ""
                    refs = []
                    ref_no = 0
                    for d in evidence_docs:
                        src = d.metadata.get("source_paper", "未知")
                        page = int(d.metadata.get("page", 0) or 0) + 1
                        snippet = d.page_content.replace("\n", " ").strip()
                        if not snippet:
                            continue
                        ref_no += 1
                        context_text += f"[资料{ref_no} | {src} (P{page})]: {snippet}\n\n"
                        refs.append(f"**[{ref_no}] {src} (P{page})**: {snippet[:100]}...")

                    ss_context = ""
                    if ss_data:
                        tldr = (
                            ss_data.get("tldr", {}).get("text", "无")
                            if ss_data.get("tldr")
                            else "无"
                        )
                        authors = ", ".join([a.get("name", "") for a in ss_data.get("authors", [])])
                        ss_context = (
                            f"\n\n【Semantic Scholar 真实元数据补充】\n"
                            f"- 标题: {ss_data.get('title')}\n"
                            f"- 作者: {authors}\n"
                            f"- 发表年份/会议: {ss_data.get('year')} / {ss_data.get('venue')}\n"
                            f"- 总引用数: {ss_data.get('citationCount')} (其中极具影响力引用: {ss_data.get('influentialCitationCount')})\n"
                            f"- 官方TLDR摘要: {tldr}\n"
                            f"**指令**：你在回答背景、影响力和核心结论时，必须优先以这部分的真实元数据为准，不要编造不存在的事实。\n"
                        )

                    # 动态 System Prompt
                    if completeness:
                        sys_prompt = (
                            "用户询问资料完整性。以下是系统生成的页级体检报告（确定性数据）。\n"
                            "要求：\n"
                            "1. 只根据报告回答，不要编造报告中没有的页码或缺失项。\n"
                            "2. 明确区分三件事：页级索引是否完整、摘要引用页是否有原文、"
                            "论文本身是否未报告某项。\n"
                            "3. 对用户当前任务给出结论：如果用于结构化对比/具体问答，说明是否够用；"
                            "如果空页或低质量页影响结论，建议 OCR 或重新上传。\n"
                            "4. 不要输出与报告无关的资料覆盖审计。\n\n"
                            f"### 页级体检报告：\n{manifest_text}"
                        )
                        context_text = ""  # 体检问题不使用检索证据，避免模型再次审计
                    elif selected_scope != "🌐 全库综合 (对比/综述)":
                        sys_prompt = (
                            f"你正在辅助用户精读论文《{selected_scope}》。\n"
                            "以下是该论文的结构化摘要（基于代表性片段）和页码级原文证据。\n"
                            "要求：\n"
                            "1. 先用结构化摘要建立整体理解，再用原文证据回答具体问题。\n"
                            "2. 回答要深入、具体，多引用数据或具体算法步骤。\n"
                            "3. 如果资料中包含公式描述，请还原为 LaTeX 格式。\n"
                            "4. 引用时给出页码，严禁编造资料中不存在的内容。\n"
                            "5. 直接回答用户问题，不要输出资料完整性审计；"
                            "只有具体事实在摘要和证据中都找不到时，才简短说明。\n"
                            "6. 摘要引用页与检索页不重合是正常的，不要据此判断资料不完整。"
                            f"{ss_context}"
                            f"\n\n### 论文结构化摘要：\n{digest_text}"
                        )
                    else:
                        sys_prompt = (
                            "你是一名学术顾问。以下是每篇论文的结构化摘要（基于代表性片段）"
                            "和页码级原文证据。\n"
                            "要求：\n"
                            "1. 先用结构化摘要建立每篇论文的整体理解，再用原文证据回答具体问题。\n"
                            "2. 必须明确指出不同观点分别来自哪篇论文，并给出页码。\n"
                            "3. 如果涉及对比，请使用 Markdown 表格形式展示。\n"
                            "4. 摘要中标注“论文未报告”的字段，不要臆测；"
                            "只有摘要和证据都缺失时，才写“资料未覆盖”。\n"
                            "5. 直接回答用户问题，不要输出资料完整性审计；"
                            "页码不重合是正常的，不要据此判断资料不完整。"
                            f"\n\n### 论文结构化摘要：\n{digest_text}"
                        )

                    # --- 核心优化 C：调用 DeepSeek (流式) ---
                    if not completeness:
                        with st.expander(
                            "📚 查看 AI (由混合管线重排推荐) 参考的原文片段", expanded=False
                        ):
                            st.markdown("\n\n".join(refs))

                    llm = get_deepseek_llm(user_api_key, temperature=0.3)

                    evidence_block = (
                        "" if completeness else f"\n\n### 页码级原文证据：\n{context_text}"
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": f"{sys_prompt}{evidence_block}",
                        },
                        {"role": "user", "content": prompt},
                    ]

                    stream = llm.stream(messages)

                    for chunk in stream:
                        if chunk.content:
                            full_response += chunk.content
                            message_placeholder.markdown(full_response)

                    message_placeholder.markdown(full_response)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": full_response}
                )

                st.session_state.pending_note = {
                    "content": full_response,
                    "question": prompt,
                }

            except Exception as e:
                logger.exception("chat_failed topic=%s error=%s", st.session_state.active_topic, e)
                st.error(f"发生错误: {str(e)}")

    if st.session_state.pending_note:
        st.markdown("---")
        c_note_1, c_note_2 = st.columns([5, 1])
        with c_note_1:
            note_tags = st.text_input(
                "给刚才的回答加个标签？(可选)",
                placeholder="例如: 核心算法, 实验结果",
                key="quick_note_tag",
            )
        with c_note_2:
            st.write("")
            if st.button("📌 存笔记"):
                tags = [t.strip() for t in note_tags.split(",")] if note_tags else []
                new_note = {
                    "id": str(uuid.uuid4())[:8],
                    "content": st.session_state.pending_note["content"],
                    "question": st.session_state.pending_note["question"],
                    "tags": tags,
                    "topic": st.session_state.active_topic,
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                st.session_state.notes.append(new_note)
                st.session_state.pending_note = None
                if save_note_to_cloud(new_note):
                    st.toast("笔记已保存并同步到云端！", icon="🎉")
                else:
                    st.toast("笔记已保存到本页，云端同步失败，稍后自动重试", icon="⚠️")
                st.rerun()
