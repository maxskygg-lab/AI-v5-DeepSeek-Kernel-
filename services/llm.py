"""DeepSeek 调用：模型句柄、核心贡献摘要与论文打分。"""

import requests
import streamlit as st

from .log import get_logger
from .semantic_scholar import _ss_headers
from .utils import get_pure_arxiv_id

logger = get_logger("llm")


def get_deepseek_llm(api_key, temperature=0.1):
    from langchain_openai import ChatOpenAI

    if not api_key:
        st.error("请先配置 DEEPSEEK_API_KEY")
        st.stop()
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=api_key,
        openai_api_base="https://api.deepseek.com",
        temperature=temperature,
    )


def get_one_line_contribution(abstract, title, api_key):
    key = title[:60]
    if key in st.session_state.contributions_cache:
        return st.session_state.contributions_cache[key]
    try:
        llm = get_deepseek_llm(api_key, temperature=0.0)
        res = llm.invoke(
            f"请用一句话（不超过40个汉字或20个英文单词）总结这篇论文的核心创新贡献。"
            f"只输出这一句话，不要前缀或解释。\n\n标题：{title}\n摘要：{abstract[:600]}"
        )
        result = res.content.strip()
    except Exception as e:
        logger.warning("contribution_failed title=%s error=%s", title[:60], e)
        result = "（生成失败）"
    st.session_state.contributions_cache[key] = result
    return result


def get_paper_score(arxiv_id, title, abstract, api_key, ss_key):
    # 此处已移除对 st.session_state 的直接读写，变为纯函数，防止多线程崩溃
    try:
        clean_id = get_pure_arxiv_id(arxiv_id)
        # 向 Semantic Scholar 请求时增加了 year 字段，获取真实发表年份
        url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{clean_id}?fields=tldr,influentialCitationCount,year"
        headers = _ss_headers(ss_key)
        ss_info = ""
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                tldr = data.get("tldr", {}).get("text", "无") if data.get("tldr") else "无"
                inf_cites = data.get("influentialCitationCount", 0)
                pub_year = data.get("year", "未知年份")
                ss_info = f"\n\n【Semantic Scholar 真实辅助数据】\n- 发表年份: {pub_year}\n- 极具影响力引用数: {inf_cites}\n- 官方TLDR摘要: {tldr}"
        except Exception as e:
            logger.debug("score_metadata_unavailable arxiv_id=%s error=%s", arxiv_id, e)

        # 锁定 temperature 为 0.0，杜绝大模型随机性，严格执行量表
        llm = get_deepseek_llm(api_key, temperature=0.0)
        prompt = (
            f"你现在是一位顶尖人工智能领域的资深论文导师。请你结合给定信息，客观且富有区分度地对这篇论文进行综合打分（满分100分）。\n"
            f"当前的打分常常偏低且雷同，请仔细发掘论文的细节和创新点，充分利用整个分数段（不要刻意压低分数），拉开合理的差距。\n\n"
            f"【学术打分量表（满分100）】：\n"
            f"1. 核心创新与突破性 (40分)：\n"
            f"   - 35-40分：提出颠覆性架构或解决领域重大难题。\n"
            f"   - 28-34分：有扎实的创新点，对SOTA有显著改进，或提供高价值数据集。\n"
            f"   - 20-27分：常规的渐进式改进，逻辑自洽，具备一定的实用价值。\n"
            f"   - 0-19分：创新性较弱，单纯的模块拼接或方法套用。\n"
            f"2. 方法严谨度与可信度 (30分)：\n"
            f"   - 25-30分：摘要明确列出量化指标、对比了强基线，提及开源或详尽实验验证。\n"
            f"   - 18-24分：提到实验效果提升，有合理的数据支持，但描述偏向概括。\n"
            f"   - 0-17分：缺乏具体数据指标支撑，结论偏主观或含糊。\n"
            f"3. 学术影响力与时效潜力 (30分)：\n"
            f"   - 综合发表年份、引用量（如有）以及研究方向的前沿热门程度给分。高引或极具潜力的热门方向给 22-30分；常规方向或普通跟进型研究给 15-21分；冷门且低引给0-14分。\n\n"
            f"【最终定档与输出规范】：\n"
            f"禁止输出计算过程、禁止输出拆项得分。只允许输出一行字：\n"
            f"【xx分】点评：一句话犀利指出核心优缺点（需一针见血，不超过30个汉字）。\n\n"
            f"标题：{title}\n摘要：{abstract[:600]}{ss_info}"
        )
        res = llm.invoke(prompt)
        result = res.content.strip()
    except Exception as e:
        logger.warning("paper_score_failed arxiv_id=%s error=%s", arxiv_id, e)
        result = "（打分失败）"
    return result
