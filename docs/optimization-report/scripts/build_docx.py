# -*- coding: utf-8 -*-
"""生成《AI 学术文献研读助手 优化专项报告》Word 版。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(BASE, "assets")
os.makedirs(ASSETS, exist_ok=True)

FONT_CN = "Microsoft YaHei"
FONT_EN = "Arial"
INK = RGBColor(0x1F, 0x1B, 0x16)
MUTED = RGBColor(0x6D, 0x65, 0x58)
ACCENT = RGBColor(0xA6, 0x43, 0x1F)
ACCENT2 = RGBColor(0x2F, 0x5D, 0x50)


# ---------------------------------------------------------------- 图表
def make_charts():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 图 2：namespace 分布
    labels = [
        "旧 · default_topic（十六进制编码）",
        "旧 · 默认主题（中文编码）",
        "旧 · 随机副本 A",
        "旧 · 随机副本 B",
        "旧 · 随机副本 C",
        "旧 · 随机副本 D",
        "旧 · 随机副本 E",
        "目标 · default_topic（稳定房间）",
        "旧 · 随机副本 F",
    ]
    values = [2118, 785, 231, 91, 91, 91, 91, 23, 17]
    colors = ["#a6431f"] * len(values)
    colors[7] = "#2f5d50"

    fig, ax = plt.subplots(figsize=(8.4, 4.3), dpi=200)
    bars = ax.barh(labels, values, color=colors, height=0.62)
    ax.invert_yaxis()
    for b, v in zip(bars, values):
        ax.text(v + 28, b.get_y() + b.get_height() / 2, str(v),
                va="center", ha="left", fontsize=9, color="#1f1b16")
    ax.set_xlabel("向量记录数（条）", fontsize=10, color="#6d6558")
    ax.set_xlim(0, 2450)
    ax.tick_params(axis="x", labelsize=9, colors="#6d6558")
    ax.tick_params(axis="y", labelsize=9, colors="#1f1b16")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#ded7c9")
    ax.grid(axis="x", color="#ded7c9", linestyle="-", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p1 = os.path.join(ASSETS, "ns_chart.png")
    fig.savefig(p1, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 图 3：能力具备矩阵
    caps = ["多篇对比覆盖", "刷新后数据可见", "跨工作区隔离",
            "扫描版 PDF 可读", "失败可诊断", "重资源缓存"]
    fig, ax = plt.subplots(figsize=(8.4, 3.7), dpi=200)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.6, len(caps) - 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["优化前", "优化后"], fontsize=11, color="#1f1b16")
    ax.set_yticks(range(len(caps)))
    ax.set_yticklabels(caps, fontsize=10, color="#1f1b16")
    ax.invert_yaxis()
    for i in range(len(caps)):
        ax.scatter([0], [i], s=300, color="#c9c2b4", zorder=3, edgecolors="none")
        ax.scatter([1], [i], s=300, color="#2f5d50", zorder=3, edgecolors="none")
        ax.text(0, i, "0", ha="center", va="center", color="white", fontsize=10, zorder=4)
        ax.text(1, i, "1", ha="center", va="center", color="white", fontsize=10, zorder=4)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color="#ded7c9", linestyle="-", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p2 = os.path.join(ASSETS, "cap_chart.png")
    fig.savefig(p2, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p1, p2


# ---------------------------------------------------------------- docx 工具
def fix_font(run, size=None, bold=None, color=None):
    run.font.name = FONT_EN
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)
    fix_font(r, size=9, color=MUTED)


# ---------------------------------------------------------------- 主流程
def main():
    p1, p2 = make_charts()

    doc = Document()

    # 页面 A4
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    add_page_number(sec)

    # 样式
    normal = doc.styles["Normal"]
    normal.font.name = FONT_EN
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_after = Pt(4)

    for name, size, color, before, after in [
        ("Heading 1", 18, INK, 18, 10),
        ("Heading 2", 13.5, ACCENT2, 14, 6),
        ("Heading 3", 11.5, INK, 10, 4),
    ]:
        st = doc.styles[name]
        st.font.name = FONT_EN
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT_CN)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.keep_with_next = False

    def P(text="", size=11, bold=False, color=None, align=None, style=None,
          before=None, after=None, italic=False):
        p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
        if text:
            r = p.add_run(text)
            fix_font(r, size=size, bold=bold, color=color)
            r.italic = italic
        if align is not None:
            p.alignment = align
        if before is not None:
            p.paragraph_format.space_before = Pt(before)
        if after is not None:
            p.paragraph_format.space_after = Pt(after)
        return p

    def H(text, level=1):
        h = doc.add_heading(text, level=level)
        for r in h.runs:
            fix_font(r)
        return h

    def KV(label, value):
        p = doc.add_paragraph()
        r1 = p.add_run(label + "：")
        fix_font(r1, size=11, bold=True, color=ACCENT2)
        r2 = p.add_run(value)
        fix_font(r2, size=11)
        p.paragraph_format.left_indent = Cm(0.3)
        p.paragraph_format.space_after = Pt(5)
        return p

    def bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style="List Bullet")
        if bold_prefix:
            r0 = p.add_run(bold_prefix)
            fix_font(r0, size=11, bold=True, color=ACCENT2)
        r = p.add_run(text)
        fix_font(r, size=11)
        p.paragraph_format.space_after = Pt(4)
        return p

    def caption(text):
        return P(text, size=9, color=MUTED, align=WD_ALIGN_PARAGRAPH.CENTER, after=10, italic=True)

    def table(headers, rows, widths):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        for i, (h, w) in enumerate(zip(headers, widths)):
            c = t.rows[0].cells[i]
            c.width = Cm(w)
            shade(c, "F1ECE1")
            para = c.paragraphs[0]
            r = para.add_run(h)
            fix_font(r, size=10, bold=True, color=INK)
            para.paragraph_format.space_after = Pt(2)
        for row in rows:
            cells = t.add_row().cells
            for i, (val, w) in enumerate(zip(row, widths)):
                cells[i].width = Cm(w)
                para = cells[i].paragraphs[0]
                r = para.add_run(str(val))
                fix_font(r, size=9.5)
                para.paragraph_format.space_after = Pt(2)
        return t

    # ============================================================ 封面
    P("", after=0)
    P("", after=0)
    P("COURSE REPORT · 项目优化专项", size=10, bold=True, color=ACCENT, align=WD_ALIGN_PARAGRAPH.LEFT)
    P("AI 学术文献研读助手", size=26, bold=True, color=INK, before=10, after=2)
    P("优化专项报告", size=26, bold=True, color=INK, after=16)
    P("从「只能读一篇、刷新就丢」到「可对比、可找回、可隔离」——一次面向真实使用场景的系统性修复与加固。",
      size=12, color=MUTED, after=28)
    P("项目类型　学术文献检索与研读系统（Streamlit + RAG）", size=10.5, color=MUTED, after=4)
    P("汇报场景　课程项目进度汇报", size=10.5, color=MUTED, after=4)
    P("报告日期　2026-10-29", size=10.5, color=MUTED, after=4)
    doc.add_page_break()

    # ============================================================ 摘要
    H("摘要", 1)
    P("本项目是一个面向科研场景的 AI 文献研读助手：用户上传论文后，系统先「翻书找相关段落」，再由大语言模型基于这些段落作答，并支持多篇论文对比。优化前，系统存在对比只认一篇、刷新页面论文消失、不同工作区数据混用、历史数据碎片化、扫描版 PDF 读不出、重复加载导致变慢、报错不透明等七类问题。本次优化围绕「检索质量、数据隔离、持久化、性能、可观测性」五条主线完成整改。优化后，多文献对比可稳定覆盖多篇论文，重开页面可自动找回已入库论文，工作区之间互不干扰，历史数据可一键去重归并。")

    # ============================================================ 第一章
    doc.add_page_break()
    H("第一章　项目背景与原始问题说明", 1)

    H("1.1　这个项目是做什么的", 2)
    P("它本质上是一个「会读论文的 AI 助手」。可以把它的工作方式理解为开卷考试：用户提问后，系统先在自己的「书架」（向量数据库）里翻出最相关的若干段落，再把这些段落交给大模型，让模型只依据这些段落作答，从而减少胡编乱造。")
    P("系统还提供「多篇论文对比」模式：用户希望 AI 同时阅读多篇论文，横向比较它们的观点、方法与结论。这一模式是本项目区别于普通问答的关键能力。")

    H("1.2　优化前存在的具体痛点", 2)
    P("以下问题均有真实的使用现场作为证据，而非假设：")

    pains = [
        ("痛点 01 · 对比失效", "「对比全部论文」，AI 却只谈一篇",
         "系统挑出「最像问题的 5 段文字」，但这 5 段往往全部来自同一篇论文，于是回答只围绕一篇展开，对比名存实亡。"),
        ("痛点 02 · 数据丢失", "今天传的论文，明天打开就不见了",
         "论文清单只存在「当前会话」里，页面一刷新就清空；而云端其实存着数据，只是系统「找不到」。"),
        ("痛点 03 · 隔离缺失", "不同用户、不同主题的数据会串在一起",
         "云端「房间号」只按主题名生成，任何用户建同名主题都会落进同一个房间，存在互相看到、甚至互相删除的风险。"),
        ("痛点 04 · 房间漂移", "同一主题，每次刷新换一个「房间」",
         "用于生成房间号的标识每次新会话都会随机变化，导致数据被写进不同房间，越用越乱。"),
        ("痛点 05 · 数据碎片", "同一批论文被存了 9 份副本",
         "实测云端散落着 9 个房间、合计约 3538 条记录，大量是重复上传产生的副本，既占空间又互相干扰。"),
        ("痛点 06 · 解析空转", "扫描版 PDF 显示「已入库」，其实一个字没读到",
         "解析器读不出文字时不报错，论文只在列表里占了一个名字，检索时永远命中不了。"),
        ("痛点 07 · 性能与可观测性", "每次提问都像「重新开机」，出错还看不清",
         "模型、检索索引、数据库连接每次请求都重新创建；异常又被静默吞掉，用户只看到「没反应」。"),
    ]
    for tag, title, desc in pains:
        p = doc.add_paragraph()
        r1 = p.add_run(tag + "　")
        fix_font(r1, size=9.5, bold=True, color=ACCENT)
        r2 = p.add_run(title)
        fix_font(r2, size=11, bold=True, color=INK)
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(2)
        P(desc, size=10.5, color=MUTED, after=4)

    H("1.3　为什么必须优化", 2)
    P("这七类问题分别击中了系统的四条生命线：可信（对比只覆盖一篇，结论不可信）、可用（刷新即丢数据，无法持续使用）、安全（工作区不加隔离，存在越权风险）、可维护（失败静默、性能浪费，难以定位与扩展）。如果不修，这个系统只能停留在「演示可用」，无法成为老师与同学愿意长期使用的工具。")

    # ============================================================ 第二章
    doc.add_page_break()
    H("第二章　优化方案设计思路", 1)
    P("下面按「一个问题 → 一个方案」的方式逐条说明。每条都包含：做了什么、为什么这样做、以及对应的技术实现。为便于非技术背景听众理解，先给生活化比喻，再给实现方式。")

    opts = [
        ("优化 01", "让「对比」真正覆盖多篇论文",
         "老师要求「每组各派一名代表发言」，而不是「谁声音大谁上 5 次」。",
         "重排打分后，第一轮每篇论文只取分数最高的 1 段，确保多篇论文都有代表；如果论文数不足 5 篇，再用剩余高分段落补齐。",
         "在 Cross-Encoder 重排之后，按来源论文（source_paper）去重选段，替代原先「纯按分数取前 5」的做法；单篇研读模式保持原逻辑不变。"),
        ("优化 02", "给每个工作区建「独立房间」（空间隔离）",
         "同一栋楼里，每个用户有自己的房间和钥匙，不能进别人的房间拿东西。",
         "房间号 = 「工作区标识 + 主题名」共同计算出的固定编码；不同工作区即便主题同名，也会落到不同房间。",
         "用「主题名 + 工作区标识」的哈希值生成云端命名空间（namespace），替换原先「仅用主题名」的方案。"),
        ("优化 03", "让「门牌号」稳定下来（工作区标识）",
         "门牌号不能每天换，否则快递永远找不到你。",
         "工作区标识改为确定性取值：优先读取配置里的固定 ID，其次使用登录邮箱，最后使用固定兜底值；绝不使用随机值。",
         "移除基于浏览器 Cookie 的随机标识（它在每次新会话首帧常返回空，导致每次都生成新 ID），改为「配置 > 邮箱 > 固定值」三级稳定取值。"),
        ("优化 04", "重开页面自动「找回论文」（持久化恢复）",
         "入住酒店后，前台能凭房号把你的行李重新送回房间，而不是让你重新打包。",
         "每次进入主题时，先从云端列出并取回该房间全部内容，重建论文清单与检索上下文。",
         "调用向量库的分页列出（list）与取回（fetch）接口，从记录元数据中还原正文与来源；同时兼容新旧两代 SDK 接口，避免旧写法失败后又被静默吞掉。"),
        ("优化 05", "一键「搬家」：把散落旧房间的数据去重归并",
         "把散落在 9 个旧储物柜里的同款物品，去重后合并进 1 个新柜子，且只搬「属于这个主题」的东西。",
         "扫描所有旧房间的向量，按正文指纹去重，只保留属于当前主题的记录，再写入新房间；旧数据不删除，可回退。",
         "以正文的哈希值作为唯一键去重，写入时使用确定性 ID，保证重复执行也不会产生新副本。"),
        ("优化 06", "重型工具只加载一次（技术栈缓存）",
         "工具箱常驻桌面，而不是每次拧螺丝都去仓库重新搬一趟。",
         "把四类「重资源」缓存复用：重排模型、关键词检索索引、向量库连接、文本向量模型。",
         "重排模型与向量库连接使用资源级缓存；关键词检索索引按「主题 + 检索范围」建立指纹缓存，语料不变则直接复用。"),
        ("优化 07", "更强的 PDF 读取与空内容拦截",
         "一位翻译看不懂扫描件，就换另一位再试；如果两位都读不出，就明确告诉你「这页是图片，需要 OCR」，而不是假装读完了。",
         "增加第二套 PDF 解析器作为回退；若最终仍提取不到文字，拒绝静默入库并给出明确提示。",
         "主解析失败时改用 PyMuPDF 重新抽取；抽取结果为空则中止入库，避免「列表里有、检索不到」的空壳记录。"),
        ("优化 08", "报错更克制，也更清楚（可观测性）",
         "不是每次开门没看到人都拉响警报，但真正出故障时要准确说出是哪一层出了问题。",
         "自动流程保持安静，只有用户主动点击「恢复 / 迁移」时才输出诊断信息；异常不再被静默吞掉，而是带原因提示。",
         "为恢复流程增加「安静 / 诊断」两种模式；诊断时同时打印「本应用应查的房间号」与「云端真实存在的房间号及记录数」，便于一眼比对。"),
    ]
    for idx, title, metaphor, action, tech in opts:
        H(idx + "　" + title, 2)
        KV("生活比喻", metaphor)
        KV("具体做法", action)
        KV("技术实现", tech)

    H("2.9　关键链路示意", 2)
    P("用户上传论文 → 解析 PDF（双解析器回退）→ 切分为段落并生成向量 → 写入本工作区专属房间 → 刷新 / 重开页面 → 按房间号列出并取回 → 重建论文清单与检索上下文 → 提问 / 多篇对比。", size=10.5)
    caption("图 1：优化后的「上传—存储—找回—使用」链路。核心是房间号稳定、内容可回取。")

    # ============================================================ 第三章
    doc.add_page_break()
    H("第三章　优化前后对比数据", 1)
    P("下表汇总本次优化的可核对变化。为避免夸大，表中标注了数据性质：「实测」表示由系统指标或云端接口直接观测；「设计」表示由新逻辑保证的确定性行为；「估算」表示依据模型加载常识给出的量级判断，未做严格压测。", size=10.5)

    H("3.1　对比总表", 2)
    rows = [
        ("多文献对比覆盖论文数", "1 篇", "按论文数覆盖，最多 5 篇", "设计"),
        ("云端房间（namespace）数量", "9 个", "1 个（目标房间）", "实测"),
        ("云端向量记录总数", "3538 条（含大量重复）", "去重后唯一记录", "实测"),
        ("重开页面后论文可见性", "0 篇（需重新上传）", "自动恢复已入库清单", "设计"),
        ("云端恢复读取", "已读单元约 0，恢复为空", "可列出并取回数据", "实测"),
        ("工作区之间数据隔离", "无（仅按主题名）", "按工作区 + 主题双重区分", "设计"),
        ("房间号稳定性", "每次新会话可能变化", "跨会话恒定", "设计"),
        ("重排模型加载", "每次请求新建 1 次", "进程内仅首次加载", "估算"),
        ("关键词检索索引构建", "每次请求重建", "语料不变即复用", "设计"),
        ("向量库连接", "每次请求重连", "按房间缓存复用", "设计"),
        ("扫描版 PDF", "可能 0 文字且静默入库", "双解析回退 + 空内容拦截", "设计"),
        ("失败可观测性", "异常被静默吞掉", "显式提示 + 按需诊断", "设计"),
        ("新建主题体验", "弹出长红字报错", "安静通过，按需诊断", "设计"),
        ("历史数据迁移", "需手工全部重传", "一键去重迁移，可不重传", "设计"),
    ]
    table(["维度", "优化前", "优化后", "数据性质"], rows, [5.4, 4.2, 4.4, 2.0])
    caption("表 1：优化前后对比总表（共 14 项）")

    H("3.2　优化前的数据碎片分布（实测）", 2)
    P("下图为优化前云端房间的真实分布。可以看到同一批论文被重复写入了多个房间，其中最大的两个房间分别有 2118 条与 785 条，另有多个 91 条左右的重复副本。这正是「刷新后找不到论文」与「空间浪费」的直接原因。", size=10.5)
    doc.add_picture(p1, width=Cm(15.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption("图 2：优化前云端房间（namespace）记录数分布")

    H("3.3　关键能力具备情况", 2)
    P("下图把七类问题归纳为六项关键能力，用「0 = 不具备或失败、1 = 已具备」对比优化前后。六项能力全部从 0 变为 1，对应上表的逐项整改。", size=10.5)
    doc.add_picture(p2, width=Cm(15.2))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption("图 3：优化前后关键能力具备情况（0/1）")

    H("3.4　支撑结论的观测证据", 2)
    bullet("优化前云端向量记录总数 3538 条（含重复）——说明「上传」环节本身正常，问题不在数据库配置。", "数据确实写进去了：")
    bullet("优化前恢复时的「已读单元」接近 0——write 指标有尖峰、read 指标接近零，直接证明「写进去了、取不出来」。", "但恢复几乎没读到数据：")
    bullet("通过稳定房间号 + 一键迁移，把散落副本归并到唯一目标房间。", "碎片化被收敛（9 → 1）：")
    bullet("每篇论文先各取一段，再补高分段落，保证横向比较有素材。", "对比从「名义」变「实质」（1 → ≥1）：")
    P("说明：上述「实测」项来自本次排查中对云端接口返回结果的直接读取；「估算」项（如模型加载耗时）未做严格性能压测，仅用于说明量级方向，不作为精确指标。", size=9, color=MUTED, before=6)

    # ============================================================ 第四章
    doc.add_page_break()
    H("第四章　最终效果与经验总结", 1)

    H("4.1　整体提升", 2)
    improve = [
        ("可用性：", "从「刷新即丢、需反复上传」变为「重开即恢复」，论文可长期沉淀。"),
        ("可信度：", "多篇对比从「只谈一篇」变为「多篇各有代表」，回答更接近真实综述需求。"),
        ("安全性：", "不同工作区拥有独立房间，同主题不再串数据，降低误删与越权风险。"),
        ("性能：", "重排模型、检索索引、数据库连接不再重复创建，问答等待更稳定。"),
        ("可维护性：", "失败不再静默；诊断信息可定位到具体房间号，排障从「猜」变为「看」。"),
        ("数据治理：", "历史碎片可一键去重归并，为后续扩展（更多论文、更多用户）打好地基。"),
    ]
    for k, v in improve:
        bullet(v, k)

    H("4.2　经验总结", 2)
    lessons = [
        ("第一，配置的「可复现」比「临时能跑」更重要。",
         "随机标识让系统短期可用，却造成了数据永久漂移。凡是会进入数据主键的东西，必须是确定性的。"),
        ("第二，失败必须可见。",
         "本次「点了没反应」的根本原因不是没有错误，而是错误被静默吞掉了。可观测性和功能同等重要。"),
        ("第三，要防御版本差异。",
         "同一个库的新旧两代接口返回结构不同（字典与对象、字符串与分页响应），不做兼容就会在真实环境里静默失效。"),
        ("第四，数据治理要尽早。",
         "重复上传会在不知不觉中制造数千条副本；用内容指纹去重，比事后人工清理省力得多。"),
        ("第五，用「用户视角」验收。",
         "技术上「写了数据」不等于用户「看得见数据」。验收标准应是用户能否刷新、重开、换设备后继续使用。"),
    ]
    for k, v in lessons:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.3)
        r = p.add_run(k + v)
        fix_font(r, size=10.5)
        p.paragraph_format.space_after = Pt(6)

    H("4.3　后续优化方向", 2)
    bullet("升级为更强的中英双语向量模型；关键词检索改为按词切分，提升中文专业术语召回。", "检索质量：")
    bullet("建立小规模评测集，用召回率、排序质量、引用忠实度量化调参效果，避免凭感觉优化。", "评估体系：")
    bullet("引入「证据不足则主动补检索 / 追问」的自纠机制，并细化为句子级引用溯源。", "生成质量：")
    bullet("为纯扫描件接入 OCR，支持表格与公式结构化，进一步降低「空壳入库」概率。", "解析能力：")

    # ============================================================ 第五章
    doc.add_page_break()
    H("第五章　佐证材料清单（用于作业提交）", 1)
    P("按课程作业要求，建议随本报告一并提交以下截屏 / 录屏 / 链接作为进度佐证：", size=10.5)
    evidence = [
        ("截图 1", "优化前「对比全部论文」的参考片段列表：可见 5 段全部来自同一篇论文。"),
        ("截图 2", "云端房间分布（9 个 namespace 与各自记录数），对应本报告图 2 的数据来源。"),
        ("截图 3", "新建主题时不再弹出长红字报错，界面安静通过。"),
        ("截图 4", "点击「迁移旧数据」后显示「写入 N 个唯一切片」，且无 TypeError。"),
        ("录屏 1", "刷新 / 重开页面后，论文列表自动恢复的完整过程（建议 15–30 秒）。"),
        ("链接 1", "项目 GitHub 仓库地址与 Streamlit 线上访问地址。"),
    ]
    for who, desc in evidence:
        p = doc.add_paragraph(style="List Bullet")
        r1 = p.add_run(who + "：")
        fix_font(r1, size=10.5, bold=True, color=ACCENT2)
        r2 = p.add_run(desc)
        fix_font(r2, size=10.5)
        p.paragraph_format.space_after = Pt(5)
    P("提交前请确保截图中不包含任何 API Key 等敏感信息。", size=9, color=MUTED, before=8)

    out = os.path.join(BASE, "AI学术文献研读助手_优化专项报告.docx")
    doc.save(out)
    print("SAVED:", out)


if __name__ == "__main__":
    main()
