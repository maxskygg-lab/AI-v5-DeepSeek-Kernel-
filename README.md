# 📖 AI 深度研读助手 v5（DeepSeek Kernel）

基于 **RAG + 混合检索重排** 的学术文献检索与深度研读工具。面向「找论文 → 读论文 → 问论文 → 记笔记」的完整科研流程，
把 arXiv 检索、Semantic Scholar 引用数据、向量入库、多论文对比问答、文献关系图谱整合在一个 Streamlit 页面里。

> 详细目录规范、命名约定与完整文件树见 [docs/README.md](docs/README.md)。

---

## 核心功能

| 标签页 | 能力 | 说明 |
|--------|------|------|
| 🔍 学术检索 & 图谱 | 关键词检索、权重调节、引用量/质量排序、关系图谱 | 语义相关度与引用量可在 0–100% 间自由配比；节点图展示引用与关联文献 |
| 📖 研读空间 | PDF 入库、单篇精读、多篇对比、自由问答 | 按「研究主题」隔离的独立向量空间，支持跨论文对比与结构化综述生成 |
| 📌 我的笔记 | 标签化笔记、按主题/标签筛选 | 研读过程中的结论可沉淀为可检索笔记 |

**研读空间的关键设计**

- **主题级空间隔离**：每个主题对应一个独立的 Pinecone namespace，数据互不干扰
- **混合检索**：BM25 稀疏检索 + Dense 向量检索，经 RRF 融合后再用 Cross-Encoder 精排
- **跨论文覆盖**：对比模式下按论文分组取段，保证结论覆盖所有已入库论文而非单篇
- **持久化恢复**：重开页面自动从云端恢复本主题已入库论文，也可手动迁移历史散落数据
- **笔记云端保存**：笔记写入独立 namespace，刷新/换设备后自动恢复；网络失败进入重试队列

---

## 技术栈

- **界面**：Streamlit、streamlit-agraph（关系图谱）
- **大模型**：DeepSeek（`deepseek-chat`，OpenAI 兼容接口）
- **向量库**：Pinecone Serverless（`pinecone>=5.0.0`、`langchain-pinecone`）
- **Embedding / 重排**：`sentence-transformers/all-MiniLM-L6-v2`、`cross-encoder/ms-marco-MiniLM-L-6-v2`
- **检索融合**：`rank-bm25` + RRF
- **数据源**：arXiv API、Semantic Scholar
- **文档解析**：PyMuPDF（主解析失败时回退）

---

## 项目结构

```text
AI-v5-DeepSeek-Kernel-/
├── app.py                  # Streamlit 主入口（初始化与路由）
├── services/               # 业务服务层（配置、向量库、笔记、PDF、检索、LLM）
├── ui/                     # 页面层（布局、侧边栏、三个 Tab、图谱）
├── tests/                  # pytest 单元测试
├── pytest.ini              # pytest 配置
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
├── .env.example            # 环境变量样例
├── .github/workflows/      # CI：black + pyflakes + pytest
├── .streamlit/             # 运行配置与 secrets（已 gitignore）
├── docs/                   # 过程文档与交付物 → 见 docs/README.md
├── legacy/                 # 已停用脚本（仅供追溯）
├── .devcontainer/  .vscode/
```

约定：`app.py` 与 `requirements.txt` **必须保留在仓库根目录**，Streamlit Cloud 以此解析入口与依赖；
业务逻辑放 `services/`，文档、交付物放 `docs/`，废弃脚本放 `legacy/`，后两者都不参与运行时。

---

## 快速开始

```bash
pip install -r requirements.txt
streamlit run app.py
```

### 开发校验

```bash
pytest -q
black --check --line-length 100 app.py services ui tests
pyflakes app.py services ui tests
```

以上命令与 `.github/workflows/ci.yml` 一致，提交前本地跑通即可。
日志级别通过环境变量 `LOG_LEVEL` 控制，默认 `INFO`；排查问题时可设为 `DEBUG`。

### 配置密钥

优先读取 `st.secrets`，其次读取系统环境变量。按 `.env.example` 中的键名，在
`.streamlit/secrets.toml`（本地，TOML 格式，值需加引号）或 Streamlit Cloud 的 Secrets 面板中填写：

| 键 | 必填 | 用途 |
|----|------|------|
| `DEEPSEEK_API_KEY` | 是 | DeepSeek 对话与打分 |
| `PINECONE_API_KEY` | 是 | 云端向量库读写 |
| `PINECONE_INDEX_NAME` | 否 | 索引名，默认 `arxiv-papers` |
| `SS_API_KEY` | 否 | Semantic Scholar 引用数据，缺失时降级 |
| `WORKSPACE_ID` | 否 | 稳定工作区标识；未配置时按登录邮箱或固定值兜底 |

> `WORKSPACE_ID` 决定 namespace 前缀，**一旦启用就不要随意更改**，否则已入库数据会「找不到」。
> secrets 文件含真实密钥，禁止提交到仓库。

---

## 常见维护操作

- **重开页面后论文消失**：侧边栏点「🔄 从云端恢复本主题」
- **数据散落在旧 namespace**：点「🛠 迁移旧数据到当前主题」，会按内容指纹去重后归并
- **切换主题后看不到之前进度**：确认当前主题名与上传时一致，namespace 由「工作区标识 + 主题名」生成
- **PDF 解析为空**：多为扫描版，系统会用 PyMuPDF 回退；仍为空则拒绝入库并提示
- **笔记刷新后不见了**：笔记页点「🔄 同步云端笔记」；若提示失败，内容仍保留在当前页面，稍后重试即可
- **图谱提示 HTTP 403**：`SS_API_KEY` 已失效，重新申请后填入 Secrets 并重启；未配置 Key 时走匿名请求，可能被限流（429）

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/README.md](docs/README.md) | 目录用途、完整文件树、命名规范、存放规则 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 后续发展方向、里程碑与验收标准 |
| [docs/optimization-report/README.md](docs/optimization-report/README.md) | 优化专项报告（网页版 / Word 版）说明与再生成方式 |
| [legacy/README.md](legacy/README.md) | 已停用脚本清单及停用原因 |
