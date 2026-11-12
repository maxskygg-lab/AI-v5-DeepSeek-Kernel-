# 文档目录说明

本目录存放项目的过程文档与交付物（报告、图表、生成脚本），不参与应用运行时。

## 目录用途

| 路径 | 用途 |
|------|------|
| `ROADMAP.md` | 后续发展方向、待优化点、里程碑与验收标准 |
| `optimization-report/` | 课程汇报用「项目优化专项报告」（网页版 + Word 版 + 生成脚本） |

## 存放规则

1. 面向人阅读的成果（报告、图表、导出文件）放本目录，按「主题」建子目录。
2. 每个主题子目录必须包含自己的 `README.md`，写清用途、文件清单、再生成方式。
3. 生成脚本放在主题子目录的 `scripts/` 下，不放在项目根目录。
4. 第三方前端库放在 `_shared/` 下；不要放进 `scripts/` 或 `assets/`。
5. 不要在本目录存放临时导出、截图草稿和中间产物。

## 完整项目结构

```text
AI-v5-DeepSeek-Kernel-/
├── app.py                          # Streamlit 主入口（初始化 + 路由）
├── services/                       # 业务服务层（不依赖页面渲染）
│   ├── config.py                   # Secrets / 环境变量读取
│   ├── embeddings.py               # Embedding 与 Cross-Encoder
│   ├── vectorstore.py              # namespace、云端恢复、旧数据迁移
│   ├── notes.py                    # 笔记云端持久化
│   ├── pdf.py                      # PDF 下载、解析与入库
│   ├── retrieval.py                # BM25 + Dense + RRF + 重排
│   ├── semantic_scholar.py         # Semantic Scholar 请求层
│   ├── llm.py                      # DeepSeek 调用与打分
│   ├── log.py                      # 统一日志（LOG_LEVEL 控制级别）
│   └── utils.py                    # arXiv ID、Excel 导出
├── ui/                             # 页面层
│   ├── layout.py                   # 页面配置、主题 CSS、标题
│   ├── sidebar.py                  # 控制台、主题管理与 PDF 上传
│   ├── search_tab.py               # Tab 1：学术检索与图谱
│   ├── read_tab.py                 # Tab 2：研读空间
│   ├── notes_tab.py                # Tab 3：笔记库
│   └── graph.py                    # 文献关联图谱渲染
├── tests/                          # pytest 单元测试
├── pytest.ini                      # pytest 配置
├── requirements.txt                # Python 依赖清单（Streamlit Cloud 从根目录读取）
├── README.md                       # 项目说明
├── .env.example                    # 环境变量样例（不含真实密钥）
├── .gitignore                      # Git 忽略规则
├── .github/workflows/ci.yml        # CI：black + pyflakes + pytest
├── .streamlit/                     # Streamlit 运行配置与 secrets（已 gitignore）
├── .devcontainer/                  # Dev Container 开发环境配置
├── .vscode/                        # 编辑器任务与设置
├── docs/                           # 过程文档与交付物（不参与运行）
│   ├── README.md                   # 本文件：文档目录规范 + 全项目结构说明
│   ├── ROADMAP.md                  # 后续发展规划与里程碑
│   └── optimization-report/        # 优化专项报告（网页 / Word / 生成脚本）
│       ├── README.md
│       ├── AI学术文献研读助手_优化专项报告.html
│       ├── AI学术文献研读助手_优化专项报告.docx
│       ├── assets/                 # 图表脚本与导出的 PNG
│       ├── scripts/                # build_docx.py 报告生成脚本
│       └── _shared/js/             # ECharts / Mermaid 第三方库（HTML 相对引用）
└── legacy/                         # 已停用但暂未删除的历史脚本
    ├── README.md
    └── offline_builder.py
```

## 命名规范

| 类型 | 规则 | 示例 |
|------|------|------|
| Python 应用代码 | 小写 + 下划线，入口固定为 `app.py` | `app.py` |
| 脚本 / 工具 | 小写 + 下划线，动词开头 | `build_docx.py` |
| 交付物（报告、导出文件） | 中文全称，同名不同格式保持一致 | `AI学术文献研读助手_优化专项报告.html/.docx` |
| 配置 / 元数据 | 小写，点号前缀 | `.env.example`、`.gitignore` |
| 目录 | 小写 + 连字符 | `optimization-report`、`_shared` |

补充约定：

- 不使用 `temp`、`new`、`final`、`v2`、`副本` 等无信息量的命名。
- 同一份内容的不同格式必须共用同一个主文件名，只改扩展名。
- 文件名不得包含空格；中文文件名仅用于对外交付物。

## 开发须知

- 不要移动 `app.py` 与 `requirements.txt`：Streamlit Cloud 从仓库根目录解析入口和依赖。
- `docs/`、`legacy/` 与运行时代码无引用关系，可整目录迁移或删除，不会影响应用启动。
- 生成报告前先确认 `docs/optimization-report/assets/` 可写（脚本会写入两张 PNG）。
