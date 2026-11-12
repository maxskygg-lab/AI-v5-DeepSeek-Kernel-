# 历史遗留脚本（已停用）

本目录存放已停止使用、但暂未删除的脚本，仅作追溯用途。**任何脚本都不应被 `app.py` 引用。**

## 文件清单

| 文件 | 原用途 | 停用原因 |
|------|--------|----------|
| `offline_builder.py` | 从 arXiv 拉取论文并构建本地 FAISS 向量库（`my_semantic_paper_db/`） | 项目已改用 Pinecone 云端向量库，`app.py` 中的 FAISS 导入已移除；脚本依赖的 `langchain_community.vectorstores.FAISS` 与当前检索链路不兼容，且 `from app import ...` 会连带执行 Streamlit 页面代码，无法独立运行 |

## 处理规则

1. 本目录文件已纳入 Git 历史，如需彻底删除可直接删除，需要时用 `git log -- legacy/offline_builder.py` 找回。
2. 不要把仍在使用的脚本移入本目录；判断标准是「`app.py` 是否有引用」。
3. 本目录内容不参与部署，也不应出现在 `requirements.txt` 的依赖判断中。
