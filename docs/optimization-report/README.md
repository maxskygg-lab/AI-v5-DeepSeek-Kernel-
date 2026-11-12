# 优化专项报告（课程汇报交付物）

面向非技术背景听众的项目优化汇报材料，内容覆盖：项目背景与原始痛点、8 项优化方案、
前后对比数据、最终效果与经验总结。所有结论按「实测 / 设计 / 估算」标注口径。

## 文件清单

| 文件 | 用途 | 说明 |
|------|------|------|
| `AI学术文献研读助手_优化专项报告.html` | 网页版，用于投屏演示 | 单文件打开即可；含 ECharts 柱状图与 Mermaid 流程图 |
| `AI学术文献研读助手_优化专项报告.docx` | Word 版，用于提交 | A4 页面，122 段 / 1 张对比表 / 2 张图表，无自动目录 |
| `assets/charts.js` | 网页版图表配置 | 由 HTML 以相对路径 `assets/charts.js` 引用 |
| `assets/ns_chart.png` | Word 版图 2「优化前数据碎片分布」 | 由生成脚本输出 |
| `assets/cap_chart.png` | Word 版图 3「关键能力具备情况」 | 由生成脚本输出 |
| `scripts/build_docx.py` | 重新生成 Word 版 | 同时生成上面两张 PNG |
| `_shared/js/echarts.min.js` | 图表库 | 被 HTML 以 `./_shared/js/...` 相对引用 |
| `_shared/js/mermaid.min.js` | 流程图库 | 被 HTML 以 `./_shared/js/...` 相对引用 |

## 重新生成 Word 版

```bash
python docs/optimization-report/scripts/build_docx.py
```

脚本基于自身位置解析输出目录（`assets/` 与 `.docx` 均写在本目录），因此必须在本目录结构完整的前提下运行。
依赖：`python-docx`、`matplotlib`。

## 展示网页版

直接用浏览器打开 `.html` 文件即可。注意 `_shared/` 与 `assets/` 必须与 HTML 保持同级，
否则图表与流程图会空白——移动文件时请整目录移动。

## 维护规则

- 正文与图表数据改动后，网页版与 Word 版必须同步更新，避免两版结论不一致。
- `_shared/` 只放第三方库，不改其内容；自定义脚本一律放 `assets/` 或 `scripts/`。
- 本目录可整目录备份或迁移，不影响 `app.py` 的运行。
