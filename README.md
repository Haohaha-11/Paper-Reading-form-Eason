# Haojiang Paper Reading

这是从 `/mnt/eason/paper-reading` 抽出的个人论文阅读工作区，只保留阅读范式、skill 和空模板，不包含原来的论文文章、PDF、图片解析结果或 Git 历史。

## 目录结构

```text
paper-reading/
├── README.md                         # 本说明
├── README.original.md                # 原仓库工作流说明备份
├── skills/PAPER-READER-SKILL.md      # 原论文批读 skill
├── templates/paper-note/             # 新论文阅读笔记模板
├── inbox/                            # 待处理 PDF/链接/任务
└── topics/                           # 按研究主题组织自己的论文笔记
```


## 当前课题

| 课题 | 论文数 | 说明 |
|------|--------|------|
| [One-Step Diffusion Super-Resolution](topics/One-Step-Diffusion-Super-Resolution/) | 5 | one-step diffusion / flow based image super-resolution，覆盖 OSEDiff、OFTSR、TinySR、RCOD-SR、CODSR |
| [VisMem for Med Image](topics/VisMem-for-Med-Image/) | 5 | latent vision memory / latent reasoning / medical image VLM，覆盖 VisMem、MedSynapse-V、AlignVLM 等 |

## 推荐流程

1. 在 `topics/<topic>/[Venue Year] PaperName/` 创建一篇新论文目录。
2. 放入 MinerU 或其他工具解析出的 `full.md`、`content_list.json`、`images/`。
3. 参考 `skills/PAPER-READER-SKILL.md` 进行批读：保留原文、按 section 拆分、内嵌批注。
4. 用 `templates/paper-note/README.md` 和 `templates/paper-note/sections/*.md` 作为空白骨架。
5. 完成后在 topic 目录维护综述、方法对比和可复用 insight。

## 不包含内容

- 原仓库中的具体论文目录，例如 `[Arxiv ...]`、`[ICLR ...]`。
- 论文 PDF、MinerU 图片、`full.md`、`layout.json` 等解析结果。
- `.git` 历史和 submodule 内容。
