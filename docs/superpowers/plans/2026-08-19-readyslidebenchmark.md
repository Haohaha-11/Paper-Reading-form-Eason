# ReadySlideBenchmark Paper Reading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a self-contained five-paper ReadySlideBenchmark topic for pathology FM selector–consumer research.

**Architecture:** The main thread prepares all source assets, copies EAGLE, and owns shared integration files. Independent workers write only their assigned paper README and section notes; the main thread performs cross-paper synthesis, QA, commits, and the direct `main` push.

**Tech Stack:** Bash, MinerU API v4, Markdown, Mermaid, Semantic Scholar API, Git.

---

### Task 1: Establish clean scope and source assets

**Files:**
- Create: `topics/ReadySlideBenchmark/`
- Create: four new paper directories and MinerU assets
- Copy: existing EAGLE paper directory to `[Nat Commun 2026] EAGLE/`

- [ ] **Step 1:** Verify `git status --short` is clean and record HEAD.
- [ ] **Step 2:** Create the five target directories without modifying existing topics.
- [ ] **Step 3:** Download the four arXiv PDFs and submit them to MinerU with formula/table extraction enabled.
- [ ] **Step 4:** Poll all jobs, download ZIP results, and verify every new directory contains `paper.pdf`, `full.md`, `content_list.json`, and `images/`.
- [ ] **Step 5:** Copy the complete existing EAGLE directory into the new topic and normalize its new title/path references.

### Task 2: Produce five isolated paper reads

**Files:**
- Create/modify only each assigned paper's `README.md` and `sections/*.md`

- [ ] **Step 1:** Split each `full.md` by the paper's original top-level sections without omitting source text.
- [ ] **Step 2:** Add concrete Chinese `claude 批注` around mechanisms, equations, figures, tables, results, limitations, and ReadySlide implications.
- [ ] **Step 3:** Use MinerU figure/table/equation images, GitHub-safe relative paths, and full multi-panel figures.
- [ ] **Step 4:** Add all required Chinese README sections, Mermaid flowchart, Q&A, and Semantic Scholar Citation Landscape.
- [ ] **Step 5:** For EAGLE, add explicit selector–consumer, fixed-pairing, budget, and real-efficiency analysis while keeping its existing complete read.

### Task 3: Integrate the benchmark topic

**Files:**
- Create: `topics/ReadySlideBenchmark/README.md`
- Modify: `README.md`

- [ ] **Step 1:** Add a five-paper table and recommended reading order.
- [ ] **Step 2:** Add the selector–consumer–budget role matrix and full-bag-vs-budgeted ranking questions.
- [ ] **Step 3:** Add unsafe novelty claims, defensible gaps, and a four-stage ReadySlide contribution ordering.
- [ ] **Step 4:** Add the new topic and all five papers to the repository root README with encoded links.

### Task 4: QA, commit, and publish

**Files:**
- Verify all new/modified ReadySlideBenchmark files plus root `README.md`

- [ ] **Step 1:** Check source-text coverage, required README headings, annotation language/author, Mermaid use, math syntax, and absence of absolute paths.
- [ ] **Step 2:** Resolve every Markdown image and local link; inspect multi-panel figures and table images.
- [ ] **Step 3:** Confirm `git diff --stat` and `git status --short` contain no unrelated files.
- [ ] **Step 4:** Commit source/notes in intentional paper-sized commits and commit shared integration separately.
- [ ] **Step 5:** Push `HEAD:main`, then verify the public remote `main` SHA equals local HEAD.
