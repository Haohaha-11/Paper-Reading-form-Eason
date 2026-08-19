# Repository README Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the repository root README into a professional personal literature-library homepage and add a cartoon-robot SVG banner.

**Architecture:** Keep detailed paper catalogs in topic READMEs while the root page becomes a concise research map. The SVG remains a repository-native asset composed only of gradients, shapes, paths, and text.

**Tech Stack:** GitHub-flavored Markdown, SVG 1.1, Mermaid, Python standard-library validation.

---

### Task 1: Redesign the SVG banner

**Files:**
- Modify: `banner.svg`

- [ ] Replace the text-only right side with two accessible cartoon research robots.
- [ ] Preserve the 1200×260 viewBox, dark research palette, and readable left-side title.
- [ ] Parse with `xml.etree.ElementTree` and verify all IDs referenced by `url(#...)` exist.

### Task 2: Rebuild the root README

**Files:**
- Modify: `README.md`

- [ ] Add title, personal-library positioning, badges, and repository metrics.
- [ ] Add the batch-reading workflow and note anatomy.
- [ ] Add featured research collections and recent updates.
- [ ] Group all 18 topics into four research domains with counts and links.
- [ ] Replace stale daily-reading and long paper tables with topic-level navigation.

### Task 3: Validate and review

**Files:**
- Verify: `README.md`
- Verify: `banner.svg`

- [ ] Resolve every local Markdown link with URL decoding.
- [ ] Confirm all 18 topic directories appear in the README and counts total 125 paper-note directories.
- [ ] Run `git diff --check` and inspect the rendered hierarchy for duplication or missing sections.
