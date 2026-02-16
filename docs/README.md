# Documentation

> **Last reviewed:** 2026-02-16

This directory is the source for the ABI documentation site.

## Contents

| Item | Description |
|------|-------------|
| `site.json` | Navigation manifest (40 pages, 10 sections) |
| `hub.html` | Landing page |
| `content/` | Markdown sources for all documentation pages |
| `assets/` | CSS and JavaScript |
| `api/` | Auto-generated API reference (`zig build gendocs`) |

## Requirements

- Zig `0.16.0-dev.2611+f996d2866` or newer (match `.zigversion`).

## Build

```bash
# Sync toolchain
zvm upgrade && zvm install master && zvm use master

# Generate the docs site
zig build docs-site
```

Output is written to `zig-out/docs/`.

Generate API reference only:

```bash
zig build gendocs
```

API docs are written to `docs/api/`.

## Layout

The site generator (`tools/docs_site/main.zig`) reads `site.json` and processes
each page's `.md` source into HTML with automatic TOC and search indexing.

### Sections (40 pages)

| Section | Pages | Topics |
|---------|-------|--------|
| Start | 3 | Home, Installation, Getting Started |
| Core | 5 | Architecture, Configuration, Framework, CLI, Migration |
| AI | 5 | Overview, Core, Inference, Training, Reasoning |
| GPU | 2 | Compute, Backends |
| Data | 4 | Database, Cache, Storage, Search |
| Infrastructure | 7 | Network, Gateway, Messaging, Pages, Web, Cloud, Mobile |
| Operations | 5 | Auth, Analytics, Observability, Deployment, Benchmarks |
| Services | 3 | Connectors, MCP, ACP |
| Reference | 6 | API, Examples, C Bindings, Troubleshooting, Contributing, Roadmap |

### Writing content

All pages use Markdown with YAML front matter:

```yaml
---
title: Page Title
description: One-line description
section: Section Name
order: 1
---
```

Cross-links between pages use `.html` extensions (the site generator does not
rewrite `.md` links). Example: `[Architecture](architecture.html)`.

The markdown parser supports: headers, code blocks, tables, bold, italic,
inline code, links, images, lists, blockquotes, and horizontal rules. It does
not support nested lists or admonitions.
