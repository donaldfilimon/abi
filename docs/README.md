# Documentation

> **Last reviewed:** 2026-02-14

This directory is the source for the ABI documentation site and project docs.

## Contents

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — doc build and layout |
| [api-reference.md](api-reference.md) | API overview and entry points |
| [content/migration-vnext.html](content/migration-vnext.html) | Legacy-to-vNext API migration guide |
| [deployment.md](deployment.md) | Deployment and production notes |
| [roadmap.md](roadmap.md) | Product and technical roadmap |
| [plan.md](plan.md) | Planning and design notes |
| [plans/](plans/) | Dated improvement and feature plans |

## Requirements

- Zig `0.16.0-dev.2611+f996d2866` or newer (match `.zigversion`).

## Build

Use the Zig toolchain pinned in `.zigversion`.

```bash
zvm upgrade
zvm install master
zvm use master
zig version
cat .zigversion
# If needed: export PATH="$HOME/.zvm/bin:$PATH"
```

Generate the docs site:

```bash
zig build docs-site
```

Output is written to `zig-out/docs/`.

Generate API reference only:

```bash
zig build gendocs
```

API docs are written to `docs/api/`. MkDocs is no longer used for the site.

## Layout

- **site.json** — Navigation and page metadata for the doc site
- **content/** — HTML fragments for each page
- **assets/** — CSS and JavaScript
- **api/** — Auto-generated API reference (`zig build gendocs`)
