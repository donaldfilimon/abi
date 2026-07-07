# MkDocs → Mintlify Migration Plan

> **Archived:** The Mintlify migration is complete. This plan remains as historical implementation context only.

> **Status: COMPLETED** — all tasks (docs.json, .mdx conversion, mkdocs.yml removal, instruction-file sync) have been executed. This file is kept as a historical record of the migration plan.

## Context / why

The user asked to move abi's documentation from **MkDocs** (Material theme, `mkdocs.yml`, `strict: true`, built with `mkdocs build`) to **Mintlify**. Mintlify's model: a repo containing a `docs.json` (site settings + navigation) and one `.md`/`.mdx` per page, built/hosted by Mintlify's GitHub app. MkDocs is a *manual* migration on Mintlify (the `@mintlify/scraping` auto-tool only supports Docusaurus/ReadMe).

## What research found

- **Content is tiny and clean:** 11 `.md` files under `docs/` (`index.md`, `spec/`, `contracts/`, `superpowers/`), **0 admonition blocks** (`!!!`), **1 mermaid** diagram (Mintlify renders mermaid natively). No MkDocs-specific syntax (`pymdownx.*`, snippets, `attr_list`) is actually exercised in the content — the extensions are enabled but unused.
- **Published nav is only 2 pages** (`mkdocs.yml` `nav:`): `Home → index.md` and `Architecture → ABI Refactor Design → spec/abi-refactor-design.md`. The other 9 files exist but aren't in nav.
- **Excludes:** `mkdocs.yml` excludes `README.md`, `skills/**`, `submodules/**`. `docs/superpowers/` (these plan files) is internal and should also be excluded from published docs.
- **Theme:** Material, palette `primary: teal`, `accent: deep orange`. Maps to Mintlify `theme: "mint"` + `colors.primary` (teal `#009688`), `colors.light/dark` accents.
- **CI:** `mkdocs build` is NOT in CI (per CLAUDE.md) — it's a manual/local gate. Mintlify builds via its GitHub app; local preview is `mint dev`, validation `mint validate`.

## Key decision to confirm before executing

**Cutover vs coexist.** Recommend a **clean cutover**: Mintlify replaces MkDocs (remove `mkdocs.yml`, `requirements-docs.txt` mkdocs note, `docs/stylesheets/`, and update the three instruction files). Running both doubles maintenance and confuses contributors. *If you'd rather keep MkDocs and add Mintlify alongside, say so — that changes Tasks 4–5.*

## Tasks

### Task 1: Create `docs.json`
- Create `docs/docs.json` (Mintlify reads it at the docs root):
  ```json
  {
    "$schema": "https://mintlify.com/docs.json",
    "theme": "mint",
    "name": "ABI Framework",
    "colors": { "primary": "#009688", "light": "#4db6ac", "dark": "#00695c" },
    "navigation": {
      "groups": [
        { "group": "Overview", "pages": ["index"] },
        { "group": "Architecture", "pages": ["spec/abi-refactor-design"] },
        { "group": "Contracts", "pages": ["contracts/external-claims-audit"] }
      ]
    },
    "navbar": { "links": [{ "label": "GitHub", "href": "https://github.com/donaldfilimon/abi" }] },
    "footer": { "socials": { "github": "https://github.com/donaldfilimon/abi" } }
  }
  ```
  (Exact `pages` list finalized in Task 2 once each file's path/slug is confirmed. `superpowers/**` deliberately excluded.)
- **Verify:** `npx mint@latest validate` (or `mint validate`) reports the config is valid.

### Task 2: Convert pages to Mintlify format
- For each published page (`index.md`, `spec/abi-refactor-design.md`, `contracts/external-claims-audit.md`, plus any others we decide to surface): ensure Mintlify **frontmatter** (`title:`, optional `description:`). MkDocs used the first `# H1` as the title; Mintlify uses frontmatter `title` and renders it, so **remove the now-duplicate leading `# H1`** to avoid a double title.
- Rename `.md` → `.mdx` (Mintlify-recommended; enables components later). Mermaid fenced blocks work unchanged.
- Confirm relative links between pages still resolve (Mintlify uses path-based, extension-less refs, e.g. `/spec/abi-refactor-design`).
- **Verify:** `mint dev` renders each page with a single title, working nav, and the mermaid diagram.

### Task 3: Assets
- If any page references local images/CSS: move images to `docs/images/` and update refs to `/images/...`. (`docs/stylesheets/extra.css` is Material-specific and does not port — Mintlify theming is via `docs.json`, so drop it.)
- **Verify:** no broken image refs in `mint dev`.

### Task 4: Retire MkDocs (cutover — only if confirmed above)
- Remove `mkdocs.yml`, `docs/stylesheets/`, and the `site/` output dir; drop mkdocs deps from `requirements-docs.txt` (or delete it if mkdocs-only).
- **Verify:** repo no longer references mkdocs in build docs.

### Task 5: Sync instruction files
- Update `CLAUDE.md` (the `pip install -r requirements-docs.txt && mkdocs build` line → the Mintlify local preview `mint dev` / `mint validate`), and mirror into `AGENTS.md` and `GEMINI.md` (the three must stay in sync — use the `instruction-sync` agent to confirm).
- **Verify:** `instruction-sync` agent reports no drift on the docs-build convention.

## Verification (end-to-end)
```bash
cd /Users/donaldfilimon/abi/docs
npx mint@latest validate          # docs.json + pages valid
npx mint@latest dev               # local server; click Home → Architecture → Contracts, confirm mermaid renders, single titles
```
Green `validate` + a clean `mint dev` walk-through of every nav entry = done. (Mintlify's own build/deploy happens via its GitHub app once `docs.json` lands on the default branch — that's a dashboard/GitHub-app setup step outside this repo change.)

## Out of scope / notes
- Connecting the Mintlify dashboard + custom domain is an account/GitHub-app step the user does in Mintlify's UI — not a code change here.
- `docs/superpowers/**` (plan files) stays internal, excluded from `docs.json` nav.
- This does not touch any Zig code, `build.zig`, or CI workflows.
