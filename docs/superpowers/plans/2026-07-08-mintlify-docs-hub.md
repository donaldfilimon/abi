# Mintlify Docs Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `docs/index.mdx` bullet hub with a Mintlify-native Card hub and align `docs/docs.json` navigation, without inventing capabilities or expanding CLI/MCP surfaces.

**Architecture:** Docs-only change. Index becomes a thin navigation shell that links to existing `docs/contracts/*`, `docs/spec/*`, and root guides. Executable contracts (`src/cli/usage.zig`, `src/mcp/handlers.zig`, `tests/contracts/`) remain source of truth.

**Tech Stack:** Mintlify MDX (`docs.json` schema), Markdown, optional `npx mint@latest validate`.

## Global Constraints

- Frozen CLI: exactly 13 top-level commands — do not document new ones.
- Frozen MCP: exactly 12 tools — do not invent tools.
- External claims: no QPS/latency/accuracy, sharding, AES/RBAC, K8s/H100, audited FHE, native ANE/CUDA kernel execution as current capabilities.
- Source wins: when prose conflicts with `build.zig` / `src/` / `tests/`, fix prose.
- Do not edit generated `src/plugin_registry.zig`.
- Do not revert or include unrelated dirty work (other skill diffs, `mcp/launcher.sh`, etc.) in commits for this plan.
- Prefer branch/worktree over bare `main` for implementation commits (SDD rule).
- Primary code gate if anything claim-adjacent moves: `./build.sh check`. Docs gate: `npx mint@latest validate`.

## File map

| File | Role |
|------|------|
| `docs/index.mdx` | Hub content (Cards, Note, groups) |
| `docs/docs.json` | Navigation groups aligned with hub |
| `docs/superpowers/specs/2026-07-08-mintlify-docs-hub-design.md` | Design reference (read-only during impl) |
| `tasks/todo.md` | Mark Phase 2 docs-hub row progress only if slice closes |

---

### Task 1: Redesign `docs/index.mdx` hub

**Files:**
- Modify: `docs/index.mdx`
- Read: `docs/contracts/public-api.mdx` (first 40 lines for accurate link labels), `docs/contracts/external-claims-audit.mdx` (title only)
- Test: visual MDX validity + mint validate in Task 3

**Interfaces:**
- Consumes: existing page paths already linked from current index
- Produces: Card-based hub; no new pages required

- [ ] **Step 1: Read current index and docs.json**

```bash
sed -n '1,50p' docs/index.mdx
python3 -m json.tool docs/docs.json > /dev/null && echo JSON_OK
```

Expected: JSON_OK; index is flat bullets.

- [ ] **Step 2: Replace body of `docs/index.mdx` with Card hub**

Keep frontmatter `title` / `description`. Body should:

1. One short welcome paragraph (orchestration framework; Zig 0.17).
2. A Mintlify warning Note for source wins pointing at `contracts/external-claims-audit`.
3. `CardGroup` cols={2} groups roughly:
   - **Contracts** → `contracts/public-api`, `contracts/external-claims-audit`
   - **Architecture / roadmap** → `spec/wdbx-north-star`, `spec/agent-wdbx-architecture`, `spec/abi-refactor-design`, `spec/multi-persona-technical`
   - **Partial design extracts** (label as PARTIAL) → `spec/sea-design-extract`, `spec/wdbx-rust-capability-extract`
4. Short **Build** section (prose, not fake metrics): link to `../README.md`, `../walkthrough.md`; mention `./build.sh check` and `./build.sh full-check`.
5. Short **Instruction files** links: `../AGENTS.md`, `../CLAUDE.md`, `../GEMINI.md`, `../CHANGELOG.md`.
6. **Historical** note: archive under `superpowers/archive/` is not active contract; link archive README only.

Use Mintlify components (valid in this docs site):

```mdx
<Note type="warning">
  **Source wins.** When docs disagree with `build.zig`, `src/cli/usage.zig`,
  `src/mcp/handlers.zig`, or `tests/contracts/`, trust the executable sources
  and `./build.sh check`. Claim boundaries:
  [External Claims Audit](contracts/external-claims-audit).
</Note>

<CardGroup cols={2}>
  <Card title="Public API Contract" href="contracts/public-api">
    Frozen CLI + MCP surfaces and honest wording.
  </Card>
  <Card title="External Claims Audit" href="contracts/external-claims-audit">
    What public collateral may and may not claim.
  </Card>
</CardGroup>
```

**Do not** paste full 13-command or 12-tool enumerations on the index — link to public-api instead.

- [ ] **Step 3: Grep index for claim red flags**

```bash
rg -n 'QPS|sharding|AES-256|HIPAA|H100|12,000|production multi-host' docs/index.mdx || echo CLEAN
```

Expected: `CLEAN` (no matches).

- [ ] **Step 4: Commit (on feature branch)**

```bash
git add docs/index.mdx
git commit -m "$(cat <<'EOF'
docs: redesign Mintlify index as Card hub

Point visitors at contracts and specs without re-listing frozen surfaces
or inventing capabilities.
EOF
)"
```

---

### Task 2: Align `docs/docs.json` navigation

**Files:**
- Modify: `docs/docs.json`
- Test: `python3 -m json.tool docs/docs.json`

**Interfaces:**
- Consumes: page slugs from Task 1 links (Mintlify paths omit `.mdx`)
- Produces: nav groups Overview / Architecture / Specs / Contracts

- [ ] **Step 1: Update navigation groups**

Ensure groups include at least:

```json
{
  "navigation": {
    "groups": [
      {
        "group": "Overview",
        "pages": ["index"]
      },
      {
        "group": "Architecture",
        "pages": [
          "spec/abi-refactor-design",
          "spec/agent-wdbx-architecture",
          "spec/multi-persona-technical",
          "spec/wdbx-north-star"
        ]
      },
      {
        "group": "Specs (partial extracts)",
        "pages": [
          "spec/sea-design-extract",
          "spec/wdbx-rust-capability-extract"
        ]
      },
      {
        "group": "Contracts",
        "pages": [
          "contracts/public-api",
          "contracts/external-claims-audit"
        ]
      }
    ]
  }
}
```

Do **not** add `superpowers/archive/*` pages to active navigation.

- [ ] **Step 2: Validate JSON**

```bash
python3 -m json.tool docs/docs.json > /dev/null && echo JSON_OK
```

Expected: `JSON_OK`

- [ ] **Step 3: Commit**

```bash
git add docs/docs.json
git commit -m "$(cat <<'EOF'
docs: align docs.json nav with Mintlify hub groups
EOF
)"
```

---

### Task 3: Validate mint + claims boundary

**Files:**
- Test only (no code unless mint reports a real path error)
- Optional read: `tests/contracts/public_docs.zig` if claim prose was edited outside index

- [ ] **Step 1: Mint validate**

```bash
npx mint@latest validate
```

Expected: exit 0. If Node/mint unavailable, record the error in the SDD report and verify:

```bash
# every href target without ../ must exist as docs/<path>.mdx
test -f docs/contracts/public-api.mdx && test -f docs/contracts/external-claims-audit.mdx
test -f docs/spec/wdbx-north-star.mdx && echo PATHS_OK
```

- [ ] **Step 2: Optional public_docs contract if claim sentences were added**

Only if Task 1 introduced new capability sentences (should not):

```bash
zig build test-contracts -Dtest-filter="public_docs"
```

- [ ] **Step 3: Repo gate if any non-docs file was touched (should be none)**

```bash
./build.sh check
```

Expected: skip if docs-only; if run, 39/39 style success.

- [ ] **Step 4: Update tracker one-liner if slice closes**

In `tasks/todo.md`, under modern-refactor Phase 2–4 notes, add that **docs hub Card redesign landed** (do not claim tools-split or flag-matrix done).

- [ ] **Step 5: Final commit if todo changed**

```bash
git add tasks/todo.md
git commit -m "$(cat <<'EOF'
docs: note Mintlify hub redesign in active board
EOF
)"
```

---

## Self-review (plan author)

| Spec requirement | Task |
|------------------|------|
| Card hub index | Task 1 |
| docs.json alignment | Task 2 |
| No archive in active nav | Task 2 |
| Claims honesty | Task 1 grep + Task 3 |
| Gates | Task 3 |
| No placeholders / TBD | none |

## Execution handoff

Plan complete at `docs/superpowers/plans/2026-07-08-mintlify-docs-hub.md`.

**1. Subagent-Driven (recommended)** — fresh implementer per task + review

**2. Inline Execution** — executing-plans in this session

Also required: feature branch or worktree (not bare `main` without explicit consent).
