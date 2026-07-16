# ABI Skill-Set Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every skill referenced by abi's config (`.opencode.json` skills array + `slash_commands` + `superpowers.plugins`) so each has a real `SKILL.md` body with valid frontmatter and honest-stub framing; remove the six dead repo-root `abi-superpower-*` orphans; and add a reusable frontmatter validator + final sync/gate.

**Architecture:** Canonical skills live in `.agents/skills/` (opencode reads them via the `.opencode/skills` symlink; the `sync-clis/launch.sh` syncer copies `SKILL.md` into `.claude/skills/` + `.grok/` for dirs that already exist there). Eight referenced skills are empty/stubbed (`opencode` has no file; `ai-plan`/`gpu`/`mcp`/`sea`/`tui`/`wdbx` have `name:`-only frontmatter + over-claiming bodies; `agent-status-reporter` has frontmatter only). We add a `tools/check_skills.sh` validator as the TDD anchor (it fails on the stubs today), author each skill so the validator goes green, delete the six repo-root orphans, then sync + run `./build.sh check`.

**Tech Stack:** Bash validator; Markdown `SKILL.md` with YAML frontmatter (`name` + `description`); abi companion smoke scripts (`*.sh`) where a skill drives a binary; `sync-clis/launch.sh` rsync-of-SKILL.md; Zig 0.17.0-dev.1398+cb5635714 for the final `./build.sh check`.

## Global Constraints

- Zig pinned by `.zigversion` to `0.17.0-dev.1398+cb5635714`; `build.sh` does NOT enforce the pin — select it with zvm/zigup first. Build from inside the repo: `./build.sh check`.
- Skill frontmatter: `name:` must equal the dir basename (kebab-case); `description:` must be non-empty, ≥20 chars, and include trigger phrases. No `TBD`/`TODO`/placeholder text anywhere.
- Honest-stub rule (verbatim from `docs/contracts/external-claims-audit.mdx`): shaders/MLIR "report `available=false`: no external shader compiler or MLIR/LLVM toolchain is linked. Do not claim real shader compilation or MLIR/LLVM lowering." WDBX demos must be "Present … honestly — **not** production multi-host distributed deployment or data sharding, **not** native local-accelerator (CUDA/Vulkan/Metal-kernel/ANE) execution, **not** a production/learned-SOTA compression codec, **not** production-secure or bootstrapped full FHE, and **not** production-ready non-loopback MCP/WDBX HTTP without TLS/authz/rate-limit review." "Do not claim WDBX encryption/RBAC." Trust `available`/`native_dispatch` flags in each `src/features/*/mod.zig` over any prose.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `chore(build):` …). Never force-push `main`.
- Sync: after authoring a canonical skill, run `.agents/skills/sync-clis/launch.sh` to propagate `SKILL.md` to `.claude/skills/` + `.grok/` (only for target dirs that already exist; the syncer does NOT create new target dirs). `.opencode/skills` is a symlink to `.agents/skills`, so opencode picks up edits with no sync.
- No energy/accuracy/QPS/latency claims anywhere without a repo test/benchmark proving them.
- `.skill-telemetry/registry.json` + `amendments.jsonl` are written by the external `skill-loop` MCP — do NOT hand-edit them.

---

## File Structure

- **Create:** `tools/check_skills.sh` — frontmatter validator (the reusable gate / "test" for every skill task).
- **Create:** `.agents/skills/opencode/SKILL.md` — opencode setup/allowlist skill (currently no file).
- **Rewrite (full body + description):** `.agents/skills/{ai-plan,gpu,mcp,sea,tui,wdbx}/SKILL.md` — six domain-index skills; current bodies are `name:`-only and over-claim.
- **Rewrite (add body):** `.agents/skills/agent-status-reporter/SKILL.md` — backs `/status`; currently frontmatter only.
- **Delete:** `abi-superpower-{ai,gpu,mcp,sea,tui,wdbx}/SKILL.md` + their six repo-root dirs — dead `name: superpowers` orphans, not a sync target, superseded by `.agents/skills/abi-superpower-*`.
- **Modify:** `tasks/todo.md` (add/close the skill-completion item), `CHANGELOG.md` (entry).
- Each canonical skill keeps its existing companion script where present (none of the 8 touched skills have one; they are routing/context skills, so no `.sh` is needed).

**Responsibility split:** `tools/check_skills.sh` is the single source of truth for "is this skill valid?" Every authoring task writes the full `SKILL.md`, then runs the validator on that one skill to prove green. The domain-index skills (`ai-plan`/`gpu`/`mcp`/`sea`/`tui`/`wdbx`) only route to the specialist skills that already exist and carry the heavy lifting — they must NOT duplicate or over-claim beyond what those specialists honestly support.

---

## Task 1: Add the skill frontmatter validator

**Files:**
- Create: `tools/check_skills.sh`

**Interfaces:**
- Produces: `tools/check_skills.sh` (exit 0 when every checked skill has `name` == dir and `description` ≥20 chars; exit non-zero with a per-skill FAIL list otherwise). Accepts an optional single skill-name arg to check one skill. This is the "test" every later task runs.

- [ ] **Step 1: Write the validator**

Create `tools/check_skills.sh` with this exact content:

```bash
#!/usr/bin/env bash
# Validate .agents/skills/*/SKILL.md frontmatter.
# Usage: tools/check_skills.sh [skill-name]   (no arg = check all canonical skills)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$ROOT/.agents/skills"
pass=0; fail=0

desc_of() { # file -> description value (single-line, or YAML folded '>' / '|')
  local f="$1"
  awk '
    /^---/{fm++; next}
    fm!=1{next}
    /^description:[ ]*/{
      v=substr($0, index($0,":")+1); sub(/^ */,"",v)
      if (v==">"||v=="|"){fold=1; next}
      print v; exit
    }
    fold{ if($0~/^[[:space:]]/){gsub(/^[[:space:]]+/,"");print;exit} else {print "";exit} }
  ' "$f"
}

check_one() {
  local name="$1" f="$DIR/$name/SKILL.md"
  if [ ! -f "$f" ]; then echo "FAIL $name: missing SKILL.md"; fail=$((fail+1)); return; fi
  local nm desc
  nm=$(awk -F': ' '/^name:/{print $2; exit}' "$f")
  desc=$(desc_of "$f")
  if [ -z "$nm" ] || [ "$nm" != "$name" ]; then
    echo "FAIL $name: name='$nm' must equal dir basename"; fail=$((fail+1)); return; fi
  if [ -z "$desc" ] || [ "${#desc}" -lt 20 ]; then
    echo "FAIL $name: description missing or <20 chars (${#desc})"; fail=$((fail+1)); return; fi
  echo "PASS $name (${#desc}c)"; pass=$((pass+1))
}

if [ "$#" -ge 1 ] && [ -n "${1:-}" ]; then
  check_one "$1"
else
  for d in "$DIR"/*/; do
    n=$(basename "$d"); [ "$n" = "sync-clis" ] && continue
    [ -d "$d" ] && check_one "$n"
  done
fi
echo "=== summary: pass=$pass fail=$fail ==="
[ "$fail" -eq 0 ]
```

- [ ] **Step 2: Make it executable and run it to verify it FAILS on the current stubs**

Run: `chmod +x tools/check_skills.sh && ./tools/check_skills.sh`
Expected: non-zero exit; output includes `FAIL opencode: missing SKILL.md`, `FAIL ai-plan: description missing or <20 chars (0)`, and the same FAIL line for `gpu`, `mcp`, `sea`, `tui`, `wdbx`, and `agent-status-reporter`. Tail: `=== summary: pass=<N> fail=8 ===` (8 = the seven `name:`-only/missing skills + `opencode`; `agent-status-reporter` already has a description so it currently PASSES — its body is added in Task 9 for correctness, not frontmatter).

- [ ] **Step 3: Commit**

```bash
git add tools/check_skills.sh
git commit -m "feat(tools): add skill frontmatter validator (tools/check_skills.sh)"
```

---

## Task 2: Author the `opencode` skill

**Files:**
- Create: `.agents/skills/opencode/SKILL.md` (the dir already exists; the file does not)

**Interfaces:**
- Consumes: facts from `.opencode.json`, `opencode.json`, `.mcp.json`, the `.opencode/skills` symlink.
- Produces: a discoverable `opencode` skill (opencode reads it via the `.opencode/skills`→`.agents/skills` symlink automatically).

- [ ] **Step 1: Write the failing-test run**

Run: `./tools/check_skills.sh opencode`
Expected: `FAIL opencode: missing SKILL.md` and non-zero exit.

- [ ] **Step 2: Write the SKILL.md**

Create `.agents/skills/opencode/SKILL.md` with this exact content:

```markdown
---
name: opencode
description: Set up and operate abi through opencode — how opencode discovers abi skills, slash-commands, and MCP servers. Use when the user asks about opencode config, why an abi skill is not loading in opencode, or how to add a skill to the opencode allowlist.
---

# opencode

opencode loads abi via three mechanisms wired in `.opencode.json` + `opencode.json`:

- **Skills**: `.opencode/skills` is a symlink to `../.agents/skills`, so every
  canonical skill dir is visible to opencode automatically. The `.opencode.json`
  `skills` array is the *allowlist* of which skills opencode advertises. To expose
  a new skill in opencode, add its dir basename to that array — no file copy.
- **Slash commands**: `.opencode.json` `slash_commands` maps `/open /diff /commit
  /context /features /learn /save /load /status /reset` to backing skills
  (`file-context-loader`, `git-diff-integration`, `git-commit-integration`,
  `context-state-reporter`, `feature-flag-display`, `sea-learning-controller`,
  `session-persister`, `session-restorer`, `agent-status-reporter`,
  `context-resetter`). Add a command by pointing `skill:` at a canonical name.
- **MCP servers**: `opencode.json` `mcp` wires `abi-mcp` (via
  `./mcp/launcher.sh stdio`) and `skill-loop` (the telemetry/registry engine).
  `.mcp.json` mirrors the same for other clients.

## To add a skill to opencode
1. Author/complete the skill under `.agents/skills/<name>/SKILL.md`.
2. Add `"<name>"` to the `.opencode.json` `skills` array.
3. Run `tools/check_skills.sh <name>` to confirm frontmatter is valid.
4. Skills auto-reload from disk — no restart needed.

## Gotchas
- `.opencode/skills` is a symlink; editing a skill there edits the canonical copy.
- `skill-loop` writes `.skill-telemetry/registry.json` + `amendments.jsonl`
  automatically (content-drift proposals). Do not hand-edit the registry.
- The `sync-clis/launch.sh` syncer does NOT create new target dirs in
  `.claude/skills/`/`.grok/`; it only updates `SKILL.md` for dirs that already
  exist there. opencode needs no sync (symlink), but Claude Code/grok do.
```

- [ ] **Step 3: Run the validator to verify it passes**

Run: `./tools/check_skills.sh opencode`
Expected: `PASS opencode (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/opencode/SKILL.md
git commit -m "docs(skills): author opencode skill (discovery + allowlist)"
```

---

## Task 3: Rewrite the `ai-plan` skill

**Files:**
- Modify: `.agents/skills/ai-plan/SKILL.md` (currently `name:`-only + generic over-claiming body)

**Interfaces:**
- Consumes: the AI-subsystem facts in `abi/CLAUDE.md` §AI Subsystem and `docs/contracts/external-claims-audit.mdx`.
- Produces: a routing skill that points at `complete-base`, `sea-learn-loop`, `agent-plan-train`, `abi-superpower-ai`, `abi-superpower-sea`, `abi-superpower-constitution`, `sea-learning-controller`.

- [ ] **Step 1: Failing-test run**

Run: `./tools/check_skills.sh ai-plan`
Expected: `FAIL ai-plan: description missing or <20 chars (0)` and non-zero exit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: ai-plan
description: Plan abi AI work — model routing, completion, SEA self-learning, and training. Use when the user asks to plan AI orchestration, choose a model, run a completion, drive the SEA learn loop, or schedule training. Routes to complete-base, sea-learn-loop, agent-plan-train, and the abi-superpower-ai/sea/constitution superpowers.
---

# ai-plan

Entry point for abi's AI subsystem (`src/features/ai/`). Routes to the
specialist skills instead of duplicating them:

| You want to… | Use |
| --- | --- |
| Smoke-test base local completion (`abi complete`) | `complete-base` |
| Drive the SEA self-learning loop (`abi complete --learn`) | `sea-learn-loop` |
| Plan + train an agent (`abi agent plan` / `train`) | `agent-plan-train` |
| Deep-dive AI routing/constitution/SEA | `abi-superpower-ai`, `abi-superpower-sea`, `abi-superpower-constitution` |
| Toggle SEA mode in the REPL | `sea-learning-controller` (`/learn`) |

## Facts that constrain any AI plan (from source, not marketing)
- Default model `claude-fable-5`; `src/features/ai/models.zig` is the single
  source of truth for ids/aliases/provider routing (mod/stub parity).
- Router `selectBestProfile` ties resolve `abbey > aviva > abi`; neutral input
  routes to `abi`. `analyzeSentiment` is **prefix-only** keyword matching.
- Constitution audit is **observability-only, not a gate** — sets
  `audit_passed` / `audit_vetoed` / `escore`; safety+privacy hard-veto only
  when either scores < 0.5. Matching is case-insensitive **substring** (infix),
  so "harm" fires on "harmless" — it cannot detect novel harm patterns.
- EMA weights persist **only** on the `--learn`/SEA path; plain `complete`
  re-runs sentiment each turn with no EMA persistence.

## Honest boundary
No energy-efficiency, accuracy, or QPS claims without a repo test/benchmark
proving them (see `docs/contracts/external-claims-audit.mdx`).
```

- [ ] **Step 3: Verify pass**

Run: `./tools/check_skills.sh ai-plan`
Expected: `PASS ai-plan (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/ai-plan/SKILL.md
git commit -m "docs(skills): rewrite ai-plan as routing skill with honest AI facts"
```

---

## Task 4: Rewrite the `gpu` skill (honest-stub)

**Files:**
- Modify: `.agents/skills/gpu/SKILL.md` (currently `name:`-only; body over-claims "high-performance hardware acceleration")

**Interfaces:**
- Consumes: `src/features/gpu/mod.zig` flags + `docs/contracts/external-claims-audit.mdx` §Shaders/MLIR.
- Produces: routing skill pointing at `backend-diagnostics` and `abi-superpower-gpu`, with the honest-stub boundary stated verbatim.

- [ ] **Step 1: Failing-test run**

Run: `./tools/check_skills.sh gpu`
Expected: `FAIL gpu: description missing or <20 chars (0)` and non-zero exit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: gpu
description: Plan abi GPU/backend work — Metal on macOS, CPU SIMD fallback, and the disclosed honest-stub backends (accelerator, shaders, mlir, mobile). Use when asked about GPU/backends, why accelerated=false, or when planning backend work. Routes to backend-diagnostics and abi-superpower-gpu. Never claims native CUDA/ANE/Metal-kernel execution — those are disclosed non-goals.
---

# gpu

Entry point for abi's GPU/backend surface (`src/features/gpu/` + the four
honest-stub feature modules). Routes to specialists:

| You want to… | Use |
| --- | --- |
| Report GPU/accelerator/shader/MLIR status + compute matrix | `backend-diagnostics` |
| Deep-dive the GPU/Metal superpower | `abi-superpower-gpu` |

## Honest status (trust the source flags over any prose)
- **Metal on macOS is real**: linked at build time; `accelerated=false` is the
  normal state until `g_metal_context.init()` succeeds at runtime; mid-run
  failure degrades to CPU. No `-Dgpu-backend` option exists.
- **Honest stubs** (`available=false` / `native_dispatch=false` in each
  `src/features/*/mod.zig`): `accelerator` (selection report + CPU SIMD
  fallback only), `shaders` (validate + checksum, no compiler), `mlir`
  (textual lower only, no LLVM toolchain), `mobile` (profile report only, no
  runtime).
- **ANE execution is a disclosed non-goal** (100% Zig constraint; requires
  CoreML/ObjC). Native CUDA/Vulkan/Metal-kernel execution is not linked.

## Hard rule
Do not claim real shader compilation, MLIR/LLVM lowering, or native
accelerator dispatch — per `docs/contracts/external-claims-audit.mdx`.
```

- [ ] **Step 3: Verify pass**

Run: `./tools/check_skills.sh gpu`
Expected: `PASS gpu (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/gpu/SKILL.md
git commit -m "docs(skills): rewrite gpu skill with honest-stub backend boundary"
```

---

## Task 5: Rewrite the `mcp` skill (loopback-only)

**Files:**
- Modify: `.agents/skills/mcp/SKILL.md` (currently `name:`-only; body implies broad "authentication and authorization")

**Interfaces:**
- Consumes: `abi/CLAUDE.md` §MCP Surface (12 tools, frozen enums, protocol limits) + `abi-mcp-transport`/`mcp-smoke` skills.
- Produces: routing skill pointing at `mcp-smoke`, `abi-superpower-mcp`, `abi-mcp-transport`, with the frozen contract + loopback boundary stated.

- [ ] **Step 1: Failing-test run**

Run: `./tools/check_skills.sh mcp`
Expected: `FAIL mcp: description missing or <20 chars (0)` and non-zero exit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: mcp
description: Plan abi MCP server work — the 12-tool JSON-RPC 2.0 surface over stdio plus optional loopback HTTP/SSE. Use when asked about abi-mcp, the tool list, transports, or middleware. Routes to mcp-smoke, abi-superpower-mcp, and abi-mcp-transport. Loopback-only; non-loopback HTTP hardening is a disclosed gap.
---

# mcp

Entry point for the abi MCP server (`src/mcp/`). Routes to specialists:

| You want to… | Use |
| --- | --- |
| Smoke-test abi-mcp + verify the 12-tool contract | `mcp-smoke` |
| Deep-dive the MCP superpower | `abi-superpower-mcp` |
| Transport / middleware / protocol limits detail | `abi-mcp-transport` |

## Frozen contract (do not change without a parity/contract-test update)
- 12 tools, in source order: `ai_run`, `ai_complete`, `ai_learn`, `ai_train`,
  `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`,
  `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- `protocol.MAX_REQUEST_SIZE` = 64 KB; `MAX_JSON_DEPTH` = 32; per-field 16 KB
  cap in `middleware.zig` (declarative validation before dispatch).
- Frozen enums: `connector_test.service` ∈ {openai, anthropic, discord,
  twilio, grok}; `ai_train.format` ∈ {jsonl, csv, text}.

## Honest boundary
Stdio exits on stdin EOF (not a long-lived daemon). Optional HTTP/SSE is
loopback-only (`127.0.0.1:8080`, `ABI_MCP_HTTP_PORT` / `ABI_MCP_HTTP_TOKEN`).
Non-loopback hardening (TLS/authz/rate-limit) is **not** done — deploy behind
a TLS-terminating proxy. `handlers.errorMessage` normalizes every `anyerror`
so `@errorName` never leaks on either transport.
```

- [ ] **Step 3: Verify pass**

Run: `./tools/check_skills.sh mcp`
Expected: `PASS mcp (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/mcp/SKILL.md
git commit -m "docs(skills): rewrite mcp skill with frozen 12-tool contract + loopback boundary"
```

---

## Task 6: Rewrite the `sea` skill

**Files:**
- Modify: `.agents/skills/sea/SKILL.md` (currently `name:`-only; body over-claims "autonomous evolution")

**Interfaces:**
- Consumes: `abi/CLAUDE.md` §AI Subsystem (SEA loop, EMA, constitution observability).
- Produces: routing skill pointing at `sea-learn-loop`, `abi-superpower-sea`, `abi-superpower-constitution`, `sea-learning-controller`.

- [ ] **Step 1: Failing-test run**

Run: `./tools/check_skills.sh sea`
Expected: `FAIL sea: description missing or <20 chars (0)` and non-zero exit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: sea
description: Plan abi SEA (Sparse Evidence Attention) self-learning work — evidence-augmented completion, 8-signal scorer, EMA modulator persistence, and constitution audit. Use when working on ai_learn / complete --learn / evidence recall. Routes to sea-learn-loop, abi-superpower-sea, abi-superpower-constitution, and sea-learning-controller.
---

# sea

Entry point for abi's SEA self-learning loop (`src/features/ai/`). Routes:

| You want to… | Use |
| --- | --- |
| Drive `abi complete --learn` end-to-end | `sea-learn-loop` |
| Deep-dive SEA scoring / adaptive modulation | `abi-superpower-sea` |
| Constitution audit (observability-only) | `abi-superpower-constitution` |
| Toggle SEA in the REPL (`/learn`) | `sea-learning-controller` |

## Facts that constrain any SEA plan
- SEA = evidence-augmented self-learning completion with an 8-signal scorer +
  budgeted greedy selection; task-aware (7 task types shift signal weights).
- `AdaptiveModulator` weights (EMA, `alpha=0.3`, key `modulator:weights`)
  persist in WDBX **only on the `--learn`/SEA path**. Plain `complete` re-runs
  sentiment each turn with no EMA persistence.
- Constitution audit (6 principles) is **observability-only, not a gate** —
  sets `audit_passed` / `audit_vetoed` / `escore`, `std.log.warn`s on violation,
  but `complete` / `run` still return the response. Safety+privacy hard-veto
  only when either < 0.5.
- `routeInputAdaptive` in `router.zig` is unreferenced; the live EMA path is
  `completeAdaptive` / `completeWithStoreAdaptive` via `runLearnLoop` only.

## Honest boundary
No accuracy / energy / learning-gain claims without a repo benchmark.
```

- [ ] **Step 3: Verify pass**

Run: `./tools/check_skills.sh sea`
Expected: `PASS sea (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/sea/SKILL.md
git commit -m "docs(skills): rewrite sea skill with EMA/constitution facts"
```

---

## Task 7: Rewrite the `tui` skill

**Files:**
- Modify: `.agents/skills/tui/SKILL.md` (currently `name:`-only + generic body)

**Interfaces:**
- Consumes: `.opencode.json` `slash_commands` map + `abi/CLAUDE.md` §CLI Surface (agent tui REPL commands).
- Produces: routing skill pointing at `run-tui`, `dashboard-smoke`, `abi-superpower-tui`, and listing the slash-command→skill map.

- [ ] **Step 1: Failing-test run**

Run: `./tools/check_skills.sh tui`
Expected: `FAIL tui: description missing or <20 chars (0)` and non-zero exit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: tui
description: Plan abi TUI/dashboard work — the interactive diagnostics dashboard and agent REPL. Use when asked about abi tui / dashboard, pane splits, slash commands, or session save/load. Routes to run-tui, dashboard-smoke, and abi-superpower-tui. A headless fallback exists; tmux is only for the interactive refresh loop.
---

# tui

Entry point for abi's TUI surface (`abi tui` / `abi dashboard` / `abi --tui`).
Routes:

| You want to… | Use |
| --- | --- |
| Drive the interactive dashboard in a real pty (screenshot) | `run-tui` |
| Non-interactive one-shot `abi dashboard` smoke (CI/headless) | `dashboard-smoke` |
| Deep-dive the TUI superpower (panes, slash commands) | `abi-superpower-tui` |

## Slash commands backed by skills (`.opencode.json` `slash_commands`)
`/open`→file-context-loader, `/diff`→git-diff-integration,
`/commit`→git-commit-integration, `/context`→context-state-reporter,
`/features`→feature-flag-display, `/learn`→sea-learning-controller,
`/save`→session-persister, `/load`→session-restorer,
`/status`→agent-status-reporter, `/reset`→context-resetter. Plugin-provided
commands come from `abi-plugin.json` `commands`.

## Gotchas
- `dashboard-smoke` reads stdin from `/dev/null` to force the non-interactive
  fallback; the only surface that needs a real terminal is the interactive
  refresh loop (`run-tui` uses tmux).
- `@file` mentions are sandboxed to cwd (8 KB budget; rejects `..` / absolute /
  symlink escape) via `file_context.zig`.
```

- [ ] **Step 3: Verify pass**

Run: `./tools/check_skills.sh tui`
Expected: `PASS tui (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/tui/SKILL.md
git commit -m "docs(skills): rewrite tui skill with slash-command map + headless note"
```

---

## Task 8: Rewrite the `wdbx` skill (honest-stub)

**Files:**
- Modify: `.agents/skills/wdbx/SKILL.md` (currently `name:`-only; body claims "Compression and encryption integration" + "Multi-store clustering" — violates the claims audit)

**Interfaces:**
- Consumes: `abi/CLAUDE.md` §WDBX + `docs/contracts/external-claims-audit.mdx` §WDBX roadmap demos (the honest-boundary sentence is quoted verbatim in the body).
- Produces: routing skill pointing at `wdbx-roundtrip`, `wdbx-api-serve`, `wdbx-cluster-serve`, `cluster-demo-guide`, `wdbx-bench`, `secure-demo`, `abi-wdbx-persistence`, `abi-superpower-wdbx`, `-wdbx-cluster`, `-wdbx-compute`, `-wdbx-secure`.

- [ ] **Step 1: Failing-test run**

Run: `./tools/check_skills.sh wdbx`
Expected: `FAIL wdbx: description missing or <20 chars (0)` and non-zero exit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: wdbx
description: Plan abi WDBX vector-store work — in-process KV+vector store, HNSW, WAL/segment persistence, loopback cluster RPC, REST serve, bench, and reference-grade secure demos. Use when asked about wdbx, vector search, persistence, clustering, or the secure demo. Routes to wdbx-roundtrip/api-serve/cluster-serve/bench, secure-demo, abi-superpower-wdbx*, and abi-wdbx-persistence. Demos are reference-grade, NOT production FHE/AES/sharding.
---

# wdbx

Entry point for abi's WDBX vector store (`src/features/wdbx/`). Routes:

| You want to… | Use |
| --- | --- |
| Insert→query round-trip smoke | `wdbx-roundtrip` |
| Serve loopback REST (`abi wdbx api serve`) | `wdbx-api-serve` |
| In-process Raft consensus demo / cluster serve | `wdbx-cluster-serve`, `cluster-demo-guide` |
| Benchmark | `wdbx-bench` |
| `abi wdbx secure demo` (quant / Huffman / autoencoder / HE / FHE) | `secure-demo` |
| Persistence (WAL / segments / recovery) deep-dive | `abi-wdbx-persistence` |
| Superpower deep-dives | `abi-superpower-wdbx`, `-wdbx-cluster`, `-wdbx-compute`, `-wdbx-secure` |

## Honest boundaries (from `docs/contracts/external-claims-audit.mdx` §WDBX)
- Real: in-process KV + vector store, HNSW, SIMD cosine, MVCC snapshots,
  WAL/segment checkpoints, loopback Raft RPC (RequestVote/AppendEntries),
  int8 quantization, order-0 Huffman, in-process autoencoder, additive HE,
  DGHV somewhat-HE (reference parameters).
- **NOT** production multi-host distributed deployment or data sharding.
- **NOT** production-secure or bootstrapped full FHE; no AES/RBAC WDBX storage.
- **NOT** a production / learned-SOTA compression codec.
- Non-loopback bind refuses without `ABI_WDBX_CLUSTER_TOKEN`; REST is
  127.0.0.1-only (`ABI_WDBX_REST_TOKEN` for bearer auth). TLS env vars are
  validated but native TLS is not linked — deploy behind a TLS-terminating proxy.
```

- [ ] **Step 3: Verify pass**

Run: `./tools/check_skills.sh wdbx`
Expected: `PASS wdbx (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/wdbx/SKILL.md
git commit -m "docs(skills): rewrite wdbx skill with honest demo-vs-production boundary"
```

---

## Task 9: Add the `agent-status-reporter` body

**Files:**
- Modify: `.agents/skills/agent-status-reporter/SKILL.md` (has valid frontmatter already; body is empty)

**Interfaces:**
- Consumes: `.skill-telemetry/runs.jsonl` + `.sessions/`, `abi scheduler status` (per `abi/CLAUDE.md` §CLI Surface).
- Produces: the skill backing the `/status` slash command.

- [ ] **Step 1: Failing-test run (frontmatter already passes — this is a body-correctness task)**

Run: `./tools/check_skills.sh agent-status-reporter`
Expected: `PASS agent-status-reporter (46c)` and exit 0 (frontmatter is fine; the gap is the empty body, which the validator does not catch). Note this in the commit: the "test" here is `cat` confirming a non-empty body after the edit.

- [ ] **Step 2: Replace the whole file with this exact content**

```markdown
---
name: agent-status-reporter
description: Report current agent/session status and system health — skill-loop run state, scheduler/memory counters, and active context. Use when the user runs /status in the abi REPL or asks what the agent is doing or for system health.
---

# agent-status-reporter

Backs the `/status` slash command in the `agent tui` REPL
(`.opencode.json` `slash_commands.status`). Reports a compact, read-only
snapshot:

## What to report
1. **Skill-loop run state** — read the last entry of
   `.skill-telemetry/runs.jsonl` and `.skill-telemetry/.sessions/` for the
   active session id + status. Summarize the latest run's status + counts; do
   not dump the file.
2. **Scheduler / memory** — run `abi scheduler status` (one-shot
   self-terminating probe): report counters + attached MemoryTracker stats +
   the always-on telemetry block. The probe is a no-op, so memory counters read
   0 by design — say so; do not fabricate load.
3. **Context** — give one line: loaded files + whether SEA mode is on (from
   `sea-learning-controller`). Defer the full context view to
   `context-state-reporter` (`/context`).

## Rules
- Read-only. Never edit source, telemetry, or session files.
- No performance / accuracy claims without a repo benchmark.
- If a telemetry file is missing, report "not initialized" — do not invent.
```

- [ ] **Step 3: Verify body is non-empty + frontmatter still passes**

Run: `test -s .agents/skills/agent-status-reporter/SKILL.md && ./tools/check_skills.sh agent-status-reporter`
Expected: `PASS agent-status-reporter (<N>c)` and exit 0.

- [ ] **Step 4: Commit**

```bash
git add .agents/skills/agent-status-reporter/SKILL.md
git commit -m "docs(skills): add agent-status-reporter body (backs /status)"
```

---

## Task 10: Delete the six repo-root `abi-superpower-*` orphans

**Files:**
- Delete: `abi-superpower-ai/`, `abi-superpower-gpu/`, `abi-superpower-mcp/`, `abi-superpower-sea/`, `abi-superpower-tui/`, `abi-superpower-wdbx/` (each contained only a stale `SKILL.md` with `name: superpowers`).

**Interfaces:**
- Consumes: proof that these dirs are not discovered by any client (`.opencode/skills` symlinks to `.agents/skills`, not repo root; the syncer targets `.claude/skills/` + `.grok/`, not repo root) and that `.opencode.json` `superpowers.plugins` names resolve via `.agents/skills/abi-superpower-*` (canonical), not these orphans.
- Produces: a clean repo root with no duplicate/dead superpower stubs.

- [ ] **Step 1: Confirm the orphans are unreferenced and differ from canonical**

Run: `for s in ai gpu mcp sea tui wdbx; do echo "--- $s ---"; head -2 "abi-superpower-$s/SKILL.md"; echo "canonical:"; head -2 ".agents/skills/abi-superpower-$s/SKILL.md"; done; echo "=== config refs to repo-root abi-superpower-* ==="; grep -rn "abi-superpower-" .opencode.json opencode.json .mcp.json .codex/ 2>/dev/null`
Expected: each repo-root file shows `name: superpowers` (stale); each canonical file shows `name: abi-superpower-<s>` + a `description:`; config refs are only the `.opencode.json` `superpowers.plugins` list of bare names (`abi-superpower-ai` etc.) which resolve through the `.opencode/skills`→`.agents/skills` symlink, not through repo-root dirs.

- [ ] **Step 2: Delete the six orphan dirs**

Run: `rm -rf abi-superpower-ai abi-superpower-gpu abi-superpower-mcp abi-superpower-sea abi-superpower-tui abi-superpower-wdbx`

- [ ] **Step 3: Verify they are gone and the canonical set is untouched**

Run: `ls -d abi-superpower-* 2>/dev/null || echo "no repo-root abi-superpower-* dirs"; ls -d .agents/skills/abi-superpower-* | wc -l`
Expected: `no repo-root abi-superpower-* dirs` and `6` (the six canonical abi-superpower-* dirs still present).

- [ ] **Step 4: Commit**

```bash
git add -A abi-superpower-ai abi-superpower-gpu abi-superpower-mcp abi-superpower-sea abi-superpower-tui abi-superpower-wdbx
git commit -m "chore(skills): remove dead repo-root abi-superpower-* orphan stubs"
```

---

## Task 11: Sync, final gate, and docs

**Files:**
- Modify: `tasks/todo.md` (add then close the skill-completion item), `CHANGELOG.md` (entry).
- Verify-only: `.claude/skills/`, `.grok/`, `./build.sh check`.

**Interfaces:**
- Consumes: all authored skills from Tasks 2–9 + the validator from Task 1.
- Produces: a green validator over all canonical skills, synced `.claude/skills/` + `.grok/`, an unchanged `./build.sh check`, and updated trackers.

- [ ] **Step 1: Run the validator across ALL canonical skills**

Run: `./tools/check_skills.sh`
Expected: every line `PASS ...`; tail `=== summary: pass=<N> fail=0 ===`; exit 0. (`<N>` should be the count of canonical skill dirs minus `sync-clis`; the eight previously-broken skills now pass.)

- [ ] **Step 2: Sync to .claude/skills and .grok**

Run: `.agents/skills/sync-clis/launch.sh`
Expected: prints `synced: claude/<name>/SKILL.md` / `synced: grok/<name>/SKILL.md` for the operational skills that already have target dirs, then `Synced canonical skills to <N> target(s).` and `done`. (The eight skills authored in this plan are routing/context skills with no existing target dirs in `.claude/skills/`, so the syncer will not create them — that is expected and fine; opencode gets them via the symlink.)

- [ ] **Step 3: Confirm the build is unaffected (no source was touched)**

Run: `./build.sh check`
Expected: green (build + tests + lint + parity + feature-off stubs + CLI smoke). This confirms the skill-authoring work did not accidentally touch `src/` or `build.zig`.

- [ ] **Step 4: Update trackers**

Append to `tasks/todo.md` under "Recently landed" (or replace the open item if one exists):
```
- Skill-set completion: authored opencode/ai-plan/gpu/mcp/sea/tui/wdbx/agent-status-reporter; added tools/check_skills.sh validator; removed 6 repo-root abi-superpower-* orphans. ./tools/check_skills.sh green; ./build.sh check green.
```

Add a `CHANGELOG.md` entry under the latest Unreleased heading:
```
- docs(skills): complete the 8 referenced-but-empty skills (opencode, ai-plan, gpu, mcp, sea, tui, wdbx, agent-status-reporter) with honest-stub framing; add tools/check_skills.sh frontmatter validator; remove 6 dead repo-root abi-superpower-* orphans.
```

- [ ] **Step 5: Commit**

```bash
git add tasks/todo.md CHANGELOG.md
git commit -m "docs: record abi skill-set completion in todo + changelog"
```

---

## Self-Review

**1. Spec coverage.** "Make all skills for ~/abi" → every skill referenced by config that was empty/stubbed is now authored: `opencode` (Task 2), `ai-plan` (3), `gpu` (4), `mcp` (5), `sea` (6), `tui` (7), `wdbx` (8), `agent-status-reporter` (9). Dead duplicates removed (Task 10). Reusable gate added (Task 1). Sync + final build gate (Task 11). The already-complete skills (run-abi, mcp-smoke, the wdbx-* operational skills, abi-superpower-*, abi-mcp-transport, abi-plugin-system, abi-wdbx-persistence, etc.) were intentionally not touched — they pass the validator and need no work. The `.skill-telemetry/registry.json` drift (agent-style `*-reviewer` names) is owned by the external `skill-loop` MCP and is out of scope for manual skill authoring, so it is correctly excluded.

**2. Placeholder scan.** Every code step contains the full file content to write — no "TBD", no "add validation", no "similar to Task N". `<N>` appears only in *expected-output* descriptions (pass counts, char counts), which is a runtime value, not a placeholder in authored content.

**3. Type/name consistency.** The validator (`tools/check_skills.sh`) is defined once in Task 1 and invoked unchanged by every later task (`./tools/check_skills.sh [name]`). Skill `name:` values equal dir basenames in every authored file. The honest-boundary sentences in Tasks 4/5/8 match the verbatim rule in Global Constraints. Slash-command→skill mappings in Task 7 and Task 9 match `.opencode.json` exactly.