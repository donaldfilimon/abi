# Codebase Analysis Checklist

Use before planning a modernization. Capture evidence (path + line + why), not vibes.

## 1. Boundaries

- [ ] Module / package boundaries identified (entrypoints, public API, generated files)
- [ ] Frozen contracts listed (CLI commands, MCP tools, public `mod`/`stub` pairs)
- [ ] Generated or host-only paths noted (do not hand-edit). **Verified 2026-09-08: no
      file under `crates/` carries a `@generated` or `DO NOT EDIT` marker, and nothing
      generates `crates/abi-plugins/src/lib.rs`** — it is *manually maintained* (`BUNDLED`,
      `registry_descriptors()`), and both `abi-plugin-system/SKILL.md` and the
      `plugin-system-reviewer` agent instruct updating it. Treat it as a source of truth,
      not a generated artifact. (This line previously named it as generated and told
      agents not to edit the one file two other instructions ask them to edit.)
- [ ] Executable sources of truth named (`Cargo.toml`, `crates/abi-cli/src/usage.rs`, contract tests, gates)

## 2. Legacy pattern scan

- [ ] Stringly-typed state / magic values (hardcoded flag lists, free-form status strings)
- [ ] Silent error swallowing (`catch {}`, broad catch → null, ignored exit codes)
- [ ] God files / procedural monoliths (>~400 LOC with multiple responsibilities)
- [ ] Tight coupling to outdated frameworks, removed std APIs, or platform-only hacks
- [ ] Missing or weak typing (unions as string, unvalidated external input)
- [ ] Manual resource management modern constructs already solve
- [ ] Historical residue: `TODO: modernize`, "was removed", archived plans linked as active

## 3. Duplication & drift

- [ ] Repeated surface lists (commands/tools/features) that should defer to source
- [ ] Docs/claims that invent capabilities not proven by tests or source
- [ ] Parallel stubs / incomplete skill or plugin resources (advertised path missing)

## 4. Testability & risk

- [ ] Can the unit be tested without network, credentials, or full process graph?
- [ ] Which gate proves behavior today? (`./tools/check.sh`, focused `./tools/cargo.sh test`)
- [ ] Blast radius if wrong: contracts, persistence, connectors, public claims

## 5. Prioritization output

For each opportunity, record:

| Target | Pattern | Evidence | Impact | Risk | Suggested strategy |
| ------ | ------- | -------- | ------ | ---- | ------------------ |
| path   | e.g. god file | file:line | high/med/low | high/med/low | direct / phased / parallel |

Hand this table to the **refactor-strategy** skill / `refactor-planner` agent.
