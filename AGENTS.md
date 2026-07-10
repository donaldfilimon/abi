# Repository Guidelines

Trust executables (`build.zig`, `tools/*.sh`, `src/`, `tests/`) over prose when they disagree.

## Project Structure & Module Organization

- **Public API**: `src/root.zig` → `@import("abi")`. CLI: `src/main.zig` → `src/cli/dispatch.zig`.
- **MCP server**: `src/mcp/` (Zig). Repo-root `mcp/` + `.mcp.json` is **host launcher glue only**.
- **Features**: `src/features/mod.zig` picks `mod.zig` (real) or `stub.zig` (disabled) per `-Dfeat-*`. All default **on**; disable with `-Dfeat-<name>=false`.
- **`feat-foundationmodels`**: comptime-gated to **arm64 macOS**; needs Xcode + macOS 26 SDK (`xcrun swiftc`). Skip with `-Dfeat-foundationmodels=false`.
- **Generated — do not hand-edit**: `src/plugin_registry.zig` (from `src/plugins/*/abi-plugin.json` via `tools/generate_plugin_registry.zig` at build time).
- **Tests**: inline `test {}` blocks in each `.zig` file (no separate unit-test tree). Contract suites under `tests/contracts/`: `surface.zig`, `feature_modules.zig`, `mcp_tools.zig`, `plugin_registry.zig`, `public_docs.zig`.

### Import Rules
- Inside `src/`: relative `.zig` imports only. Always include the `.zig` extension.
- **Only** the MCP handler graph may `@import("abi")`: `src/mcp/main.zig`, `handlers.zig`, `ai_tools.zig`, `connector_tools.zig`, `plugin_tools.zig`, `state.zig`.
- From outside `src/`: use `@import("abi")` and `@import("build_options")`.

## Build, Test, and Development Commands

- Toolchain pin: `.zigversion` → `0.17.0-dev.1252+e4b325c19`. `build.sh`/`tools/build.sh` use whatever `zig` is on PATH (they do **not** switch). Zig 0.16 fails on WDBX/MCP listeners (`std.Io.net.Stream`). Select the pin with zvm/zigup before building.
- Prefer `./build.sh …` on macOS (documented Darwin/Metal workflow). Plain `zig build` works with a matching toolchain but bypasses the wrapper.

| Command | What it does |
|---------|-------------|
| `./build.sh check` | Primary gate: build CLI+MCP, module tests, contracts, feature-off stubs, CLI smoke, lint, mod/stub parity |
| `./build.sh full-check` | check + integration + benchmarks + dashboard/agent-TUI smoke |
| `./build.sh cli` | Build `zig-out/bin/abi` |
| `./build.sh mcp` | Build `zig-out/bin/abi-mcp` |
| `zig build test -Dtest-filter="<pattern>"` | Single test (post-`--` form silently ignored) |
| `zig build test-cli` \| `test-plugins` \| `test-contracts` \| `test-mcp-server` \| `test-integration` | Focused test suites |
| `zig build benchmarks` | Benchmark suite |
| `zig build lint` \| `fix` | Check/apply formatting |
| `zig build check-parity` | Verify mod/stub public declaration-name parity |
| `zig build cross-smoke` | Opt-in cross-compile (Linux/Windows/macOS; slow) |
| `npx mint@latest validate` | Docs site validation (not in CI) |

CI (`.github/workflows/ci.yml`) runs `zig build check` + `cross-smoke` on macOS; keep its Zig version in sync with `.zigversion` when either moves.

## Coding Style & Naming Conventions

### Naming
- Functions/variables: `camelCase`
- Types/structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- Enum variants: `snake_case`

### Zig 0.17 Patterns (agents commonly miss these)
- Entry point: `pub fn main(init: std.process.Init) !void`
- `ArrayListUnmanaged(T).empty` (not `.init(allocator)`)
- `std.mem.trimEnd` (not `trimRight`); `splitScalar` / `splitAny` / `splitSequence` (not `split`)
- Timestamps: `foundation.time.unixMs()` (not `std.time.milliTimestamp`)
- End modules with `std.testing.refAllDecls(@This())` in a `test {}` block
- No silent empty `catch {}` on persistence / inference / connector / data-access paths — propagate or log
- Feature gates: `build_options.feat_*` at comptime (not runtime checks)
- Explicit `std.mem.Allocator` passed; no global allocator

## Testing Guidelines

- Tests are inline `test {}` blocks — no separate test/ directory.
- Each module must include `std.testing.refAllDecls(@This())` in its test block.
- Run single test: `zig build test -Dtest-filter="<pattern>"` (or `./build.sh test -Dtest-filter="…"`). The post-`--` form (`-- --test-filter`) is **silently ignored**.
- Public feature API change → update **both** `mod.zig` and `stub.zig`, then `zig build check-parity`.
- Parity scanner: column-0 `pub const` / `pub fn` names only (not `pub var`, threadlocal, or nested decls). Missing `stub.zig` for a `mod.zig` leaf also fails.

## Commit & Pull Request Guidelines

- Commit messages: Conventional Commits style (e.g., `docs:`, `chore(build):`, `fix:`, `refactor:`, `feat:`).
- `origin/main` shares **no common ancestor** with local `main` — never force-push to reconcile.
- PR requirements:
  - Run `./build.sh check` before and after changes as baseline.
  - Public API change → update both `mod.zig` + `stub.zig` → run `zig build check-parity`.
  - When commands, contracts, feature flags, or Zig patterns change, update all three sibling instruction files: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`.

### CLI Surface (frozen — `tests/contracts/surface.zig`)
13 top-level commands (order matters for the freeze test):
`help`, `complete`, `train`, `agent`, `backends`, `plugin`, `auth`, `twilio`, `tui`, `dashboard`, `wdbx`, `scheduler`, `nn`. Specs: `src/cli/usage.zig`.
- `help --json` → typed metadata; `help --completion <bash|zsh|fish>` → shell scripts.
- `complete`: `--live`, `--model`, `--confirm` (apple-fm), `--learn` (SEA).
- `agent`: `plan`, `train`, `tui`, `multi`, `spawn`, `browser`, `os`. `multi`/`spawn` are local scheduler orchestration; `browser` emits a reviewed local plan and never embeds or launches a browser.
- `abi --tui` → `abi tui`. Malformed numeric args → usage + exit **2**.
- **Do not resurrect** legacy names: `version`, `doctor`, `features`, `platform`, `connectors`, `search`, `info`, `chat`, `db`, `serve`.

### MCP Surface (12 tools — same contract file)
`ai_run`, `ai_complete`, `ai_learn`, `ai_train`, `wdbx_query`, `scheduler_stats`, `scheduler_info`, `connector_test`, `gpu_status`, `plugin_list`, `wdbx_stats`, `plugin_run`.
- Primary: JSON-RPC 2.0 over stdio; **64 KB** request cap (`protocol.MAX_REQUEST_SIZE`).
- Optional HTTP/SSE on `127.0.0.1:8080` (`ABI_MCP_HTTP_PORT`, `ABI_MCP_HTTP_TOKEN`). WDBX REST bearer: `ABI_WDBX_REST_TOKEN`. Loopback-only — not a production non-loopback claim.

## Session Start Checklist

- `tasks/lessons.md` + `tasks/todo.md` are auto-loaded via `opencode.json` `instructions` — follow the checklist there.
- `git status --short --branch`; never revert unrelated dirty work.
- Baseline: `./build.sh check` before and after changes.
- Identify touched modules; confirm mod/stub pairs if changing public APIs.
- Update `tasks/todo.md` as work begins and completes.

## Domain Constraints (Claim-Safe)

- **WDBX**: in-process store + segment/WAL. Cluster RPC is real TCP RequestVote/AppendEntries (`ABI_WDBX_CLUSTER_TOKEN` required for non-loopback bind; optional `ABI_WDBX_CLUSTER_PEERS` allowlist). **Not** production multi-host or sharding.
- **GPU**: capability report + deterministic CPU/SIMD fallback. No `-Dgpu-backend` option; native kernel dispatch is not linked.
- **Connectors**: live path needs credentials + explicit `.live` transport. Local tests stay offline. Discord: numeric snowflake IDs. Twilio: `AC`+32-hex SID, 32-hex token, `.live`.
- **SEA** (`feat-sea`): evidence-augmented self-learning completion; persists `AdaptiveModulator` weights in WDBX.
- **Claims**: no unproven sharding / production FHE·AES·RBAC / non-loopback hardening / K8s·H100 / Swift·Python·TF stacks / QPS·latency·accuracy·energy·certs. Wording: `docs/contracts/external-claims-audit.mdx`. Threat model: `abi-threat-model.md`.

## OpenCode

- Config: `opencode.json` (schema `https://opencode.ai/config.json`).
- `.opencode/skills/` → symlink to `.agents/skills/` (canonical).
- MCP: `abi-mcp` = `./mcp/launcher.sh stdio`; `skill-loop` = `@stylusnexus/skill-loop-cli@0.3.3`.
- OpenCode MCP shape: `type: "local"`, `enabled: true`, single `command` array — **not** `.mcp.json`'s `command` + `args` split.

## Keep in Sync

Root siblings `CLAUDE.md` and `GEMINI.md` restate the same conventions. When commands, contracts, feature flags, or Zig patterns change, update all three.
