# ABI Phase 2 Completion + MCP Transport Depth Implementation Plan

**Status:** Completed on 2026-07-09. The tracked implementation also includes
the follow-on local `agent multi|spawn|browser` orchestration slice. The local
`analysis/abi/AI_NATIVE_SPEC.md` copy records the default P0 decision but remains
excluded by the repository's Markdown allowlist; this plan and `tasks/todo.md`
are the tracked records.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close modern-refactor Phase 2 remaining gates: land the tools refactor on `docs/mintlify-hub`, deepen MCP HTTP transport regression tests (wrong bearer, oversized POST body), document loopback auth in contracts, and commit the `abi` coordinator agent—without adding CLI commands or MCP tools.

**Architecture:** Test-first additions in existing `src/mcp/http_transport.zig` inline tests and contract docs only; no new module roots. Tools layout stays `tools/run_contract_cli.sh` + `tools/contract_cli/*` + `tools/feature_flags.sh`. Reimagine greenfield (`modernized/abi-reimagined/`) is **out of scope** until `REIMAGINED_ARCHITECTURE.md` is approved.

**Tech Stack:** Zig `0.17.0-dev.1252+e4b325c19` (pin); `./build.sh check`; `zig build test-mcp-server`; `zig build test-contracts`.

## Global Constraints

- Frozen CLI: exactly **13** top-level commands in `src/cli/usage.zig` — do not add commands.
- Frozen MCP: exactly **12** tools in `src/mcp/handlers.zig` / `tests/contracts/surface.zig` — do not add tools.
- mod/stub: public API changes require both `mod.zig` and `stub.zig` + `zig build check-parity`.
- Claims: no QPS/latency/sharding/production multi-host/native kernel wording without proof (`docs/contracts/external-claims-audit.mdx`).
- MCP may `@import("abi")` only in the handler group; do not merge REST/MCP HTTP framing into `foundation/`.
- Primary gate: `./build.sh check` after every task.
- macOS: prefer `./build.sh` over raw `zig build` for CLI/MCP link workflow.

---

## File map

| File | Responsibility |
|------|----------------|
| `tools/run_contract_cli.sh` | Thin orchestrator sourcing `contract_cli/*.sh` |
| `tools/contract_cli/common.sh` | Shared `require_substring`, `ABI`, `ABI_WDBX_PERSIST=0` |
| `tools/feature_flags.sh` | `abi_read_disabled_feature_flags` from `build.zig` |
| `tools/check_feature_stubs.sh` | Feature-off matrix driven by `feature_flags.sh` |
| `src/mcp/http_transport.zig` | Loopback HTTP/SSE; bearer + size limits; inline transport tests |
| `tests/contracts/mcp_tools.zig` | Tool dispatch/middleware error contracts (already broad) |
| `docs/contracts/public-api.mdx` | Human-readable MCP/REST auth companion |
| `.claude/agents/abi.md` | Coordinator agent for slice discipline |
| `tasks/todo.md` | Active board — update when Phase 2 slice closes |
| `analysis/abi/AI_NATIVE_SPEC.md` | Reimagine spec — record default P0 when HITL absent |

---

### Task 1: Commit Phase 2 tools refactor (isolated)

**Files:**
- Modify: `tools/run_contract_cli.sh`, `tools/check_feature_stubs.sh`
- Create: `tools/contract_cli/common.sh`, `help.sh`, `complete_through_wdbx.sh`, `dashboard_tui.sh`, `nn.sh`, `tools/feature_flags.sh`
- Modify: `tasks/todo.md` (already notes factoring — verify accuracy only)

**Interfaces:**
- Consumes: `build.zig` feat-* option lines
- Produces: unchanged CLI contract smoke behavior (`run_contract_cli: ok`)

- [x] **Step 1: Verify contract smoke**

```bash
cd /Users/donaldfilimon/abi
bash tools/run_contract_cli.sh
```

Expected: `run_contract_cli: ok`

- [x] **Step 2: Verify feature stub matrix starts**

```bash
bash tools/check_feature_stubs.sh 2>&1 | head -3
```

Expected: lines like `check_feature_stubs: zig build cli -Dfeat-ai=false` (no `mapfile` error on macOS bash 3.2)

- [x] **Step 3: Run primary gate**

```bash
./build.sh check
```

Expected: exit 0

- [x] **Step 4: Commit (tools only)**

```bash
git add tools/run_contract_cli.sh tools/check_feature_stubs.sh tools/feature_flags.sh tools/contract_cli/
git add tasks/todo.md
git commit -m "$(cat <<'EOF'
tools: factor contract CLI smoke and read feat flags from build.zig

Split run_contract_cli into contract_cli sections; drive check_feature_stubs
from build.zig via feature_flags.sh (bash 3.2 compatible).
EOF
)"
```

---

### Task 2: MCP HTTP rejects wrong bearer token (TDD)

**Files:**
- Modify: `src/mcp/http_transport.zig` (append test after `MCP HTTP transport accepts configured bearer token`)
- Test: inline `test` in same file (picked up by `zig build test-mcp-server`)

**Interfaces:**
- Consumes: `handleHttpConnectionWithAuth`, `bindLoopback`, `readHttpResponse`
- Produces: regression test `MCP HTTP transport rejects wrong bearer token`

- [x] **Step 1: Write the failing test**

Add at end of `src/mcp/http_transport.zig` test section (before `test { refAllDecls }`):

```zig
test "MCP HTTP transport rejects wrong bearer token" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const body = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}";
    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nAuthorization: Bearer wrong-token\r\nContent-Length: {d}\r\n\r\n{s}",
        .{ body.len, body },
    );
    defer allocator.free(request);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var wb: [512]u8 = undefined;
        var sw = client.writer(io, &wb);
        try sw.interface.writeAll(request);
        try sw.interface.flush();
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnectionWithAuth(allocator, io, conn, "local-token");

    var resp_buf: [2048]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "401 Unauthorized") != null);
}
```

- [x] **Step 2: Run test to verify it passes (or fails if auth bug)**

```bash
zig build test-mcp-server -Dtest-filter="MCP HTTP transport rejects wrong bearer token"
```

Expected: PASS (if fail, fix `hasBearerToken` in `http_transport.zig` only—no behavior change beyond correct 401)

- [x] **Step 3: Commit**

```bash
git add src/mcp/http_transport.zig
git commit -m "test(mcp): reject wrong HTTP bearer token on loopback transport"
```

---

### Task 3: MCP HTTP 413 for oversized POST body (TDD)

**Files:**
- Modify: `src/mcp/http_transport.zig`

**Interfaces:**
- Consumes: `MAX_REQUEST_SIZE` from `protocol.zig` (`64 * 1024`)
- Produces: test `MCP HTTP transport returns 413 for oversized POST body`

- [x] **Step 1: Write the test**

```zig
test "MCP HTTP transport returns 413 for oversized POST body" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;

    var bound = try bindLoopback(io);
    defer bound.server.deinit(io);

    const oversized_body_len = MAX_REQUEST_SIZE + 1;
    var body = try allocator.alloc(u8, oversized_body_len);
    defer allocator.free(body);
    @memset(body, 'x');

    const request = try std.fmt.allocPrint(
        allocator,
        "POST /message HTTP/1.1\r\nContent-Length: {d}\r\n\r\n",
        .{oversized_body_len},
    );
    defer allocator.free(request);

    var caddr = try std.Io.net.IpAddress.parseIp4("127.0.0.1", bound.port);
    const client = try caddr.connect(io, .{ .mode = .stream });
    defer client.close(io);

    {
        var hdr_wb: [256]u8 = undefined;
        var sw = client.writer(io, &hdr_wb);
        try sw.interface.writeAll(request);
        try sw.interface.flush();
        // body bytes (may be sent in chunks; server reads until Content-Length)
        var sent: usize = 0;
        while (sent < oversized_body_len) {
            const chunk = @min(1024, oversized_body_len - sent);
            try client.write(io, body[sent..][0..chunk]);
            sent += chunk;
        }
    }

    const conn = try bound.server.accept(io);
    try handleHttpConnection(allocator, io, conn);

    var resp_buf: [512]u8 = undefined;
    const resp = try readHttpResponse(io, client, &resp_buf);
    try std.testing.expect(std.mem.indexOf(u8, resp, "413") != null);
}
```

**Note:** If `client.write` API differs on Zig 0.17, use the same write pattern as neighboring tests in `http_transport.zig` (match existing `client.writer` usage). Adjust only the test—do not weaken the 413 path in production code.

- [x] **Step 2: Run filtered test**

```bash
zig build test-mcp-server -Dtest-filter="413 for oversized"
```

Expected: PASS

- [x] **Step 3: Full MCP server tests**

```bash
zig build test-mcp-server
```

Expected: exit 0

- [x] **Step 4: Commit**

```bash
git add src/mcp/http_transport.zig
git commit -m "test(mcp): assert HTTP 413 for POST body over 64KB cap"
```

---

### Task 4: Document loopback bearer auth in public API contract

**Files:**
- Modify: `docs/contracts/public-api.mdx` (add subsection under MCP or security)
- Test: `zig build test-contracts` (if `public_docs.zig` asserts phrases—grep before editing)

**Interfaces:**
- Consumes: `ABI_MCP_HTTP_TOKEN`, `ABI_WDBX_REST_TOKEN`, BR-P03/P07 from `analysis/abi/BUSINESS_RULES.md`
- Produces: prose only; no new env vars

- [x] **Step 1: Grep contract tests for forbidden/new strings**

```bash
rg -n 'MCP_HTTP|REST_TOKEN|bearer' tests/contracts/public_docs.zig docs/contracts/public-api.mdx
```

- [x] **Step 2: Add honest subsection** (example text to insert):

```markdown
### Loopback HTTP authentication (optional)

MCP HTTP/SSE (`127.0.0.1`, default port 8080) and WDBX REST (`abi wdbx api serve`) accept an optional bearer token via `ABI_MCP_HTTP_TOKEN` and `ABI_WDBX_REST_TOKEN`. When set, missing or wrong `Authorization: Bearer` receives HTTP 401. Stdio MCP is not tokenized. This is loopback hardening only—not a production multi-tenant or non-loopback exposure claim.
```

- [x] **Step 3: Validate contracts**

```bash
zig build test-contracts
./build.sh check
```

Expected: exit 0

- [x] **Step 4: Commit**

```bash
git add docs/contracts/public-api.mdx
git commit -m "docs(contracts): document optional MCP/REST loopback bearer tokens"
```

---

### Task 5: Land ABI coordinator agent

**Files:**
- Create: `.claude/agents/abi.md` (already drafted in working tree)

**Interfaces:**
- Consumes: `AGENTS.md`, `tasks/todo.md`, `analysis/abi/AI_NATIVE_SPEC.md`
- Produces: committed agent definition for Claude Code

- [x] **Step 1: Review agent for claim violations**

```bash
rg -n 'production multi-host|sharding|QPS|native.*kernel' .claude/agents/abi.md || echo CLEAN
```

Expected: CLEAN or only "do not claim" negations

- [x] **Step 2: Commit**

```bash
git add .claude/agents/abi.md
git commit -m "chore(claude): add abi coordinator agent for slice discipline"
```

---

### Task 6: Close Phase 2 row on active board

**Files:**
- Modify: `tasks/todo.md`
- Modify: `analysis/abi/AI_NATIVE_SPEC.md` §6 (record default P0 if user HITL still open)

- [x] **Step 1: Update todo row** — mark MCP transport depth + tools factoring done; remaining: cluster ops smoke, full reimagine architecture (blocked on HITL).

- [x] **Step 2: Record default P0 in spec** (if no user answer yet):

```markdown
P0 (default until HITL): C1–C7
Drop: C15, C16
Defer: C10 multi-node ops polish, C14 demos
```

- [x] **Step 3: Commit**

```bash
git add tasks/todo.md analysis/abi/AI_NATIVE_SPEC.md
git commit -m "docs(tasks): close Phase 2 tools+MCP transport depth slice"
```

---

### Task 7: Final integration gate

**Files:** none

- [x] **Step 1: Run full smoke skills**

```bash
./.agents/skills/run-abi/smoke.sh
./.agents/skills/mcp-smoke/smoke.sh
```

Expected: `SMOKE OK` / MCP PASS

- [x] **Step 2: Run primary gate**

```bash
./build.sh check
```

Expected: exit 0

---

## Spec self-review

| Spec requirement | Task |
|----------------|------|
| Slice 2 factor `run_contract_cli` | Task 1 |
| Slice 3 data-driven feature flags | Task 1 |
| Slice 4 HTTP bearer edge cases | Tasks 2–3 |
| Slice 4 no new MCP tools | All tasks |
| Slice 5 docs/agents alignment | Tasks 4–5 |
| AI_NATIVE_SPEC P0 record | Task 6 |
| Reimagine scaffold | **Not covered** (await architecture approval) |

**Placeholder scan:** No TBD steps in task bodies; Task 3 notes API adjustment only if compile fails.

**Gaps (follow-on plans):** WDBX cluster multi-node ops smoke, dashboard/TUI slice, GPU honesty docs, full `modernized/` reimagine — separate plans per brainstorming decomposition.

---

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-07-09-abi-phase2-completion-and-mcp-transport-depth.md`.

**Two execution options:**

1. **Subagent-driven (recommended)** — fresh subagent per task, review between tasks

2. **Inline execution** — run tasks in this session with checkpoints (`executing-plans`)

Which approach do you want?
