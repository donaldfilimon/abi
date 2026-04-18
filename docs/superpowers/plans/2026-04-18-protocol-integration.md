# Protocol Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate, update, and verify the MCP, LSP, and ACP protocol services within the Zig 0.17 migration branch.

**Architecture:** Services are buildable modules within `src/protocols/`. We will use existing build targets to ensure they are production-ready under 0.17.

**Tech Stack:** Zig 0.17.0, ABI framework protocol modules.

---

### Task 1: Verify MCP Server Integration
**Files:**
- Build: `src/mcp_main.zig`
- Command: `./build.sh mcp`

- [ ] **Step 1: Build MCP Server with Zig 0.17**
Run: `./build.sh mcp`
Expected: Build success (compiling with 0.17).

- [ ] **Step 2: Run smoke test for MCP**
Run: `./zig-out/bin/abi-mcp --help`
Expected: Help message prints correctly.

- [ ] **Step 3: Commit verification**
```bash
git add src/mcp_main.zig
git commit -m "feat(protocols): verify MCP server compatibility with Zig 0.17"
```

---

### Task 2: Verify LSP and ACP Services
**Files:**
- Build Targets: `src/protocols/lsp/`, `src/protocols/acp/`

- [ ] **Step 1: Audit LSP/ACP for 0.17 breaking changes**
Analyze `src/protocols/lsp/mod.zig` and `src/protocols/acp/mod.zig` for 0.16 -> 0.17 diffs.

- [ ] **Step 2: Build LSP/ACP modules**
Run: `zig build lsp-tests acp-tests`
Expected: Pass.

- [ ] **Step 3: Commit**
```bash
git add src/protocols/lsp/ src/protocols/acp/
git commit -m "feat(protocols): verify LSP and ACP service compatibility"
```

---

### Task 3: Final Integration Validation
- [ ] **Step 1: Run full check**
Run: `./build.sh check`

- [ ] **Step 2: Final parity confirmation**
Run: `./build.sh check-parity`
