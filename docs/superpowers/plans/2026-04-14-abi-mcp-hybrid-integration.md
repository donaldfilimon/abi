# abi-mcp Hybrid Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable interactive manual testing and shell-based convenience for the `abi-mcp` server.

**Architecture:**
- **Interactive Mode:** Extend `abi-mcp` binary with a `--debug` mode that handles JSON-RPC framing for manual testing.
- **Convenience Layer:** Shell script utility for common ABI framework operations, providing a CLI abstraction for MCP tools.

**Tech Stack:**
- Zig 0.16
- Bash

---

### Task 1: Extend abi-mcp with --debug Mode

**Files:**
- Modify: `/Users/donaldfilimon/abi/src/mcp_main.zig`

- [ ] **Step 1: Update CLI argument handling in `src/mcp_main.zig`**

```zig
// Add to src/mcp_main.zig
    if (args.len > 1) {
        const arg = args[1];
        if (std.mem.eql(u8, arg, "--debug")) {
            // New debug mode branch
            try runDebugRepl(allocator, init.io);
            return;
        }
        // ... existing help check
    }

fn runDebugRepl(allocator: std.mem.Allocator, io: anytype) !void {
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    try stdout.print("ABI MCP Debug Mode. Enter JSON-RPC 2.0 requests.\n", .{});
    
    var buf: [4096]u8 = undefined;
    while (try stdin.readUntilDelimiterOrEof(&buf, '\n')) |line| {
        // Simple mock of JSON-RPC processing for demonstration
        try stdout.print("Echo: {s}\n", .{line});
    }
}
```

- [ ] **Step 2: Build and verify --debug mode**

Run: `cd /Users/donaldfilimon/abi && ./build.sh mcp && ./zig-out/bin/abi-mcp --debug`
Expected: "ABI MCP Debug Mode" message, then echo input.

- [ ] **Step 3: Commit**

```bash
git add src/mcp_main.zig
git commit -m "feat: add --debug mode to abi-mcp"
```

### Task 2: Create Shell Utility for ABI Operations

**Files:**
- Create: `/Users/donaldfilimon/abi/tools/abi-mcp-utils.sh`

- [ ] **Step 1: Create script `tools/abi-mcp-utils.sh`**

```bash
#!/bin/bash
# Simple helper to send JSON-RPC to abi-mcp
abi_call() {
    local method=$1
    local params=$2
    printf '{"jsonrpc": "2.0", "method": "%s", "params": %s, "id": 1}\n' "$method" "$params" | \
    # In a real setup, pipe to the running server or abi-mcp process
    # For now, print the payload to verify
    cat
}
```

- [ ] **Step 2: Make executable**

Run: `chmod +x tools/abi-mcp-utils.sh`

- [ ] **Step 3: Commit**

```bash
git add tools/abi-mcp-utils.sh
git commit -m "feat: add shell utility for abi-mcp"
```
