# abi-dev-agents Plugin Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a Claude Code plugin with 6 specialized agents optimized for the ABI Zig codebase.

**Architecture:** Markdown-based plugin with agents folder containing ABI-specific system prompts, tools, and whenToUse descriptions. Each agent incorporates critical ABI patterns like Zig 0.16 idioms, feature flags, and stub patterns.

**Tech Stack:** Claude Code plugin system (markdown frontmatter + system prompts)

---

## Plugin Structure

```
~/.claude/plugins/abi-dev-agents/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   ├── abi-planner.md
│   ├── abi-explorer.md
│   ├── abi-architect.md
│   ├── abi-code-explorer.md
│   ├── abi-code-reviewer.md
│   └── abi-issue-analyzer.md
└── README.md
```

---

## Task 1: Create Plugin Directory Structure

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/.claude-plugin/plugin.json`

**Step 1: Create plugin.json**

```json
{
  "name": "abi-dev-agents",
  "description": "Specialized agents for the ABI Zig framework - planning, exploration, architecture, code review, and debugging with Zig 0.16 and ABI patterns expertise",
  "version": "1.0.0",
  "author": {
    "name": "Donald Filimon"
  }
}
```

---

## Task 2: Create abi-planner Agent

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/agents/abi-planner.md`

**Agent Frontmatter:**
```yaml
---
name: abi-planner
description: Design implementation plans for the ABI Zig framework. Use when planning new features, modules, or significant changes that need step-by-step implementation guidance following ABI patterns.
tools: Glob, Grep, Read, WebFetch, TodoWrite
model: sonnet
color: blue
---
```

**System Prompt Key Points:**
- Reference CLAUDE.md for coding conventions
- Include Zig 0.16 patterns (std.Io.Threaded, Timer API, {t} format)
- Feature flag system with mod.zig/stub.zig parity
- Table-driven build system patterns
- Comptime generic patterns for code deduplication

---

## Task 3: Create abi-explorer Agent

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/agents/abi-explorer.md`

**Agent Frontmatter:**
```yaml
---
name: abi-explorer
description: Explore and understand ABI codebase patterns. Use when learning how features work, finding similar implementations, or understanding the architecture of specific modules.
tools: Glob, Grep, LS, Read, NotebookRead
model: haiku
color: cyan
---
```

**System Prompt Key Points:**
- Quick exploration focused on pattern discovery
- Module structure understanding (src/ai/, src/gpu/, src/database/)
- Finding similar implementations for reference
- Identifying existing conventions before making changes

---

## Task 4: Create abi-architect Agent

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/agents/abi-architect.md`

**Agent Frontmatter:**
```yaml
---
name: abi-architect
description: Design feature architectures for the ABI framework. Use when creating new modules, planning GPU backends, designing AI subsystems, or making architectural decisions that affect multiple files.
tools: Glob, Grep, LS, Read, NotebookRead, WebFetch, TodoWrite
model: sonnet
color: green
---
```

**System Prompt Key Points:**
- ABI architecture patterns (flat domain structure)
- GPU backend abstraction layer design
- AI module sub-feature organization
- Configuration system with builder pattern
- Registry integration for feature toggling

---

## Task 5: Create abi-code-explorer Agent

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/agents/abi-code-explorer.md`

**Agent Frontmatter:**
```yaml
---
name: abi-code-explorer
description: Deep analysis of ABI features and modules. Use when tracing execution paths, mapping module dependencies, understanding complex subsystems like GPU dispatching or AI persona routing.
tools: Glob, Grep, LS, Read, NotebookRead, WebFetch
model: sonnet
color: purple
---
```

**System Prompt Key Points:**
- Execution path tracing (GPU dispatch chain, AI routing)
- Dependency mapping between modules
- Understanding comptime generic patterns
- Analyzing test infrastructure (stress, property, e2e tests)

---

## Task 6: Create abi-code-reviewer Agent

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/agents/abi-code-reviewer.md`

**Agent Frontmatter:**
```yaml
---
name: abi-code-reviewer
description: |
  Review code for ABI framework best practices and Zig 0.16 compliance. Use when: (1) Code has been written and needs review, (2) Checking for common ABI pitfalls, (3) Verifying mod.zig/stub.zig parity, (4) Ensuring Zig 0.16 patterns are correct.
tools: Glob, Grep, LS, Read, NotebookRead
model: sonnet
color: yellow
---
```

**System Prompt Key Points:**
- Critical Zig 0.16 gotchas:
  - std.Io.Dir.cwd() not std.fs.cwd()
  - ArrayListUnmanaged.empty not .init()
  - {t} format specifier not @errorName()
  - Timer.start() not Instant.now()
- Feature flag verification (mod.zig/stub.zig sync)
- Test patterns (error.SkipZigTest for hardware tests)
- Code style (4 spaces, PascalCase types, camelCase functions)

---

## Task 7: Create abi-issue-analyzer Agent

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/agents/abi-issue-analyzer.md`

**Agent Frontmatter:**
```yaml
---
name: abi-issue-analyzer
description: Analyze errors, test failures, and issues in ABI context. Use when debugging compilation errors, test failures, runtime issues, or investigating build problems specific to the ABI framework.
tools: Read, Grep, Glob, Bash
model: sonnet
color: red
---
```

**System Prompt Key Points:**
- Common ABI compilation errors and fixes
- Feature flag issues (-Denable-* flags)
- GPU backend conflicts
- WASM limitations
- libc linking requirements
- Slow build troubleshooting (.zig-cache, parallelism)

---

## Task 8: Create README.md

**Files:**
- Create: `~/.claude/plugins/abi-dev-agents/README.md`

**Content:**
- Plugin overview
- Agent descriptions and use cases
- Installation instructions
- ABI-specific patterns each agent knows

---

## Task 9: Verification

**Steps:**
1. Verify plugin loads: Check if agents appear in Claude Code
2. Test each agent with a sample prompt
3. Verify ABI patterns are correctly applied

---

## Critical ABI Patterns for All Agents

### Zig 0.16 I/O (MUST KNOW)

```zig
// Initialize I/O backend
var io_backend = std.Io.Threaded.init(allocator, .{
    .environ = std.process.Environ.empty,  // .empty for library
});
defer io_backend.deinit();
const io = io_backend.io();

// File operations
const content = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(10 * 1024 * 1024));
```

### Feature Flags

```bash
zig build -Denable-ai=true -Denable-gpu=true -Dgpu-backend=cuda,vulkan
```

### Stub Pattern

```zig
// mod.zig (real) and stub.zig (disabled) must have identical public APIs
// Stub returns error.<Feature>Disabled for all functions
```

### Format Specifiers

```zig
// Use {t} not @errorName()/@tagName()
std.debug.print("Error: {t}, State: {t}", .{err, state});
```

### ArrayListUnmanaged

```zig
var list = std.ArrayListUnmanaged(u8).empty;  // NOT .init()
```

### Timer API

```zig
var timer = std.time.Timer.start() catch return error.TimerFailed;
const elapsed_ns = timer.read();
```

---

## Post-Implementation Checklist

- [ ] Plugin directory created at correct location
- [ ] plugin.json has valid JSON
- [ ] All 6 agent files have correct frontmatter
- [ ] Each agent has ABI-specific system prompt
- [ ] README.md documents all agents
- [ ] Plugin loads in Claude Code
- [ ] Agents respond with ABI-aware guidance
