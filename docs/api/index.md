# ABI Framework API Reference

> Comprehensive API documentation auto-generated from source code.

---

## Quick Links
| Module | Category | Description | Build Flag |
| --- | --- | --- | --- |
| [config](config.md) | Core Framework | Unified configuration system. | `always-on` |
| [errors](errors.md) | Core Framework | Composable error hierarchy for framework operations. | `always-on` |
| [feature_catalog](feature_catalog.md) | Core Framework | Canonical feature catalog for ABI. | `always-on` |
| [framework](framework.md) | Core Framework | Framework orchestration with builder pattern. | `always-on` |
| [registry](registry.md) | Core Framework | Plugin registry for feature management. | `always-on` |
| [runtime](runtime.md) | Compute & Runtime | Runtime infrastructure (thread pool, channels, scheduling). | `always-on` |
| [simd](simd.md) | Compute & Runtime | SIMD operations (shorthand for `shared.simd`). | `always-on` |
| [acp](acp.md) | Infrastructure | ACP (Agent Communication Protocol) for agent-to-agent communication. | `always-on` |
| [ha](ha.md) | Infrastructure | High availability (replication, backup, PITR). | `always-on` |
| [mcp](mcp.md) | Infrastructure | MCP (Model Context Protocol) server for WDBX database. | `always-on` |
| [connectors](connectors.md) | Utilities | External service connectors (OpenAI, Anthropic, Ollama, etc.). | `always-on` |
| [lsp](lsp.md) | Utilities | LSP (ZLS) client utilities. | `always-on` |
| [platform](platform.md) | Utilities | Platform detection and abstraction. | `always-on` |
| [shared](shared.md) | Utilities | Shared utilities (SIMD, time, sync, security, etc.). | `always-on` |
| [tasks](tasks.md) | Utilities | Task management system. | `always-on` |

---

## Core Framework

### [config](config.md)

Unified configuration system.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

### [errors](errors.md)

Composable error hierarchy for framework operations.

**Source:** [`src/core/errors.zig`](../../src/core/errors.zig)

### [feature_catalog](feature_catalog.md)

Canonical feature catalog for ABI.

Centralizes feature descriptions, compile-time flag mappings, parent-child
relationships, and real/stub module paths used by parity checks.

**Source:** [`src/core/feature_catalog.zig`](../../src/core/feature_catalog.zig)

### [framework](framework.md)

Framework orchestration with builder pattern.

**Source:** [`src/core/framework.zig`](../../src/core/framework.zig)

### [registry](registry.md)

Plugin registry for feature management.

**Source:** [`src/core/registry/mod.zig`](../../src/core/registry/mod.zig)

## Compute & Runtime

### [runtime](runtime.md)

Runtime infrastructure (thread pool, channels, scheduling).

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

### [simd](simd.md)

SIMD operations (shorthand for `shared.simd`).

**Source:** [`src/services/shared/simd/mod.zig`](../../src/services/shared/simd/mod.zig)

## Infrastructure

### [acp](acp.md)

ACP (Agent Communication Protocol) for agent-to-agent communication.

**Source:** [`src/services/acp/mod.zig`](../../src/services/acp/mod.zig)

### [ha](ha.md)

High availability (replication, backup, PITR).

**Source:** [`src/services/ha/mod.zig`](../../src/services/ha/mod.zig)

### [mcp](mcp.md)

MCP (Model Context Protocol) server for WDBX database.

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

## Utilities

### [connectors](connectors.md)

External service connectors (OpenAI, Anthropic, Ollama, etc.).

**Source:** [`src/services/connectors/mod.zig`](../../src/services/connectors/mod.zig)

### [lsp](lsp.md)

LSP (ZLS) client utilities.

**Source:** [`src/services/lsp/mod.zig`](../../src/services/lsp/mod.zig)

### [platform](platform.md)

Platform detection and abstraction.

**Source:** [`src/services/platform/mod.zig`](../../src/services/platform/mod.zig)

### [shared](shared.md)

Shared utilities (SIMD, time, sync, security, etc.).

**Source:** [`src/services/shared/mod.zig`](../../src/services/shared/mod.zig)

### [tasks](tasks.md)

Task management system.

**Source:** [`src/services/tasks/mod.zig`](../../src/services/tasks/mod.zig)

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
