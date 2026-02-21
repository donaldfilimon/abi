# ABI Framework API Reference

> Comprehensive API documentation auto-generated from source code.

---

## Quick Links

| Module | Category | Description |
|--------|----------|-------------|
| [config](config.md) | Core Framework | migration. Compatibility is preserved for one release cycle. |
| [feature_catalog](feature_catalog.md) | Core Framework | — |
| [framework](framework.md) | Core Framework | - `abi.Config` -> `abi.vnext.AppConfig.framework` |
| [errors](errors.md) | Core Framework | Composable error hierarchy for framework operations. |
| [vnext](vnext.md) | Utilities | vNext forward API surface (staged compatibility release). |
| [registry](registry.md) | Core Framework | Plugin registry for feature management. |
| [runtime](runtime.md) | Compute & Runtime | Runtime infrastructure (thread pool, channels, scheduling). |
| [platform](platform.md) | Utilities | Platform detection and abstraction. |
| [shared](shared.md) | Utilities | Shared utilities (SIMD, time, sync, security, etc.). |
| [connectors](connectors.md) | Utilities | External service connectors (OpenAI, Anthropic, Ollama, etc.). |
| [ha](ha.md) | Infrastructure | High availability (replication, backup, PITR). |
| [tasks](tasks.md) | Utilities | Task management system. |
| [mcp](mcp.md) | Infrastructure | MCP (Model Context Protocol) server for WDBX database. |
| [acp](acp.md) | Infrastructure | ACP (Agent Communication Protocol) for agent-to-agent communication. |
| [simd](simd.md) | Compute & Runtime | SIMD operations (shorthand for `shared.simd`). |

---

## Core Framework

### [config](config.md)

migration. Compatibility is preserved for one release cycle.

**Source:** [`src/core/config/mod.zig`](../../src/core/config/mod.zig)

### [feature_catalog](feature_catalog.md)

**Source:** [`src/core/feature_catalog.zig`](../../src/core/feature_catalog.zig)

### [framework](framework.md)

- `abi.Config` -> `abi.vnext.AppConfig.framework`

**Source:** [`src/core/framework.zig`](../../src/core/framework.zig)

### [errors](errors.md)

Composable error hierarchy for framework operations.

**Source:** [`src/core/errors.zig`](../../src/core/errors.zig)

### [registry](registry.md)

Plugin registry for feature management.

**Source:** [`src/core/registry/mod.zig`](../../src/core/registry/mod.zig)

## Compute & Runtime

### [runtime](runtime.md)

Runtime infrastructure (thread pool, channels, scheduling).

**Source:** [`src/services/runtime/mod.zig`](../../src/services/runtime/mod.zig)

### [simd](simd.md)

SIMD operations (shorthand for `shared.simd`).

**Source:** [`src/services/shared/simd.zig`](../../src/services/shared/simd.zig)

## Infrastructure

### [ha](ha.md)

High availability (replication, backup, PITR).

**Source:** [`src/services/ha/mod.zig`](../../src/services/ha/mod.zig)

### [mcp](mcp.md)

MCP (Model Context Protocol) server for WDBX database.

**Source:** [`src/services/mcp/mod.zig`](../../src/services/mcp/mod.zig)

### [acp](acp.md)

ACP (Agent Communication Protocol) for agent-to-agent communication.

**Source:** [`src/services/acp/mod.zig`](../../src/services/acp/mod.zig)

## Utilities

### [vnext](vnext.md)

vNext forward API surface (staged compatibility release).

**Source:** [`src/vnext/mod.zig`](../../src/vnext/mod.zig)

### [platform](platform.md)

Platform detection and abstraction.

**Source:** [`src/services/platform/mod.zig`](../../src/services/platform/mod.zig)

### [shared](shared.md)

Shared utilities (SIMD, time, sync, security, etc.).

**Source:** [`src/services/shared/mod.zig`](../../src/services/shared/mod.zig)

### [connectors](connectors.md)

External service connectors (OpenAI, Anthropic, Ollama, etc.).

**Source:** [`src/services/connectors/mod.zig`](../../src/services/connectors/mod.zig)

### [tasks](tasks.md)

Task management system.

**Source:** [`src/services/tasks/mod.zig`](../../src/services/tasks/mod.zig)

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
