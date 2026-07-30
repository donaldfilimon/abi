---
name: abi-plugin-system
description: ABI plugin system superpower. Manifest validation, generated registry, CLI/MCP symmetric loading, 16 bundled fixtures.
superpower:
  command: "execute"
  parameters:
    - name: "action"
      type: "string"
      enum: ["list", "run", "validate", "generate", "info"]
      description: "Plugin action"
    - name: "name"
      type: "string"
      description: "Plugin name to run/inspect"
    - name: "input"
      type: "string"
      description: "Input for plugin run"
---

# ABI Superpower: Plugin System

Exposes the plugin manager and registry as a superpower. Generated registry, manifest validation, CLI/MCP symmetric plugin loading.

## Actions

### list
List all registered plugins with metadata:
```
/abi-plugin-system list
```

### run
Execute a plugin by name:
```
/abi-plugin-system run --name example-plugin --input "test input"
```

### validate
Validate a plugin manifest:
```
/abi-plugin-system validate --manifest ./crates/abi-plugins/plugins/example-plugin/abi-plugin.json
```

### generate
Regenerate plugin registry from manifests:
```
/abi-plugin-system generate
```

### info
Show plugin metadata:
```
/abi-plugin-system info --name example-plugin
```

## Plugin Manifest (`abi-plugin.json`)

```json
{
  "name": "example-plugin",
  "version": "1.0.0",
  "description": "Example plugin",
  "target_feature": "feat-example",
  "entry_point": "mod.rs",
  "commands": [
    { "name": "greet", "summary": "Say hello", "aliases": ["hi"] }
  ],
  "context_providers": [
    { "name": "time", "summary": "Current timestamp" }
  ]
}
```

Required fields: `name`, `version`, `description`, `target_feature`, `entry_point`
Aliases accepted: `targetFeature` / `entryPoint`
`entry_point` must be a safe relative `.rs` path under the plugin directory.

## Bundled Fixtures (16)

| Plugin | Target Feature | Purpose |
|--------|---------------|---------|
| example-plugin | baseline | Baseline fixture |
| example-wdbx-plugin | feat-wdbx | WDBX-targeted |
| accelerator-plugin | feat-accelerator | Backend selection |
| ai-plugin | feat-ai | AI pipeline |
| foundationmodels-plugin | feat-foundationmodels | Apple FM connector |
| gpu-plugin | feat-gpu | GPU vector ops |
| hash-plugin | feat-hash | Stable hashing |
| metrics-plugin | feat-metrics | Observability counters |
| mlir-plugin | feat-mlir | Textual MLIR lowering |
| mobile-plugin | feat-mobile | Mobile profile |
| nn-plugin | feat-nn | Char-LM trainer |
| os-control-plugin | feat-os-control | OS command policy |
| sea-plugin | feat-sea | SEA self-learning |
| shader-plugin | feat-shader | Shader validation |
| telemetry-exporter | feat-telemetry | Telemetry export |
| tui-plugin | feat-tui | Agent TUI / slash-command surface |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              crates/abi-plugins/src/lib.rs                  │
│  `BUNDLED` + `registry_descriptors()` / `register_all()`    │
│  (edit when adding/removing bundled plugins)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│         crates/abi-plugins/plugins/<name>/                  │
│  abi-plugin.json + mod.rs + stub.rs                         │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│  CLI: abi plugin        │         │  MCP: plugin_list/      │
│  - plugin list          │         │  plugin_run             │
└─────────────────────────┘         └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
                   ┌──────────────────────────────────────┐
                   │  crates/abi-plugins/src/manager.rs   │
                   │  - Load/unload/list                  │
                   │  - Manifest validation               │
                   │  - Safe .rs entry_point              │
                   └──────────────────────────────────────┘
```

## Registry

- Bundled metadata lives in `crates/abi-plugins/src/lib.rs` (`BUNDLED`,
  `registry_descriptors`, `register_all`) — update it when plugins change.
- Contract tests under `crates/abi-plugins/tests/` pin the plugin count.

## Plugin Manager (`crates/abi-plugins/src/manager.rs`)

| Method | Description |
|--------|-------------|
| `load` | Load plugin from manifest + entry_point |
| `unload` | Unload plugin, free resources |
| `list` | Return all loaded plugin metadata |
| `run` | Execute plugin's `run()` function |

Validates:
- Manifest required fields present
- `target_feature` matches enabled feature
- `entry_point` exists as `.rs` under plugin dir
- No path traversal in `entry_point`

## CLI Surface

| Command | Description |
|---------|-------------|
| `abi plugin list` | List bundled plugins |
| `abi plugin run <name> [input]` | Execute plugin |

## MCP Tools

| Tool | Description |
|------|-------------|
| `plugin_list` | List plugin metadata (matches CLI) |
| `plugin_run` | Execute plugin with input |

Both return same metadata shape — contract test asserts symmetry.

## Slash Commands (Agent TUI)

Plugins can declare `commands[]` in manifest:
- Registers as `/command` in `agent tui` REPL
- Aliases supported (`aliases: ["hi"]`)

Plugins can declare `context_providers[]`:
- Injects snippets into REPL prompt via `__context__:<name>` dispatch

## Feature Gates

- Requires `feat-tui=true` (default) for CLI/MCP plugin tools
- Plugin's `target_feature` must be enabled
- When disabled: `plugin_list` returns empty, `plugin_run` returns `FeatureDisabled`

## Claim Boundary

- ✅ 16 bundled fixtures with manifest validation
- ✅ Generated registry (build-time, not hand-edited)
- ✅ CLI/MCP symmetric loading and metadata
- ✅ Slash commands + context providers from manifests
- ✅ Safe entry_point path validation
- ❌ NOT a dynamic plugin marketplace
- ❌ NOT sandboxed execution (loads as a Rust module)
- ❌ NOT hot-reload capable (build-time generation)