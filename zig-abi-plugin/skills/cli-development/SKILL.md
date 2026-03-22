---
name: cli-development
description: This skill should be used when adding CLI commands, creating command files, refreshing the CLI registry, building TUI dashboards, or writing CLI tests. Triggers on "add CLI command", "create command", "refresh registry", "TUI", "dashboard", "CLI test", "command file", "terminal UI", "cli smoke test", "cli-tests".
---

# CLI Development Guide

## Overview

The ABI CLI is a separate `cli` named module rooted at `tools/cli/mod.zig`. It imports the `abi` module for access to all framework functionality. Commands live in `tools/cli/commands/`, organized into domain groups. A comptime registry auto-derives help text, completions, and dispatch from command metadata declarations. The TUI subsystem in `tools/cli/terminal/` provides dashboard, panel, and widget primitives for interactive terminal interfaces.

## Directory Layout

```
tools/cli/
├── mod.zig                        # CLI module root
├── command.zig                    # Meta struct, CommandDescriptor, handler wrappers
├── framework/
│   ├── context.zig                # CommandContext (allocator, env, config)
│   └── types.zig                  # CommandDescriptor, CommandHandler, enums
├── utils/
│   ├── mod.zig                    # Utility re-exports
│   ├── args.zig                   # ArgParser
│   └── io_backend.zig             # CLI I/O abstraction
├── commands/
│   ├── mod.zig                    # Command registry (re-exports from generated snapshot)
│   ├── ai/                        # AI commands: agent, brain, chat, embed, llm/*, mcp, model, etc.
│   ├── core/                      # Core commands: config, discord, init, plugins, profile, ui/
│   ├── db/                        # Database commands
│   ├── dev/                       # Developer tool commands
│   └── infra/                     # Infrastructure commands
├── registry/
│   └── overrides.zig              # Per-command descriptor overrides
├── generated/
│   └── cli_registry_snapshot.zig  # Auto-generated; do NOT edit by hand
├── terminal/                      # TUI subsystem
│   ├── mod.zig                    # TUI re-exports (panels, widgets, events, themes, etc.)
│   ├── dashboard.zig              # Generic Dashboard(PanelType) comptime wrapper
│   ├── async_loop.zig             # Non-blocking event loop
│   ├── terminal.zig               # Raw terminal control, size detection
│   ├── events.zig                 # Key, Mouse, Event types
│   ├── themes.zig                 # ThemeManager, built-in themes
│   ├── widgets.zig                # Reusable widget primitives
│   ├── layout.zig                 # Rect, Constraint layout system
│   ├── panel.zig                  # Panel protocol
│   ├── keybindings.zig            # KeyAction enum, binding maps
│   ├── render_utils.zig           # Drawing helpers, ButtonHitZone
│   ├── component.zig              # Component abstraction
│   └── *_panel.zig                # Domain panels (agent, bench, brain, db, gpu, model, network, training)
└── tests/
    └── build_options_stub.zig     # Standalone stub for build_options (must stay in sync)
```

## Import Rules for CLI Files

CLI files belong to the `cli` named module. Apply these rules strictly:

- Use `@import("abi")` to access framework types and namespaces. Never use relative paths into `src/`.
- Use relative paths for intra-CLI imports: `@import("../../command.zig")`, `@import("../../utils/mod.zig")`.
- Access feature namespaces through `abi`: `abi.foundation.app_paths`, `abi.ai.memory.Message`, `abi.gpu`.
- Import `build_options` is not available in CLI files at runtime. Feature-gated behavior in the CLI must go through the `abi` module's comptime-resolved exports.

Example import block for a typical command file:

```zig
const std = @import("std");
const abi = @import("abi");
const command_mod = @import("../../command.zig");
const context_mod = @import("../../framework/context.zig");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;
```

## Adding a New Command

### Step 1: Create the command file

Place the file in the appropriate group directory under `tools/cli/commands/`. Choose an existing group (`ai/`, `core/`, `db/`, `dev/`, `infra/`) or create a new subdirectory if the command belongs to a new domain.

Every command file must export two things:

1. `pub const meta: command_mod.Meta` -- declarative metadata.
2. `pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void` -- the handler.

### Step 2: Define the Meta struct

The `Meta` struct in `tools/cli/command.zig` controls how the registry discovers and presents the command:

```zig
pub const meta: command_mod.Meta = .{
    .name = "mycommand",
    .description = "Short description of what this command does",
    // Optional fields with defaults:
    // .aliases = &.{"mc"},
    // .kind = .action,           // .action (default) or .group
    // .subcommands = &.{},       // For .group kind
    // .children = &.{},          // ChildMeta structs for group subcommands
    // .options = &.{},           // OptionInfo for help/completions
    // .ui = .{},                 // UiMeta (category, icon)
    // .visibility = .public,     // .public, .internal, .hidden
    // .risk = .safe,             // .safe, .moderate, .dangerous
};
```

For group commands with subcommands, set `.kind = .group` and populate `.subcommands` and `.children`:

```zig
pub const meta: command_mod.Meta = .{
    .name = "config",
    .description = "Configuration management (init, show, validate)",
    .kind = .group,
    .subcommands = &.{ "init", "show", "validate", "help" },
    .children = &.{
        .{ .name = "init", .description = "Generate default config", .handler = wrapInit },
        .{ .name = "show", .description = "Display current config", .handler = wrapShow },
        .{ .name = "validate", .description = "Validate config file", .handler = wrapValidate },
    },
};
```

Each child handler has the signature `fn(*const context_mod.CommandContext, []const [:0]const u8) !void`. Use thin wrappers to adapt internal functions to this signature.

### Step 3: Implement the run function

The `run` function receives a `CommandContext` (with `.allocator`, environment, and config) and the remaining CLI arguments:

```zig
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = args;
    const writer = cli_io.getStdOut();
    try writer.writeAll("Hello from mycommand\n");
    _ = ctx;
}
```

For commands that parse flags, use `utils.args.ArgParser`:

```zig
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    var parser = @import("../../utils/args.zig").ArgParser.init(ctx.allocator, args);
    const verbose = parser.flag("--verbose");
    // ...
}
```

Alternatively, wrap the handler with `command_mod.parserHandler` or `command_mod.contextParserHandler` to receive an `ArgParser` directly.

### Step 4: Refresh the registry

After adding or modifying any command file, regenerate the snapshot:

```bash
zig build refresh-cli-registry
```

This updates `tools/cli/generated/cli_registry_snapshot.zig`. The generated file is the single source of truth for comptime dispatch. Never edit it by hand. If the registry refresh step is skipped, the build will not see the new command.

Verify the command appears in help output:

```bash
zig build run -- --help
zig build run -- mycommand --help
```

### Step 5: Add CLI tests

Add smoke test vectors for the new command. The CLI test infrastructure lives in `build/cli_tests.zig` and `build/cli_smoke_runner.zig`. Smoke tests exercise ~53 vectors covering command dispatch, help flags, and error paths.

Run existing tests to confirm nothing breaks:

```bash
zig build cli-tests              # Smoke test coverage (~53 vectors)
zig build cli-tests-full         # Exhaustive integration vectors from matrix manifest
```

For exhaustive integration testing, add vectors to `tests/integration/matrix_manifest.zig`.

## Sub-Module Re-Export Requirement

When a CLI command accesses a sub-module of a comptime-gated feature (e.g., `abi.ai.memory`, `abi.gpu.backends`), that sub-module must be re-exported from **both** `mod.zig` and `stub.zig` of the feature. If the stub omits the re-export, the CLI fails to compile when the feature is disabled.

Verify by building with the feature disabled:

```bash
zig build test -Dfeat-ai=false --summary all
```

If the CLI references `abi.ai.some_submodule` and `stub.zig` does not re-export it, this build will fail with a missing field error.

## build_options_stub Sync

The file `tools/cli/tests/build_options_stub.zig` provides a standalone `build_options` module for CLI matrix generation outside `build.zig`. When adding new `feat_*` flags to `build/options.zig`, also add matching entries to the stub:

```zig
// In tools/cli/tests/build_options_stub.zig
pub const feat_newfeature = true;  // Match default from build/options.zig
```

The stub currently tracks all 27+ feature flags plus GPU backend flags (`gpu_cuda`, `gpu_vulkan`, `gpu_stdgpu`, `gpu_metal`). Forgetting to update it causes standalone CLI test generation to fail with an unresolved import error.

## TUI Patterns

### Dashboard Architecture

The TUI subsystem uses a comptime-generic `Dashboard(PanelType)` pattern defined in `tools/cli/terminal/dashboard.zig`. This eliminates duplicated state management across dashboard commands. The generic owns terminal lifecycle, theme management, notification rendering, toolbar chrome, help overlay, and the async event loop.

To create a new dashboard, define a panel type that satisfies the duck-typed interface:

```zig
const MyPanel = struct {
    theme: *const themes_mod.Theme,
    // ... panel-specific state

    pub fn deinit(self: *MyPanel) void { ... }
    pub fn update(self: *MyPanel) !void { ... }
    pub fn render(self: *MyPanel, x: usize, y: usize, w: usize, h: usize) !void { ... }
};
```

Then instantiate the dashboard:

```zig
const tui = @import("../../terminal/mod.zig");
const MyDashboard = tui.dashboard.Dashboard(MyPanel);
```

The `Dashboard` provides built-in toolbar buttons (Run, Stop, Refresh, Find, Settings, Theme, Help), keybinding dispatch, and theme cycling. Panel-specific key handling is delegated to the panel's own methods.

### Event Loop

`async_loop.zig` provides the non-blocking event loop. It polls for terminal input events (key presses, mouse clicks, resize) and dispatches them to the dashboard's event handler. The loop runs at a configurable tick rate and calls `panel.update()` each tick for live data refresh (metrics, streaming output, etc.).

### Terminal Primitives

- `terminal.zig` -- raw mode enter/exit, cursor control, size detection (`TerminalSize`), platform capabilities.
- `events.zig` -- `Key`, `Mouse`, `Event` types for input handling.
- `themes.zig` -- `ThemeManager` with built-in light/dark/high-contrast themes. Panels receive a `*const Theme` reference.
- `widgets.zig` -- reusable primitives (progress bars, tables, sparklines).
- `layout.zig` -- `Rect` and `Constraint` for flexible panel sizing.
- `render_utils.zig` -- drawing helpers, ANSI escape sequences, `ButtonHitZone` for toolbar click regions.

### Existing Panels

Reference these for implementation patterns:

| Panel | File | Purpose |
|-------|------|---------|
| `agent_panel.zig` | AI agent monitoring | Session state, message history |
| `bench_panel.zig` | Benchmark results | Suite progress, timing data |
| `brain_panel.zig` | Neural visualization | Brain animation, training mapping |
| `db_panel.zig` | Database metrics | Query stats, index health |
| `gpu_monitor.zig` | GPU utilization | Memory, compute, temperature |
| `model_panel.zig` | Model info | Architecture, parameters, status |
| `network_panel.zig` | Network metrics | Connections, throughput, latency |
| `training_panel.zig` | Training progress | Loss curves, epoch tracking |

### Editor and Launcher

- `tools/cli/terminal/editor/` -- built-in text editor component for inline editing.
- `tools/cli/terminal/launcher/` -- application launcher UI.
- `tools/cli/terminal/dsl/` -- DSL for declarative TUI layout construction.

## Checklist for Adding a CLI Command

1. Create the command file in the correct group under `tools/cli/commands/`.
2. Export `pub const meta: command_mod.Meta` with name, description, and kind.
3. Export `pub fn run(ctx, args) !void` implementing the command logic.
4. Use `@import("abi")` for framework access; relative paths for CLI-internal imports.
5. Run `zig build refresh-cli-registry` to regenerate the snapshot.
6. Verify with `zig build run -- <command> --help`.
7. Ensure any accessed feature sub-modules are re-exported from both `mod.zig` and `stub.zig`.
8. If new `feat_*` flags were added, update `tools/cli/tests/build_options_stub.zig`.
9. Run `zig build cli-tests` to confirm smoke test coverage passes.
10. For group commands, define `ChildMeta` entries with handler wrappers for each subcommand.
