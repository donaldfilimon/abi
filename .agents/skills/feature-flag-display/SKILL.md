---
name: feature-flag-display
description: Show all build-time feature flags and their current state. Maps to `/features` slash command in abi agent tui.
---

# Feature Flag Display

Shows the 15 feature flags and their current build-time state.

## Usage

```
/features
```

## Output

Shows table with:
- Feature name (e.g., feat-ai, feat-gpu, feat-wdbx, etc.)
- Enabled/Disabled state
- Description of what it controls
- Whether it's a stub or real implementation

## Implementation

Reads `build_options` from Zig build system:
- 15 flags: ai, gpu, tui, accelerator, shader, mlir, mobile, wdbx, os_control, hash, metrics, telemetry, nn, sea, foundationmodels
- Each has `mod.zig` (real) + `stub.zig` (disabled) pair

## Skill Integration

Maps to `abi agent tui` REPL `/features` command and `abi backends` CLI output.