# AGENTS.md

Zig 0.17-dev framework for AI services, semantic vector storage, GPU acceleration, and distributed runtime.

## Build Commands

**macOS 26.4+**: Always use `./build.sh`, never `zig build` directly — stock Zig's LLD cannot link.
**Linux / older macOS**: Use `zig build` directly.

| Command | Description |
|---------|-------------|
| `./build.sh cli` / `zig build cli` | Build CLI binary (`zig-out/bin/abi`) |
| `./build.sh mcp` / `zig build mcp` | Build MCP server (`zig-out/bin/abi-mcp`) |
| `./build.sh test --summary all` | Run all tests |
| `zig build test -- --test-filter "pat"` | Run single test |
| `./build.sh check` | Lint + test + stub parity |
| `zig build check-parity` | Verify mod/stub declaration parity |
| `zig build fix` | Auto-format |

**Test lanes**: `zig build {messaging,secrets,pitr,agents,gpu,network,web,search,auth,storage,cloud,cache,database,connectors,lsp,acp,ha,tasks,documents,compute,desktop,pipeline}-tests`

## Critical Rules

1. **Never `@import("abi")` from `src/`** — causes circular import. Use relative imports only.
2. **Mod/stub contract**: Every feature has `mod.zig` (real), `stub.zig` (no-ops), `types.zig` (shared). Update both together for any public API change.
3. **After any public API change**: Run `zig build check-parity` before committing.
4. **Feature gates**: `if (build_options.feat_X) @import("features/X/mod.zig") else @import("features/X/stub.zig")`
5. **String ownership**: Use `allocator.dupe()` for string literals in structs with `deinit()`.

## Pointer Cast Refactoring - COMPLETED

All GPU subsystem `@ptrCast(@alignCast(ptr))` patterns centralized. Total: ~115+ casts across ~20 files.

### Centralized Helper

`src/features/gpu/pointer_cast.zig`:
```zig
pub fn implCast(comptime Impl: type, ptr: *anyopaque) *Impl {
    return @ptrCast(@alignCast(ptr));
}
```

### Files Refactored (20 files)

**Core Interface:** `interface.zig` (12), `adapters.zig` (15)

**GPU Backends:** `opengl.zig` (8), `opengles.zig` (8), `webgpu.zig` (7), `vulkan.zig` (3), `stdgpu.zig` (3), `cuda/mod.zig` (3), `fallback.zig` (4), `tpu/mod.zig` (2), `fpga/vtable.zig` (2), `cuda/memory.zig` (1), `std_gpu_integration.zig` (1), `peer_transfer/vulkan.zig` (2)

**AiOps Subsystem:** `coordinator_ai_ops.zig` (local helpers for slice casts), `cpu_fallback.zig` (local helpers), `simulated.zig` (local helpers)

**Metal Backends:** `metal_buffers.zig` (2), `metal_compute.zig` (2)

### Files Retaining Direct Casts (Technical Reasons)

- `cuda/native.zig` - Direct FFI handling
- `simulated.zig` - Internal operations
- `metal/unified_memory.zig` - Metal memory ops
- `metal_device.zig` - ObjC dispatch
- `dispatch/coordinator/launch.zig` - Kernel uniforms
- `peer_transfer/metal.zig` - objc_msgSend lookups

Many-pointer casts (`[*]T`) and generic `comptime T` casts keep `@ptrCast(@alignCast)`.

### Usage

```zig
const PointerCast = @import("pointer_cast.zig");
pub fn callback(ptr: *anyopaque) callconv(.C) void {
    const self: *Impl = PointerCast.implCast(Impl, ptr);
}
```

### Verification

- `zig build check-parity` - PASSED
- `zig build test --summary all` - 3781/3787 (6 skipped)

## Toolchain

- **Zig**: pinned in `.zigversion` (0.17.0-dev)
- **Toolchain manager**: `tools/zigly --status` shows current Zig path
- **Cross-compile check**: `zig build cross-check` validates linux/wasi targets

## MCP Server

- Entry: `src/mcp_main.zig` → `zig-out/bin/abi-mcp`
- Config: `.mcp.json` (root)

Would an agent likely miss this without help? See ENTRA-ID.md for Entra onboarding guidance.
Would an agent likely miss this without help? Yes. See GEMINI.md for ABI Framework overview.
Would an agent likely miss this without help? Yes. See CUSTOMIZE-MODEL-DEPLOYMENT.md for detailed customization workflow.
Would an agent likely miss this without help? Yes. See DEPLOY-MODEL-OPTIMAL-REGION.md for deployment to optimal region guidance.
Would an agent likely miss this without help? Yes. See DEPLOY-MODEL.md for Deploy Model routing guidance.
