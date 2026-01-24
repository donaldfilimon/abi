# AGENTS.md

This file provides guidance for AI agents (Claude, GPT, Gemini, Copilot, and others) working with the ABI framework codebase.

> **Codebase Status:** Synced with repository as of 2026-01-24. Build: `zig build test --summary all` passes 194/198 tests.
> **Core Mandates:** See [PROMPT.md](PROMPT.md) for strict requirements and KPIs.

## Project Overview

**ABI Framework** is a Zig 0.16 multi-persona AI system with WDBX distributed memory:
- **Persona System**: Abbey (high-EQ), Aviva (unfiltered expert), Abi (moderator)
- **WDBX Memory**: Block-chained conversational memory with MVCC and version vectors
- **Distributed Architecture**: Raft consensus, intelligent sharding, block exchange
- **GPU Acceleration**: VTable-based backends (CUDA, Vulkan, Metal, FPGA)

## Essential Commands

### Build & Test
```bash
# Build and test everything
zig build                              # Build project
zig build test --summary all           # Run all tests (194/198 must pass)
zig fmt --check .                      # Check formatting (always run zig fmt . after edits)

# Single file testing
zig test src/file.zig                  # Test specific file
zig test src/file.zig --test-filter "pattern"  # Run specific tests
zig build test src/file.zig            # Alternative: use build system

# Development
zig build bench-competitive            # CRITICAL: Run before/after performance changes
zig build benchmarks                   # Complete benchmark suite
zig build run -- --help                # CLI help
```

### Linting & Validation
```bash
# Must run after every edit
zig fmt .                              # Format all files (ALWAYS)
zig build test --summary all           # Regression test (194/198 baseline)
zig build typecheck                    # Type checking without running tests
```

## Critical Performance Rules

**FAILURE TO COMPLY BREAKS KPI CONTRACTS:**
- **Latency**: GPU dispatch overhead < 50µs (verify with `zig build bench-competitive`)
- **Throughput**: Kernels > 80% theoretical peak bandwidth
- **Memory**: Zero leaks; use `GeneralPurposeAllocator` in tests
- **Formatting**: Zero `zig fmt` diffs allowed

## Code Style & Conventions

### Module Imports & Structure
```zig
// Parent modules export dependencies for children
// src/database/distributed/mod.zig:
pub const time = @import("../../shared/time.zig");
pub const network = @import("../../network/mod.zig");

// Child modules import via parent
// src/database/distributed/shard_manager.zig:
const parent = @import("./mod.zig");
const time = parent.time;
const network = parent.network;
```

### Type Definitions & Exports
```zig
// Export primary types at module level
pub const ConversationBlock = struct { ... };
pub const BlockConfig = struct { ... };
pub const BlockChain = struct { ... };

// Error types use Error suffix
pub const BlockChainError = error{ ... };
pub const DistributedBlockChainError = error{ ... };

// Configuration types use Config suffix  
pub const DistributedConfig = struct { ... };
pub const ShardConfig = struct { ... };
```

### Error Handling Pattern
```zig
// Use error sets, not generic error
pub const MyError = error{
    NotFound,
    InvalidInput,
    OutOfMemory,
};

// Return error!Type
pub fn init(allocator: std.mem.Allocator) MyError!Self {
    const ptr = allocator.create(Self) catch return error.OutOfMemory;
    // ...
}

// Error formatting (Zig 0.16)
std.debug.print("Error: {t}", .{err});
```

### Memory Management
```zig
// Always defer cleanup in init functions
pub fn init(allocator: std.mem.Allocator) !Self {
    const data = try allocator.alloc(u8, size);
    errdefer allocator.free(data);  // Cleanup on error
    
    const obj = try allocator.create(MyType);
    errdefer allocator.destroy(obj);
    
    return Self{
        .allocator = allocator,
        .data = data,
        .obj = obj,
    };
}

// Always implement deinit
pub fn deinit(self: *Self) void {
    self.allocator.free(self.data);
    self.allocator.destroy(self.obj);
}
```

### Testing Standards
```zig
// Test files should be in `tests/` subdirectory or named `*_test.zig`
test "function name description" {
    const allocator = std.testing.allocator;
    
    // Setup with proper cleanup
    const data = try allocator.alloc(u8, 100);
    defer allocator.free(data);
    
    // Test assertion
    try std.testing.expect(data.len == 100);
    
    // Test error cases
    try std.testing.expectError(error.NotFound, functionThatFails());
}
```

## Architecture Patterns

### VTable Backend Implementation
```zig
// Follow src/gpu/interface.zig pattern
pub const MyBackend = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) interface.BackendError!*Self {
        const self = allocator.create(Self) catch return error.OutOfMemory;
        self.* = .{ .allocator = allocator };
        return self;
    }
    
    pub fn deinit(self: *Self) void {
        self.allocator.destroy(self);
    }
    
    // Implement all interface methods
    pub fn allocate(self: *Self, size: usize) interface.BackendError!*anyopaque { ... }
};
```

### WDBX Block Chain Pattern
```zig
// Use src/database/block_chain.zig as reference
// Conversation blocks: B_t = {V_t, M_t, T_t, R_t, H_t}
// - V_t: Embeddings (query/response)
// - M_t: Metadata (persona tag, routing weights, intent)
// - T_t: Temporal markers (MVCC timestamps)
// - R_t: References (parent, skip pointers)
// - H_t: Integrity (cryptographic hash)
```

### Distributed Components
```zig
// Sharding: src/database/distributed/shard_manager.zig
// - Tenant → session → semantic clustering hierarchy
// - Consistent hashing ring for placement
// - Locality-aware replication

// Block Exchange: src/database/distributed/block_exchange.zig  
// - Version vectors for causal consistency
// - Anti-entropy synchronization
// - MVCC conflict resolution

// Raft Consensus: src/database/distributed/raft_block_chain.zig
// - Wraps local block chain with distributed coordination
// - Leader election and log replication
```

## Common Workflows

### Adding New Feature
1. **Research Alignment**: Check against research documents (WDBX, FPGA roadmap)
2. **Module Structure**: Follow existing patterns in same domain
3. **Import Paths**: Use parent module export pattern (critical for `src/database/distributed/`)
4. **Testing**: Write unit tests with `GeneralPurposeAllocator`
5. **Validation**: `zig fmt .` → `zig build test --summary all` → `zig build bench-competitive`

### Debugging Import Issues
```bash
# If "import of file outside module path" error:
# 1. Check if parent module exports dependency
# 2. Use parent = @import("./mod.zig"); pattern
# 3. Example: src/database/distributed/ files must import via mod.zig
```

### Performance Verification
```bash
# BEFORE and AFTER any performance-critical change:
zig build bench-competitive  # Must maintain < 50µs dispatch latency
zig build benchmarks         # Full benchmark suite
```

## Quality Gates (Non-Negotiable)

1. **✅ zig fmt .** - No formatting diffs
2. **✅ zig build test --summary all** - 194/198 tests must pass (regression)
3. **✅ GeneralPurposeAllocator in tests** - Zero memory leaks
4. **✅ Parent module export pattern** - For nested modules
5. **✅ Performance benchmarks** - Before/after critical changes

## Quick Reference

| Task | Command |
|------|---------|
| Format code | `zig fmt .` |
| Run all tests | `zig build test --summary all` |
| Test single file | `zig test src/file.zig` |
| Benchmark | `zig build bench-competitive` |
| Check formatting | `zig fmt --check .` |
| Build project | `zig build` |
| Run CLI | `zig build run -- --help` |

**Remember**: This codebase implements research-mandated WDBX architecture. All changes must align with research documents and maintain 194/198 test pass rate.
