# runtime-memory API Reference

> Memory pools and custom allocators

**Source:** [`src/runtime/memory/mod.zig`](../../src/runtime/memory/mod.zig)

---

Memory Management for Runtime

This module provides memory management utilities:

- `MemoryPool` / `FixedPool` - Fixed-size object pool
- `ArenaAllocator` / `WorkerArena` - Arena-style allocation
- `SlabAllocator` - Multi-size class pool for hot paths
- `ZeroCopyBuffer` - Avoid unnecessary copies
- `ScopedArena` - RAII-style temporary allocation scope
- Buffer management utilities

---

## API

---

*Generated automatically by `zig build gendocs`*
