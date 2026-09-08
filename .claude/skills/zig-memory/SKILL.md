---
name: zig-memory
description: Zig allocators and memory ownership on master (0.17.0-dev). Use when writing or reviewing std.mem.Allocator implementations and callers, choosing between SafeAllocator, ArenaAllocator, FixedBufferAllocator, BufferFirstAllocator, page_allocator, smp_allocator and c_allocator, pairing alloc/free and dupe/free, defer and errdefer discipline, alignment and std.mem.Alignment, ArrayList on master, or writing a leak-checked test with std.testing.allocator.
---

# Zig memory and allocators on master

Ground truth is the installed toolchain, not older tutorials:

- `zig` `0.17.0-dev.2018+ab30a0b9a` at `/Users/donaldfilimon/.zvm/bin/zig`
- stdlib at `/Users/donaldfilimon/.zvm/master/lib/std/`

Compile anything uncertain in `/private/tmp/zig-skill-scratch`.

Related skills: `zig-types` for pointer, slice and alignment types, `zig-errors` for `errdefer` discipline in loops and constructors, `zig-testing` for the test runner's leak reporting.

## Changed on master (check this first)

1. **`std.heap.GeneralPurposeAllocator` does not exist.** It fails with
   `root source file struct 'heap' has no member named 'GeneralPurposeAllocator'`.
   Any `var gpa = std.heap.GeneralPurposeAllocator(.{}){};` snippet is stale.

2. **`DebugAllocator` is itself now deprecated, in favor of `SafeAllocator`.**
   `lib/std/heap.zig:23` reads `Deprecated; use SafeAllocator.` and
   `lib/std/heap.zig:21` says the same of `DebugAllocatorConfig`.
   `lib/std/heap.zig:25` marks the `Check { ok, leak }` enum deprecated too:
   `SafeAllocator.deinit` returns a **`usize` leak count**
   (`lib/std/heap/SafeAllocator.zig:653`), not `Check`, so
   `if (gpa.deinit() == .leak)` no longer compiles.

3. **`std.ArrayList(T)` is the UNMANAGED list.** `lib/std/std.zig:52` returns
   `array_list.Aligned(T, null)`, and `lib/std/std.zig:62` makes
   `ArrayListUnmanaged` a deprecated alias of `ArrayList`. So
   `std.ArrayList(u8).init(gpa)` fails with `has no member named 'init'`. The
   allocator-carrying list survives as `std.array_list.Managed(T)`, marked
   `Deprecated.` at `lib/std/array_list.zig:11`.

4. **`std.testing.allocator` is backed by `SafeAllocator`**
   (`lib/std/testing.zig:20`), and it has **no `detectLeaks`**. The old mid-suite
   `testing.allocator_instance.detectLeaks()` call is a compile error. Scope a private
   `SafeAllocator` when one test must assert its own leak count.

5. **`std.heap.MemoryPool` is unmanaged too**: `.empty`, then
   `create(allocator)` / `deinit(allocator)` (`lib/std/heap/memory_pool.zig:55, 63, 71, 113`).
   `MemoryPool(T).init(gpa)` does not exist.

6. **The vtable takes `std.mem.Alignment`, a log2 enum, not a byte count and not a
   `Log2Align` integer.** See the signatures below.

Two more that change how `main` is written: `callMain` (`lib/std/start.zig:786`) accepts
`fn main()` and also `fn main(init: std.process.Init)` or
`fn main(init: std.process.Init.Minimal)`, where `Init` (`lib/std/process.zig:31`)
hands you `gpa`, `arena`, `io`, `environ_map`, and `preopens` directly. And the default
process allocator selected there is `SafeAllocator` in Debug and Safe modes, falling back
to `c_allocator`, `wasm_allocator`, or `smp_allocator`.

## The Allocator interface

`std.mem.Allocator` is `{ ptr: *anyopaque, vtable: *const VTable }`, two words.
The vtable has exactly four members (`lib/std/mem/Allocator.zig:25`):

```zig
alloc:  *const fn (*anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8,
resize: *const fn (*anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool,
remap:  *const fn (*anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8,
free:   *const fn (*anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void,
```

Semantics, from the doc comments in that file:

- `alloc` returns `null` on failure. `len` must be greater than zero.
- `resize` is in place or fail. `true` means same address, new length.
- `remap` may relocate. `null` means "this would be equivalent to alloc plus copy plus
  free, so do it yourself".
- `free`'s `memory.len` must equal the length from the most recent successful
  `alloc`/`resize`/`remap`, and `alignment` must equal the original `alloc` alignment.

Stubs for the members you do not implement are provided:
`Allocator.noAlloc`, `noResize`, `noRemap`, `noFree` (`lib/std/mem/Allocator.zig:89-132`).
`Allocator.failing` (`lib/std/mem/Allocator.zig:535`) always returns `error.OutOfMemory`.

A minimal conforming bump allocator:

```zig
fn allocator(self: *Bump) Allocator {
    return .{ .ptr = self, .vtable = &.{
        .alloc = alloc,
        .resize = Allocator.noResize,
        .remap = Allocator.noRemap,
        .free = Allocator.noFree,
    } };
}

fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
    _ = ret_addr;
    const self: *Bump = @ptrCast(@alignCast(ctx));
    const start = alignment.forward(self.used);
    if (start + len > self.buf.len) return null;
    self.used = start + len;
    return self.buf.ptr + start;
}
```

## Caller-side methods

From `lib/std/mem/Allocator.zig`: `create` / `destroy` for single items,
`alloc` / `free` for slices, `alignedAlloc`, `allocSentinel`, `allocWithOptions`,
`resize`, `remap`, `realloc`, `dupe`, `dupeSentinel`, `print`, `printSentinel`.

```zig
const one = try gpa.create(u32);   defer gpa.destroy(one);
const many = try gpa.alloc(u32, 3); defer gpa.free(many);
const copy = try gpa.dupe(u8, "hello"); defer gpa.free(copy);
const z = try gpa.dupeSentinel(u8, "abc", 0); defer gpa.free(z);   // [:0]u8
const s = try gpa.print("{d}-{s}", .{ 42, "x" }); defer gpa.free(s);
const over = try gpa.alignedAlloc(u8, .fromByteUnits(64), 128); defer gpa.free(over);
```

`gpa.print` is the master spelling of "allocate a formatted string"; the caller owns it.

## Which allocator

| Allocator | Init | Use for |
|---|---|---|
| `std.heap.SafeAllocator` | `.init(backing, .{})`, `deinit()` returns leak count | development, tests, anything where you want double-free and use-after-free caught |
| `std.heap.ArenaAllocator` | `.init(child)`, `deinit()` | many allocations with one lifetime; `free` is a no-op, `reset(.retain_capacity)` reuses the backing memory |
| `std.heap.FixedBufferAllocator` | `.init(&buf)` | no heap at all; returns `error.OutOfMemory` when the buffer is full, `reset()` rewinds |
| `std.heap.BufferFirstAllocator` | `.init(&buf, fallback)` | stack buffer first, real allocator when it does not fit |
| `std.heap.MemoryPool(T)` | `.empty`, `create(gpa)`, `deinit(gpa)` | many objects of one type |
| `std.heap.page_allocator` | singleton | one syscall per allocation, coarse, thread-safe |
| `std.heap.smp_allocator` | singleton | the ReleaseFast multithreaded general purpose choice |
| `std.heap.c_allocator` | singleton | only when linking libc |
| `std.heap.brk_allocator` / `wasm_allocator` | singleton | single-threaded WebAssembly and Linux |

`SafeAllocator`'s header states its guarantees directly
(`lib/std/heap/SafeAllocator.zig:1-10`): `deinit` reports all leaks and frees all backing
memory; allocation mismatches panic or segfault; allocations from another instance with a
different `canary` panic; double frees and racing operations panic or segfault; and given
a non-reusing backing allocator, most writes after free are detected. It is thread-safe.

## Ownership conventions

- The function that allocates does not own the memory; the value's documented owner does.
  A constructor returning by value transfers ownership to the caller, so the caller writes
  the `defer`.
- `deinit` takes the allocator on master's unmanaged containers, because the container
  does not store one. Store the allocator in the owning struct if its `deinit` should
  take none.
- `toOwnedSlice(gpa)` transfers the buffer out of a container and leaves it empty; the
  caller then owns it and frees it with `gpa.free`.
- Set `self.* = undefined;` at the end of a `deinit` so use-after-free is caught in safe
  builds rather than reading stale but valid-looking data.
- Pair every acquisition with an `errdefer` on the failure path and a `defer` (or an owner
  `deinit`) on the success path. See the `zig-errors` skill for the loop and
  returned-by-value cases, which are where this goes wrong.

## Alignment

`std.mem.Alignment` (`lib/std/mem.zig:25`) is `enum(math.Log2Int(usize))`, so
`@intFromEnum(Alignment.of(u64))` is `3`, not `8`.

```zig
Alignment.of(u64).toByteUnits()      // 8
Alignment.fromByteUnits(8)           // .of(u64)
a.forward(addr) / a.backward(addr)   // round up / down
a.check(addr)                        // is addr aligned
Alignment.max(x, y) / .min(x, y) / a.compare(.gt, b)
```

`gpa.alignedAlloc(u32, .fromByteUnits(64), 4)` returns `[]align(64) u32`. Over-aligned
memory must be freed through the same slice type, because the vtable's `free` requires
the original alignment.

## ArrayList on master

```zig
var list: std.ArrayList(u32) = .empty;      // lib/std/array_list.zig:657
defer list.deinit(gpa);                     // :691, takes the allocator

try list.append(gpa, 1);                    // :1023
try list.appendSlice(gpa, &.{ 2, 3 });
_ = list.pop();                             // :1496, returns ?T
const owned = try list.toOwnedSlice(gpa);   // :748
```

Other entry points worth knowing: `initCapacity(gpa, n)` then `appendAssumeCapacity`,
`initBuffer(&backing)` for a fixed-capacity list that never touches an allocator (every
allocator-taking method on such a list is illegal behavior), `ensureTotalCapacity`,
`ensureUnusedCapacity`, and `std.array_list.Aligned(T, .fromByteUnits(16))` for an
over-aligned list. `toManaged(gpa)` converts to the deprecated managed form.

The list also carries `pointer_stability: debug.SafetyLock`; `lockPointers()` turns any
call that would invalidate existing element pointers into an assertion failure, which is
the cheapest way to catch a stale `&list.items[i]`.

## Leak-checked tests

```zig
test "the standard pattern" {
    const gpa = std.testing.allocator;
    const buf = try gpa.alloc(u8, 16);
    defer gpa.free(buf);
    ...
}
```

The test runner deinits `std.testing.allocator_instance` after the suite and fails the run
if anything leaked. Verified negative control: a test that allocates 16 bytes and never
frees them prints `test ... OK` and then

```
[SafeAllocator] (err): leaked [addr: 102738010, len: 16 (0x10) align: 1] allocated at:
/private/tmp/zig-skill-scratch/negative/neg-leak.zig:3:48: ... in test.deliberate leak
```

with exit status 1. Two things follow: a per-test `OK` line is not evidence the test
passed the suite, and the leak report names the allocation site, so you do not need to
bisect.

Related tools:

- `std.testing.checkAllAllocationFailures(gpa, fn, args)` (`lib/std/testing.zig:1142`)
  reruns the body with each allocation index failing in turn and asserts no leak on any
  of those paths. This is the cheapest way to prove every `errdefer` is correct.
- `std.testing.FailingAllocator.init(backing, .{ .fail_index = n })`
  (`lib/std/testing.zig:13`) makes one specific allocation fail.
- `std.testing.failing_allocator` (`lib/std/testing.zig:14`) fails immediately.
- A private `SafeAllocator` when a single test must assert its own leak count:
  `try testing.expectEqual(0, safe.deinit());`

## Verified snippets

Compiled with `zig 0.17.0-dev.2018+ab30a0b9a` in `/private/tmp/zig-skill-scratch/memory/`
via `zig test <file>`. 6 of 6 pass:

`01-allocator-interface.zig`, `02-which-allocator.zig`, `03-ownership.zig`,
`04-arraylist.zig`, `05-leak-check.zig`, `06-alignment.zig`.

Negative controls that must fail, and do:
`std.heap.GeneralPurposeAllocator(.{})` gives
`root source file struct 'heap' has no member named 'GeneralPurposeAllocator'`;
`std.ArrayList(u8).init(gpa)` gives `struct 'array_list.Aligned(u8,null)' has no member named 'init'`;
`std.heap.MemoryPool(u64).init(gpa)` gives `has no member named 'init'`;
`testing.allocator_instance.detectLeaks()` gives
`no field or member function named 'detectLeaks' in 'heap.SafeAllocator'`;
and `/private/tmp/zig-skill-scratch/negative/neg-leak.zig` exits 1 with the leak report above.

The scratch directory is disposable (`/private/tmp` is wiped on reboot). Every construct
above appears inline in this skill, so re-verification means pasting a block into a fresh
scratch file, not recovering these paths.
