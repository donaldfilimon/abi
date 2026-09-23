---
name: zig
description: Start here for any Zig work on this Mac. Routes to the eleven zig-* skills covering the toolchain, types, errors, memory, comptime, builtins, the standard library, the std.Io rewrite, the build system, testing, and C interop. Load this before writing Zig from memory, because the installed toolchain is master and has removed constructs that every tutorial and every model still teaches.
---

# Zig on this Mac

The installed toolchain is **master**, not a release: `0.17.0-dev.2018+ab30a0b9a`
through zvm at `/Users/donaldfilimon/.zvm/bin/zig`. Measured 2026-09-06.

## Read this before writing a line of Zig

Master has **removed** constructs that pretraining, tutorials, and Stack Overflow
all still teach. These are compiler errors, not warnings, and each was confirmed
by compiling on this machine:

| You will reach for | It gives | Use instead |
|---|---|---|
| `@cImport` / `@cInclude` | `invalid builtin function` | `zig translate-c`, or `b.addTranslateC` |
| `@Type(...)` | `invalid builtin function` | `@Int`, `@Struct`, `@Enum`, `@Union`, `@Fn`, `@Pointer`, `@Tuple` |
| `errdefer \|err\|` | `expected block or expression, found '\|'` | plain `errdefer`, destructure in a wrapper |
| `callconv(.C)` | `no member named 'C'` | `callconv(.c)` |
| `std.heap.GeneralPurposeAllocator` | `no member named` | `std.heap.SafeAllocator` |
| `std.io.getStdOut()` | gone | `Io.File.Writer` off `init.io` |
| `std.fs.cwd()` | `no member named 'cwd'` | `Io.Dir.cwd()` |
| `ArrayList(T).init(gpa)` | gone | `ArrayList` is unmanaged; pass the allocator per call |
| `@typeInfo(S).Struct.fields` | gone | `.@"struct"` with `field_names` / `field_types` |
| `std.meta.fields` | `@compileError` | the parallel arrays above |

`std.builtin` is a deprecated alias for `std.lang`, and `std.fs` is a 21-line
shim. If a Zig answer feels familiar, that is a reason to check it, not to trust
it.

## Two authorities, and they answer different questions

**Compiling proves removal. The langref proves deprecation.** `@intFromEnum`
compiles and passes today while `~/.zvm/master/doc/langref.html` says
"Deprecated. Use @backingInt or @bitCast instead", so a green test is not
evidence a construct is current. The langref ships with the toolchain and its
version stamp matches `zig version` exactly. Read both.

Never answer a Zig question from memory. Read
`/Users/donaldfilimon/.zvm/master/lib/std/`, then compile the answer in
`/private/tmp`.

## Where to go

- **[[zig-toolchain]]** first for anything environmental: which zig is active,
  the dangling `~/.zvm/current` symlink, the second zig on PATH, where the
  langref and stdlib source live, scratch-path rules.
- **[[zig-types]]** for the type system: pointers, slices, sentinels, optionals,
  error unions, structs, enums, unions, vectors, coercion, casting.
- **[[zig-errors]]** for error sets, `try`, `catch`, `errdefer`, error return
  traces, and when to use `unreachable` or `assert` instead of an error.
- **[[zig-memory]]** for allocators, ownership discipline, alignment, and the
  leak-checked test pattern.
- **[[zig-comptime]]** for generics, `inline`, `@TypeOf`, `@typeInfo`,
  reflection, and the comptime failure modes.
- **[[zig-builtins]]** for the `@`-builtin reference: 128 recognized, 126 usable.
- **[[zig-std]]** for the standard library map: containers, `fmt`, `mem`, `json`,
  `http`, `crypto`, `math`, `sort`, threads, time.
- **[[zig-io]]** for the `std.Io` rewrite: readers, writers, buffering, printing
  to stdout, and async under `Io.Threaded`. **The most-changed surface; go here
  before any I/O or concurrency work.**
- **[[zig-build]]** for `build.zig`, `build.zig.zon`, modules, steps,
  dependencies, and cross-compilation.
- **[[zig-testing]]** for test blocks, filters, `std.testing`, and fuzzing.
- **[[zig-c-interop]]** for `extern`, `export`, the C ABI, linking, and using
  `zig cc` as a cross compiler.

## False greens seen on this toolchain

A `--test-filter` matching nothing prints `All 0 tests passed` and exits 0. A
leaking test prints `OK` on its own line and fails only in the summary. Zig
analysis is lazy, so a snippet nothing references can compile without being
checked: give every test an explicit driver. And `cmd | tail` reports tail's exit
status, not the command's, which has manufactured false green claims here before.

## Projects using Zig here

`~/dev/active/cell-lang` is the Cell language compiler; read its `AGENTS.md`
first, and always build it with `-Dswift=false`. `~/string` is a small
unversioned Zig package at the home root. `~/dev/archive/wdbx-zig-scaffold` is a
historical scaffold and not the Rust WDBX substrate.
