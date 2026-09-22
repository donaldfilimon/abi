---
name: zig
description: Start here for any Zig work on this Mac. Covers, through its eleven references/ files, the toolchain, types, errors, memory, comptime, builtins, the standard library, the std.Io rewrite, the build system, testing, and C interop. Load this before writing Zig from memory, because the installed toolchain is master and has removed constructs that every tutorial and every model still teaches. Also use for Zig type-system, error-set, allocator, comptime/reflection, @-builtin, std library, std.Io readers/writers, build.zig/build.zig.zon, test-block, and extern/export/zig cc questions (these were the separate zig-* skills).
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

Each topic below is a file in this skill's `references/` directory (merged from the former `zig-*` skills on 2026-09-21); read the one you need rather than all of them.

- **[references/toolchain.md](references/toolchain.md)** first for anything environmental: which zig is active,
  the dangling `~/.zvm/current` symlink, the second zig on PATH, where the
  langref and stdlib source live, scratch-path rules.
- **[references/types.md](references/types.md)** for the type system: pointers, slices, sentinels, optionals,
  error unions, structs, enums, unions, vectors, coercion, casting.
- **[references/errors.md](references/errors.md)** for error sets, `try`, `catch`, `errdefer`, error return
  traces, and when to use `unreachable` or `assert` instead of an error.
- **[references/memory.md](references/memory.md)** for allocators, ownership discipline, alignment, and the
  leak-checked test pattern.
- **[references/comptime.md](references/comptime.md)** for generics, `inline`, `@TypeOf`, `@typeInfo`,
  reflection, and the comptime failure modes.
- **[references/builtins.md](references/builtins.md)** for the `@`-builtin reference: 128 recognized, 126 usable.
- **[references/std.md](references/std.md)** for the standard library map: containers, `fmt`, `mem`, `json`,
  `http`, `crypto`, `math`, `sort`, threads, time.
- **[references/io.md](references/io.md)** for the `std.Io` rewrite: readers, writers, buffering, printing
  to stdout, and async under `Io.Threaded`. **The most-changed surface; go here
  before any I/O or concurrency work.**
- **[references/build.md](references/build.md)** for `build.zig`, `build.zig.zon`, modules, steps,
  dependencies, and cross-compilation.
- **[references/testing.md](references/testing.md)** for test blocks, filters, `std.testing`, and fuzzing.
- **[references/c-interop.md](references/c-interop.md)** for `extern`, `export`, the C ABI, linking, and using
  `zig cc` as a cross compiler.

## False greens seen on this toolchain

A `--test-filter` matching nothing prints `All 0 tests passed` and exits 0. A
leaking test prints `OK` on its own line and fails only in the summary. Zig
analysis is lazy, so a snippet nothing references can compile without being
checked: give every test an explicit driver. And `cmd | tail` reports tail's exit
status, not the command's, which has manufactured false green claims here before.

## Projects using Zig here

`~/dev/active/cell-lang` is the Cell language compiler; read its `AGENTS.md`
first, and always build it with `-Dswift=false`. `~/dev/archive/string-zig-nogit-20260906` is a small
unversioned Zig package (formerly `~/string`, moved 2026-09-21). `~/dev/archive/wdbx-zig-scaffold` is a
historical scaffold and not the Rust WDBX substrate.
