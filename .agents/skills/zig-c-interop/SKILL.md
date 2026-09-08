---
name: zig-c-interop
description: >-
  Use when calling C from Zig or exposing Zig to C on master (0.17.0-dev), on
  "invalid builtin function: '@cImport'", when replacing @cImport/@cInclude with
  translate-c or b.addTranslateC, on "union 'lang.CallingConvention' has no
  member named 'C'", when exe.linkLibC / exe.linkSystemLibrary / exe.addCSourceFiles
  no longer exist on Step.Compile, writing extern fn or extern struct or export
  or @export, converting [*c]T and NUL-terminated C strings to Zig slices,
  adding C source files or include paths from build.zig, building a static or
  shared library for C consumers with installHeader, or using zig cc / zig c++
  as a drop-in cross compiler. Do not use for build.zig structure in general
  (zig-build) or std.testing (zig-testing).
---

# Zig C interop on master

Verified against the zvm master toolchain on this Mac:
`/Users/donaldfilimon/.zvm/bin/zig`, version `0.17.0-dev.2018+ab30a0b9a`.
Everything below was compiled, linked and run; the round-trip project at the end
built, tested and executed clean.

Scratch work goes in `/private/tmp/zig-build-scratch/`, never iCloud, never home root.

## Changed on master

| Older idiom | Status on this toolchain |
|---|---|
| `const c = @cImport({ @cInclude("stdlib.h"); });` | **`@cImport` is removed.** `error: invalid builtin function: '@cImport'`. Use `zig translate-c` at the CLI or `b.addTranslateC(...)` in `build.zig`, then `@import` the resulting module. |
| `callconv(.C)` | `callconv(.c)`. `.C` errors with `union 'lang.CallingConvention' has no member named 'C'`. `CallingConvention` is declared at `lib/std/lang.zig:164`; `std.builtin.CallingConvention` still resolves to it. |
| `@export(decl, .{ .name = "x" })` | `@export(&decl, .{ .name = "x", .linkage = .strong })`, taking a **pointer**. Verified failure without the `&`: `error: expected pointer type, found 'fn (c_int) callconv(.c) c_int'`. |
| `exe.linkLibC()` | `link_libc: ?bool = null` in `Module.CreateOptions` (`lib/std/Build/Module.zig:208`), or `Module.linkSystemLibrary` resolving a libc name. (`Step.TranslateC.Options.link_libc` is the non-optional `bool = true`, `Build/Step/TranslateC.zig:26`.) |
| `exe.linkSystemLibrary("z")` | `mod.linkSystemLibrary("z", .{})` with `LinkSystemLibraryOptions` (`lib/std/Build/Module.zig:341`). Not a method on `*Step.Compile` any more. |
| `exe.addCSourceFiles(.{...})`, `exe.addIncludePath(...)`, `exe.linkLibrary(lib)` | Module-level: `Module.addCSourceFiles` (`:388`), `Module.addCSourceFile` (`:412`), `Module.addIncludePath` (`:473`), `Module.addLibraryPath` (`:503`), `Module.addObjectFile` (`:445`), `Module.linkLibrary` (`:456`), `Module.linkFramework` (`:371`). |
| `b.addStaticLibrary` / `b.addSharedLibrary` | `b.addLibrary(.{ .linkage = .static \| .dynamic, .name, .root_module })` (`lib/std/Build.zig:635`). |
| translate-c is clang-based | translate-c is now **Aro**-backed: its output contains `pub const __VERSION__ = "Aro aro-zig";`. `zig cc` is still clang. Behavior on hostile headers can differ from older releases. |

`installHeader` and `installHeadersDirectory` are the exceptions that stayed on
`*Step.Compile` (`lib/std/Build/Step/Compile.zig:476` and `:491`), because they
are about installation, not compilation.

## @cImport is gone; translate-c is the workflow

Verified on this toolchain:

```
$ zig test t.zig -lc
t.zig:2:11: error: invalid builtin function: '@cImport'
const c = @cImport({
          ^~~~~~~~
```

There is no in-language C-header import. Two replacements.

### CLI, for inspection and one-offs

```bash
zig translate-c -I. -lc hdr.h > out.zig
```

Exit 0, output is ordinary Zig you can read and check in. For a header
declaring `struct Point { int x; int y; };` and `int point_sum(struct Point);`
it emits:

```zig
const __root = @This();
pub const __builtin = @import("std").zig.c_translation.builtins;
pub const __helpers = @import("std").zig.c_translation.helpers;
pub const struct_Point = extern struct {
    x: c_int = 0,
    y: c_int = 0,
    pub const point_sum = __root.point_sum;
    pub const sum = __root.point_sum;
};
pub extern fn point_sum(p: struct_Point) c_int;
pub extern fn c_strlen_wrapper(s: [*c]const u8) usize;
```

Two things to notice. A `struct Foo` becomes `struct_Foo` (a `typedef` gets the
bare name too). And functions whose first parameter is a struct are re-exposed as
methods on it, including a prefix-stripped alias, so `p.sum()` works alongside
`point_sum(p)`.

The output also carries several hundred lines of predefined macros. Committing
translate-c output is a legitimate strategy when the header is stable and you
want to hand-edit the awkward parts; regenerating it in the build is better when
the header moves.

### build.zig, for a live module

`b.addTranslateC(options) *Step.TranslateC` (`lib/std/Build.zig:950`).
`Step.TranslateC.Options` (`lib/std/Build/Step/TranslateC.zig:22`), copied from
source:

```zig
pub const Options = struct {
    root_source_file: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.Optimize,
    link_libc: bool = true,
};
```

Then configure and turn it into a module:

```zig
const translate = b.addTranslateC(.{
    .root_source_file = b.path("include/mathc.h"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,
});
translate.addIncludePath(b.path("include"));
const cmath_mod = translate.createModule();   // private to this package
// or: translate.addModule("cmath")            // public to consumers
```

Configuration methods on `*Step.TranslateC`: `addIncludePath` (`:114`),
`addSystemIncludePath` (`:106`), `addAfterIncludePath` (`:98`),
`addFrameworkPath` (`:138`), `addSystemFrameworkPath` (`:130`),
`addConfigHeader` (`:122`), `defineCMacro(name, ?value)` (`:156`),
`defineCMacroRaw` (`:166`), `linkSystemLibrary` (`:174`),
`getOutput() LazyPath` (`:57`), and `addCheckFile(expected_matches)` (`:146`)
if you want to assert on the generated Zig.

Translating a header does **not** compile the C. You still need the object code,
from `addCSourceFiles`, a system library, or a prebuilt object.

## Declaring the boundary by hand

For a handful of symbols, skip translate-c and write the declarations:

```zig
pub extern fn point_sum(p: Point) c_int;
pub extern "z" fn compress(dest: [*c]u8, destLen: *c_ulong, src: [*c]const u8, srcLen: c_ulong) c_int;

pub const Point = extern struct { x: c_int, y: c_int };
```

`extern` on a function means "defined elsewhere, C calling convention by default".
The optional string is the library name. `extern struct` guarantees C ABI layout:
field order preserved, C alignment and padding rules, no field reordering. A
plain Zig `struct` has no layout guarantee and must never cross the boundary.
`packed struct` is a different thing: it is bit-packed and backed by an integer,
not equivalent to a C `__attribute__((packed))` struct.

`extern union` follows C union layout; a bare Zig `union` does not.

Enums crossing the boundary should be `enum(c_int)` or whatever the C
implementation actually uses, and C enums are not exhaustive from Zig's side, so
prefer a non-exhaustive `enum(c_int) { a, b, _ }` when the C side may add values.

## Calling convention

```zig
export fn zig_add(a: c_int, b: c_int) callconv(.c) c_int { return a + b; }
```

`.C` is dead:

```
error: union 'lang.CallingConvention' has no member named 'C'
    export fn f(a: c_int) callconv(.C) c_int { return a; }
                                   ~^
lib/std/lang.zig:164: note: union declared here
```

`CallingConvention` is a tagged union now, not a flat enum
(`lib/std/lang.zig:164`). `.c` is an alias for the target's C convention
(`builtin.target.cCallingConvention().?`). Also available: `.winapi` (dispatches
per arch to `x86_64_win`, `x86_stdcall`, `aarch64_aapcs_win`, `arm_aapcs_vfp`),
`.kernel` (amdgcn/nvptx/spirv), `.naked`, `.auto`, `.async`, and arch-specific
variants with payloads. `callconv(std.builtin.CallingConvention.c)` compiles and
is the same thing.

`export fn` and `extern fn` both default to `.c`, so the explicit `callconv(.c)`
is documentation rather than a requirement. Write it anyway at an ABI boundary.

## Exporting Zig to C

```zig
export fn zig_add(a: c_int, b: c_int) callconv(.c) c_int { return a + b; }
```

`export` gives the symbol strong external linkage under its own name. To export
under a different name, or to export a private declaration, use `@export`, which
now takes a **pointer**:

```zig
fn internalAdd(a: c_int, b: c_int) callconv(.c) c_int { return a + b; }
comptime { @export(&internalAdd, .{ .name = "renamed_add", .linkage = .strong }); }
```

Verified: `zig build-obj` on that file, then `nm` shows
`0000000000000000 T _renamed_add`.

Exported functions must use C-representable types in their signature: no Zig
error unions, no slices, no `anytype`, no non-`extern` structs by value. Return
an `c_int` status and an out-parameter instead of `!T`.

## C ABI types and pointers

`c_char`, `c_short`, `c_ushort`, `c_int`, `c_uint`, `c_long`, `c_ulong`,
`c_longlong`, `c_ulonglong`, `c_longdouble` are the target's C types. Never
assume `c_int == i32` across targets; use the `c_` type at the boundary and
convert inside.

`[*c]T` is the **C pointer**: it may be null, has unknown length, and coerces
implicitly to and from `*T`, `[*]T`, `?*T`. That implicit coercion is exactly
what makes it dangerous, so convert it to a real Zig type at the edge of your
code and never let it propagate inward.

Conversions:

| C shape | Zig at the boundary | Convert with |
|---|---|---|
| `const char *` (NUL-terminated) | `[*c]const u8` | `std.mem.span(@as([*:0]const u8, @ptrCast(s)))` gives `[:0]const u8` |
| `const char *` + explicit length | `[*c]const u8`, `usize` | `s[0..len]` |
| `char *` out-buffer + capacity | `[*c]u8`, `usize` | `buf[0..cap]`, write, return the used length |
| Zig `[]const u8` to C | needs NUL | `try allocator.dupeZ(u8, slice)` then `.ptr`, or a `[:0]const u8` literal |
| nullable object pointer | `?*T` | `if (p) \|ptr\| ...` |

String literals in Zig are `*const [N:0]u8`, already NUL-terminated, so passing
one straight to a `[*c]const u8` parameter works. A `[]const u8` slice from
runtime is **not** NUL-terminated and must be duplicated before it crosses.

Verified export taking a C string and returning a count:

```zig
pub export fn zig_upper_count(s: [*c]const u8) callconv(.c) usize {
    const slice = std.mem.span(@as([*:0]const u8, @ptrCast(s)));
    var n: usize = 0;
    for (slice) |ch| { if (std.ascii.isUpper(ch)) n += 1; }
    return n;
}
```

Called from C, `zig_upper_count("HeLLo")` returned `3`.

## Linking from build.zig

All of this is module-level now.

```zig
const mod = b.createModule(.{
    .root_source_file = b.path("src/root.zig"),
    .target = target,
    .optimize = optimize,
    .link_libc = true,          // Module.CreateOptions:208; replaces exe.linkLibC()
});
mod.addIncludePath(b.path("include"));
mod.addCSourceFiles(.{
    .root = b.path("c"),        // files are relative to this
    .files = &.{"mathc.c"},
    .flags = &.{ "-std=c17", "-Wall", "-Wextra" },
});
mod.linkSystemLibrary("z", .{});
mod.linkLibrary(other_compile_step);
```

`AddCSourceFilesOptions` (`lib/std/Build/Module.zig:377`), from source:

```zig
pub const AddCSourceFilesOptions = struct {
    root: ?LazyPath = null,
    files: []const []const u8,
    flags: []const []const u8 = &.{},
    language: ?CSourceLanguage = null,
};
```

Paths in `files` **must be relative**; an absolute path panics with an explicit
message. `language` overrides the per-file extension sniffing when you need to
force C vs C++ vs Objective-C.

`LinkSystemLibraryOptions` (`:322`) carries `needed`, `weak`, `use_pkg_config`,
`preferred_link_mode` and `search_strategy`. Note that
`mod.linkSystemLibrary("c", .{})` is special-cased: it sets `link_libc = true`
and returns rather than adding a link object (`lib/std/Build/Module.zig:350`),
and the same holds for libc++ names.

## Building a library for C consumers

```zig
const lib = b.addLibrary(.{
    .name = "cinterop",
    .linkage = .static,          // or .dynamic
    .root_module = root_mod,
});
lib.installHeader(b.path("include/mathc.h"), "mathc.h");
b.installArtifact(lib);
```

A shared library needs position-independent code, so give it its own module with
`.pic = true` rather than reusing the static one. With
`.version = .{ .major = 0, .minor = 1, .patch = 0 }` the install produces the
usual soname chain; verified on this Mac:

```
zig-out/lib/libcinterop.a
zig-out/lib/libcinterop.0.1.0.dylib
zig-out/lib/libcinterop.0.dylib
zig-out/lib/libcinterop.dylib
zig-out/include/mathc.h
```

## zig cc and zig c++

Drop-in clang, with the whole cross-compilation matrix attached. Verified:

```bash
zig cc -o hello consumer.c zig-out/lib/libcinterop.a     # links a Zig static lib
zig cc -target x86_64-linux-musl -c mathc.c -Iinclude -o mathc-linux.o
zig c++ -o tcpp t.cpp
```

The cross object came out as `ELF 64-bit LSB relocatable, x86-64`. This is the
easiest way to sanity-check a C side in isolation, and the reason `CC="zig cc"`
works as a cross toolchain for autotools and CMake projects. Also available as
drop-ins: `zig ar`, `zig ranlib`, `zig objcopy`, `zig dlltool`, `zig lib`,
`zig rc`. Note `zig cc` is clang while `zig translate-c` is Aro, so the two can
disagree on a pathological header.

## Verified round-trip project

Zig calls C, C calls Zig, one static library, one shared library, header
installed, all through `build.zig`. Gates, all exit 0 on this toolchain:

```
cd /private/tmp/zig-build-scratch/cinterop
zig build                      # exit 0
zig build test --summary all   # exit 0, 4/4 steps, 2/2 tests passed
zig build run-consumer         # exit 0, printed zig_add(20,22)=42 / zig_upper_count("HeLLo")=3
zig build check                # exit 0
```

`include/mathc.h`:

```c
#ifndef MATHC_H
#define MATHC_H
#include <stddef.h>
struct Point { int x; int y; };
int point_sum(struct Point p);
size_t count_chars(const char *s, char needle);
#endif
```

`src/root.zig`:

```zig
const std = @import("std");
const cmath = @import("cmath");

/// Hand-written extern declaration; equivalent to what translate-c emits.
pub extern fn point_sum(p: cmath.struct_Point) c_int;

/// Exported for C consumers.
pub export fn zig_add(a: c_int, b: c_int) callconv(.c) c_int {
    return a + b;
}

/// Takes a NUL-terminated C string, returns a Zig slice.
pub export fn zig_upper_count(s: [*c]const u8) callconv(.c) usize {
    const slice = std.mem.span(@as([*:0]const u8, @ptrCast(s)));
    var n: usize = 0;
    for (slice) |ch| { if (std.ascii.isUpper(ch)) n += 1; }
    return n;
}

test "call C from Zig via translate-c module" {
    const p: cmath.struct_Point = .{ .x = 4, .y = 5 };
    try std.testing.expectEqual(@as(c_int, 9), cmath.point_sum(p));
    try std.testing.expectEqual(@as(c_int, 9), point_sum(p));
    try std.testing.expectEqual(@as(usize, 2), cmath.count_chars("banana", 'n'));
}

test "exported functions are callable from Zig too" {
    try std.testing.expectEqual(@as(c_int, 7), zig_add(3, 4));
    try std.testing.expectEqual(@as(usize, 2), zig_upper_count("aBcD"));
}
```

`build.zig`:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. Translate the C header into a Zig module.
    const translate = b.addTranslateC(.{
        .root_source_file = b.path("include/mathc.h"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    translate.addIncludePath(b.path("include"));
    const cmath_mod = translate.createModule();

    // 2. A module that owns both the Zig source and the C source files.
    const root_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "cmath", .module = cmath_mod }},
    });
    root_mod.addIncludePath(b.path("include"));
    root_mod.addCSourceFiles(.{
        .root = b.path("c"),
        .files = &.{"mathc.c"},
        .flags = &.{ "-std=c17", "-Wall", "-Wextra" },
    });

    // 3. A static library for C consumers, with its header installed.
    const lib = b.addLibrary(.{
        .name = "cinterop",
        .linkage = .static,
        .root_module = root_mod,
    });
    lib.installHeader(b.path("include/mathc.h"), "mathc.h");
    b.installArtifact(lib);

    // 4. A C program linked against the Zig static library.
    const consumer = b.addExecutable(.{
        .name = "consumer",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    consumer.root_module.addCSourceFiles(.{
        .root = b.path("c"),
        .files = &.{"consumer.c"},
        .flags = &.{"-std=c17"},
    });
    consumer.root_module.linkLibrary(lib);
    b.installArtifact(consumer);

    const run_consumer = b.addRunArtifact(consumer);
    b.step("run-consumer", "Run the C program that links the Zig library")
        .dependOn(&run_consumer.step);

    // 5. A shared library for C consumers (separate module: shared needs PIC).
    const shared_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .pic = true,
        .imports = &.{.{ .name = "cmath", .module = cmath_mod }},
    });
    shared_mod.addIncludePath(b.path("include"));
    shared_mod.addCSourceFiles(.{ .root = b.path("c"), .files = &.{"mathc.c"} });
    const dylib = b.addLibrary(.{
        .name = "cinterop",
        .linkage = .dynamic,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
        .root_module = shared_mod,
    });
    b.installArtifact(dylib);

    // 6. A custom step backed by a system command.
    const check = b.addSystemCommand(&.{ "sh", "-c", "echo custom-step-ran" });
    b.step("check", "Custom step").dependOn(&check.step);

    const t = b.addTest(.{ .root_module = root_mod });
    b.step("test", "Run tests").dependOn(&b.addRunArtifact(t).step);
}
```

Note item 4: a C-only executable is an `addExecutable` whose root module has
**no** `root_source_file`. `Module.CreateOptions.root_source_file` is `?LazyPath`
and null means the module is made up of link objects only
(`lib/std/Build/Module.zig:195`).

## Diagnosing

- Undefined symbol at link time from a translated header: the header was
  translated but the C was never compiled. Add `addCSourceFiles`, a
  `linkSystemLibrary`, or an `addObjectFile`.
- `error: dependency on libc must be explicitly specified in the build command`:
  set `link_libc = true` on the module, or `-lc` on a bare `zig test`/`zig build-exe`.
- Header not found during translate-c but found during compilation: include paths
  are configured separately on the `*Step.TranslateC` and on the `*Module`. Set
  both.
- Silent ABI corruption at a boundary: check that every struct crossing it is
  `extern struct`, and that integer widths use the `c_` types rather than fixed
  Zig widths.
- Never pipe a build or test gate into `head`/`tail` and read `$?`; that is the
  pipe tail's status, not the command's.
