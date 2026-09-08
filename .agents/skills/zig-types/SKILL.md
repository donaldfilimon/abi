---
name: zig-types
description: Zig type system on master (0.17.0-dev). Use when writing or reviewing Zig types, choosing between [N]T / []T / [*]T / [*c]T, sentinel-terminated types, optionals, structs (packed, extern, tuples), enums (non-exhaustive), unions (tagged, bare, packed), opaque, @Vector, anytype, comptime type values, @typeInfo reflection, coercion and peer type resolution, align, or deciding between @intCast, @truncate, @bitCast, @intFromFloat, @ptrCast and @alignCast.
---

# Zig types on master

Ground truth for this skill is the locally installed toolchain, not the 0.13/0.14
documentation and not model memory:

- `zig` `0.17.0-dev.2018+ab30a0b9a` at `/Users/donaldfilimon/.zvm/bin/zig`
- stdlib at `/Users/donaldfilimon/.zvm/master/lib/std/`
- langref for this exact build at `/Users/donaldfilimon/.zvm/master/doc/langref.html`

Verify anything uncertain by compiling it in `/private/tmp/zig-skill-scratch`, never
in the home root and never on an iCloud path.

Related skills: `zig-comptime` for generic functions and comptime evaluation, `zig-errors` for error unions, `zig-memory` for allocators and alignment in practice, `zig-toolchain` for which `zig` binary is active.

## Changed on master (check this first)

These contradict widely repeated older Zig idioms. Each was verified by compiling
against the installed toolchain.

1. **`std.builtin` is deprecated and is now an alias of `std.lang`.**
   `lib/std/std.zig:71` reads `pub const builtin = lang;` with the comment
   `Deprecated; use lang.`, and `lib/std/std.zig:72` is `pub const lang = @import("lang.zig")`.
   There is no `lib/std/builtin.zig` on master. `@import("builtin")` (the compiler-generated
   module, distinct from `std.builtin`) still exists and its types are spelled
   `std.lang.OutputMode`, `std.lang.CompilerBackend`, and so on.

2. **`@Type` does not exist.** `@Type(.{ .int = ... })` fails with
   `error: invalid builtin function: '@Type'`. Reification is now per-kind:
   `@Int`, `@Struct`, `@Union`, `@Enum`, `@Fn`, `@Pointer`, `@Tuple`, `@EnumLiteral`,
   `@Vector`, `@SpirvType`. The authoritative list of all 128 builtins for this build is
   `lib/std/zig/BuiltinFn.zig`.

3. **`@typeInfo` payloads were restructured into parallel arrays.**
   `lang.Type.Struct` (`lib/std/lang.zig:751`) has `field_names`, `field_types`, and
   `field_attrs`, not a `fields: []const StructField` slice. `lang.Type.Enum`
   (`lib/std/lang.zig:808`) has `field_names`, `field_values`, `decl_names`, and a
   `mode: .exhaustive | .nonexhaustive`. `lang.Type.Pointer` (`lib/std/lang.zig:683`)
   moved `is_const`, `is_volatile`, `alignment`, and `address_space` under
   `attrs: Attributes`, so it is now `info.attrs.@"const"`.

4. **`std.meta.fields` is a hard compile error**, not a deprecation warning:
   `lib/std/meta.zig:248` is `pub const fields = @compileError("deprecated in favor of @typeInfo")`.
   `std.meta.declarationInfo` is the same at `lib/std/meta.zig:246`.

5. **`@backingInt` and `@fromBackingInt` are new and take a VALUE, not a type.**
   `@backingInt(Flags)` on a type fails with
   `error: expected enum, tagged union, packed union or packed struct, found 'type'`.
   Use `@backingInt(flags_value)`. To get the backing integer *type* of a packed struct,
   read `@typeInfo(Flags).@"struct".backing_integer.?`.

Also worth knowing: `Type.Pointer.Size` tags are lowercase (`one`, `many`, `slice`, `c`),
and `@enumFromInt` in an `anytype` argument position fails with
`error: @enumFromInt must have a known result type`, so wrap it in `@as(E, ...)`.

## Primitives

Widths measured on aarch64-macos with this toolchain (`types/01-primitives.zig`):

| Fact | Value |
|---|---|
| `uN` / `iN` | arbitrary `N` from 0 to 65535, `@bitSizeOf(i7) == 7` |
| `u0` | `@bitSizeOf` 0, `@sizeOf` 0 |
| `u1` | `@bitSizeOf` 1, `@sizeOf` 1 |
| `u24` | `@bitSizeOf` 24, `@sizeOf` **4**, not 3 |
| `usize` / `isize` | 64 bits here, target dependent |
| `f16 f32 f64 f80 f128` | 16/32/64/80/128 bits, `@sizeOf(f80) == 16` |
| `bool` | 1 bit, 1 byte |
| `void` | `@sizeOf` 0 |
| `c_int` / `c_long` | 32 / 64 bits on LP64 |

`comptime_int` and `comptime_float` have no runtime representation until coerced.

## Arrays, slices, pointers

```zig
[N]T      // value type, length is part of the type, copied on assignment
[]T       // slice: { ptr, len }, @sizeOf == 2 * @sizeOf(usize)
[*]T      // many-pointer: just an address, no length, supports [] and +
*[N]T     // pointer to array, coerces to []T and to [*]T
*T        // single item, no pointer arithmetic, no indexing
[*c]T     // C pointer: nullable, allowzero, coerces to and from *T unchecked
```

Slicing rule that surprises people: with both bounds comptime-known you get a
**pointer to array**, not a slice.

```zig
var a: [4]u8 = .{ 1, 2, 3, 4 };
const p = a[1..3];     // *[2]u8
var i: usize = 1; _ = &i;
const s = a[i..3];     // []u8
```

Const-ness is part of the pointer type. `[]u8` coerces to `[]const u8` but never the
reverse; `@constCast` removes it, and writing through the result is illegal behavior
unless the underlying memory really is mutable.

## Sentinel-terminated types

```zig
[N:x]T    // array with a sentinel at index N
[:x]T     // slice guaranteeing element at index len is x
[*:x]T    // many-pointer guaranteeing a terminating x
```

String literals are `*const [N:0]u8`, so `"abc".len == 3` and `"abc"[3] == 0`.
`[:0]const u8` coerces to `[]const u8` (the guarantee is dropped); the reverse needs a
re-slice such as `s[0..n :0]`, which asserts the sentinel is actually present.
`std.mem.span(ptr)` walks a `[*:0]T` to produce a `[:0]T`.

Slicing to a sentinel with comptime bounds gives `*[N:0]T`, which then coerces to `[:0]T`.

## Optionals

`?T` plus the unwrap forms:

```zig
maybe orelse default        // value or default
maybe orelse return err     // orelse can take control flow
maybe.?                     // asserts non-null, panics in safe modes
if (maybe) |v| { ... } else { ... }
if (maybe) |*p| p.* = 2;    // capture by pointer to mutate in place
while (next()) |v| { ... }  // loop until null
```

`?*T` is pointer-sized (null pointer optimization). `?u8` is not: a non-pointer
optional carries a tag.

## Structs

Field defaults, `.{ ... }` with inferred result type, and three layouts:

```zig
const Point  = struct { x: i32, y: i32 = 0 };            // .auto, no layout guarantee
const Ext    = extern struct { a: u8, b: u32 };          // C ABI, @sizeOf 8, @offsetOf(b) 4
const Flags  = packed struct(u8) { a: bool, b: bool, rest: u6 };  // no padding
```

Only `extern` and `packed` pin the layout. Never `@bitCast` an `.auto` struct and expect
a stable encoding. For a packed struct:

```zig
try expectEqual(u8, @typeInfo(Flags).@"struct".backing_integer.?);
const raw = @backingInt(flags);              // value in, integer out
const back: Flags = @fromBackingInt(raw);    // integer in, value out
```

Tuples are structs whose field names are numbers: `t[0]`, `t.len`, and
`@typeInfo(@TypeOf(t)).@"struct".is_tuple == true`.

## Enums

```zig
const Color = enum(u8) { red = 1, green = 2, blue = 4 };
const Status = enum(u8) { ok = 0, err = 1, _ };   // non-exhaustive
```

An exhaustive enum switch needs no `else` when every tag is listed. A non-exhaustive
enum accepts any value of its tag type and its switch needs an `_` arm (or `else`):

```zig
switch (s) { .ok => ..., .err => ..., _ => ... }
```

`@intFromEnum`, `@enumFromInt`, `@tagName`. In an `anytype` argument,
`@enumFromInt` needs `@as(E, @enumFromInt(n))`.
`@typeInfo(E).@"enum"` gives `tag_type`, `mode`, `field_names`, `field_values`.

## Unions

```zig
union(enum) { int: i32, float: f64, none }   // tagged, inferred tag enum
union(Tag)  { a: u8, b: u16 }                // tagged with an explicit enum
union       { i: i32, f: f32 }               // bare: reading the inactive field is illegal behavior
packed union { a: u32, b: packed struct(u32) { lo: u16, hi: u16 } }
```

Switch on a tagged union to capture the payload, and use `|*p|` to get a pointer to the
active payload. `std.meta.activeTag(v)` and `@as(Tag, v)` both read the tag.
`@unionInit(U, "field", value)` builds one from a comptime field name.
A `packed union` may only contain types with a bit-packed representation, so
`b: [4]u8` is rejected; use a `packed struct` instead.

## opaque and anyopaque

`opaque {}` is a type of unknown size, usable only behind a pointer. It is how C handles
are modeled. `anyopaque` is the type-erased pointee: `*anyopaque` is what
`std.mem.Allocator` stores for its implementation state. Restore with
`@ptrCast(@alignCast(erased))`, and keep constness: a `*const T` must be erased to
`*const anyopaque`.

## @Vector

```zig
const a: @Vector(4, i32) = .{ 1, 2, 3, 4 };
const b: @Vector(4, i32) = @splat(10);
const c = a + b;                     // element-wise
@reduce(.Add, a)                     // fold to a scalar
@select(i32, pred_vec, a, b)
@shuffle(i32, a, b, mask)
```

Vectors coerce to and from fixed-size arrays of the same length and element type.
`@Vector(8, bool)` is bit-packed: `@bitSizeOf` is 8.

## type as a value, anytype, reflection

`type` is a comptime value, so a generic container is a function returning `type`:

```zig
fn Pair(comptime T: type) type { return struct { a: T, b: T }; }
```

`anytype` is resolved per call site. `@TypeOf(x)` and `@typeName(T)` are the usual
introspection entry points; `@FieldType(S, "x")` gets one field's type without walking
`@typeInfo`.

Master-shaped reflection:

```zig
const info = @typeInfo(S).@"struct";
info.field_names[0]        // [:0]const u8
info.field_types[0]        // type
@typeInfo(*const u8).pointer.attrs.@"const"   // true
@typeInfo([]u8).pointer.size                  // .slice
const T = @Int(.unsigned, 12);                // u12, replaces @Type
const Tup = @Tuple(&.{ u8, bool });
```

## Coercion and peer type resolution

Implicit coercions that always hold: integer and float widening, comptime-known values
that fit, stricter qualification only (mutable to const, non-volatile to volatile),
`T` into `?T` and into `E!T` (including through both layers), `*[N]T` into `[]T` and into
`[*]T`, tuples into arrays, sentinel slices into plain slices.

Narrowing is never implicit. `const c: u8 = some_u16;` is a compile error; write
`@intCast`.

Peer type resolution runs in `switch`, `if`, `while`, `for`, multiple `break` values,
and some binary operations (langref section `Peer-Type-Resolution` in this build's
`doc/langref.html`). It picks one type every branch coerces into:

```zig
if (b) @as(i8, 1) else @as(i16, 2)   // i16
if (b) @as(u8, 1) else null          // ?u8
if (b) "true" else "false"           // []const u8, two different array lengths
```

## Casting: which builtin, and when

| Need | Builtin | Fails how |
|---|---|---|
| Same value, narrower integer | `@intCast` | panics in Debug and ReleaseSafe if it does not fit |
| Keep the low bits | `@truncate` | cannot fail |
| Reinterpret the bits, same bit width | `@bitCast` | compile error if widths differ |
| Float to integer, truncating toward zero | `@intFromFloat` | safety-checked |
| Integer to float | `@floatFromInt` | rounds |
| Between float widths | `@floatCast` | rounds |
| Change the pointee type | `@ptrCast` | unchecked, you own the aliasing rules |
| Recover a stricter alignment | `@alignCast` | runtime-checked in safe modes |
| Address to and from integer | `@intFromPtr`, `@ptrFromInt` | |
| Pin a result type | `@as` | |
| Drop const or volatile | `@constCast`, `@volatileCast` | writing through it may still be illegal behavior |
| Narrow an error set | `@errorCast` | checked in safe modes |

`@intCast` and `@truncate` are not interchangeable: `@as(u8, @truncate(0xDEADBEEF))` is
`0xEF`, while `@as(u8, @intCast(0xDEADBEEF))` panics.

## align

Alignment is part of the pointer type. Relaxing it is implicit, tightening it needs
`@alignCast`:

```zig
var x: u32 align(16) = 1;
const p = &x;                       // *align(16) u32
const relaxed: *align(4) u32 = p;   // fine
const under: *align(1) u32 = @ptrCast(&byte_buf[0]);
```

`std.mem.Alignment` (`lib/std/mem.zig:25`) is an `enum(math.Log2Int(usize))`, so
`@intFromEnum` gives a log2 value, not bytes. Use `.of(T)`, `.fromByteUnits(n)`,
`.toByteUnits()`, `.forward(addr)`, `.backward(addr)`, `.check(addr)`, `.max`, `.min`.
This is the type the allocator vtable takes.

## Verified snippets

Compiled with `zig 0.17.0-dev.2018+ab30a0b9a` in `/private/tmp/zig-skill-scratch/types/`
via `zig test <file>`. 13 of 13 pass:

`01-primitives.zig`, `02-arrays-slices.zig`, `03-sentinel.zig`, `04-optional.zig`,
`05-struct.zig`, `06-enum.zig`, `07-union.zig`, `08-opaque.zig`, `09-vector.zig`,
`10-coercion.zig`, `11-casting.zig`, `12-align.zig`, `13-type-as-value.zig`.

Negative controls that must fail, and do:
`@Type(...)` gives `invalid builtin function: '@Type'`;
`@backingInt(SomeType)` gives `expected enum, tagged union, packed union or packed struct, found 'type'`;
`std.meta.fields(E)` gives `deprecated in favor of @typeInfo`.

The scratch directory is disposable (`/private/tmp` is wiped on reboot). Every construct
above appears inline in this skill, so re-verification means pasting a block into a fresh
scratch file, not recovering these paths.
