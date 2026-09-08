---
name: zig-comptime
description: >-
  Use when writing or debugging Zig comptime code on the zvm master toolchain:
  comptime parameters, comptime blocks or comptime var, generic functions,
  functions that return types, anytype and duck typing, inline fn / inline for /
  inline while, @TypeOf, @typeInfo reflection, building types with @Struct /
  @Enum / @Union / @Int, @hasDecl / @hasField / @field / @FieldType,
  @compileError diagnostics, std.meta helpers, @setEvalBranchQuota, or the
  errors "unable to resolve comptime value", "runtime value contains reference
  to comptime var", "evaluation exceeded 1000 backwards branches", "invalid
  builtin function: '@Type'", or "no field named 'fields' in struct
  'lang.Type.Struct'".
---

# Zig comptime on master

Ground truth for this file: the toolchain at `/Users/donaldfilimon/.zvm/bin/zig`,
version `0.17.0-dev.2018+ab30a0b9a`, its stdlib at
`/Users/donaldfilimon/.zvm/master/lib/std/`, and its shipped
`doc/langref.html` (same version stamp). Every runnable snippet below was
compiled and passes with `zig test` against that toolchain. Signature listings
and quoted `lang.zig` excerpts are reference, not programs, and the failure
demonstrations are shown together with the error they actually produce.

Verify anything you add:

```sh
mkdir -p /private/tmp/zig-comptime-scratch && cd /private/tmp/zig-comptime-scratch
/Users/donaldfilimon/.zvm/bin/zig test snippet.zig
```

Zig analysis is lazy. An unreferenced `fn` or type is never checked, so a
snippet only counts as verified if a `test` block actually drives it.

## Changed on master (read this first)

Master has moved away from several idioms that are still everywhere online.

| Older idiom | On this toolchain |
|---|---|
| `std.builtin.Type` | `std.lang.Type`. `std.builtin` is a deprecated alias: `pub const builtin = lang;` at `lib/std/std.zig:71`, marked "To be removed after Zig 0.17.0" at `lib/std/std.zig:68-70`. Both names resolve to the same namespace today. |
| `@Type(info)` | **Removed.** `@Type` is not a builtin at all: `error: invalid builtin function: '@Type'`. Use the per-kind constructors `@Int`, `@Struct`, `@Enum`, `@Union`, `@Pointer`, `@Fn`, `@Tuple`, `@SpirvType`, `@EnumLiteral`. |
| `@typeInfo(T).Struct`, `.Int`, `.Pointer` | snake_case tags: `.@"struct"`, `.int`, `.pointer`. TitleCase gives `error: no field named 'Int' in union 'lang.Type'`. The casing question is settled: **snake_case**, with `@"..."` quoting only where the tag collides with a keyword (`@"struct"`, `@"enum"`, `@"union"`, `@"fn"`, `@"opaque"`, `@"anyframe"`). |
| `info.@"struct".fields` (a `[]const StructField`) | **Gone.** `error: no field named 'fields' in struct 'lang.Type.Struct'`. Replaced by parallel arrays `field_names`, `field_types`, `field_attrs` (`lib/std/lang.zig:751`). `Type.StructField`, `Type.UnionField`, `Type.EnumField`, `Type.Declaration`, and `Type.Error` no longer exist. |
| `info.@"struct".decls` (a `[]const Declaration`) | `decl_names: []const [:0]const u8` (`lib/std/lang.zig:763`). Just names. |
| `ptr_info.is_const`, `.is_volatile`, `.alignment`, `.address_space` | Folded into `attrs: Attributes` (`lib/std/lang.zig:712`): `attrs.@"const"`, `attrs.@"volatile"`, `attrs.@"allowzero"`, `attrs.@"align"`, `attrs.@"addrspace"`. |
| `enum_info.is_exhaustive: bool` | `mode: Mode` where `Mode = enum { exhaustive, nonexhaustive }` (`lib/std/lang.zig:820`). |
| `field.default_value` | `field_attrs[i].default_value_ptr`, an opaque pointer. Read it with `field_attrs[i].defaultValue(FieldType)` (`lib/std/lang.zig:780`). |
| `@intFromEnum` / `@enumFromInt` | Deprecated in the langref. Use `@backingInt` / `@fromBackingInt`, which also cover packed structs and packed unions. |
| `@intFromFloat` | Deprecated; the langref says "Equivalent to `@trunc`". `@trunc`, `@round`, `@floor`, `@ceil` infer their result type, so an integer destination performs the conversion. |
| `std.meta.fields(T)` | `pub const fields = @compileError("deprecated in favor of @typeInfo");` at `lib/std/meta.zig:248`. Same for `declarationInfo` at `:246`. |
| Error set names sorted alphabetically | Source order. `@typeInfo(error{Foo, Bar}).error_set.error_names.?[0]` is `"Foo"`. |
| `comptime` on a container-level `const` | `error: redundant comptime keyword in already comptime scope`. Container scope is already comptime; the keyword is only for function bodies. |
| `@inComptime()` inside an explicit `comptime` block | `error: redundant '@inComptime' in comptime scope`. Call it from a helper function instead. |

Also note `@import("builtin")` and `std.builtin` are different things.
`@import("builtin")` is the compiler-generated module describing this build
(`target`, `mode`, `cpu`, `zig_backend`, `zig_version`, `is_test`).
`std.builtin` is the deprecated alias for `std.lang`, the language-level types.
The rename makes these easy to garble.

## `comptime` in each position

```zig
// 1. comptime parameter carrying a type.
fn Box(comptime T: type) type {
    return struct { value: T };
}

// 2. comptime parameter carrying a value: forces the argument comptime-known,
//    which lets it size a return type.
fn repeat(comptime n: usize, byte: u8) [n]u8 {
    return @splat(byte);
}

// 3. comptime block. At CONTAINER scope the keyword is redundant and is a
//    compile error, so it is omitted:
const table = blk: {
    var t: [8]u16 = undefined;
    for (&t, 0..) |*slot, i| slot.* = @intCast(i * i);
    break :blk t;
};

//    Inside a function body the keyword is required:
fn scaled(x: u8) u16 {
    const scale = comptime blk: {
        var v: u16 = 1;
        for (0..4) |_| v *= 3;
        break :blk v;
    };
    return scale * x;
}

// 4. comptime var: mutable during analysis only.
fn sumOfSquares(comptime n: usize) usize {
    comptime var acc: usize = 0;
    inline for (0..n) |i| acc += i * i;
    return acc;
}
```

## Generics: `comptime T: type` versus `anytype`

Use `comptime T: type` when the type must be nameable elsewhere in the
signature (two parameters that must match, a return type derived from it).
Use `anytype` when the caller's type is incidental and you recover what you
need with `@TypeOf`.

```zig
fn maxOf(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

fn sumAll(list: anytype) @TypeOf(list[0]) {
    var acc: @TypeOf(list[0]) = 0;
    for (list) |x| acc += x;
    return acc;
}
```

A generic container is a function returning a type. Type functions are
memoized on their arguments, so `Stack(u8, 4) == Stack(u8, 4)` and
`Stack(u8, 4) != Stack(u8, 5)`:

```zig
fn Stack(comptime T: type, comptime cap: usize) type {
    return struct {
        items: [cap]T = undefined,
        len: usize = 0,

        const Self = @This();
        pub const Item = T;

        pub fn push(self: *Self, v: T) !void {
            if (self.len == cap) return error.Full;
            self.items[self.len] = v;
            self.len += 1;
        }
        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            self.len -= 1;
            return self.items[self.len];
        }
    };
}
```

## `@typeInfo` shape on master

`@typeInfo(comptime T: type) std.lang.Type`. The union is at
`lib/std/lang.zig:641`:

```zig
pub const Type = union(enum) {
    type, void, bool, noreturn,
    int: Int,               // lang.zig:670  { signedness: Signedness, bits: u16 }
    float: Float,           // { bits: u16 }
    pointer: Pointer,       // lang.zig:683
    array: Array,           // lang.zig:723  { len: comptime_int, child: type, sentinel_ptr }
    @"struct": Struct,      // lang.zig:751
    comptime_float, comptime_int, undefined, null,
    optional: Optional,     // lang.zig:789  { child: type }
    error_union: ErrorUnion,// lang.zig:795  { error_set: type, payload: type }
    error_set: ErrorSet,    // lang.zig:802  { error_names: ?[]const [:0]const u8 }
    @"enum": Enum,          // lang.zig:808
    @"union": Union,        // lang.zig:825
    @"fn": Fn,              // lang.zig:848
    @"opaque": Opaque,      // lang.zig:925  { decl_names: []const [:0]const u8 }
    frame: Frame,
    @"anyframe": AnyFrame,
    vector: Vector,         // lang.zig:943  { len: comptime_int, child: type }
    enum_literal,
    spirv: Spirv,           // lang.zig:872
};
```

The container payloads, copied from source:

```zig
// lib/std/lang.zig:751
pub const Struct = struct {
    is_tuple: bool,
    layout: ContainerLayout,
    /// Always `null` if `layout != .@"packed"`.
    backing_integer: ?type,

    field_names: []const [:0]const u8,
    /// Guaranteed to have the same length as `field_names`.
    field_types: []const type,
    /// Guaranteed to have the same length as `field_names`.
    field_attrs: []const FieldAttributes,

    decl_names: []const [:0]const u8,

    pub const FieldAttributes = struct {       // lang.zig:765
        @"comptime": bool = false,
        @"align": ?usize = null,
        default_value_ptr: ?*const anyopaque = null,
        pub inline fn defaultValue(comptime attrs: FieldAttributes, comptime FieldType: type) ?FieldType { ... }
    };
};

// lib/std/lang.zig:808
pub const Enum = struct {
    tag_type: type,
    mode: Mode,
    field_names: []const [:0]const u8,
    field_values: []const comptime_int,
    decl_names: []const [:0]const u8,
    pub const Mode = enum { exhaustive, nonexhaustive };   // lang.zig:820
};

// lib/std/lang.zig:683
pub const Pointer = struct {
    size: Size,                 // .one, .many, .slice, .c   (lang.zig:703)
    attrs: Attributes,          // lang.zig:712
    child: type,
    sentinel_ptr: ?*const anyopaque,
    pub inline fn sentinel(comptime ptr: Pointer) ?ptr.child { ... }   // lang.zig:696
};

// lib/std/lang.zig:848
pub const Fn = struct {
    attrs: Attributes,          // { @"callconv", varargs }  (lang.zig:864)
    is_generic: bool,
    /// `null` means the return type is generic.
    return_type: ?type,
    /// A `null` element is an `anytype` or generic parameter.
    param_types: []const ?type,
    param_attrs: []const ParamAttributes,   // lang.zig:860
};
```

Reading it, compiled and passing:

```zig
const std = @import("std");
const testing = std.testing;

test "struct info is parallel arrays, not a fields slice" {
    const S = struct { a: u32, b: bool = true, c: []const u8 };
    const info = @typeInfo(S).@"struct";

    try testing.expect(info.layout == .auto);
    try testing.expect(info.backing_integer == null);
    try testing.expect(!info.is_tuple);
    try testing.expectEqual(@as(usize, 3), info.field_names.len);
    try testing.expectEqualStrings("b", info.field_names[1]);
    try testing.expectEqual(bool, info.field_types[1]);
    try testing.expectEqual(@as(?bool, true), info.field_attrs[1].defaultValue(bool));
    try testing.expectEqual(@as(?u32, null), info.field_attrs[0].defaultValue(u32));
    try testing.expectEqual(@as(usize, 0), info.decl_names.len);
}

test "pointer, fn and error set shapes" {
    const pi = @typeInfo([*:0]const u8).pointer;
    try testing.expect(pi.size == .many);
    try testing.expect(pi.attrs.@"const");
    try testing.expect(!pi.attrs.@"volatile");
    try testing.expectEqual(@as(?u8, 0), pi.sentinel());

    const fi = @typeInfo(@TypeOf(std.mem.eql)).@"fn";
    try testing.expect(fi.is_generic);
    try testing.expectEqual(@as(?type, type), fi.param_types[0]);

    // Error set names are in SOURCE order, not sorted alphabetically.
    const esi = @typeInfo(error{ Foo, Bar }).error_set;
    try testing.expectEqualStrings("Foo", esi.error_names.?[0]);
    try testing.expectEqualStrings("Bar", esi.error_names.?[1]);
}
```

## Building types: `@Type` is gone

Signatures from the shipped langref:

```zig
@Int(comptime signedness: std.lang.Signedness, comptime bits: u16) type
@Tuple(comptime field_types: []const type) type
@Pointer(comptime size, comptime attrs, comptime Element: type, comptime sentinel: ?Element) type
@Fn(comptime param_types: []const type,
    comptime param_attrs: *const [param_types.len]std.lang.Type.Fn.ParamAttributes,
    comptime ReturnType: type,
    comptime attrs: std.lang.Type.Fn.Attributes) type
@Struct(comptime layout: std.lang.Type.ContainerLayout,
        comptime BackingInt: ?type,
        comptime field_names: []const []const u8,
        comptime field_types: *const [field_names.len]type,
        comptime field_attrs: *const [field_names.len]std.lang.Type.Struct.FieldAttributes) type
@Union(comptime layout, comptime ArgType: ?type, comptime field_names,
       comptime field_types, comptime field_attrs) type
@Enum(comptime TagInt: type, comptime mode: std.lang.Type.Enum.Mode,
      comptime field_names: []const []const u8,
      comptime field_values: *const [field_names.len]TagInt) type
@SpirvType(comptime options: std.lang.Type.Spirv) type
@EnumLiteral() type
```

Verified construction:

```zig
const std = @import("std");
const testing = std.testing;

test "build types with the per-kind constructors" {
    try testing.expectEqual(u18, @Int(.unsigned, 18));
    try testing.expectEqual(i7, @Int(.signed, 7));

    const S = @Struct(.auto, null, &.{ "x", "y" }, &.{ u8, u16 }, &.{ .{}, .{} });
    var s: S = .{ .x = 1, .y = 2 };
    s.y += 1;
    try testing.expectEqual(@as(u16, 3), s.y);

    const E = @Enum(u8, .exhaustive, &.{ "red", "green" }, &.{ 10, 20 });
    try testing.expectEqual(@as(u8, 20), @backingInt(E.green));

    const U = @Union(.auto, null, &.{ "n", "f" }, &.{ u32, f32 }, &.{ .{}, .{} });
    const u: U = .{ .f = 1.5 };
    try testing.expectEqual(@as(f32, 1.5), u.f);

    const T = @Tuple(&.{ u8, bool });
    const t: T = .{ 5, true };
    try testing.expectEqual(@as(u8, 5), t[0]);

    try testing.expectEqual([]const u8, @Pointer(.slice, .{ .@"const" = true }, u8, null));
    try testing.expectEqual(fn (u8, u8) u8, @Fn(&.{ u8, u8 }, &.{ .{}, .{} }, u8, .{}));
}
```

Round-tripping needs one mechanical step: `@typeInfo` hands you **slices**
(`[]const type`) but the constructors want **pointers to sized arrays**
(`*const [N]type`). Slice a comptime-known length back into an array pointer:

```zig
fn Mirror(comptime T: type) type {
    const info = @typeInfo(T).@"struct";
    return @Struct(
        info.layout,
        info.backing_integer,
        info.field_names,                              // [:0]const u8 coerces to []const u8
        info.field_types[0..info.field_names.len],     // slice -> *const [N]type
        info.field_attrs[0..info.field_names.len],
    );
}
```

When every field shares one type and one attribute set, use the stdlib's
`&@splat(...)` form. This is exactly `std.enums.EnumFieldStruct`
(`lib/std/enums.zig:37`):

```zig
fn AllOf(comptime E: type, comptime Data: type) type {
    const names = @typeInfo(E).@"enum".field_names;
    return @Struct(.auto, null, names, &@splat(Data), &@splat(.{}));
}
```

## `inline fn`, `inline for`, `inline while`

`inline for` and `inline while` are **required**, not optimizations, whenever
the loop body needs the loop variable comptime-known or the iterations have
different types. Reflection loops are the canonical case: `@field` demands a
comptime field name, and each field has its own type.

```zig
const std = @import("std");
const Rec = struct { a: u8, b: u16, c: bool };

fn describe(rec: Rec, buf: []u8) ![]u8 {
    var w: std.Io.Writer = .fixed(buf);
    inline for (@typeInfo(Rec).@"struct".field_names) |name| {
        try w.print("{s}={any};", .{ name, @field(rec, name) });
    }
    return w.buffered();
}

test describe {
    var buf: [64]u8 = undefined;
    const out = try describe(.{ .a = 1, .b = 2, .c = true }, &buf);
    try std.testing.expectEqualStrings("a=1;b=2;c=true;", out);
}
```

`inline while` covers the same ground when you need an index into
heterogeneous comptime data:

```zig
fn firstTypeName(comptime types: []const type) []const u8 {
    comptime var i: usize = 0;
    inline while (i < types.len) : (i += 1) {
        if (@sizeOf(types[i]) == 2) return @typeName(types[i]);
    }
    return "none";
}
```

`inline fn` forces the body into the caller. It is **not** needed merely to use
a call in a comptime position: an array-length slot is already a comptime
context, so a plain function works there. What `inline` buys is that the
result of a call made from a **runtime** scope stays comptime-known.

```zig
const std = @import("std");

fn plain(comptime T: type) u16 {
    return @typeInfo(T).int.bits;
}
inline fn inl(comptime T: type) u16 {
    return @typeInfo(T).int.bits;
}

// Both work directly in an array-length position, inline or not.
const direct: [plain(u5)]u8 = undefined;

test "inline is what keeps a runtime-scope call result comptime-known" {
    _ = direct;
    const b = inl(u24);
    const arr_b: [b]u8 = undefined; // ok: inl was inlined, b is comptime-known
    try std.testing.expectEqual(@as(usize, 24), arr_b.len);
}
```

Drop the `inline` and bind the result in a runtime scope and it stops working:

```zig
fn plain(comptime T: type) u16 { return @typeInfo(T).int.bits; }
test "not inline" {
    const a = plain(u24);
    const arr: [a]u8 = undefined;
    _ = arr;
}
```
```
error: unable to resolve comptime value
note: types must be comptime-known
```

A plain `for` over a comptime-known slice still produces runtime values. That
is fine, and preferable, when every element has the same type. Reach for
`inline` only when it buys comptime-knownness.

## Reflection builtins

```zig
@TypeOf(...) type
@typeName(T: type) *const [N:0]u8
@This() type
@hasDecl(comptime Namespace: type, comptime name: []const u8) bool
@hasField(comptime T: type, comptime name: []const u8) bool
@field(lhs: anytype, comptime field_name: []const u8) (field)
@FieldType(comptime Type: type, comptime field_name: []const u8) type
@src() std.lang.SourceLocation        // lang.zig:627
@inComptime() bool
```

A decl is not a field: `@hasDecl(Widget, "kind")` is true while
`@hasField(Widget, "kind")` is false. `@field` is also an lvalue:

```zig
const std = @import("std");
const Widget = struct { id: u32, label: []const u8 };

test "@field as an lvalue" {
    var w = Widget{ .id = 1, .label = "a" };
    inline for (@typeInfo(Widget).@"struct".field_names) |name| {
        if (@FieldType(Widget, name) == u32) @field(w, name) = 42;
    }
    try std.testing.expectEqual(@as(u32, 42), w.id);
}
```

`@src()` returns `{ module, file, fn_name, line, column }`, all
`[:0]const u8` except the two `u32`s. Inside a test its `fn_name` is the full
test name, for example `"test.@src and @inComptime"`.

## Good compile errors

`@compileError(comptime msg: []const u8) noreturn`. Gate a generic on the
contract you actually need, in a `comptime` block at the top of the type
function, so the caller gets one sentence instead of a cascade of type errors
from deep inside the body.

```zig
fn Serializer(comptime T: type) type {
    comptime {
        if (@typeInfo(T) != .@"struct")
            @compileError("Serializer requires a struct, got " ++ @typeName(T));
        if (!@hasDecl(T, "encode"))
            @compileError(@typeName(T) ++ " is missing `pub fn encode(self: @This()) u64`");
        const info = @typeInfo(@TypeOf(@field(T, "encode"))).@"fn";
        if (info.return_type != u64)
            @compileError(@typeName(T) ++ ".encode must return u64");
    }
    return struct {
        pub fn run(v: T) u64 { return v.encode(); }
    };
}
```

Message strings must be comptime-concatenated with `++`. For formatted
diagnostics use `std.fmt.comptimePrint`. `@compileLog(...)` prints during
analysis but is itself a compile error if left in, so it is a debugging tool
only.

## `std.meta` helpers that exist on master

Verified present in `lib/std/meta.zig`:

| Helper | Line | Note |
|---|---|---|
| `stringToEnum(T, name) ?T` | 18 | |
| `alignment(T) comptime_int` | 33 | |
| `Child(T) type` | 56 | element of pointer/array/optional/vector |
| `Elem(T) type` | 75 | |
| `sentinel(T) ?Elem(T)` | 107 | |
| `Sentinel(T, val) type` | 143 | |
| `containerLayout(T)` | 170 | |
| `declarations(T) []const [:0]const u8` | 204 | names only |
| `fieldNames(T) []const [:0]const u8` | 309 | deprecated in favor of `@typeInfo` |
| `fieldTypes(T) []const type` | 350 | deprecated in favor of `@typeInfo` |
| `fieldInfo(T, field)` | 253 | deprecated in favor of `@typeInfo` |
| `tags(T) *const [N]T` | 379 | |
| `FieldEnum(T) type` | 406 | |
| `DeclEnum(T) type` | 481 | |
| `BareUnion(T) type` | 514 | |
| `BackingInt(T) type` | 524 | |
| `Tag(T) type` | 553 | |
| `activeTag(u)` | 576 | |
| `eql(a, b) bool` | 601 | |
| `fieldIndex(T, name) ?comptime_int` | 729 | fields only, not decls |
| `Float(bit_count) type` | 737 | |
| `ArgsTuple(Function) type` | 762 | |
| `isError(error_union) bool` | 829 | |
| `hasFn(T, name) bool` | 840 | |
| `hasMethod(T, name) bool` | 869 | |
| `hasUniqueRepresentation(T) bool` | 914 | |
| `Slice(Pointer) type` | 1086 | |
| `AbsorbSentinel(Pointer) type` | 1107 | |
| `TrailerFlags` | 8 | |

Removed, and they fail loudly rather than silently:
`std.meta.fields` and `std.meta.declarationInfo` are `@compileError` stubs
(`lib/std/meta.zig:246` and `:248`).

## Comptime-known versus runtime-known

A value is comptime-known if analysis can compute it without running the
program. Literals, `const` bound to comptime expressions, container-scope
declarations, and `comptime` parameters are comptime-known. A `var`, a
function parameter without `comptime`, and anything derived from I/O are not,
regardless of how obvious the value looks.

```zig
const std = @import("std");

test "comptime-known versus runtime-known" {
    const a = 10; // comptime_int, comptime-known
    var b: usize = 10; // runtime-known
    b += 0;

    const arr: [a]u8 = @splat(0); // ok: array length must be comptime-known
    const slice = arr[0..b]; // b is fine here: slicing takes runtime bounds
    try std.testing.expectEqual(@as(usize, 10), slice.len);
}
```

Only comptime-known values can size an array, select a type, index a tuple,
or feed a `comptime` parameter.

## Classic failure modes

Each of these was compiled to capture the exact message.

**Runtime value in a comptime context.** The giveaway is that the note names
the comptime-only return type.

```zig
fn pick(n: usize) type { return [n]u8; }
var n: usize = 3; n += 0;
_ = pick(n);
```
```
error: unable to resolve comptime value
note: call to function with comptime-only return type 'type' is evaluated at comptime
note: types are not available at runtime
```
Fix: mark the parameter `comptime n: usize`, or make the caller's value
comptime-known.

**A comptime var escaping into runtime.**

```zig
fn leak() *u32 {
    comptime var x: u32 = 1;
    return &x;
}
```
```
error: runtime value contains reference to comptime var
```
Fix: freeze the value into a `const` first, then take that address. This is
the `std.meta.tags` idiom (`lib/std/meta.zig:379`):

```zig
fn bakedTable(comptime n: usize) *const [n]u16 {
    return comptime blk: {
        var t: [n]u16 = undefined;
        for (&t, 0..) |*slot, i| slot.* = @intCast(i * 3);
        const frozen = t;
        break :blk &frozen;
    };
}
```

**Quota exhaustion.** The default budget is 1000 backwards branches per
comptime evaluation.

```zig
const total = comptime blk: {
    var acc: u64 = 0; var i: u64 = 0;
    while (i < 100_000) : (i += 1) acc += i;
    break :blk acc;
};
```
```
error: evaluation exceeded 1000 backwards branches
note: use @setEvalBranchQuota() to raise the branch limit from 1000
```
Fix: `@setEvalBranchQuota(comptime new_quota: u32)` as the first statement of
the evaluation. It only ever raises the limit and applies to the whole
enclosing comptime evaluation, not just the following statement. The stdlib
sizes it from the data, for example `@setEvalBranchQuota(2 * @typeInfo(E).@"enum".field_names.len)`
throughout `lib/std/enums.zig`.

**Reaching for a removed spelling.** `@Type(...)` gives
`invalid builtin function: '@Type'`; `info.Int` gives
`no field named 'Int' in union 'lang.Type'`; `info.@"struct".fields` gives
`no field named 'fields' in struct 'lang.Type.Struct'`. All three mean you are
applying pre-master documentation. See the table at the top.

**Result-type-inferring builtins with no destination.** `@truncate`,
`@intCast`, `@bitCast`, `@floatCast`, `@floatFromInt`, `@fromBackingInt`,
`@ptrCast`, `@errorCast` all infer their result type. Passing one directly to
an `anytype` parameter fails:

```
error: @truncate must have a known result type
note: result type is unknown due to anytype parameter
note: use @as to provide explicit result type
```
Fix: bind to a typed `const`, or wrap in `@as(T, ...)`.

## See also

`zig-builtins` for the full `@`-builtin reference on this same toolchain.
