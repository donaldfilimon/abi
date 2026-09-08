---
name: zig-builtins
description: >-
  Use when reaching for a Zig @-builtin on the zvm master toolchain and needing
  its exact signature or the right one for the job: casting (@as, @intCast,
  @bitCast, @ptrCast, @truncate, @backingInt, @fromBackingInt), reflection
  (@typeInfo, @TypeOf, @hasDecl, @FieldType), type construction (@Int, @Struct,
  @Enum, @Union, @Pointer, @Fn, @Tuple), memory (@memcpy, @memset, @splat,
  @sizeOf, @offsetOf, @fieldParentPtr), atomics (@atomicLoad, @atomicRmw,
  @cmpxchgStrong), math and overflow (@addWithOverflow, @divFloor, @mod, @abs),
  vectors (@Vector, @reduce, @shuffle, @select), control (@call, @branchHint,
  @setEvalBranchQuota, @compileError, @panic), C interop (@cVaArg, @extern,
  @export), or target introspection. Also use for "invalid builtin function",
  "must have a known result type", or when a builtin named in older docs no
  longer exists.
---

# Zig `@`-builtins on master

Ground truth for this file: the toolchain at `/Users/donaldfilimon/.zvm/bin/zig`,
version `0.17.0-dev.2018+ab30a0b9a`, and its stdlib at
`/Users/donaldfilimon/.zvm/master/lib/std/`.

## How this list was sourced

The list is **complete for documented builtins**, not a guessed subset.

1. The toolchain ships `doc/langref.html`, whose own version stamp is
   `0.17.0-dev.2018+ab30a0b9a`, matching `zig version` exactly. Its
   "Builtin Functions" table of contents enumerates **124** builtins, and every
   signature below is copied from that document.
2. Cross-checked against the compiler binary itself:
   `/usr/bin/strings -n 3 ~/.zvm/master/zig | grep -E '^@[A-Za-z][A-Za-z0-9]*$'`.
   All 124 langref names appear in the binary; nothing in the langref is absent
   from the binary. Use `/usr/bin/strings` explicitly, since `~/.swiftly/bin`
   shims `strings` on this machine.
3. Every one of the 400 other `@`-strings in the binary was then probed by
   compiling `_ = @NAME();` and checking for `invalid builtin function`.
   Exactly **four** are recognized but undocumented in the TOC:
   `@Frame` and `@frame`, which reject with
   `error: async has not been implemented in the self-hosted compiler yet`,
   and `@disableInstrumentation` and `@disableIntrinsics`, which both compile
   and are used in the stdlib (`lib/std/debug.zig:441`, `lib/std/pie.zig:319`,
   `lib/compiler_rt.zig:605`). So the recognized set is **128**, of which 126
   are usable.

Every group below has a compiled-and-passing test behind it. Blocks written as
signature listings or member listings are reference, not programs. Regenerate
the runnable ones with:

```sh
mkdir -p /private/tmp/zig-comptime-scratch && cd /private/tmp/zig-comptime-scratch
/Users/donaldfilimon/.zvm/bin/zig test snippet.zig
```

## Changed on master

- **`@Type` no longer exists.** `error: invalid builtin function: '@Type'`.
  It was split into `@Int`, `@Struct`, `@Enum`, `@Union`, `@Pointer`, `@Fn`,
  `@Tuple`, `@SpirvType`, `@EnumLiteral`.
- **`@intFromEnum` and `@enumFromInt` are deprecated.** The langref says to use
  `@backingInt` and `@fromBackingInt`, which additionally cover packed structs
  and packed unions rather than enums alone.
- **`@intFromFloat` is deprecated**, "Equivalent to `@trunc`". `@trunc`,
  `@round`, `@floor`, `@ceil` infer their result type, so an integer
  destination performs the float-to-int conversion directly.
- **`std.builtin` is a deprecated alias for `std.lang`**
  (`lib/std/std.zig:68-72`). Builtin argument types like `AtomicOrder`,
  `CallModifier`, `ReduceOp`, `Signedness`, `BranchHint` live in
  `lib/std/lang.zig`.
- **`std.lang.Optimize` variants are `.debug`, `.safe`, `.fast`, `.small`**
  (`lib/std/lang.zig:115`). The TitleCase `.Debug` / `.ReleaseSafe` /
  `.ReleaseFast` / `.ReleaseSmall` survive only as deprecated consts at
  `lib/std/lang.zig:127+`, "to be removed after 0.18.0". `OptimizeMode` is an
  alias for `Optimize` (`lib/std/lang.zig:111`).
- `@typeInfo` returns `std.lang.Type` with **snake_case** tags and parallel
  `field_names` / `field_types` / `field_attrs` arrays. See the `zig-comptime`
  skill for the full shape.

## Type operations and construction

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@TypeOf` | `@TypeOf(...) type` | Recover the type of an expression; with several arguments it gives their peer-resolved type. Operands are not evaluated. |
| `@typeInfo` | `@typeInfo(comptime T: type) std.lang.Type` | The only way to inspect a type's structure. |
| `@typeName` | `@typeName(T: type) *const [N:0]u8` | Diagnostics. Fully qualified, so match with `endsWith`, not equality. |
| `@This` | `@This() type` | The innermost enclosing struct/union/enum, for `Self` aliases in generated types. |
| `@FieldType` | `@FieldType(comptime Type: type, comptime field_name: []const u8) type` | One field's type by name, without unpacking `@typeInfo`. |
| `@hasDecl` | `@hasDecl(comptime Namespace: type, comptime name: []const u8) bool` | Duck-typing gate on a declaration. A decl is not a field. |
| `@hasField` | `@hasField(comptime T: type, comptime name: []const u8) bool` | Duck-typing gate on a field. |
| `@field` | `@field(lhs: anytype, comptime field_name: []const u8) (field)` | Field access by comptime string; also an lvalue. |
| `@Int` | `@Int(comptime signedness: std.lang.Signedness, comptime bits: u16) type` | Build an integer type of computed width. |
| `@Tuple` | `@Tuple(comptime field_types: []const type) type` | Build an anonymous tuple type. |
| `@Pointer` | `@Pointer(comptime size, comptime attrs, comptime Element: type, comptime sentinel: ?Element) type` | Build a pointer type with computed constness/alignment/sentinel. |
| `@Fn` | `@Fn(comptime param_types: []const type, comptime param_attrs: *const [N]ParamAttributes, comptime ReturnType: type, comptime attrs: Attributes) type` | Build a function type. |
| `@Struct` | `@Struct(comptime layout, comptime BackingInt: ?type, comptime field_names: []const []const u8, comptime field_types: *const [N]type, comptime field_attrs: *const [N]FieldAttributes) type` | Build a struct type. |
| `@Union` | `@Union(comptime layout, comptime ArgType: ?type, comptime field_names, comptime field_types, comptime field_attrs) type` | Build a union type. `ArgType` is the tag enum for `auto`, the backing int for `packed`. |
| `@Enum` | `@Enum(comptime TagInt: type, comptime mode: Mode, comptime field_names: []const []const u8, comptime field_values: *const [N]TagInt) type` | Build an enum type. |
| `@SpirvType` | `@SpirvType(comptime options: std.lang.Type.Spirv) type` | SPIR-V samplers and images only. |
| `@EnumLiteral` | `@EnumLiteral() type` | The type of a bare `.foo` literal. |
| `@Vector` | `@Vector(len: comptime_int, Element: type) type` | SIMD vector type. |

The constructors want `*const [N]T` where `@typeInfo` gives `[]const T`. Slice a
comptime-known length (`info.field_types[0..info.field_names.len]`) or build the
array with `&@splat(...)`, which is what `lib/std/enums.zig:37` does.

## Casting and conversion

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@as` | `@as(comptime T: type, expression) T` | The default. Only permits unambiguous, safe coercions; prefer it whenever it compiles. |
| `@intCast` | `@intCast(int: anytype) anytype` | Change integer width **preserving value**. Out of range is safety-checked Illegal Behavior. |
| `@truncate` | `@truncate(integer: anytype) anytype` | Change integer width **discarding high bits**. Never fails; use when wrapping is intended. |
| `@bitCast` | `@bitCast(value: anytype) anytype` | Reinterpret bits between same-size types with defined layout. Rejects pointers and layout-less types (bare structs, slices, optionals, error unions). |
| `@floatCast` | `@floatCast(value: anytype) anytype` | Between float widths. |
| `@floatFromInt` | `@floatFromInt(int: anytype) anytype` | Int to float. |
| `@trunc` / `@round` / `@floor` / `@ceil` | `@trunc(value: anytype) @TypeOf(value)` | Float rounding. The `@TypeOf(value)` signature is the default only: the langref adds "When the inferred result type is an integer, the integer part is extracted from the truncated result", so an integer destination performs the float-to-int conversion. Out of range is safety-checked Illegal Behavior. `@intFromFloat` is the deprecated spelling of `@trunc`. All four verified against `i32` destinations. |
| `@intFromBool` | `@intFromBool(value: bool) u1` | |
| `@backingInt` | `@backingInt(enum_or_bitpack: T) BackingInt(T)` | Enum, packed struct, or packed union to its integer. Also acts on a tagged union's active tag. Replaces `@intFromEnum`. |
| `@fromBackingInt` | `@fromBackingInt(backing_int: BackingInt(T)) T` | The inverse. Invalid enum values are safety-checked Illegal Behavior. Replaces `@enumFromInt`. |
| `@ptrCast` | `@ptrCast(value: anytype) anytype` | Change pointee type only. Cannot change const, volatile, address space, alignment, or slice-ness. |
| `@constCast` | `@constCast(value: anytype) DestType` | Remove `const`. Writing through it when the original was truly const is Illegal Behavior. |
| `@volatileCast` | `@volatileCast(value: anytype) DestType` | Remove `volatile`. |
| `@alignCast` | `@alignCast(ptr: anytype) anytype` | Increase the claimed alignment. Safety-checked. |
| `@addrSpaceCast` | `@addrSpaceCast(ptr: anytype) anytype` | Change address space. |
| `@intFromPtr` | `@intFromPtr(value: anytype) usize` | |
| `@ptrFromInt` | `@ptrFromInt(address: usize) anytype` | |
| `@errorCast` | `@errorCast(value: anytype) anytype` | Narrow `anyerror` to a subset error set. |
| `@intFromError` | `@intFromError(err: anytype) @Int(.unsigned, @bitSizeOf(anyerror))` | |
| `@errorFromInt` | `@errorFromInt(value: @Int(.unsigned, @bitSizeOf(anyerror))) anyerror` | |
| `@errorName` | `@errorName(err: anyerror) [:0]const u8` | |

Most of these **infer** their result type. Passing one straight into an
`anytype` parameter fails with `must have a known result type` plus
`result type is unknown due to anytype parameter`. Bind to a typed `const` or
wrap in `@as(T, ...)`.

Verified:

```zig
const std = @import("std");
const testing = std.testing;

const Color = enum(u8) { red = 1, blue = 7 };
const Flags = packed struct(u8) { a: bool, b: bool, rest: u6 };

test "casting family" {
    const truncated: u8 = @truncate(@as(u16, 0x0102));
    const widened: u16 = @intCast(@as(u8, 5));
    const bits: u32 = @bitCast(@as(f32, 1.0));
    const trunced: i32 = @trunc(@as(f32, 3.7));
    const rounded: i32 = @round(@as(f32, 3.7));
    try testing.expectEqual(@as(u8, 2), truncated);
    try testing.expectEqual(@as(u16, 5), widened);
    try testing.expectEqual(@as(u32, 0x3f800000), bits);
    try testing.expectEqual(@as(i32, 3), trunced);
    try testing.expectEqual(@as(i32, 4), rounded);

    try testing.expectEqual(@as(u8, 7), @backingInt(Color.blue));
    const c: Color = @fromBackingInt(@as(u8, 1));
    try testing.expectEqual(Color.red, c);

    const f: Flags = @fromBackingInt(@as(u8, 0b0000_0011));
    try testing.expect(f.a and f.b);

    // A many-pointer needs @ptrCast before @alignCast can retype it.
    var backing: [16]u8 align(8) = @splat(0);
    const raw: [*]u8 = &backing;
    const one: *u8 = @ptrCast(raw);
    const aligned: *align(8) u8 = @alignCast(one);
    aligned.* = 9;
    try testing.expectEqual(@as(u8, 9), backing[0]);
}
```

## Size, layout and memory

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@sizeOf` | `@sizeOf(comptime T: type) comptime_int` | In-memory byte size including padding. |
| `@bitSizeOf` | `@bitSizeOf(comptime T: type) comptime_int` | Bit width; the one that matters for packed types. |
| `@alignOf` | `@alignOf(comptime T: type) comptime_int` | |
| `@offsetOf` | `@offsetOf(comptime T: type, comptime field_name: []const u8) comptime_int` | Byte offset. `auto` layout **reorders fields**, so this is not declaration order; only `extern` and `packed` pin it. |
| `@bitOffsetOf` | `@bitOffsetOf(comptime T: type, comptime field_name: []const u8) comptime_int` | Bit offset, for packed structs. |
| `@memcpy` | `@memcpy(noalias dest, noalias source) void` | Non-overlapping copy; lengths must match. |
| `@memmove` | `@memmove(dest, source) void` | The one that tolerates overlap. |
| `@memset` | `@memset(dest, elem) void` | Fill a slice or array pointer with one element value. |
| `@splat` | `@splat(element: anytype) anytype` | Produce a whole array or vector of one value. The idiomatic initializer: `const zeros: [4]u8 = @splat(0);`. |
| `@fieldParentPtr` | `@fieldParentPtr(comptime field_name: []const u8, field_ptr: *T) anytype` | Recover the container from a pointer to one of its fields. Result type is inferred. |
| `@unionInit` | `@unionInit(comptime Union: type, comptime active_field_name: []const u8, init_expr) Union` | Union initialization when the field name is a comptime string rather than an identifier. |
| `@tagName` | `@tagName(value: anytype) [:0]const u8` | Enum or tagged-union active field name. |
| `@embedFile` | `@embedFile(comptime path: []const u8) *const [N:0]u8` | Bake a file into the binary at compile time. |

## Math, overflow and bit twiddling

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@addWithOverflow` | `@addWithOverflow(a: anytype, b: anytype) struct { @TypeOf(a, b), u1 }` | Wrapping result plus an overflow bit, as a tuple. Same shape for `@subWithOverflow`, `@mulWithOverflow`, and `@shlWithOverflow(a, shift_amt)`. |
| `@divTrunc` | `@divTrunc(numerator: T, denominator: T) T` | Round toward zero. `-7 / 3 == -2`. |
| `@divFloor` | `@divFloor(numerator: T, denominator: T) T` | Round toward negative infinity. `-7 / 3 == -3`. |
| `@divCeil` | `@divCeil(numerator: T, denominator: T) T` | Round toward positive infinity. |
| `@divExact` | `@divExact(numerator: T, denominator: T) T` | Assert the division is exact; fastest, safety-checked. |
| `@rem` | `@rem(numerator: T, denominator: T) T` | Remainder taking the **numerator's** sign. `@rem(-7, 3) == -1`. |
| `@mod` | `@mod(numerator: T, denominator: T) T` | Modulus taking the **denominator's** sign. `@mod(-7, 3) == 2`. |
| `@shlExact` / `@shrExact` | `@shlExact(value: T, shift_amt: Log2T) T` | Shift asserting no bits are lost. |
| `@clz` / `@ctz` / `@popCount` | `@clz(operand: anytype) anytype` | Leading zeros, trailing zeros, set bits. |
| `@byteSwap` / `@bitReverse` | `@byteSwap(operand: anytype) T` | Endian swap, bit reversal. |
| `@abs` | `@abs(value: anytype) anytype` | For a signed integer the result is the **unsigned** type of the same width, so it cannot overflow. |
| `@min` / `@max` | `@min(...) T` | Variadic, peer-resolved. |
| `@mulAdd` | `@mulAdd(comptime T: type, a: T, b: T, c: T) T` | Fused multiply-add; a single rounding. |
| `@sqrt`, `@sin`, `@cos`, `@tan`, `@exp`, `@exp2`, `@log`, `@log2`, `@log10` | `@sqrt(value: anytype) @TypeOf(value)` | Hardware float math where available. |

## Vectors

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@Vector` | `@Vector(len: comptime_int, Element: type) type` | |
| `@splat` | `@splat(element: anytype) anytype` | Broadcast a scalar. |
| `@reduce` | `@reduce(comptime op: std.lang.ReduceOp, value: anytype) E` | Horizontal fold. Ops are TitleCase (`.Add`, `.Mul`, `.Min`, `.Max`, `.And`, `.Or`, `.Xor`), defined at `lib/std/lang.zig:46`. |
| `@shuffle` | `@shuffle(comptime E: type, a, b, comptime mask: @Vector(mask_len, i32)) @Vector(mask_len, E)` | Permute across two vectors. Non-negative mask indices select from `a`, negative ones from `b`. |
| `@select` | `@select(comptime T: type, pred: @Vector(len, bool), a, b) @Vector(len, T)` | Per-lane pick between two vectors. |

Verified:

```zig
const std = @import("std");
const testing = std.testing;

test "vector builtins" {
    const a: @Vector(4, i32) = .{ 1, 2, 3, 4 };
    const b: @Vector(4, i32) = @splat(10);
    try testing.expectEqual(@as(i32, 10), @reduce(.Add, a));
    try testing.expectEqual(
        @Vector(4, i32){ 1, 10, 3, 10 },
        @select(i32, @Vector(4, bool){ true, false, true, false }, a, b),
    );
    try testing.expectEqual(
        @Vector(4, i32){ 1, 10, 4, 10 },
        @shuffle(i32, a, b, @Vector(4, i32){ 0, -1, 3, -1 }),
    );
}
```

## Atomics

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@atomicLoad` | `@atomicLoad(comptime T: type, ptr: *const T, comptime ordering: AtomicOrder) T` | |
| `@atomicStore` | `@atomicStore(comptime T: type, ptr: *T, value: T, comptime ordering: AtomicOrder) void` | |
| `@atomicRmw` | `@atomicRmw(comptime T: type, ptr: *T, comptime op: AtomicRmwOp, operand: T, comptime ordering: AtomicOrder) T` | Returns the value **before** the operation. |
| `@cmpxchgStrong` | `@cmpxchgStrong(comptime T: type, ptr: *T, expected_value: T, new_value: T, success_order: AtomicOrder, fail_order: AtomicOrder) ?T` | Returns `null` on **success**, the current value on failure. Use in a non-loop context. |
| `@cmpxchgWeak` | same shape | May fail spuriously; only correct inside a retry loop, where it is cheaper. |

`AtomicOrder` is at `lib/std/lang.zig:35`
(`unordered`, `monotonic`, `acquire`, `release`, `acq_rel`, `seq_cst`);
`AtomicRmwOp` at `lib/std/lang.zig:58` and its variants are TitleCase
(`.Add`, `.Xchg`, `.Min`, ...).

For anything beyond a single operation, prefer `std.atomic.Value(T)`, which
wraps these.

## Control flow, analysis and diagnostics

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@call` | `@call(modifier: std.lang.CallModifier, function: anytype, args: anytype) anytype` | Call with an args tuple, or force a modifier. `.compile_time` folds the call so the result is comptime-known; `.always_inline`, `.never_inline`, `.always_tail`, `.never_tail` are hard requirements that error if impossible. `CallModifier` at `lib/std/lang.zig:1005`. |
| `@compileError` | `@compileError(comptime msg: []const u8) noreturn` | Reject an unsupported instantiation with one clear sentence. Concatenate with `++` or `std.fmt.comptimePrint`. |
| `@compileLog` | `@compileLog(...) void` | Print during analysis. It is itself a compile error if left in, so it is a debugging tool only. |
| `@setEvalBranchQuota` | `@setEvalBranchQuota(comptime new_quota: u32) void` | Raise the 1000-branch comptime budget. Applies to the whole enclosing comptime evaluation, and only ever raises. |
| `@inComptime` | `@inComptime() bool` | Branch on whether analysis is comptime. Writing it directly inside an explicit `comptime` block is a "redundant" compile error. |
| `@branchHint` | `@branchHint(hint: BranchHint) void` | Must be the **first statement** in a function or in a branch body. `BranchHint` at `lib/std/lang.zig:1245`. |
| `@setRuntimeSafety` | `@setRuntimeSafety(comptime safety_on: bool) void` | Scope-local safety toggle. |
| `@setFloatMode` | `@setFloatMode(comptime mode: FloatMode) void` | `.strict` or `.optimized` (`lib/std/lang.zig:951`). |
| `@prefetch` | `@prefetch(ptr: anytype, comptime options: PrefetchOptions) void` | |
| `@panic` | `@panic(message: []const u8) noreturn` | |
| `@trap` | `@trap() noreturn` | Illegal instruction; no message, no unwinding. |
| `@breakpoint` | `@breakpoint() void` | |
| `@disableInstrumentation` | `@disableInstrumentation() void` | Undocumented in the langref but real. Opt a function out of compiler instrumentation; used at `lib/std/debug.zig:441` and `lib/std/pie.zig:319`. |
| `@disableIntrinsics` | `@disableIntrinsics() void` | Undocumented in the langref but real. Stop the backend lowering calls in this function to intrinsics, which is how `lib/compiler_rt.zig:605` avoids recursing into itself. |
| `@src` | `@src() std.lang.SourceLocation` | `{ module, file, fn_name, line, column }` (`lib/std/lang.zig:627`). |
| `@returnAddress` | `@returnAddress() usize` | |
| `@frameAddress` | `@frameAddress() usize` | |
| `@errorReturnTrace` | `@errorReturnTrace() ?*std.lang.StackTrace` | |
| `@import` | `@import(comptime target: []const u8) anytype` | |

Verified:

```zig
const std = @import("std");

fn addThree(a: u8, b: u8, c: u8) u8 {
    return a + b + c;
}

fn hinted(x: u8) u8 {
    @branchHint(.likely); // first statement of the function
    if (x > 100) {
        @branchHint(.cold); // first statement of the branch
        return 0;
    }
    return x + 1;
}

test "@call modifiers and @branchHint placement" {
    const folded = @call(.compile_time, addThree, .{ 1, 2, 3 });
    const arr: [folded]u8 = undefined; // proof the result is comptime-known
    try std.testing.expectEqual(@as(usize, 6), arr.len);
    try std.testing.expectEqual(@as(u8, 2), hinted(1));
}
```

## C interop and export

| Builtin | Signature | When it is the right choice |
|---|---|---|
| `@extern` | `@extern(T: type, comptime options: std.lang.ExternOptions) T` | Reference a symbol from another object without a declaration. |
| `@export` | `@export(comptime ptr: *const anyopaque, comptime options: std.lang.ExportOptions) void` | Publish a symbol under a chosen name and linkage. |
| `@cVaStart` | `@cVaStart() std.lang.VaList` | Only inside a variadic function. |
| `@cVaArg` | `@cVaArg(operand: *std.lang.VaList, comptime T: type) T` | |
| `@cVaCopy` | `@cVaCopy(src: *std.lang.VaList) std.lang.VaList` | |
| `@cVaEnd` | `@cVaEnd(src: *std.lang.VaList) void` | |

## GPU and wasm targets

| Builtin | Signature |
|---|---|
| `@workGroupId` | `@workGroupId(comptime dimension: u32) u32` |
| `@workGroupSize` | `@workGroupSize(comptime dimension: u32) u32` |
| `@workItemId` | `@workItemId(comptime dimension: u32) u32` |
| `@wasmMemorySize` | `@wasmMemorySize(index: u32) usize` |
| `@wasmMemoryGrow` | `@wasmMemoryGrow(index: u32, delta: usize) isize` |

## Build and target introspection is not a builtin

There is no builtin for "what am I compiling for". That lives on the
compiler-generated module:

```zig
const builtin = @import("builtin");
builtin.target.cpu.arch
builtin.mode          // .debug, .safe, .fast, .small  (NOT .Debug/.ReleaseSafe)
builtin.zig_version
builtin.zig_backend
builtin.is_test
```

Do not confuse this with `std.builtin`, which is the deprecated alias for
`std.lang` and holds the language-level types the builtins above take as
arguments (`AtomicOrder`, `CallModifier`, `ReduceOp`, `Signedness`,
`ContainerLayout`, `Type`).

## Recognized but unusable

`@Frame(func)` and `@frame()` parse but reject with
`error: async has not been implemented in the self-hosted compiler yet`.
They are two of the four builtins the compiler recognizes that the langref
TOC omits; the other two, `@disableInstrumentation` and `@disableIntrinsics`,
work and are listed above.
The `Type` union still carries `frame` and `@"anyframe"` variants
(`lib/std/lang.zig:931` and `:937`) for the same reason.

## See also

`zig-comptime` for `comptime` semantics, generics, `@typeInfo` shape, and the
comptime failure modes on this same toolchain.
