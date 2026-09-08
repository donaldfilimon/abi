---
name: zig-errors
description: Zig error handling on master (0.17.0-dev). Use when writing or reviewing error sets, inferred error sets, the || set-union operator, try, catch, errdefer, error return traces, switching on errors, @errorCast, or when choosing between returning an error, unreachable, std.debug.assert, and @panic. Also covers errdefer inside loops and errdefer around a resource returned by value.
---

# Zig error handling on master

Ground truth is the installed toolchain, not older blog posts:

- `zig` `0.17.0-dev.2018+ab30a0b9a` at `/Users/donaldfilimon/.zvm/bin/zig`
- stdlib at `/Users/donaldfilimon/.zvm/master/lib/std/`
- langref for this exact build at `/Users/donaldfilimon/.zvm/master/doc/langref.html`

Compile anything uncertain in `/private/tmp/zig-skill-scratch`.

Related skills: `zig-types` for the error-union type itself and `@errorCast`, `zig-memory` for the allocator side of the ownership patterns, `zig-testing` for `expectError` and leak reporting.

## Changed on master (check this first)

1. **`errdefer |err|` no longer parses.** The capture form is gone. The parser rule at
   `lib/std/zig/Parse.zig:991` builds an `errdefer` node with a single expression payload
   and no capture token, so both `errdefer |err| doThing(err);` and
   `errdefer |err| { ... }` fail with `error: expected block or expression, found '|'`.
   This build's `doc/langref.html` `errdefer` section documents only the plain form.
   Replacement idiom is below.

2. **Discarding an error value is a compile error.** `_ = err;` on anything of error set
   type gives `error: error set is discarded`. Use the value (log it, panic with it,
   store it, return it) or do not bind it.

3. **`@typeInfo(E).error_set` is a struct, not an optional slice.**
   `lib/std/lang.zig:802` defines `ErrorSet` with one field,
   `error_names: ?[]const [:0]const u8`. So the master spelling is
   `@typeInfo(E).error_set.error_names.?.len`. The old `.error_set.?.len` fails with
   `expected optional type, found 'lang.Type.ErrorSet'`. For `anyerror` the set is open,
   so `error_names` is `null`.

4. **`std.builtin` is deprecated in favor of `std.lang`** (`lib/std/std.zig:71`), which is
   where `StackTrace` lives (`lib/std/lang.zig:11`). There is no `lib/std/builtin.zig`.

## Error sets and error unions

An error set is a type; its members are error values.

```zig
const FileError = error{ NotFound, PermissionDenied };
const NetError  = error{ Timeout, Refused };
const Both      = FileError || NetError;   // set union, 4 members
```

Coercion goes narrow to wide only: `FileError` into `Both` into `anyerror`.
`anyerror` is the global open set. `@errorName(e)` gives the tag as `[:0]const u8`.

A bare `error.Whatever` has an inferred one-member set until context narrows it.

`E!T` is the error union:

```zig
@typeInfo(FileError!u32).error_union.payload      // u32
@typeInfo(FileError!u32).error_union.error_set    // FileError
```

`@errorCast` narrows a wider union or error value to a subset, checked in safe builds.

### Inferred error sets

`fn f() !T` with no set makes the compiler infer the union of everything the body can
return. It is convenient inside a module and a liability across a public API boundary,
because the set silently grows when a callee grows. Name the set on anything a caller
switches on exhaustively.

Read the inferred set back with:

```zig
const R = @typeInfo(@TypeOf(f)).@"fn".return_type.?;
const set = @typeInfo(R).error_union.error_set;
```

## try, catch, and the if/else capture

```zig
try f(x)                               // exactly: f(x) catch |err| return err
f(x) catch 0                           // supply a value
f(x) catch |err| switch (err) { ... }  // handle per error
f(x) catch return 0                    // catch can take control flow
f(x) catch unreachable                 // assert this cannot fail

if (f(x)) |v| { ... } else |err| { ... }

while (step()) |v| { ... } else |err| { ... }   // loop until the first error
```

Switching on an error set is exhaustive without an `else` when every member is listed,
which is the main reason to name a set rather than infer it.

## errdefer

`errdefer` runs on block exit if and only if the block is left with an error.
Registered errdefers run in reverse order, and only those registered before the failing
statement run at all.

```zig
fn build(...) E!Thing {
    const a = try acquireA();
    errdefer releaseA(a);

    const b = try acquireB();
    errdefer releaseB(b);

    if (bad) return error.Step3Failed;   // runs releaseB then releaseA
    return .{ .a = a, .b = b };          // runs neither
}
```

### errdefer inside a loop

An `errdefer` in a loop body is scoped to one iteration and is discarded when that
iteration ends normally. It will **not** clean up resources acquired by earlier
iterations. Verified: failing on iteration 2 of 5 runs exactly one cleanup, not three.
When each iteration acquires something, either push into a container whose `deinit`
frees everything and put a single `errdefer container.deinit(gpa)` outside the loop, or
track a count and free that many in one outer `errdefer`.

### errdefer around a resource returned by value

The constructor's errdefers cover only the failure paths inside the constructor. Once it
returns successfully, ownership transfers and the **caller** pairs the `defer`:

```zig
fn init(gpa: Allocator, name: []const u8, size: usize) Allocator.Error!Node {
    const owned_name = try gpa.dupe(u8, name);
    errdefer gpa.free(owned_name);

    const payload = try gpa.alloc(u8, size);
    errdefer gpa.free(payload);

    return .{ .name = owned_name, .payload = payload };
}

// caller
var n = try Node.init(gpa, "abc", 8);
defer n.deinit(gpa);
```

Prove every failure path with `std.testing.checkAllAllocationFailures`
(`lib/std/testing.zig:1142`), which reruns the body with each allocation failing in turn
and asserts nothing leaked on any of those paths.

### Replacing the removed `errdefer |err|`

Wrap the call and destructure the error union where you need to see which error unwound:

```zig
fn classify(fail: bool, seen: *?anyerror) E!u32 {
    if (inner(fail)) |v| {
        return v;
    } else |err| {
        seen.* = err;
        return err;
    }
}
```

## Error return traces

A `try` chain records each hop. Returning an error from `main` prints the message plus
the trace. Actual output from `zig build-exe exe-06-error-trace.zig` and running it:

```
error: Deep
/private/tmp/zig-skill-scratch/errors/exe-06-error-trace.zig:6:5: 0x1009e988b in level3 (exe-06-error-trace)
    return error.Deep;
    ^
/private/tmp/zig-skill-scratch/errors/exe-06-error-trace.zig:10:5: 0x1009e9867 in level2 (exe-06-error-trace)
    try level3();
    ^
/private/tmp/zig-skill-scratch/errors/exe-06-error-trace.zig:14:5: 0x1009e972b in level1 (exe-06-error-trace)
    try level2();
    ^
/private/tmp/zig-skill-scratch/errors/exe-06-error-trace.zig:20:5: 0x1009e84f3 in main (exe-06-error-trace)
    try level1();
    ^
```

Exit status is 1.

How tracing is controlled:

- On by default in Debug, off by default in release modes.
- `-ferror-tracing` enables it in release builds; `-fno-error-tracing` disables it in
  Debug. Verified: `-fno-error-tracing` and `-O ReleaseFast` both print bare
  `error: Deep` with no frames.
- `std.options.allow_stack_tracing` (`lib/std/std.zig:185`, default
  `!@import("builtin").strip_debug_info`) gates capture entirely. With it false, traces
  are empty.
- `@errorReturnTrace()` returns `?*std.lang.StackTrace` (verified by `@TypeOf`) for manual inspection;
  `std.debug.dumpStackTrace(st)` prints one (`lib/std/debug.zig:914`).

The trace is not a stack trace. It shows where the error travelled, which is what you
want when the failing call is three layers down.

## unreachable, assert, panic, or an error

Pick by who can act on it.

| Situation | Use | Behavior |
|---|---|---|
| The caller could plausibly handle or report it | `E!T` | caller decides |
| A precondition the caller was required to satisfy | `std.debug.assert` | body is `@disableInstrumentation(); if (!ok) unreachable;` (`lib/std/debug.zig:440`), so it disappears in ReleaseFast. Its own doc comment says to prefer `std.testing` inside a test block, because it may not register a test failure in fast and small modes |
| A branch you have already proven cannot be taken | `unreachable` | panic in Debug and ReleaseSafe, undefined behavior in ReleaseFast and ReleaseSmall |
| An error you have proven cannot occur here | `catch unreachable` | same as above |
| A runtime condition that must abort in every build | `@panic` / `std.debug.panic` | `lib/std/debug.zig:460`, present in all optimization modes |

Two consequences worth stating outright: never put a side effect inside
`std.debug.assert`, because the whole call vanishes in ReleaseFast; and `unreachable` in
ReleaseFast is not a crash, it is undefined behavior, so use it only for facts the code
has genuinely established.

## Verified snippets

Compiled with `zig 0.17.0-dev.2018+ab30a0b9a` in `/private/tmp/zig-skill-scratch/errors/`.
6 of 6 pass (`zig test`, except the last which is `zig build-exe` plus a run):

`01-error-sets.zig`, `02-try-catch.zig`, `03-errdefer.zig`, `04-errdefer-ownership.zig`,
`05-unreachable-assert-panic.zig`, `exe-06-error-trace.zig`.

Dropped because they do not compile on master, and kept here as the evidence:
`errdefer |err| seen.* = err;` and `errdefer |err| { seen.* = err; }` both give
`expected block or expression, found '|'`; `_ = err;` gives `error: error set is discarded`;
`@typeInfo(E).error_set.?` gives `expected optional type, found 'lang.Type.ErrorSet'`.

The scratch directory is disposable (`/private/tmp` is wiped on reboot). Every construct
above appears inline in this skill, so re-verification means pasting a block into a fresh
scratch file, not recovering these paths.
