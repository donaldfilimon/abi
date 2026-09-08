---
name: zig-testing
description: >-
  Use when writing or debugging Zig tests on master (0.17.0-dev), choosing a
  std.testing assertion, deciding between `zig test file.zig` and `zig build
  test`, filtering with --test-filter, tracking down a testing.allocator leak
  report, using testing.tmpDir or testing.io for file I/O in tests, calling
  refAllDecls or checkAllAllocationFailures, skipping with error.SkipZigTest,
  writing a `test Decl {}` block, reading a test failure stack trace, or setting
  up coverage-guided fuzzing with std.testing.fuzz / Smith / `zig build test
  --fuzz`. Do not use for build.zig structure (zig-build) or C interop
  (zig-c-interop).
---

# Zig testing on master

Verified against the zvm master toolchain on this Mac:
`/Users/donaldfilimon/.zvm/bin/zig`, version `0.17.0-dev.2018+ab30a0b9a`.
Source of truth is `/Users/donaldfilimon/.zvm/master/lib/std/testing.zig`
(1427 lines) plus `lib/std/testing/Smith.zig` and
`lib/std/compiler/test_runner.zig`. Every assertion listed below was compiled
and run; nothing here is recalled.

Scratch work goes in `/private/tmp/zig-build-scratch/`, never iCloud, never home root.

## Changed on master

| Older idiom | Status on this toolchain |
|---|---|
| `tmp.dir.writeFile(.{ .sub_path, .data })` via `std.fs` | `Io`-parameterized: `tmp.dir.writeFile(std.testing.io, .{ .sub_path = "a.txt", .data = "payload" })`. `TmpDir.dir` is an `Io.Dir` and `tmpDir` takes `Io.Dir.OpenOptions` (`lib/std/testing.zig:603`). `std.testing.io` (`:24`) is the test `Io` instance. |
| `std.testing.FailingAllocator` for leak reporting | Leaks are reported by the `SafeAllocator` behind `std.testing.allocator` (`:21`); `FailingAllocator` (`:13`) is for injected allocation failure, a different job. |
| no fuzzing, or `std.testing.fuzz(context, testOne, .{ .corpus })` taking `[]const u8` | `std.testing.fuzz(context, testOne, options)` (`:1254`) where `testOne` receives a `*std.testing.Smith` (`:1247`), not a byte slice. |
| `var list = std.ArrayList(u8).init(allocator)` inside tests | `var list: std.ArrayList(u8) = .empty;` with the allocator passed per call: `list.append(gpa, x)`, `list.deinit(gpa)`. This is what `zig init` generates. |
| `-Doptimize=ReleaseFast` for fuzz runs | `-Doptimize=fast`. The old spellings still resolve (deprecated aliases, `lib/std/lang.zig:127-133`, removal planned after 0.18.0) but only the lowercase names are advertised. See the `zig-build` skill. |

## Test block forms

Three forms, all verified compiling and running in one file:

```zig
test "named test" { ... }          // reported as `file.test.named test`
test { ... }                       // anonymous, reported as `file.test_0`
test Thing { ... }                 // decltest, reported as `file.decltest.Thing`
```

The `test Decl {}` form works on this toolchain. It attaches the test to a
declaration so documentation tooling can pair them, and the runner labels it
`decltest.<Name>`.

Only `test` blocks in the **root module of the test binary** and in files
reachable from it via analyzed `@import` run. A `test` in a file that nothing
imports never executes. `std.testing.refAllDecls(@This())` (`lib/std/testing.zig:1240`)
forces analysis of every public declaration in a container, which is how you pull
in tests and compile errors that lazy analysis would otherwise skip:

```zig
test { std.testing.refAllDecls(@This()); }
```

It is a no-op outside `builtin.is_test`.

## Running tests

```bash
zig test src/root.zig                 # one file, direct
zig test src/root.zig -lc             # with libc
zig build test                        # whatever `test` step build.zig defines
zig build test --summary all          # per-step timing, cache state, pass counts
```

`zig build test` is the real gate for a package because it covers every module
the build script wired up. A test binary covers exactly **one** module's root, so
an N-module package needs N `addTest` calls hung off one `test` step. See the
`zig-build` skill.

### Filtering

```bash
zig test t.zig --test-filter expectError            # substring match on the full name
zig test t.zig --test-filter alpha --test-filter beta   # multiple flags are OR'd
```

For a `build.zig` test binary, filters go in `TestOptions.filters`
(`lib/std/Build.zig:651`) as `[]const []const u8`.

**False-green trap, verified:** a filter that matches nothing does not fail.

```
$ zig test named.zig --test-filter zzz
All 0 tests passed.
$ echo $?
0
```

Never conclude a suite passed from an exit code alone when a filter is in play.
Read the `N/M` count.

## Assertions that exist on master

Checked by compiling and running each one. Line numbers are in
`lib/std/testing.zig`.

| Function | Line | Notes |
|---|---|---|
| `expect(ok: bool) !void` | 583 | fails with `error.TestUnexpectedResult` |
| `expectEqual(expected, actual) !void` | 55 | `inline`; peer-resolves types, so `expectEqual(4, 2 + 2)` works |
| `expectEqualDeep(expected, actual)` | 766 | `inline`; structural, follows pointers and slices |
| `expectEqualSlices(comptime T, expected, actual)` | 331 | prints a diff with index of first mismatch |
| `expectEqualStrings(expected, actual)` | 668 | `[]const u8` specialization with visible-newline output |
| `expectEqualSentinel(comptime T, comptime sentinel, expected, actual)` | 541 | for `[:s]const T` |
| `expectStringStartsWith(actual, expected_starts_with)` | 703 | |
| `expectStringEndsWith(actual, expected_ends_with)` | 729 | |
| `expectError(expected_error: anyerror, actual_error_union)` | 628 | fails if the union holds a payload |
| `expectApproxEqAbs(expected, actual, tolerance)` | 258 | `inline`; absolute tolerance |
| `expectApproxEqRel(expected, actual, tolerance)` | 294 | `inline`; relative tolerance |
| `expectFmt(expected, comptime template, args)` | 656 | asserts formatted output |
| `checkAllAllocationFailures(backing_allocator, test_fn, extra_args)` | 1142 | runs `test_fn` once per allocation site with that site forced to fail |
| `refAllDecls(comptime T)` | 1240 | forces analysis of public decls |
| `tmpDir(opts: Io.Dir.OpenOptions) TmpDir` | 603 | |
| `allocator` | 21 | leak-detecting; `@compileError` outside tests |
| `failing_allocator` | 14 | always fails |
| `io` | 24 | the `Io` instance for test file/dir work |
| `FailingAllocator` | 13 | injected-failure allocator type |
| `Smith` | 1247 | fuzzer input source |
| `Reader` / `ReaderIndirect` / `WriterIndirect` | 1263 / 1306 / 1366 | scripted `Io.Reader`/`Io.Writer` fakes for stream tests |

There is **no** `expectEqualStructs`, no `expectNotEqual`, no `expectTrue`.
Negations go through `expect(!cond)`.

Verified example, all 11 blocks in one file, 10 passed and 1 skipped:

```zig
const std = @import("std");
const testing = std.testing;

pub const Thing = struct {
    v: u32,
    pub fn double(t: Thing) u32 { return t.v * 2; }
};

fn mayFail(x: u8) error{TooBig}!u8 {
    if (x > 10) return error.TooBig;
    return x;
}

test "expect / expectEqual" {
    try testing.expect(1 + 1 == 2);
    try testing.expectEqual(4, 2 + 2);
    try testing.expectEqual(@as(u32, 6), (Thing{ .v = 3 }).double());
}

test "expectEqualStrings and slices" {
    try testing.expectEqualStrings("abc", "abc");
    try testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3 }, &[_]u8{ 1, 2, 3 });
    try testing.expectEqualSentinel(u8, 0, "hi", "hi");
    try testing.expectStringStartsWith("hello world", "hello");
    try testing.expectStringEndsWith("hello world", "world");
}

test "expectError" {
    try testing.expectError(error.TooBig, mayFail(200));
}

test "approx" {
    try testing.expectApproxEqAbs(@as(f64, 1.0), 1.0 + 1e-9, 1e-6);
    try testing.expectApproxEqRel(@as(f64, 100.0), 100.0001, 1e-5);
}

test "expectFmt" {
    try testing.expectFmt("x=7", "x={d}", .{7});
}

test "expectEqualDeep" {
    try testing.expectEqualDeep(Thing{ .v = 1 }, Thing{ .v = 1 });
}

test "allocator leak detection" {
    const gpa = testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try list.appendSlice(gpa, "zig");
    try testing.expectEqualStrings("zig", list.items);
}

test "tmpDir" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "payload" });
    var buf: [16]u8 = undefined;
    const contents = try tmp.dir.readFile(testing.io, "a.txt", &buf);
    try testing.expectEqualStrings("payload", contents);
}

test "skip" {
    return error.SkipZigTest;
}

test Thing {
    try testing.expectEqual(@as(u32, 10), (Thing{ .v = 5 }).double());
}

test {
    testing.refAllDecls(@This());
}
```

```
$ zig test t.zig
1/11 t.test.expect / expectEqual...OK
...
9/11 t.test.skip...SKIP
10/11 t.decltest.Thing...OK
11/11 t.test_0...OK
10 passed; 1 skipped; 0 failed.
```

## Skipping

`return error.SkipZigTest;` marks the test SKIP rather than FAIL, and the run
still exits 0. Use it for a platform or capability that genuinely is not present,
guarded on `builtin.os.tag`, `builtin.target`, or a runtime probe. A skipped test
is a hole in coverage that nothing else reports, so do not use it to park a
failing test.

## Allocator and leak detection

`std.testing.allocator` (`lib/std/testing.zig:21`) is backed by a
`SafeAllocator`, which tracks every live allocation and reports leaks at the end
of the run with the **allocation-site** stack trace, not the leak-detection site.

**Trap, verified:** a leaking test prints `OK` on its own line. The failure only
appears in the summary.

```
1/2 fail.test.leak...OK
[SafeAllocator] (err): leaked [addr: 103658010, len: 8 (0x8) align: 1] allocated at:
/private/tmp/zig-build-scratch/tst/fail.zig:4:30: 0x1030f704b in test.leak (test)
    const buf = try gpa.alloc(u8, 8);
                             ^
...
1 passed; 0 skipped; 1 failed.
1 errors were logged.
1 tests leaked memory.
error: the following test command failed with exit code 1:
```

So a leak makes the run fail, but scanning for per-test `FAIL` lines misses it.
Read the summary lines.

`checkAllAllocationFailures` complements this by proving your error paths do not
leak either. Verified:

```zig
fn buildList(gpa: std.mem.Allocator, n: usize) !void {
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try list.appendNTimes(gpa, 'x', n);
}

test "checkAllAllocationFailures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, buildList, .{@as(usize, 64)});
}
```

It runs `buildList` once with unlimited memory to count allocation sites, then
once per site with that site forced to return `error.OutOfMemory`, asserting the
function propagates the error and leaks nothing (`lib/std/testing.zig:1142`).
The function under test must take `std.mem.Allocator` as its first parameter.

## Temporary directories

```zig
var tmp = std.testing.tmpDir(.{});
defer tmp.cleanup();
```

`tmpDir` creates a randomly named directory under `.zig-cache/tmp/` in the cwd
(`lib/std/testing.zig:603`), so it never touches the user's filesystem outside
the cache. `TmpDir` holds `dir: Io.Dir`, `parent_dir: Io.Dir` and the encoded
`sub_path`; `cleanup()` closes both handles and `deleteTree`s the directory.
Every operation on `tmp.dir` takes `std.testing.io` as its first argument.

## Failure output and stack traces

A failing assertion prints the diagnostic, then `FAIL (<ErrorName>)`, then a
stack trace with source lines and carets, walking from inside `std.testing`
out through your test to `test_runner.zig`:

```
2/2 fail.test.bad equal...expected 1, found 2
FAIL (TestExpectedEqual)
/Users/donaldfilimon/.zvm/master/lib/std/testing.zig:93:17: 0x1030f694f in expectEqualInner__func_997 (test)
                return error.TestExpectedEqual;
                ^
/Users/donaldfilimon/.zvm/master/lib/std/testing.zig:57:5: 0x1030f69df in expectEqual (test)
    return expectEqualInner(T, expected, actual);
    ^
/private/tmp/zig-build-scratch/tst/fail.zig:8:5: 0x1030f6a1f in test.bad equal (test)
    try std.testing.expectEqual(@as(u32, 1), @as(u32, 2));
    ^
```

The frame that matters is the deepest one under your own path. The runner then
prints the exact command it ran, including the `--seed`, so you can rerun the
same binary by hand:

```
error: the following test command failed with exit code 1:
/Users/donaldfilimon/.cache/zig/o/<hash>/test --seed=0xd8e4aaef
```

`std.testing.backend_can_print` (`:32`) gates the pretty output on backends that
cannot format, so on exotic targets failures degrade to a bare error return.

## Fuzzing is real on this toolchain

Confirmed by running it, not inferred from the presence of `lib/fuzzer.zig`.

```zig
test "fuzz finds the bug" {
    try std.testing.fuzz({}, testOne, .{});
}

fn testOne(_: void, smith: *std.testing.Smith) !void {
    var buf: [4]u8 = undefined;
    smith.bytes(&buf);
    if (buf[0] == 'z' and buf[1] == 'i' and buf[2] == 'g' and buf[3] == '!') {
        return error.FoundIt;
    }
}
```

Signature (`lib/std/testing.zig:1254`), copied from source:

```zig
pub const FuzzInputOptions = struct {
    corpus: []const []const u8 = &.{},
};

/// Inline to avoid coverage instrumentation.
pub inline fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), smith: *Smith) anyerror!void,
    options: FuzzInputOptions,
) anyerror!void
```

`Smith` (`lib/std/testing/Smith.zig`) is a structured input source, not a byte
slice. It hands you typed values from the fuzzer's entropy:

- `smith.bytes(out: []u8)` fills a buffer
- `smith.value(T)` produces a `T`, including enums and integers
- `smith.valueRangeAtMost(T, at_least, at_most)` and `valueRangeLessThan`
- `smith.index(len)` picks an index
- `smith.eos()` asks whether the input is exhausted, so you can drive a loop of
  random operations
- weighted variants (`valueWeighted`, `boolWeighted`, `eosWeighted`) bias the
  distribution

The `zig init` template's `src/main.zig` ships a working `Smith`-driven fuzz test
against `std.ArrayList`; read it for idiom.

### Two run modes

```bash
zig build test                              # runs the corpus once, ordinary test
zig build test --fuzz -Doptimize=fast       # coverage-guided fuzzing loop
```

Under plain `zig build test` a fuzz test counts as one passing test. Under
`--fuzz` the build runner starts a web UI and fuzzes until you stop it:

```
info(web_server): web interface listening at http://[::1]:63333/
info(web_server): hint: pass '--webui=[::1]:63333' to use the same port next time
```

It does not exit on its own, so fence it in scripts (`timeout 60 zig build test
--fuzz ...`). On a finding it reports the failing test and **saves the input**:

```
test
+- run test failure
failed with error.FoundIt
error: test 'root.test.fuzz finds the bug' exited with code 1; input saved to '.zig-cache/f/crash'
```

That is a real run against the planted four-byte magic above, found in under 60
seconds. Fuzzing instrumentation is per-module: `Module.CreateOptions` has a
`fuzz: ?bool` field (`lib/std/Build/Module.zig:222`) if you need to force it on
or off for a module.

## Checklist before claiming a suite is green

1. Run the package gate (`zig build test`), not just one file.
2. Read the `N passed; M skipped; K failed` line, not the exit code alone.
3. Confirm no `tests leaked memory` line in the summary.
4. If you passed `--test-filter`, confirm the match count is non-zero.
5. Never pipe the gate into `head`/`tail` and read `$?`; that reports the pipe
   tail's status. Redirect to a file and echo the command's own status.
