---
name: zig-io
description: The std.Io model on Zig master (0.17.0-dev). Use when writing or reviewing anything that prints to stdout/stderr, reads stdin, opens files or directories, opens sockets, sleeps, gets the time, gets randomness, spawns concurrent work, or awaits and cancels it. Also use when `std.io` does not exist, `std.fs.cwd` does not exist, `getStdOut().writer()` fails to compile, a Writer's output vanishes because nothing flushed, `main` needs an `Io` and does not have one, or a reader loop reads the delimiter twice. Covers the Io interface and its vtable, Io.Reader/Io.Writer buffering, Io.Limit, Io.File/Io.Dir/Io.net, Io.Threaded and Io.Evented, async/await/concurrent/Group/Select, and cancellation.
---

# std.Io on Zig master

Ground truth for this file is the installed toolchain, measured 2026-09-06:

- binary `/Users/donaldfilimon/.zvm/bin/zig`, version `0.17.0-dev.2018+ab30a0b9a`
- stdlib source `/Users/donaldfilimon/.zvm/master/lib/std/`
- every snippet below was compiled with that binary before it was written down

Citations are `lib/std/<path>.zig:<line>` against that tree. Line numbers drift on
master; the names are the durable part. Read the source when a number does not land.

For the rest of the standard library see the `zig-std` skill. For allocators see
`zig-memory`, for the toolchain itself see `zig-toolchain`.

## The one-paragraph model

`Io` is a fat interface value: a `userdata` pointer plus a `vtable` pointer
(`lib/std/Io.zig:24-26`, `:51`). It abstracts the filesystem, networking,
processes, time, sleeping, randomness, async/await/concurrent/cancel, queues,
wait groups, select, mutexes, futexes, conditions, and memory-mapped files
(`lib/std/Io.zig:1-13`). Anything that touches the outside world takes an `io: Io`
parameter. `Io.Reader` and `Io.Writer` are a separate, non-vtable-of-`Io` concern:
they are plain byte-stream interfaces with an explicit caller-owned buffer, and
they are what `print`, `readAll`, and friends hang off.

## Changed on master: the old idioms and their replacements

These are the things a model trained before the rewrite will reach for first.
Each "gone" line below was compiled and the error text is the real diagnostic.

| Old idiom | Status on this toolchain | Replacement |
|---|---|---|
| `std.io` | **Does not exist.** `error: root source file struct 'std' has no member named 'io'` | `std.Io` (capital I) |
| `std.io.getStdOut().writer()` | gone with `std.io` | `Io.File.Writer.init(.stdout(), io, &buf)` then `&fw.interface` |
| `std.io.getStdErr()`, `getStdIn()` | gone | `Io.File.stderr()`, `Io.File.stdin()` (`lib/std/Io/File.zig:104`, `:117`) |
| `std.fs.cwd()` | **Does not exist.** `error: root source file struct 'fs' has no member named 'cwd'` | `std.Io.Dir.cwd()` (`lib/std/Io/Dir.zig:88`) |
| `std.fs.File`, `std.fs.Dir` | gone; `std.fs` is a deprecation shim of 16 lines plus `path` (`lib/std/fs.zig:5-16`) | `std.Io.File`, `std.Io.Dir` |
| `std.fs.path` | deprecated alias; `lib/std/fs.zig:5-6` says "Deprecated, use `std.Io.Dir.path`" | `std.Io.Dir.path` (same module, canonical name moved; `lib/std/Io/Dir.zig:17`) |
| `AnyReader` / `AnyWriter` / `GenericReader` | gone (absent from the `pub const` list of `Io.zig:28-49`; verified by grep, not by a compile) | `*Io.Reader`, `*Io.Writer` are already the type-erased form |
| `reader.readUntilDelimiterOrEof` | gone (absent from the `pub fn` list of `Io/Reader.zig`; verified by grep, not by a compile) | `Io.Reader.takeDelimiter` / `takeDelimiterExclusive` / `takeDelimiterInclusive` |
| `writer.writeAll` on an unbuffered writer | still exists, but buffering is now **explicit and caller-owned**; a `Writer` with a non-empty buffer writes nothing until `flush` | always `try w.flush()` |
| `std.time.sleep`, `std.time.Timer`, `milliTimestamp` | `std.time` is now 35 lines of unit constants plus `epoch` | `io.sleep`, `Io.Clock.now`, `Io.Timestamp`, `Io.Duration` |
| `std.Thread.Mutex` / `.Condition` / `.Pool` / `.WaitGroup` / `.Semaphore` / `.ResetEvent` / `.Futex` | **all gone from `std.Thread`** | `Io.Mutex`, `Io.Condition`, `Io.Group`, `Io.Semaphore`, `Io.RwLock`, `io.futexWait`/`futexWake` |
| `std.crypto.random` | gone | `io.randomSecure(buf)` (`lib/std/Io.zig:2679`) or `io.random(buf)` (`:2662`) |
| `std.crypto.sign.Ed25519.KeyPair.generate()` | now takes an `Io` | `.generate(io)` (`lib/std/crypto/25519/ed25519.zig:336`) |
| `pub fn main() !void` as the only form | still legal, but then you must build your own `Io` | `pub fn main(init: std.process.Init) !void` hands you `init.io` |

## Hello world, verified

This is the exact idiom on this toolchain. `zig build-exe hello.zig && ./hello`
prints `Hello, world!`.

```zig
const std = @import("std");
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print("Hello, {s}!\n", .{"world"});
    try stdout.flush();
}
```

Four things are load-bearing and all four are new:

1. `main` takes `std.process.Init`. That is where `io` comes from.
2. `.stdout()` is `Io.File.stdout()` (`lib/std/Io/File.zig:91`), a `File`, not a writer.
3. The buffer is yours. `Io.File.Writer.init(file, io, buffer)` (`lib/std/Io/File/Writer.zig:36`)
   stores it; the streaming `Io.Writer` is the `interface` field (`:19`), so you write
   through `&fw.interface`, not through `fw`.
4. **`flush` or your output does not exist.** With a 1024-byte buffer and a short
   line, nothing reaches the fd until `flush`.

## Where `io` comes from

### From `main`

`std/start.zig:786-792` and `:826` dispatch on the first parameter type of `root.main`.
Three forms are accepted:

```zig
pub fn main() !void                              // no Init at all
pub fn main(init: std.process.Init.Minimal) !void // args + environ only, no Io
pub fn main(init: std.process.Init) !void         // the full thing
```

`std.process.Init` (`lib/std/process.zig:31`) carries:

```zig
minimal: Minimal,              // .args, .environ
arena: *std.heap.ArenaAllocator, // process-lifetime, threadsafe
gpa: Allocator,                // leak-checked in Debug
io: Io,                        // target-appropriate default implementation
environ_map: *Environ.Map,
preopens: Preopens,
```

Verified `_main` snippet: `args = try init.minimal.args.toSlice(arena)`,
`init.environ_map.get("PATH")`, `std.process.currentPath(io, &buf)` all work.

### Building one yourself

`Io.Threaded` (`lib/std/Io/Threaded.zig`) is the portable implementation and the
one `start.zig` uses by default.

```zig
var gpa_state: std.heap.DebugAllocator(.{}) = .init;
defer _ = gpa_state.deinit();
const gpa = gpa_state.allocator();

var threaded: Io.Threaded = .init(gpa, .{});   // :1609
defer threaded.deinit();                        // :1714
const io = threaded.io();                       // :1809
```

`InitOptions` fields, all defaulted: `stack_size`, `async_limit: ?Io.Limit`
(default: CPU count minus one), `concurrent_limit: Io.Limit = .unlimited`,
`argv0`, `environ`, `disable_memory_mapping`. The `gpa` is used only for
`async`/`concurrent`/`groupAsync`/`groupConcurrent`; pass `Allocator.failing` if
you avoid those. `Io.Threaded.init_single_threaded` (`:1676`) is a comptime constant instance.

In tests, `std.testing.io` is a live `Io` already (`lib/std/testing.zig:23-24`,
backed by `io_instance: Io.Threaded`). Use it rather than standing up your own,
unless the test needs its own async limits.

`Io.failing` (`lib/std/Io.zig:2706`) is an implementation whose every operation
errors or is `unreachable`. Useful for proving a code path does no I/O.

### Which implementations exist here

`lib/std/Io.zig:28-39`:

```zig
pub const Threaded = @import("Io/Threaded.zig");
pub const fiber = @import("Io/fiber.zig");
pub const Evented = if (fiber.supported) switch (builtin.os.tag) {
    .linux => Uring,
    .dragonfly, .freebsd, .netbsd, .openbsd => Kqueue,
    .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => Dispatch,
    else => void,
} else void; // context-switching code not implemented yet
pub const Dispatch = @import("Io/Dispatch.zig");
pub const Kqueue = @import("Io/Kqueue.zig");
pub const Uring = @import("Io/Uring.zig");
```

`fiber.supported` is `true` on aarch64, riscv64, and x86_64 (`lib/std/Io/fiber.zig:1`).

**Measured on this Mac (aarch64-macos): `Io.Evented == Io.Dispatch`, and it does
not compile.** A program that calls `ev.init(gpa, .{})` and then `ev.io()` fails
inside the standard library:

```
lib/std/Io/Dispatch.zig:2067:38: error: switch must handle all possibilities
lib/std/Io.zig:253:5: note: unhandled enumeration value: 'net_send'
```

So on this toolchain the evented backend is in flux and `Io.Threaded` is the only
working implementation. A compile-time probe of the *type* (`Io.Evented == Io.Dispatch`,
`Io.Evented != void`) does pass; instantiating it does not. Do not promise an
event-loop backend here without recompiling that probe first.

## Io.Writer

`lib/std/Io/Writer.zig:14-18` is the whole state:

```zig
vtable: *const VTable,
/// If this has length zero, the writer is unbuffered, and `flush` is a no-op.
buffer: []u8,
/// In `buffer` before this are buffered bytes, after this is `undefined`.
end: usize = 0,
```

That is the buffering model in three fields. The buffer belongs to the caller,
`end` is the fill level, and `flush` is what moves bytes to the sink. A writer
with a zero-length buffer is unbuffered and `flush` is a no-op.

`VTable` (`:20`) has one required method and three defaulted ones:

```zig
drain:    *const fn (w: *Writer, data: []const []const u8, splat: usize) Error!usize,   // :45
sendFile: *const fn (w: *Writer, file_reader: *File.Reader, limit: Limit) FileError!usize
          = unimplementedSendFile,                                                       // :63
flush:    *const fn (w: *Writer) Error!void = defaultFlush,                              // :78
rebase:   *const fn (w: *Writer, preserve: usize, capacity: usize) Error!void
          = defaultRebase,                                                               // :86
```

`drain` consumes `buffer[0..end]` first, then each slice of `data` in order, with
the last element repeated `splat` times. It returns bytes consumed **excluding**
the buffer. `Error` is exactly `error{WriteFailed}` (`:91`); real diagnostics live
on the concrete implementation (`File.Writer.err`, `Io/File/Writer.zig:12`).

Ready-made writers:

- `Io.Writer.fixed(buffer)` (`:125`) writes into a slice, fails with `WriteFailed` when full.
- `Io.Writer.Allocating` (`:2643`) grows an allocation; `.init(gpa)`, `.written()`, `.deinit()`.
- `Io.Writer.Discarding` (`:2345`) counts and throws away; `.init(&.{})`, `.count`.
- `Io.Writer.hashed(hasher, buf)` (`:136`) tees into a hasher.
- `Io.Writer.failing` (`:140`) errors on everything.
- `Io.Writer.fromArrayList` / `toArrayList` (`:2482`, `:2496`).

Verified:

```zig
var buf: [64]u8 = undefined;
var w: Io.Writer = .fixed(&buf);
try w.print("{d}-{s}", .{ 42, "ok" });
try std.testing.expectEqualStrings("42-ok", w.buffered());  // :155

var a: Io.Writer.Allocating = .init(std.testing.allocator);
defer a.deinit();
try a.writer.print("x={d}", .{7});
try std.testing.expectEqualStrings("x=7", a.written());
```

### Implementing one

Embed an `Io.Writer` as a field named whatever you like, recover the outer struct
with `@fieldParentPtr`, and supply `drain`. This compiled and ran:

```zig
const Upper = struct {
    interface: Io.Writer,
    sink: []u8,
    written: usize = 0,

    fn init(buffer: []u8, sink: []u8) Upper {
        return .{ .interface = .{ .vtable = &vtable, .buffer = buffer }, .sink = sink };
    }

    const vtable: Io.Writer.VTable = .{ .drain = drain };

    fn drain(w: *Io.Writer, data: []const []const u8, splat: usize) Io.Writer.Error!usize {
        const u: *Upper = @alignCast(@fieldParentPtr("interface", w));
        u.push(w.buffered());          // buffered bytes go first
        var consumed: usize = 0;
        for (data[0 .. data.len - 1]) |bytes| {
            u.push(bytes);
            consumed += bytes.len;
        }
        const last = data[data.len - 1];
        for (0..splat) |_| {           // last element repeats `splat` times
            u.push(last);
            consumed += last.len;
        }
        _ = w.consume(w.end);          // report the buffer as taken
        return consumed;               // return count EXCLUDES the buffer
    }

    fn push(u: *Upper, bytes: []const u8) void { /* ... */ }
};
```

Two traps encoded there: the returned count excludes buffered bytes, and the
implementation is responsible for advancing past what it took from `buffer`.

## Io.Reader

`lib/std/Io/Reader.zig:16-21`:

```zig
vtable: *const VTable,
buffer: []u8,
/// Number of bytes which have been consumed from `buffer`.
seek: usize,
/// In `buffer` before this are buffered bytes, after this is `undefined`.
end: usize,
```

`buffer[seek..end]` is what is readable without touching the source. `VTable` (`:23`):

```zig
stream:  *const fn (r: *Reader, w: *Writer, limit: Limit) StreamError!usize,          // :45
discard: *const fn (r: *Reader, limit: Limit) Error!usize = defaultDiscard,           // :66
readVec: *const fn (r: *Reader, data: [][]u8) Error!usize = defaultReadVec,           // :84
rebase:  *const fn (r: *Reader, capacity: usize) RebaseError!void = defaultRebase,    // :98
```

Only `stream` is required, and it writes into a `*Writer` rather than a byte
slice. That is the inversion behind the rewrite: readers push into writers.
`Error` is `error{ReadFailed, EndOfStream}` family (`:102`, `:112`).

Ready-made: `Io.Reader.fixed(bytes)` (`:152`), `Io.Reader.limited(limit, buf)`
(`:147`, returns a `Reader.Limited` whose `.interface` is the reader),
`Io.Reader.failing` (`:132`), `Io.Reader.ending` (`:145`).

### The delimiter trap

There are three delimiter takers and they differ in exactly the way that bites:

| Call | Returns | Consumes the delimiter |
|---|---|---|
| `takeDelimiterExclusive(d)` (`:876`) | text without `d` | **No** |
| `takeDelimiterInclusive(d)` (`:806`) | text **with** `d` | Yes |
| `takeDelimiter(d)` (`:899`) | text without `d`, `null` at end | Yes |

Verified, and the second line here is the one that surprises people:

```zig
var r: Io.Reader = .fixed("alpha\nbeta\ngamma\n");
try expectEqualStrings("alpha", try r.takeDelimiterExclusive('\n'));
try expectEqualStrings("\n",    try r.takeDelimiterInclusive('\n')); // the leftover delimiter
try expectEqualStrings("beta\n", try r.takeDelimiterInclusive('\n'));
try expectEqualStrings("gamma", (try r.takeDelimiter('\n')).?);
try expectEqual(@as(?[]u8, null), try r.takeDelimiter('\n'));
```

For a line loop, `takeDelimiter` is the one you want. If you use
`takeDelimiterExclusive`, follow it with `r.toss(1)` (`:549`).

Other primitives, all verified: `peek(n)` (`:515`), `take(n)` (`:563`),
`takeByte()` (`:1161`), `toss(n)`, `tossBuffered()` (`:555`), `fill(n)` (`:1109`),
`streamRemaining(w)` (`:262`), `streamExact(w, n)` (`:216`),
`streamDelimiter(w, d)` (`:956`), `allocRemaining(gpa, limit)` (`:296`),
`appendRemaining(...)` (`:351`), `takeStructPointer`, `takeEnum`, `takeLeb128`.

## Io.Limit

`lib/std/Io.zig:728`. A `usize`-backed non-exhaustive enum, not a plain integer:

```zig
pub const Limit = enum(usize) { nothing = 0, unlimited = math.maxInt(usize), _ };
```

Construct with `.limited(n)` or `.limited64(n)`. Useful methods: `toInt()` returns
`?usize` and is `null` for `.unlimited`; `slice(s)` and `sliceConst(s)` truncate a
slice; `minInt(n)`; `min`/`max`; `subtract(n)` returns `null` on overshoot;
`nonzero()`; `slice1(buf)` leaves one byte of headroom so callers can tell
"hit the limit" apart from "end of stream". Verified:

```zig
const l: Io.Limit = .limited(4);
try expectEqual(@as(?usize, 4), l.toInt());
try expectEqual(@as(?usize, null), Io.Limit.unlimited.toInt());
var bytes = "abcdefgh".*;
try expectEqualStrings("abcd", l.slice(&bytes));
```

## Files and directories

Everything takes `io` as its second parameter, right after the receiver.

```zig
const io = std.testing.io;
var tmp = std.testing.tmpDir(.{});
defer tmp.cleanup();

try tmp.dir.writeFile(io, .{ .sub_path = "hello.txt", .data = "line one\n" });

const contents = try tmp.dir.readFileAlloc(io, "hello.txt", gpa, .limited(1024));
defer gpa.free(contents);
```

`Io.Dir.writeFile` is `lib/std/Io/Dir.zig:661`, `readFileAlloc` is `:1330` and
takes an `Io.Limit` rather than a `usize` max size. `createFile` is `:641`,
`openFile` is `:580`.

Adapters between a `File` and the stream interfaces (`lib/std/Io/File.zig`):

```zig
pub fn reader(file: File, io: Io, buffer: []u8) Reader;           // :563  positional
pub fn readerStreaming(file: File, io: Io, buffer: []u8) Reader;  // :589
pub fn writer(file: File, io: Io, buffer: []u8) Writer;           // :597  positional
pub fn writerStreaming(file: File, io: Io, buffer: []u8) Writer;  // :604
```

or construct directly: `Io.File.Writer.init(file, io, buf)` (`Io/File/Writer.zig:36`),
`.initStreaming` (`:48`), `.initDetect` (`:58`); `Io.File.Reader.init` (`Io/File/Reader.zig:84`),
`.initSize` (`:92`), `.initStreaming` (`:104`). Both expose the stream as the
`interface` field (`Io/File/Writer.zig:19`, `Io/File/Reader.zig:27`) and keep the
real error in `err: ?Error` beside it (`Io/File/Writer.zig:12`), because
`Io.Writer.Error` is only `error{WriteFailed}`.

Verified round trip:

```zig
var file = try tmp.dir.createFile(io, "nums.txt", .{});
defer file.close(io);
var wbuf: [64]u8 = undefined;
var fw: Io.File.Writer = .init(file, io, &wbuf);
for (0..3) |i| try fw.interface.print("{d}\n", .{i});
try fw.interface.flush();
```

`Io.Dir.cwd()` (`Io/Dir.zig:88`) is a sentinel handle (`AT_FDCWD` on POSIX). It
works for relative operations such as `access`, `openFile`, `createDirPathOpen`,
but **`realPath` on it fails with `error.FileNotFound`**. For the actual working
directory string use `std.process.currentPath(io, &buf)` or `currentPathAlloc`.

## stdin, stdout, stderr

stdout: see hello world above. Same shape for stdin, with `initStreaming` because
a pipe is not seekable. This compiled and ran (`printf 'a\nb\n' | ./io_stdin_main`
prints `1: a` / `2: b`):

```zig
var in_buf: [4096]u8 = undefined;
var stdin_reader: Io.File.Reader = .initStreaming(.stdin(), io, &in_buf);
const stdin = &stdin_reader.interface;

var out_buf: [4096]u8 = undefined;
var stdout_writer: Io.File.Writer = .init(.stdout(), io, &out_buf);
const stdout = &stdout_writer.interface;
defer stdout.flush() catch {};

var n: usize = 0;
while (try stdin.takeDelimiter('\n')) |line| {
    n += 1;
    try stdout.print("{d}: {s}\n", .{ n, line });
}
```

stderr has three tiers, all verified in one program:

```zig
std.debug.print("unbuffered, ignores errors\n", .{});  // std/debug.zig:323
std.log.info("structured, honors std_options\n", .{});

var buf: [256]u8 = undefined;
const locked = try io.lockStderr(&buf, null);          // std/Io.zig:2633
defer io.unlockStderr();                                // :2646
try locked.file_writer.interface.print("locked write\n", .{});
try locked.file_writer.interface.flush();
```

`lockStderr` returns `LockedStderr { file_writer: *File.Writer, terminal_mode: Terminal.Mode }`
(`:2598`) and clears any in-progress `std.Progress` line for you. `tryLockStderr`
(`:2640`) is the non-blocking form. This lock is what keeps `std.debug.print`,
`std.log`, `std.Progress`, and application stderr writes from interleaving.

## Concurrency

`async` and `await` are ordinary identifiers on master, not keywords. `Io.Threaded`'s
own vtable literally contains `.async = async`.

### async and await

`lib/std/Io.zig:2520`:

```zig
pub fn async(
    io: Io,
    function: anytype,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) Future(@typeInfo(@TypeOf(function)).@"fn".return_type.?)
```

`Future(Result)` (`:1290`) is `{ any_future: ?*AnyFuture, result: Result }` with two
methods, both idempotent and both not threadsafe:

```zig
pub fn await(f: *@This(), io: Io) Result
pub fn cancel(f: *@This(), io: Io) Result
```

Verified, including the error-union case:

```zig
var f1 = io.async(add, .{ 2, 3 });
var f2 = io.async(add, .{ 10, 20 });
try expectEqual(@as(i32, 5),  f1.await(io));
try expectEqual(@as(i32, 30), f2.await(io));

var f = io.async(fail, .{});                 // fn fail() error{Boom}!i32
try expectError(error.Boom, f.await(io));
```

`async` is a hint. An implementation is allowed to run the function inline, so it
does not by itself guarantee overlap.

### concurrent

`:2559`. Same shape, but returns `ConcurrentError!Future(...)` and carries a real
guarantee: the caller can make progress while the task runs. `ConcurrentError`
is `error{ConcurrencyUnavailable}` (`:2546`), returned when resources are
exhausted or the implementation cannot do it.

```zig
var f = try io.concurrent(add, .{ 7, 8 });
try expectEqual(@as(i32, 15), f.await(io));
```

Use `concurrent` when the task must actually run alongside you (a server accept
loop feeding a client in the same test, for instance). Use `async` when you only
want the option of parallelism.

### Group

`:1332`. An unordered set of tasks awaited or canceled as a whole. Per-task
resources are released when that task returns, so a long-lived group fed by
`Group.concurrent` does not leak; a group fed by `Group.async` and never awaited
can.

```zig
var group: Io.Group = .init;
defer group.cancel(io);
for (0..8) |_| group.async(io, bump, .{&counter});   // returns void
try group.await(io);                                  // Cancelable!void, NOT void
```

`Group.await` returns `Cancelable!void`; forgetting the `try` is a compile error
("error union ... is ignored"). `Group.concurrent` returns `ConcurrentError!void`.

### Select

`:1490`. `Select(U)` where `U` is a tagged union of the possible results. Spawn
with `s.async(.field, fn, args)` and take whichever finishes first.

```zig
const Result = union(enum) { a: u32, b: u64 };
var buffer: [2]Result = undefined;
var s: Io.Select(Result) = .init(io, &buffer);
defer s.cancelDiscard();
s.async(.a, one, .{});
s.async(.b, two, .{});
for (0..2) |_| switch (try s.await()) { .a => |v| ..., .b => |v| ... }
```

Also `awaitMany(buffer, min)`, `cancel()` returning `?U`, and `concurrent`.

### Queue

`:2378`. A bounded, blocking, threadsafe channel over a caller-supplied buffer.
`putOne` / `putAll` / `put(elements, min)`, `getOne` / `get(buffer, min)`,
`close(io)`. A closed queue drains first and then returns `error.Closed`.

```zig
var qbuf: [2]u32 = undefined;
var q: Io.Queue(u32) = .init(&qbuf);
var f = try io.concurrent(producer, .{ io, &q });
defer f.await(io);
while (q.getOne(io)) |v| total += v else |err| switch (err) {
    error.Closed => {},
    else => return err,
}
```

### Locks and conditions

`Io.Mutex` (`:1710`, `extern struct` over an atomic tri-state, `.init`, `tryLock`
takes no `io`, `lock`/`unlock` do), `Io.Condition` (`:1776`, `.init`, `wait(io, *Mutex)`,
`signal`, `broadcast`), `Io.RwLock` (`Io/RwLock.zig:15`, `.init`, `lock`/`unlock`,
`lockShared`/`unlockShared`), `Io.Semaphore` (`Io/Semaphore.zig`, no `init` function:
it is `.{ .permits = n }`, then `wait(io)` / `post(io)`), `Io.Event` (`:1960`), and
raw `io.futexWait` / `futexWake` (`:1675`, `:1699`).

Verified condition-variable pattern:

```zig
try s.mutex.lock(io);
while (s.count < 4) try s.cond.wait(io, &s.mutex);
s.mutex.unlock(io);
```

## Cancellation

Cancellation is cooperative and delivered at **cancelation points**, which are
calls into `Io` that can return `error.Canceled`. `Cancelable` is
`error{Canceled}` (`:813`).

- `future.cancel(io)` (`:1290`) requests cancellation **and** awaits, returning the
  task's result. Verified: a task blocked in `io.sleep(.fromSeconds(3600), .awake)`
  returns `error.Canceled` from `f.cancel(io)`.
- Only the **next** cancelation point in that task signals; later ones do not
  re-signal. Swallowing `error.Canceled` is therefore usually a bug.
- `io.checkCancel()` (`:1479`) is an explicit cancelation point for compute loops.
- `io.recancel()` (`:1433`) re-arms the request you deferred.
- `Io.CancelProtection` (`:1445`) and `io.swapCancelProtection(new)` (`:1465`)
  defer delivery across a critical section.
- `Group.cancel(io)` (`:1332`) and `Select.cancel` / `cancelDiscard` cancel a whole set.

## Time, sleeping, randomness

`std.time` is now only unit constants. Clocks live on `Io`.

`Io.Clock` (`:830`) members are `real`, `awake`, `boot`, `cpu_process`, `cpu_thread`.
**There is no `.monotonic`.** `.awake` is the monotonic clock that excludes
suspend (`CLOCK_UPTIME_RAW` on macOS, `CLOCK_MONOTONIC` on Linux); `.boot`
includes suspend.

```zig
const t0: Io.Timestamp = Io.Clock.now(.awake, io);   // Io.zig:887, note the receiver order
try io.sleep(.fromMilliseconds(5), .awake);          // :2591
const elapsed: Io.Duration = t0.untilNow(io, .awake);
try expect(elapsed.toMilliseconds() >= 4);

const ct: Io.Clock.Timestamp = .now(io, .real);      // :913, bundles the clock
```

`Io.Duration` (`:1081`): `.zero`, `.max`, `fromNanoseconds/Microseconds/Milliseconds/Seconds`,
`toNanoseconds/...`, and a `format` method so `{f}` works.
`Io.Timestamp` (`:1015`): `durationTo`, `addDuration`, `subDuration`, `untilNow`,
`compare`, `withClock`.
`Io.Timeout` (`:1245`) is the union accepted by `waitTimeout`-style calls.
`Clock.resolution(io)` (`:898`) reports granularity and may be zero.

Randomness: `io.random(buf)` (`:2662`) for fast bytes, `io.randomSecure(buf)`
(`:2679`, `RandomSecureError = error{EntropyUnavailable} || Cancelable`) for
crypto entropy. `std.crypto.random` no longer exists. `std.Random` still exists
but is the PRNG-engine namespace (`DefaultPrng`, `ChaCha`, ...), not an OS source.

## Networking

`Io.net` (`lib/std/Io/net.zig`). Verified TCP loopback round trip:

```zig
const any: Io.net.IpAddress = try .parseLiteral("127.0.0.1:0");
var server = try any.listen(io, .{});          // net.zig:248 -> Server
defer server.deinit(io);

var addr = server.socket.address;              // resolved ephemeral port; the
                                               // field is socket.address, there
                                               // is no `listen_address`
var f = try io.concurrent(serve, .{ io, &server });
defer f.await(io) catch {};

var stream = try addr.connect(io, .{ .mode = .stream });  // .mode is REQUIRED
defer stream.close(io);
var buf: [64]u8 = undefined;
var r = stream.reader(io, &buf);
try expectEqualStrings("pong", try r.interface.take(4));
```

Server side: `server.accept(io)` returns a `Stream`; `stream.writer(io, buf)` and
`stream.reader(io, buf)` give you the `Io.Writer` / `Io.Reader` interfaces through
their `.interface` fields, exactly like files.

`std.http.Client` takes both an allocator and an `Io` as struct fields
(`lib/std/http/Client.zig:26-28`); `std.http.Server.init(in: *Reader, out: *Writer)`
takes the two stream interfaces and therefore works over anything, including a
fixed buffer.

## The Io vtable, briefly

`lib/std/Io.zig:51` onwards. The shape of a method is
`*const fn (userdata: ?*anyopaque, ...) ...`. Groups present on this toolchain:

`crashHandler`; `async` / `concurrent` / `await` / `cancel`; `groupAsync` /
`groupConcurrent` / `groupAwait` / `groupCancel`; `recancel` /
`swapCancelProtection` / `checkCancel`; `futexWait` / `futexWaitUncancelable` /
`futexWake`; `operate` and the `Batch` family (`Operation`, `:246`; `operate`,
`:553`; `Batch`, `:576`); `dir*` (create/open/stat/access/createFile/openFile/read/
realPath/delete/rename/symLink/readLink/setOwner/setPermissions/setTimestamps/hardLink);
`file*` (stat/length/close/writePositional/readPositional/seek/sync/isTty/
setLength/lock/realPath/hardLink/memoryMap*); `process*` (executableOpen/
executablePath/currentPath/setCurrentDir/replace/spawn) and `child*`;
`lockStderr` / `tryLockStderr` / `unlockStderr`; `random` / `randomSecure`;
`now` / `clockResolution` / `sleep`; `net*` (listenIp/accept/bindIp/connectIp/
listenUnix/connectUnix/socketCreatePair/writeFile/close/shutdown/interfaceName*/lookup).

You almost never write one. Read `Io.failing` (`:2706` and the `failing*` /
`unreachable*` / `no*` helpers after it) as the minimal reference implementation
before attempting a real one.

## Still in flux, say so rather than guessing

- `Io.Evented` / `Io.Dispatch` does not compile on this toolchain (unhandled
  `net_send` in `Io/Dispatch.zig:2067`). `Io.Kqueue` and `Io.Uring` were not
  compiled here at all: **unverified**.
- `Io.Operation` / `Io.Batch` (`:246`, `:576`) are the batched-submission surface.
  Not exercised by any snippet here: **unverified**.
- `Io.Terminal` (`Io/Terminal.zig`) beyond `Terminal.Mode` appearing in
  `lockStderr`: **unverified**.
- `Io.File.MemoryMap`, `Io.File.Atomic`, `Io.File.MultiReader`, `Io.Dir.Reader`:
  present in the tree, **not compiled here**.
- `std.process.Child` spawn through `io`: present in the vtable, **not exercised**.

## Verification recipe

Never write a Zig `std.Io` snippet from memory. Compile it:

```sh
mkdir -p /private/tmp/zig-std-scratch && cd /private/tmp/zig-std-scratch
/Users/donaldfilimon/.zvm/bin/zig test snippet.zig        # tests
/Users/donaldfilimon/.zvm/bin/zig build-exe snippet_main.zig  # programs with main
```

Scratch goes in `/private/tmp`, never iCloud and never the home root. Redirect to
a file and read `$?` from zig itself: `zig test x.zig > log 2>&1; echo "EXIT: $?"`.
A piped `| tail` reports tail's exit status, not zig's, and zsh has no
`${PIPESTATUS[0]}` (it is `$pipestatus`). Also remember that Zig analyzes lazily:
a `pub fn` that no `test` or `main` calls is never type-checked, so a passing
`zig test` on a file with no test block proves nothing.
