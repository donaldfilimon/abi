---
name: zig-std
description: Routing map of the Zig standard library on master (0.17.0-dev). Use when picking an import path or a type in std, when unsure whether a name still exists on master, when choosing between ArrayList and MultiArrayList or HashMap and ArrayHashMap, when writing a format string or a custom `format` method, and when a familiar name has moved or vanished (std.fs, std.time, std.Thread.Mutex, std.crypto.random, std.builtin, std.meta.fields, ArrayListUnmanaged, GeneralPurposeAllocator). Covers mem, containers, fmt, fs/Io.Dir, process, posix, json, http, crypto, hash, math, sort, Thread, atomic, time, testing, log, debug, heap, and text handling. For anything touching I/O, concurrency, or an `io` parameter, use zig-io instead.
---

# The Zig standard library on master

Ground truth is the installed toolchain, measured 2026-09-06:

- binary `/Users/donaldfilimon/.zvm/bin/zig`, version `0.17.0-dev.2018+ab30a0b9a`
- stdlib source `/Users/donaldfilimon/.zvm/master/lib/std/`, 100 top-level entries
- every code sample below was compiled and run with that binary

Citations are `lib/std/<path>.zig:<line>` against that tree. Names are durable,
line numbers are not. Verify a name with `grep -n "^pub fn <name>\|^pub const <name>"`
in the module before using it, rather than trusting recall.

Companions: `zig-io` for `std.Io` (all I/O, files, sockets, time, concurrency),
`zig-memory` for allocators, `zig-testing`, `zig-comptime`, `zig-types`,
`zig-errors`, `zig-build`, `zig-c-interop`, `zig-toolchain`.

## Changed on master

The load-bearing renames. Anything trained on 0.13 or 0.14 will get these wrong.

| Was | Now | Evidence |
|---|---|---|
| `std.io` | `std.Io` and nothing else; `std.io` **does not exist** | `std.zig:21`; compiling `std.io.getStdOut()` gives `error: root source file struct 'std' has no member named 'io'` |
| `std.fs.cwd`, `std.fs.File`, `std.fs.Dir` | `std.Io.Dir.cwd()`, `std.Io.File`, `std.Io.Dir`. `std.fs` is a 16-line deprecation shim plus `fs/path.zig` | `fs.zig:5-16`; `std.fs.cwd()` gives `error: root source file struct 'fs' has no member named 'cwd'` |
| `std.fs.path` | `std.Io.Dir.path` (same module, canonical name moved) | `fs.zig:5`, `Io/Dir.zig:17` |
| `ArrayList` carried its allocator | `std.ArrayList(T)` is **unmanaged**: `.empty`, and every mutator takes `gpa`. `std.ArrayListUnmanaged` is a deprecated alias **for it** | `std.zig:52`, `:62` |
| `ArrayList(T).init(gpa)` | `std.array_list.Managed(T).init(gpa)` if you really want the old shape | `array_list.zig:11` |
| `std.ArrayHashMapUnmanaged`, `AutoArrayHashMapUnmanaged`, `StringArrayHashMapUnmanaged` | `std.array_hash_map.Custom` / `.Auto` / `.String` | `std.zig:39-44` |
| `std.builtin` (the reflection namespace) | `std.lang`; `std.builtin` is deprecated, "to be removed after Zig 0.17.0" | `std.zig:69-72` |
| `@typeInfo(S).@"struct".fields` / `.decls` | **split into parallel arrays**: `.field_names`, `.field_types`, `.field_attrs`, `.decl_names` | `lang.zig:751-763` |
| `std.meta.fields` | `@compileError("deprecated in favor of @typeInfo")` | `meta.zig:248` |
| `std.GeneralPurposeAllocator` | `std.heap.DebugAllocator(.{})`, `.init`, `.deinit()` returns `Check` | `heap.zig:24`, `:26` |
| `std.time.sleep`, `Timer`, `milliTimestamp`, `nanoTimestamp` | `std.time` is 35 lines: unit constants plus `epoch`. Clocks are `std.Io.Clock` / `Io.Timestamp` / `Io.Duration`, sleeping is `io.sleep` | `time.zig` (whole file) |
| `std.Thread.Mutex`, `.Condition`, `.Pool`, `.WaitGroup`, `.Semaphore`, `.ResetEvent`, `.Futex` | **all removed**. `std.Thread` is only spawn/join/detach/yield/getCpuCount/getCurrentId/setName/getName | `Thread.zig:344`, `:370`, `:364`, `:380`, `:293`, `:279`, `:52`, `:159` |
| those primitives | `std.Io.Mutex`, `Io.Condition`, `Io.RwLock`, `Io.Semaphore`, `Io.Group`, `io.futexWait`/`futexWake` | `Io.zig:1710`, `:1776`, `:48`, `:49`, `:1332` |
| `std.crypto.random` | **removed**. `io.randomSecure(buf)` for entropy, `io.random(buf)` for fast bytes. `std.Random` is the PRNG-engine namespace only | `Io.zig:2679`, `:2662` |
| `Ed25519.KeyPair.generate()` | `.generate(io)` | `crypto/25519/ed25519.zig:336` |
| `mem.indexOf`, `indexOfScalar`, `lastIndexOf`, `indexOfAny`, ... | renamed to `find`, `findScalar`, `findLast`, `findAny`, ...; old names are deprecated aliases that still resolve | `mem.zig:1436`/`:1441`, `:1238`/`:1241`, `:1522`/`:1528`, `:1340`/`:1344` |
| format `{}` on a `[]const u8` | **compile error**: "cannot format slice without a specifier (i.e. {s}, {x}, {b64}, or {any})" | `Io/Writer.zig:1502` |
| a type's `format` being called by `{}` | only `{f}` delegates to `format`; the method signature is now `fn format(self, w: *Io.Writer) Io.Writer.Error!void` (no fmt string, no options, no anytype writer) | `Io/Writer.zig:685-686` |
| `std.SegmentedList` | **not present** on this toolchain | absent from `std.zig` and from `lib/std/` |
| `std.PriorityQueue(...).init(gpa, ctx)` | `.empty` or `.initContext(ctx)`, then `push(gpa, x)` / `pop()` / `deinit(gpa)` | `priority_queue.zig:26`, `:33`, `:48`, `:88` |
| `std.heap.MemoryPool(T).init(gpa)` | `.empty`, then `create(gpa)` / `destroy(ptr)` / `deinit(gpa)` | `heap/memory_pool.zig:55`, `:113`, `:71` |

## std.mem

The slice-and-bytes toolbox. It no longer contains the file or stream API.

**Allocator.** `std.mem.Allocator` (`mem.zig:22`) is the interface every allocating
API takes. `alloc`/`free`, `create`/`destroy`, `dupe`/`dupeZ`, `realloc`, `resize`.
See `zig-memory` for which concrete allocator to pick.

**Comparison and search.** `eql` (`:753`), `order` (`:673`), `lessThan` (`:724`),
`startsWith` (`:3208`), `endsWith` (`:3219`), `allEqual` (`:1194`), `count` (`:1646`),
`containsAtLeast` (`:1714`). Search is the `find*` family; the `indexOf*` names are
deprecated aliases pointing at them:

| Deprecated | Canonical | Line |
|---|---|---|
| `indexOf` | `find` | `:1436` / `:1441` |
| `indexOfPos` | `findPos` | `:1555` / `:1558` |
| `lastIndexOf` | `findLast` | `:1522` / `:1528` |
| `indexOfScalar` | `findScalar` | `:1238` / `:1241` |
| `lastIndexOfScalar` | `findScalarLast` | `:1246` / `:1249` |
| `indexOfAny` | `findAny` | `:1340` / `:1344` |
| `indexOfNone` | `findNone` | `:1380` / `:1385` |
| `indexOfDiff` | `findDiff` | `:854` / `:858` |
| `indexOfMin` / `indexOfMax` | `findMin` / `findMax` | `:3807` / `:3830` |

**Copying.** `copyForwards` (`:251`) and `copyBackwards` (`:259`) for overlapping
ranges; use `@memcpy` when they cannot overlap. `swap` (`:3885`), `reverse` (`:3924`),
`rotate` (`:4078`), `replace` (`:4096`), `replaceScalar` (`:4151`).

**Trimming.** `trim` (`:1224`), `trimStart` (`:1202`), `trimEnd` (`:1213`). Note the
names: not `trimLeft`/`trimRight`.

**Splitting.** The rule is one line: **tokenize skips empty runs, split does not.**

```zig
var t = mem.tokenizeScalar(u8, "a,,b,", ',');  // "a", "b"
var s = mem.splitScalar(u8, "a,,b,", ',');     // "a", "", "b", ""
```

Six splitters and six tokenizers, by delimiter kind:
`tokenizeScalar`/`splitScalar` (one byte, `:2472`/`:2667`),
`tokenizeAny`/`splitAny` (any byte from a set, `:2428`/`:2647`),
`tokenizeSequence`/`splitSequence` (a multi-byte delimiter, `:2450`/`:2626`),
plus `splitBackwardsScalar`/`Any`/`Sequence` (`:2866`/`:2846`/`:2825`).
Iterators are `TokenIterator` (`:3327`), `SplitIterator` (`:3401`),
`SplitBackwardsIterator` (`:3467`). Also `window` (`:3042`).

**Cutting.** `cut` (`:3255`), `cutLast` (`:3273`), `cutScalar` (`:3292`),
`cutPrefix` (`:3229`), `cutSuffix` (`:3239`). **`cut` returns a tuple, not a struct**:

```zig
const before, const after = mem.cut(u8, "key=value", "=").?;
```

**Joining.** `join` (`:3521`), `joinZ` (`:3527`), `concat` (`:3611`).

**Bytes and alignment.** `asBytes` (`:4383`), `toBytes` (`:4446`), `bytesAsValue`
(`:4470`), `bytesToValue` (`:4530`), `bytesAsSlice` (`:4550`), `sliceAsBytes` (`:4649`),
`alignForward` (`:4810`), `alignBackward` (`:4933`), `Alignment` (`:25`).
Endianness: `littleToNative` (`:4249`), `nativeToLittle` (`:4281`), `bigToNative`,
`nativeToBig`, `byteSwap` (`:2240`). Bit-level: `readPackedInt` (`:2003`),
`writePackedInt` (`:2151`), `readVarInt` (`:1792`).

**Sorting entry points live here, not only in std.sort:** `mem.sort` (`:632`),
`mem.sortUnstable` (`:647`), and the `Context` variants.

**Sentinels.** `span` (`:917`), `sliceTo` (`:974`), `len` (`:1118`),
`findSentinel` (`:1149`, deprecated alias `indexOfSentinel`), `absorbSentinel` (`:4746`).

## Containers

Pick by shape, not by habit.

| Need | Type | Import |
|---|---|---|
| growable array | `std.ArrayList(T)` (unmanaged) | `std.zig:52` |
| growable array carrying its allocator | `std.array_list.Managed(T)` | `array_list.zig:11` |
| over-aligned array | `std.array_list.Aligned(T, alignment)` | `array_list.zig:633` |
| hash lookup, any key | `std.AutoHashMap(K,V)` / `AutoHashMapUnmanaged` | `hash_map.zig:46`, `:50` |
| hash lookup, string key | `std.StringHashMap(V)` / `StringHashMapUnmanaged` | `hash_map.zig:64`, `:70` |
| custom hash/eql | `std.HashMap(K,V,Context,max_load)` / `HashMapUnmanaged` | `hash_map.zig:135` |
| hash lookup with **insertion order** and `.keys()`/`.values()` slices | `std.array_hash_map.Auto(K,V)` / `.String(V)` / `.Custom(...)` | `array_hash_map.zig:15`, `:20`, `:70` |
| many rows, few fields touched at a time (struct of arrays) | `std.MultiArrayList(T)` | `std.zig:22` |
| double-ended queue | `std.Deque(T)` | `std.zig:9` |
| intrusive lists | `std.DoublyLinkedList`, `std.SinglyLinkedList` (no element type: the node is embedded) | `std.zig:10`, `:28` |
| fixed-size bit set | `std.StaticBitSet(N)` (`.empty`, `.full`) | `bit_set.zig:38` |
| runtime-sized bit set | `std.DynamicBitSetUnmanaged` (`.initEmpty(gpa, n)`), `std.DynamicBitSet` | `bit_set.zig:676`, `:1082` |
| set/map keyed by an enum | `std.EnumSet(E)` (`.empty`), `std.EnumMap(E,V)`, `std.EnumArray(E,V)`, `std.EnumMultiset` | `enums.zig:241`, `:423`, `:1058`, `:661` |
| heap by priority | `std.PriorityQueue(T,Ctx,cmp)`, `std.PriorityDequeue` | `std.zig:23`, `:24` |
| comptime string lookup | `std.StaticStringMap(V).initComptime(...)` | `std.zig:7` |
| balanced tree | `std.Treap(K,cmp)` | `std.zig:36` |
| env-var-shaped string maps | `std.BufMap`, `std.BufSet` | `std.zig:5`, `:6` |

**`std.SegmentedList` does not exist on this toolchain.** If you need stable
element pointers across growth, use `std.MultiArrayList` plus indices, or a
`MemoryPool`, and say so rather than reaching for the old name.

Verified shapes:

```zig
var list: std.ArrayList(u32) = .empty;
defer list.deinit(gpa);
try list.append(gpa, 1);
try list.appendSlice(gpa, &.{ 2, 3 });
_ = list.pop();                       // returns ?T

var um: std.AutoHashMapUnmanaged(u32, u32) = .empty;
defer um.deinit(gpa);
try um.put(gpa, 2, 20);
const gop = try um.getOrPut(gpa, 2);  // gop.found_existing

var m: std.array_hash_map.Auto(u32, u32) = .empty;   // TWO type params, not three
defer m.deinit(gpa);
try m.put(gpa, 10, 1);
try m.put(gpa, 5, 2);
try expectEqualSlices(u32, &.{ 10, 5 }, m.keys());   // insertion order preserved

var mal: std.MultiArrayList(Row) = .empty;
defer mal.deinit(gpa);
try mal.append(gpa, .{ .id = 1, .ok = true });
const slice = mal.slice();
_ = slice.items(.id);                 // []u32, one column

var pq: std.PriorityQueue(u32, void, lt) = .empty;
defer pq.deinit(gpa);
try pq.push(gpa, 5);
_ = pq.pop();                         // ?T

var set: std.EnumSet(Color) = .empty; // NOT .initEmpty()
var static: std.StaticBitSet(64) = .empty;
```

## std.fmt and format strings

`std.fmt` (`fmt.zig`) is the parsing and number side. Rendering lives on
`Io.Writer.print` (`Io/Writer.zig:697`), and the specifier list is documented
above it at `Io/Writer.zig:631-696`. Entry points: `std.fmt.bufPrint` (`:600`),
`allocPrint` (`:625`), `count` (`:615`), `parseInt` (`:322`), `parseFloat` (`:568`),
`bytesToHex` (`:1140`), `hexToBytes` (`:1156`).

Placeholder grammar, unchanged in shape:

```
{[argument][specifier]:[fill][alignment][width].[precision]}
```

Specifiers on this toolchain, all verified by compiling:

| Spec | Meaning | Verified |
|---|---|---|
| `{d}` | decimal | `"{d}", .{42}` gives `42` |
| `{x}` / `{X}` | hex, or a byte slice as hex | `2a` / `2A` |
| `{b}` / `{o}` | binary / octal | `101010` / `52` |
| `{s}` | u8 slice as text, or a C string via a many/C pointer | `hi` |
| `{c}` | integer as one ASCII byte | `*` |
| `{u}` | integer as a UTF-8 sequence | |
| `{e}` | scientific | `"{e}", .{@as(f64,100)}` gives `1e2` |
| `{t}` | **tag name** of an enum or tagged union, **name** of an error | `"{t}", .{E.beta}` gives `beta` |
| `{f}` | **delegate to the type's `format` method** | `(1,2)` for a Point |
| `{any}` | default reflection rendering, bypasses `format` | `.{ .x = 1, .y = 2 }` |
| `{q}` | double-quote escaped string | `"a\nb"` |
| `{qf}` | delegate to `format` while quote-escaping | |
| `{b64}` | standard base64 | `aGk=` for `"hi"` |
| `{B}` / `{Bi}` | bytes in SI / IEC units | `1.5kB` / `1.46KiB` at `.2` precision |
| `{?}` | optional: value or `null`; may carry an inner spec, e.g. `{?d}` | `null`, `5` |
| `{!}` | error union: value or error; e.g. `{!d}` | `5` |
| `{*}` | the address instead of the value | |

Width, fill, alignment, precision: `"{d:3}"` gives `  7`, `"{d:.<3}"` gives `7..`,
`"{d:.2}"` on `3.14159` gives `3.14`. Indexed and named arguments work:
`"{1s} {0s}"` and `"{[n]d}"` with `.{ .n = 9 }`.

**Two traps, both compiled:**

1. Bare `{}` on a `[]const u8` is a **compile error**, not a fallback:
   `cannot format slice without a specifier (i.e. {s}, {x}, {b64}, or {any})`
   (`Io/Writer.zig:1502`). Use `{s}`.
2. Bare `{}` on a struct that **has** a `format` method compiles and does **not**
   call it. It renders `.{ .x = 1, .y = 2 }`, the same as `{any}`. Only `{f}`
   delegates. Note also that `{any}` prints anonymous-literal syntax with **no
   type name prefix** on master.

### The custom `format` method

The signature changed. On master it takes exactly one writer and returns
`Io.Writer.Error!void` (`Io/Writer.zig:685-686`). No format string, no
`std.fmt.FormatOptions`, no `anytype` writer, no `_ = fmt;` boilerplate.

```zig
const Point = struct {
    x: i32,
    y: i32,
    pub fn format(p: Point, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("({d},{d})", .{ p.x, p.y });
    }
};
// "{f}", .{Point{ .x = 1, .y = 2 }}  ->  "(1,2)"
```

Variants seen in the tree: `formatNumber(self, w, std.fmt.Number)` for types that
honor width and precision (`Io.Timestamp` does, `Io.zig:1060`).

## Files, directories, paths, processes

All of this is in `zig-io`; the routing entry is:

- files and directories: `std.Io.Dir`, `std.Io.File`, both taking `io` as the
  second parameter. `std.fs` is a shim (`fs.zig:5-16`).
- path string manipulation: `std.Io.Dir.path` (`Io/Dir.zig:17`), with `basename`,
  `dirname`, `extension`, `join`, `resolve`, `isAbsolute`.
- `std.process`: `Init` (`process.zig:31`) is what `main` receives;
  `currentPath(io, buf)`, `currentPathAlloc(io, gpa)`, `Environ` and `Environ.Map`
  (`process/Environ.zig:100`, `:506`), `Child`, `Preopens`, `fatal`.
- `std.posix` is the raw syscall and constant layer (`AF`, `O`, `E`, `S`, `SIG`,
  `SEEK`, `STDOUT_FILENO`, `iovec`, `fd_t`, `sockaddr`, ...). It is **not** the
  portable file API. Reach for it only for a constant or a struct layout that
  `std.Io` does not expose.

Verified:

```zig
try expectEqualStrings("b", Io.Dir.path.basename("/a/b"));
try expectEqualStrings("/a", Io.Dir.path.dirname("/a/b").?);
try expectEqualStrings(".zig", Io.Dir.path.extension("m.zig"));
const n = try std.process.currentPath(io, &buf);
```

## std.json

`json.zig`. Three ways in, one way out.

```zig
// typed, arena-backed, caller owns `parsed`
const parsed = try std.json.parseFromSlice(Config, gpa, src, .{});   // json.zig:83
defer parsed.deinit();
parsed.value.port;

// dynamic
const p = try std.json.parseFromSlice(std.json.Value, gpa, "{\"a\":[1,2]}", .{});
p.value.object.get("a").?.array.items[1].integer;

// out, to any Io.Writer
try std.json.Stringify.value(cfg, .{}, &w);                          // json.zig:94

// or through a format string
try w.print("{f}", .{std.json.fmt(.{ .a = 1 }, .{})});               // json.zig:97
```

Also `parseFromSliceLeaky` (`:84`, for an arena you already own),
`parseFromTokenSource` (`:85`), `parseFromValue` (`:88`), `Scanner` (`:69`),
`Diagnostics` (`:75`), `ParseOptions` (`:81`). `std.zon` mirrors this shape for
ZON (`zon.zig:42-47`).

## std.http

`http.zig:7-8`. `Client` and `Server` both work over the `Io` stream interfaces.

- `std.http.Client` is a struct you fill in, and it needs **both** an allocator
  and an `Io` (`http/Client.zig:26-28`): `.{ .allocator = gpa, .io = io }`.
  Then `client.fetch(.{ .location = .{ .url = "..." } })` (`:1807`) for one-shot,
  or `client.request(...)` (`:1684`) for control. `deinit` at `:1307`.
- `std.http.Server.init(in: *Io.Reader, out: *Io.Writer)` (`http/Server.zig:25`),
  then `receiveHead()` (`:46`). Because it takes the stream interfaces rather than
  a socket, it parses out of a fixed buffer just as happily. Verified:

```zig
var in: Io.Reader = .fixed("GET /health HTTP/1.1\r\nHost: x\r\n\r\n");
var out: Io.Writer = .fixed(&out_buf);
var server: std.http.Server = .init(&in, &out);
var req = try server.receiveHead();
req.head.method;   // .GET
req.head.target;   // "/health"
var it = req.iterateHeaders();
```

Enums: `Method` (`http.zig:25`), `Status` (`:96`, with `.phrase()` and `.class()`),
`Version` (`:13`), `Header` (`:313`), `TransferEncoding`, `ContentEncoding`.
`std.Uri` parses URLs. Networking primitives are `std.Io.net`.

## std.crypto

`crypto.zig`. Namespaces, not a flat list: `crypto.hash` (`sha2.Sha256`,
`sha3`, `blake2`, `Blake3`, `Md5`, `Sha1`), `crypto.auth.hmac`, `crypto.aead`
(`chacha_poly`, `aes_gcm`), `crypto.sign` (`Ed25519`, `ecdsa`), `crypto.dh`
(`X25519`), `crypto.kdf` (`hkdf`), `crypto.pwhash` (`argon2`, `bcrypt`, `scrypt`),
`crypto.tls`, `crypto.Certificate`, `crypto.timing_safe`, `crypto.random_stream`.

**`std.crypto.random` is gone.** Entropy comes from `io.randomSecure(buf)`.
Key generation now takes an `Io`: `Ed25519.KeyPair.generate(io)`
(`crypto/25519/ed25519.zig:336`). Verified:

```zig
var digest: [Sha256.digest_length]u8 = undefined;
Sha256.hash("abc", &digest, .{});
try w.print("{x}", .{digest});    // ba7816bf...

HmacSha256.create(&out, "msg", &key);
const kp = std.crypto.sign.Ed25519.KeyPair.generate(io);
const sig = try kp.sign("msg", null);
try sig.verify("msg", kp.public_key);
```

## std.hash

Non-cryptographic. `hash.zig`: `Wyhash` (`:31`, the default for `AutoHashMap`),
`XxHash3` / `XxHash64` / `XxHash32` (`:34-36`), `Crc32` (`:9`, an alias for
`crc.@"CRC-32/ISO-HDLC"`), `Adler32` (`:1`), `Fnv1a_32/64/128` (`:12-14`),
`Murmur2_32/64`, `Murmur3_32` (`:21-24`), `CityHash32/64` (`:27-28`),
`SipHash64` / `SipHash128` (`:17-18`), `autoHash` (`:4`), `autoHashStrat` (`:5`).
All follow the same shape: a one-shot `Type.hash(seed, bytes)` and an incremental
`var h: Type = .init(seed); h.update(...); h.final()`.

## std.math and std.sort

`math.zig`: constants (`pi`, `e`, `tau`, `sqrt2`, `log2e`), `maxInt`/`minInt`,
checked `add`/`sub`/`mul` returning `error{Overflow}`, `divCeil`/`divFloor`/
`divExact`, `clamp`, `order` returning `Order`, `lossyCast`, `cast`,
`approxEqAbs` (`:77`) / `approxEqRel` (`:105`), `Log2Int`, `IntFittingRange`,
float predicates (`isNan` `:200`, `isInf`, `isFinite`, `isNormal`, `signbit`),
`nan`/`inf`/`floatEps`/`floatMax`/`floatMin`, and the libm surface
(`pow` `:218`, `sqrt` `:220`, `cbrt`, `atan2` `:225`, `hypot` `:226`, `log`/`log2`/`log10`,
`frexp` `:202`, `modf` `:204`, `ldexp` `:217`, `copysign` `:206`, `nextAfter` `:214`).
`std.simd` for vector helpers.

`sort.zig` holds the algorithms and the search; the ergonomic sort entry points
are on `std.mem`:

```zig
std.mem.sort(u32, &xs, {}, std.sort.asc(u32));          // stable  (mem.zig:632)
std.mem.sortUnstable(u32, &xs, {}, std.sort.desc(u32)); // pdqsort (mem.zig:647)
std.sort.isSorted(u32, &xs, {}, std.sort.asc(u32));     // sort.zig:950
```

Search takes an **order function** returning `std.math.Order`, not a `lessThan`:
`binarySearch` (`:450`), `lowerBound` (`:534`), `upperBound` (`:604`),
`partitionPoint` (`:675`), `equalRange` (`:771`). Also `min`/`max`/`argMin`/`argMax`
(`:878`, `:930`, `:846`, `:898`), and the low-level `insertion` (`:16`), `heap` (`:57`),
`pdq` (`:10`), `block` (`:9`).

## std.Thread

Threads only. Verified by compiling `@hasDecl` assertions in both directions.

```zig
pub fn spawn(config: SpawnConfig, comptime function: anytype, args: anytype) SpawnError!Thread  // :344
pub fn join(self: Thread) void        // :370
pub fn detach(self: Thread) void      // :364
pub fn yield() YieldError!void        // :380
pub fn getCpuCount() CpuCountError!usize  // :293
pub fn getCurrentId() Id              // :279
pub fn setName(self: Thread, io: Io, name: []const u8) SetNameError!void  // :52
pub fn getName(self: Thread, buffer_ptr: *[max_name_len:0]u8) GetNameError!?[]const u8  // :159
```

`SpawnConfig` (`:298`) carries `stack_size` and `allocator`.

Everything else you remember from `std.Thread` moved to `std.Io`:

| Gone from std.Thread | Use |
|---|---|
| `Mutex` | `std.Io.Mutex` (`Io.zig:1710`) |
| `Condition` | `std.Io.Condition` (`Io.zig:1776`) |
| `RwLock` | `std.Io.RwLock` (`Io.zig:48`) |
| `Semaphore` | `std.Io.Semaphore` (`Io.zig:49`), constructed as `.{ .permits = n }` |
| `Pool`, `WaitGroup` | `std.Io.Group` (`Io.zig:1332`), or `io.async` / `io.concurrent` |
| `ResetEvent` | `std.Io.Event` (`Io.zig:1960`) |
| `Futex` | `io.futexWait` / `io.futexWake` (`Io.zig:1675`, `:1699`) |

For anything beyond a bare `spawn`+`join`, prefer the `Io` primitives: they are
cancelable and they work under every `Io` implementation. See `zig-io`.

## std.atomic

`atomic.zig`. `std.atomic.Value(T)` (`:9`) with `.init(x)`, `load(order)`,
`store(x, order)`, `swap`, `fetchAdd`/`fetchSub`/`fetchAnd`/`fetchOr`/`fetchXor`,
`cmpxchgStrong`/`cmpxchgWeak` (both return `?T`, `null` on success), `rmw`, and a
`raw` field for non-atomic access. Orders are `std.lang.AtomicOrder`
(`lang.zig:35`): `.unordered`, `.monotonic`, `.acquire`, `.release`, `.acq_rel`,
`.seq_cst`. `std.atomic.cache_line` (`:519`) for padding. `std.atomic.spinLoopHint`.

```zig
var v: std.atomic.Value(u32) = .init(0);
_ = v.fetchAdd(5, .monotonic);
try expectEqual(@as(?u32, null), v.cmpxchgStrong(5, 10, .acq_rel, .monotonic));
```

## std.time

35 lines. Unit constants only: `ns_per_us`/`ms`/`s`/`min`/`hour`/`day`/`week`,
`us_per_*`, `ms_per_*`, `s_per_*`, plus `std.time.epoch` (`time.zig:1`) for
calendar arithmetic and `std.Tz` for zone files.

There is no `Timer`, no `sleep`, no `milliTimestamp`, no `Instant`. Clocks and
sleeping are `std.Io`:

```zig
const t0: Io.Timestamp = Io.Clock.now(.awake, io);   // Io.zig:887
try io.sleep(.fromMilliseconds(5), .awake);          // Io.zig:2591
const elapsed = t0.untilNow(io, .awake);             // Io.Duration
```

`Io.Clock` members are `real`, `awake`, `boot`, `cpu_process`, `cpu_thread`.
**There is no `.monotonic`;** `.awake` is it.

## std.testing

`testing.zig`. `allocator` (`:21`, leak-checked), `io` (`:24`, a live
`Io.Threaded`), `tmpDir(opts)` (`:603`, creates under `.zig-cache/tmp` and cleans
up via `.cleanup()`), `FailingAllocator`, `checkAllAllocationFailures`,
`expect`, `expectEqual`, `expectEqualStrings`, `expectEqualSlices`,
`expectEqualDeep`, `expectError`, `expectApproxEqAbs`/`Rel`, `expectFmt`,
`expectStringStartsWith`, `refAllDecls`, `fuzz` with `std.testing.Smith`.

**A `pub const std_options` in a test file is ignored.** `std.options` is read
from the **root** source file (`std.zig:119`), and under `zig test` the root is
the test runner, not your file. Verified: `log_level = .warn` in a test file had
no effect, while the same declaration in a `main` program did. Put logging
configuration in an executable, or override the test runner.

Second trap: Zig analyzes lazily. A `pub fn` that no `test` or `main` calls is
never type-checked, so a `zig test` that passes on a file with no test block
proves nothing. See `zig-testing`.

## std.log and std.debug

`log.zig`. `std.log.err` / `warn` / `info` / `debug`, and `std.log.scoped(.name)`
(`:137`) for a scoped logger. Level order is `err < warn < info < debug`
(`log.zig:30`), and a message is emitted when its level is `<=` the configured
level (`logEnabled`, `:76`). Configure from the root file:

```zig
pub const std_options: std.Options = .{
    .log_level = .warn,
    .log_scope_levels = &.{ .{ .scope = .net, .level = .info } },
};
```

Verified output from a compiled program with exactly that:

```
error: err: shown
warning: warn: shown
info(net): net info: shown, scope raised to .info
```

`std.Options` (`std.zig:121`) also carries `logFn`, `page_size_min`/`max`,
`fmt_max_depth`, `enable_segfault_handler`, `allow_stack_tracing`, `networking`,
`http_disable_tls`, `side_channels_mitigations`.

`debug.zig`: `print` (`:323`, unbuffered stderr, ignores errors), `assert` (`:440`),
`panic` (`:460`), `dumpHex` (`:344`), `lockStderr`/`unlockStderr` (`:298`, `:307`),
`dumpCurrentStackTrace` (`:821`), `FullPanic` (`:103`), `simple_panic` (`:97`),
`no_panic` (`:98`), `SelfInfo` (`:64`), `Dwarf`, `Pdb`, `ElfFile`, `MachOFile`.

## Reflection: std.lang, std.meta, @import("builtin")

Three distinct things, routinely confused:

- `std.lang` (`std.zig:72`) is the **reflection type namespace**: `Type`,
  `TypeId`, `Endian`, `Signedness`, `AtomicOrder`, `CallingConvention`,
  `Optimize`, `SourceLocation`, `StackTrace`, `panic`. `std.builtin` is a
  deprecated alias for it (`std.zig:69-71`, "to be removed after Zig 0.17.0").
- `@import("builtin")` is the per-compilation module: `cpu`, `os`, `abi`, `mode`,
  `single_threaded`, `link_libc`, `zig_backend`, `is_test`. Different thing.
- `std.meta` is helpers over the above: `FieldEnum` (`meta.zig:406`),
  `stringToEnum` (`:18`), `Tag`, `Child`, `Elem`, `hasFn`, `ArgsTuple`, `Tuple`,
  `eql`. **`std.meta.fields` is a `@compileError` now** (`:248`): use `@typeInfo`.

Struct type info is split into parallel arrays on master (`lang.zig:751-763`):

```zig
const info: std.lang.Type = @typeInfo(S);
info.@"struct".field_names[0];   // [:0]const u8   -- NOT .fields[0].name
info.@"struct".field_types[0];   // type
info.@"struct".field_attrs[0];   // .@"comptime", .@"align", .default_value_ptr
info.@"struct".decl_names;       // NOT .decls
```

## Text, encodings, and misc

- `std.ascii`: `isDigit`, `isAlphabetic`, `isWhitespace`, `toUpper`/`toLower`,
  `eqlIgnoreCase`, `upperString`/`lowerString`, `indexOfIgnoreCase`.
- `std.unicode`: `utf8ValidateSlice`, `utf8CountCodepoints`, `Utf8View` and its
  `.iterator()`, `utf8Encode`/`utf8Decode`, WTF-8 and UTF-16 conversions.
- `std.base64`: `standard`, `url_safe`, `standard_no_pad` codecs, each with an
  `Encoder` (`encode`, `calcSize`) and `Decoder` (`decode`, `calcSizeForSlice`).
- `std.Uri`, `std.SemanticVersion`, `std.Progress`, `std.Random` (engines:
  `DefaultPrng` = Xoshiro256, `DefaultCsprng` = ChaCha, `Pcg`, `Sfc64`, ...),
  `std.compress` (`flate`, `lzma`, `lzma2`, `xz`, `zstd`), `std.tar`, `std.zip`,
  `std.leb`, `std.simd`, `std.Target`, `std.Build`, `std.zig` (the compiler's own
  tokenizer/parser/Ast), `std.c`, `std.os`, `std.elf`/`macho`/`coff`/`dwarf`/`pdb`,
  `std.valgrind`, `std.wasm`, `std.spirv`.

## Verification recipe

Never write a Zig snippet from memory on this toolchain. Compile it:

```sh
mkdir -p /private/tmp/zig-std-scratch && cd /private/tmp/zig-std-scratch
/Users/donaldfilimon/.zvm/bin/zig test snippet.zig
/Users/donaldfilimon/.zvm/bin/zig build-exe snippet_main.zig
```

Scratch goes in `/private/tmp`, never iCloud and never the home root. Redirect to
a file and read `$?` from zig itself: `zig test x.zig > log 2>&1; echo "EXIT: $?"`.
A piped `| tail` reports tail's exit status, not zig's, and zsh has no
`${PIPESTATUS[0]}` (it is `$pipestatus`).

Before listing a name in a doc or a review comment, confirm it:

```sh
grep -n '^pub fn <name>\|^pub const <name>' /Users/donaldfilimon/.zvm/master/lib/std/<module>.zig
```

## Marked unverified

Named here because they exist in the tree, but **not compiled** for this skill,
so do not quote a signature from memory: `std.Build` and the whole build-system
surface (see `zig-build`), `std.compress`, `std.tar`, `std.zip`, `std.Target`,
`std.dwarf`/`elf`/`macho`/`coff`/`pdb`, `std.zig` (the self-hosted compiler
frontend), `std.c`, `std.valgrind`, `std.Progress`, `std.Treap`, `std.EnumMultiset`,
`std.BufMap`/`BufSet`, `std.zon`, `std.crypto.tls`/`Certificate`/`pwhash`,
`std.http.Client.fetch` against a live server, and `std.process.Child`.
