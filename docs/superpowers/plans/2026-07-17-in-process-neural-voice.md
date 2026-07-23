# In-Process Neural Voice (`--neural` flag) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing pure-Zig char-level LM (`src/features/nn/`) into `abi complete` as an honest, small, in-process neural voice behind a `--neural` flag, using a bundled pre-trained checkpoint.

**Architecture:** A new binary save/load module lets a trained `Model` persist to and load from disk. A new streaming-sample function in `nn/` emits chunked output via callback. CLI wiring adds `--neural` as a mutually-exclusive alternative to `--model`, dispatching to a new handler that loads the bundled checkpoint and streams generation to stdout with honest disclosure text.

**Tech Stack:** Zig 0.17.0-dev.1442+972627084 (pinned, `.zigversion`), existing `src/features/nn/` char-LM, existing CLI arg-parsing engine (`src/cli/arg.zig` + `src/cli/wiring.zig`).

## Global Constraints

- Zig 0.17 patterns throughout: `ArrayListUnmanaged(T).empty`, explicit `std.mem.Allocator`, `std.mem.readInt`/`writeInt` for binary fields, every module test block ends with `std.testing.refAllDecls(@This())`.
- File I/O follows this repo's existing convention (see `src/foundation/credentials.zig`): use `std.Options.debug_io` as the IO context, `std.Io.Dir.cwd()` for relative paths. Verify against that file's exact call shapes if the compiler disagrees with a call below — that file is the ground-truth reference for this repo's current `std.Io` usage, more authoritative than any single API guess in this plan.
- `src/features/nn/` stays `std`-only (per its own `types.zig` doc comment) — do not import `features.ai` types into any `nn/` file. The streaming callback type is defined locally in `nn/` using only `std.mem.Allocator`/`*anyopaque`/`anyerror!void`, not `features.ai.types.StreamCallback`.
- No silent `catch {}` in the checkpoint load path — a missing/corrupt checkpoint must surface a distinct error, never silently fall back.
- Claims discipline (`AGENTS.md`): help text and any printed output for `--neural` must state it is a small in-process character-level demo model, not comparable to the template/live paths — never imply general-purpose assistant quality.
- `--neural` and `--model` are mutually exclusive — passing both is a CLI usage error, not a silent override.
- Frozen CLI surface: this is a new flag on the existing `complete` command, not a new top-level command — the 13-command set is unchanged.
- After any public API change in `src/features/nn/` or `src/cli/`, run `zig build check-parity` (note: `src/features/nn/` and `src/cli/` are not both necessarily under the mod/stub parity contract — `nn` has a `stub.zig`, confirm parity still passes after each task regardless).

---

### Task 1: Model binary save/load (`src/features/nn/persist.zig`)

**Files:**
- Create: `src/features/nn/persist.zig`
- Modify: `src/features/nn/mod.zig` (add `pub const saveModel = persist_mod.saveModel;` and `pub const loadModel = persist_mod.loadModel;` exports)
- Test: inline in `src/features/nn/persist.zig`

**Interfaces:**
- Consumes: `Model` (`src/features/nn/model.zig`, fields: `allocator`, `vocab: [256]i16`, `id_to_byte: []u8`, `vocab_size: usize`, `seq_len: usize`, `embed_dim: usize`, `hidden: usize`, `activation: Activation`, `embed/w1/w2: Matrix`, `b1/b2: []f32`, `corpus: []u8`, `report: TrainReport`), `Matrix.alloc(allocator, rows, cols) !Matrix` and `Matrix.free(self, allocator) void` (`model.zig`), `train_mod.trainModel` (test only).
- Produces: `pub fn saveModel(allocator: std.mem.Allocator, model: *const Model, path: []const u8) !void` and `pub fn loadModel(allocator: std.mem.Allocator, path: []const u8) !Model` — later tasks call these by these exact names/signatures.

- [ ] **Step 1: Write the failing test**

Create `src/features/nn/persist.zig` with just the module skeleton and this test:

```zig
const std = @import("std");
const model_mod = @import("model.zig");
const types = @import("types.zig");
const train_mod = @import("train.zig");

const Model = model_mod.Model;
const Matrix = model_mod.Matrix;
const Activation = types.Activation;

pub const MAGIC = "ABINN001";

test {
    std.testing.refAllDecls(@This());
}

test "saveModel/loadModel round-trips a trained model exactly" {
    const a = std.testing.allocator;
    var model = try train_mod.trainModel(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 8,
        .embed_dim = 4,
        .epochs = 10,
        .seed = 1,
    });
    defer model.deinit();

    const path = ".zig-cache/nn-persist-test.bin";
    try saveModel(a, &model, path);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    var loaded = try loadModel(a, path);
    defer loaded.deinit();

    try std.testing.expectEqual(model.vocab_size, loaded.vocab_size);
    try std.testing.expectEqual(model.seq_len, loaded.seq_len);
    try std.testing.expectEqual(model.embed_dim, loaded.embed_dim);
    try std.testing.expectEqual(model.hidden, loaded.hidden);
    try std.testing.expectEqualStrings(model.corpus, loaded.corpus);
    try std.testing.expectEqualSlices(f32, model.embed.data, loaded.embed.data);
    try std.testing.expectEqualSlices(f32, model.w1.data, loaded.w1.data);
    try std.testing.expectEqualSlices(f32, model.b1, loaded.b1);
    try std.testing.expectEqualSlices(f32, model.w2.data, loaded.w2.data);
    try std.testing.expectEqualSlices(f32, model.b2, loaded.b2);
}

test "loadModel rejects a corrupted checksum" {
    const a = std.testing.allocator;
    var model = try train_mod.trainModel(a, "hello world ", .{ .seq_len = 2, .hidden = 4, .embed_dim = 4, .epochs = 1, .seed = 1 });
    defer model.deinit();

    const path = ".zig-cache/nn-persist-corrupt-test.bin";
    try saveModel(a, &model, path);
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    const file = try std.Io.Dir.cwd().openFile(std.Options.debug_io, path, .{ .mode = .read_write });
    defer file.close(std.Options.debug_io);
    try file.writePositionalAll(std.Options.debug_io, "\x00", 8);

    try std.testing.expectError(error.ChecksumMismatch, loadModel(a, path));
}

test "loadModel rejects a bad magic header" {
    const a = std.testing.allocator;
    const path = ".zig-cache/nn-persist-badmagic-test.bin";
    try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{ .sub_path = path, .data = "NOTVALID" });
    defer std.Io.Dir.cwd().deleteFile(std.Options.debug_io, path) catch {};

    try std.testing.expectError(error.InvalidMagic, loadModel(a, path));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `zig build test -Dtest-filter="saveModel"`
Expected: FAIL with "saveModel not defined" (or equivalent compile error) — `saveModel`/`loadModel` don't exist yet.

- [ ] **Step 3: Write minimal implementation**

Add above the tests in `src/features/nn/persist.zig` (after the existing `const`/`MAGIC` lines):

```zig
pub const PersistError = error{
    InvalidMagic,
    TruncatedCheckpoint,
    ChecksumMismatch,
};

fn appendU32(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

fn readU32At(data: []const u8, off: usize) !u32 {
    if (off + 4 > data.len) return error.TruncatedCheckpoint;
    return std.mem.readInt(u32, data[off..][0..4], .little);
}

pub fn saveModel(allocator: std.mem.Allocator, model: *const Model, path: []const u8) !void {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(allocator);

    try out.appendSlice(allocator, MAGIC);
    try appendU32(&out, allocator, @intCast(model.vocab_size));
    try appendU32(&out, allocator, @intCast(model.seq_len));
    try appendU32(&out, allocator, @intCast(model.embed_dim));
    try appendU32(&out, allocator, @intCast(model.hidden));
    try out.append(allocator, if (model.activation == .tanh) 0 else 1);
    try appendU32(&out, allocator, @intCast(model.corpus.len));
    try out.appendSlice(allocator, model.corpus);
    try out.appendSlice(allocator, std.mem.sliceAsBytes(&model.vocab));
    try out.appendSlice(allocator, model.id_to_byte);
    try out.appendSlice(allocator, std.mem.sliceAsBytes(model.embed.data));
    try out.appendSlice(allocator, std.mem.sliceAsBytes(model.w1.data));
    try out.appendSlice(allocator, std.mem.sliceAsBytes(model.b1));
    try out.appendSlice(allocator, std.mem.sliceAsBytes(model.w2.data));
    try out.appendSlice(allocator, std.mem.sliceAsBytes(model.b2));

    const crc = std.hash.Crc32.hash(out.items);
    try appendU32(&out, allocator, crc);

    try std.Io.Dir.cwd().writeFile(std.Options.debug_io, .{ .sub_path = path, .data = out.items });
}

pub fn loadModel(allocator: std.mem.Allocator, path: []const u8) !Model {
    const data = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(data);

    if (data.len < 8 + 4 * 4 + 1 + 4 + 4) return error.TruncatedCheckpoint;
    if (!std.mem.eql(u8, data[0..8], MAGIC)) return error.InvalidMagic;

    const stored_crc = try readU32At(data, data.len - 4);
    const computed_crc = std.hash.Crc32.hash(data[0 .. data.len - 4]);
    if (stored_crc != computed_crc) return error.ChecksumMismatch;

    var off: usize = 8;
    const vocab_size: usize = try readU32At(data, off);
    off += 4;
    const seq_len: usize = try readU32At(data, off);
    off += 4;
    const embed_dim: usize = try readU32At(data, off);
    off += 4;
    const hidden: usize = try readU32At(data, off);
    off += 4;
    if (off >= data.len) return error.TruncatedCheckpoint;
    const activation: Activation = if (data[off] == 0) .tanh else .relu;
    off += 1;
    const corpus_len: usize = try readU32At(data, off);
    off += 4;

    if (off + corpus_len > data.len) return error.TruncatedCheckpoint;
    const corpus = try allocator.dupe(u8, data[off .. off + corpus_len]);
    errdefer allocator.free(corpus);
    off += corpus_len;

    var vocab: [256]i16 = undefined;
    const vocab_bytes = std.mem.sliceAsBytes(&vocab);
    if (off + vocab_bytes.len > data.len) return error.TruncatedCheckpoint;
    @memcpy(vocab_bytes, data[off .. off + vocab_bytes.len]);
    off += vocab_bytes.len;

    if (off + vocab_size > data.len) return error.TruncatedCheckpoint;
    const id_to_byte = try allocator.dupe(u8, data[off .. off + vocab_size]);
    errdefer allocator.free(id_to_byte);
    off += vocab_size;

    var embed = try Matrix.alloc(allocator, vocab_size, embed_dim);
    errdefer embed.free(allocator);
    {
        const b = std.mem.sliceAsBytes(embed.data);
        if (off + b.len > data.len) return error.TruncatedCheckpoint;
        @memcpy(b, data[off .. off + b.len]);
        off += b.len;
    }

    var w1 = try Matrix.alloc(allocator, hidden, seq_len * embed_dim);
    errdefer w1.free(allocator);
    {
        const b = std.mem.sliceAsBytes(w1.data);
        if (off + b.len > data.len) return error.TruncatedCheckpoint;
        @memcpy(b, data[off .. off + b.len]);
        off += b.len;
    }

    const b1 = try allocator.alloc(f32, hidden);
    errdefer allocator.free(b1);
    {
        const b = std.mem.sliceAsBytes(b1);
        if (off + b.len > data.len) return error.TruncatedCheckpoint;
        @memcpy(b, data[off .. off + b.len]);
        off += b.len;
    }

    var w2 = try Matrix.alloc(allocator, vocab_size, hidden);
    errdefer w2.free(allocator);
    {
        const b = std.mem.sliceAsBytes(w2.data);
        if (off + b.len > data.len) return error.TruncatedCheckpoint;
        @memcpy(b, data[off .. off + b.len]);
        off += b.len;
    }

    const b2 = try allocator.alloc(f32, vocab_size);
    errdefer allocator.free(b2);
    {
        const b = std.mem.sliceAsBytes(b2);
        if (off + b.len > data.len) return error.TruncatedCheckpoint;
        @memcpy(b, data[off .. off + b.len]);
    }

    return Model{
        .allocator = allocator,
        .vocab = vocab,
        .id_to_byte = id_to_byte,
        .vocab_size = vocab_size,
        .seq_len = seq_len,
        .embed_dim = embed_dim,
        .hidden = hidden,
        .activation = activation,
        .embed = embed,
        .w1 = w1,
        .b1 = b1,
        .w2 = w2,
        .b2 = b2,
        .corpus = corpus,
        .report = .{ .initial_loss = 0, .final_loss = 0, .steps = 0, .improved = true },
    };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `zig build test -Dtest-filter="saveModel/loadModel round-trips"` then `-Dtest-filter="rejects a corrupted checksum"` then `-Dtest-filter="rejects a bad magic header"`.
Expected: all three PASS. If a `std.Io` call signature mismatches (e.g. `openFile`/`writePositionalAll`/`deleteFile`/`writeFile`/`readFileAlloc` argument order), fix by matching the exact call shape used in `src/foundation/credentials.zig` (search that file for the same function names) — that file is this repo's current source of truth for `std.Io` usage on this Zig pin.

- [ ] **Step 5: Wire exports and verify parity**

Edit `src/features/nn/mod.zig` — add after the existing `pub const extractCorpusFromJsonl = train_mod.extractCorpusFromJsonl;` line:

```zig
const persist_mod = @import("persist.zig");
pub const saveModel = persist_mod.saveModel;
pub const loadModel = persist_mod.loadModel;
pub const PersistError = persist_mod.PersistError;
```

Run: `zig build check-parity`
Expected: exit 0. If `src/features/nn/stub.zig` fails to compile because it lacks these three new symbols, add matching stub entries there — check `stub.zig`'s existing pattern for how it stubs `trainModel`/`sample` (likely returning `error.FeatureDisabled` or similar) and mirror that for `saveModel`/`loadModel`.

- [ ] **Step 6: Commit**

```bash
git add src/features/nn/persist.zig src/features/nn/mod.zig src/features/nn/stub.zig
git commit -m "feat(nn): add binary save/load for trained Model checkpoints"
```

---

### Task 2: Streaming sample generation (`src/features/nn/mod.zig`)

**Files:**
- Modify: `src/features/nn/mod.zig` (add `StreamCallback` type and `sampleStreaming` function, alongside the existing `sample` function)
- Test: inline in `src/features/nn/mod.zig`

**Interfaces:**
- Consumes: `Model`, `Scratch.alloc` (`model.zig`), `model_mod.forwardLossNoTarget` (`model.zig`) — same primitives the existing `sample()` function already uses; read `sample()`'s current body in `mod.zig` before writing this task, since `sampleStreaming` is structurally the same loop with chunked-callback emission added.
- Produces: `pub const StreamCallback = *const fn (ctx: *anyopaque, chunk: []const u8, done: bool) anyerror!void;` and `pub fn sampleStreaming(allocator: std.mem.Allocator, model: *const Model, seed_char: u8, max_len: usize, chunk_size: usize, on_chunk: ?StreamCallback, callback_ctx: ?*anyopaque) ![]u8`. Task 3's CLI handler calls this by this exact name/signature.

- [ ] **Step 1: Write the failing test**

Add to `src/features/nn/mod.zig`, after the existing `test "greedy sampling reproduces the learned repeating pattern"` test block:

```zig
test "sampleStreaming emits chunk_size-batched callbacks and matches non-streaming output" {
    const a = std.testing.allocator;
    var model = try trainModel(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 16,
        .embed_dim = 8,
        .epochs = 400,
        .lr = 0.5,
        .seed = 7,
    });
    defer model.deinit();

    const Ctx = struct {
        buf: std.ArrayListUnmanaged(u8) = .empty,
        chunk_count: usize = 0,
        saw_done: bool = false,
        allocator: std.mem.Allocator,
        fn cb(ctx: *anyopaque, chunk: []const u8, done: bool) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            if (done) {
                self.saw_done = true;
                return;
            }
            try self.buf.appendSlice(self.allocator, chunk);
            self.chunk_count += 1;
        }
    };
    var ctx = Ctx{ .allocator = a };
    defer ctx.buf.deinit(a);

    const streamed = try sampleStreaming(a, &model, 'h', 24, 4, Ctx.cb, &ctx);
    defer a.free(streamed);

    try std.testing.expect(ctx.saw_done);
    try std.testing.expect(ctx.chunk_count >= 6); // 24 chars / 4-char chunks = 6 callbacks
    try std.testing.expectEqualStrings(streamed, ctx.buf.items);

    const non_streamed = try sample(a, &model, 'h', 24);
    defer a.free(non_streamed);
    try std.testing.expectEqualStrings(non_streamed, streamed);
}

test "sampleStreaming with no callback still returns full output" {
    const a = std.testing.allocator;
    var model = try trainModel(a, "hello world ", .{ .seq_len = 2, .hidden = 8, .embed_dim = 4, .epochs = 10, .seed = 1 });
    defer model.deinit();

    const out = try sampleStreaming(a, &model, 'h', 12, 4, null, null);
    defer a.free(out);
    try std.testing.expectEqual(@as(usize, 12), out.len);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `zig build test -Dtest-filter="sampleStreaming"`
Expected: FAIL — `sampleStreaming` not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `src/features/nn/mod.zig`, directly after the existing `sample()` function (reuses the same forward-pass/argmax loop structure, adding chunk buffering and callback invocation):

```zig
pub const StreamCallback = *const fn (ctx: *anyopaque, chunk: []const u8, done: bool) anyerror!void;

pub fn sampleStreaming(
    allocator: std.mem.Allocator,
    model: *const Model,
    seed_char: u8,
    max_len: usize,
    chunk_size: usize,
    on_chunk: ?StreamCallback,
    callback_ctx: ?*anyopaque,
) ![]u8 {
    const out = try allocator.alloc(u8, max_len);
    errdefer allocator.free(out);
    if (max_len == 0) {
        if (on_chunk) |cb| {
            const ctx = callback_ctx orelse return error.MissingStreamContext;
            try cb(ctx, "", true);
        }
        return out;
    }

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var sc = try Scratch.alloc(arena.allocator(), model);

    const start = std.mem.indexOfScalar(u8, model.corpus, seed_char) orelse 0;
    _ = model.sampleAt(start, sc.ctx);

    var chunk_start: usize = 0;
    for (out, 0..) |*slot, i| {
        model_mod.forwardLossNoTarget(model, &sc);
        var best: usize = 0;
        var best_p: f32 = sc.probs[0];
        for (sc.probs[1..], 1..) |pv, idx| {
            if (pv > best_p) {
                best_p = pv;
                best = idx;
            }
        }
        slot.* = model.id_to_byte[best];
        if (model.seq_len > 1) {
            std.mem.copyForwards(usize, sc.ctx[0 .. model.seq_len - 1], sc.ctx[1..model.seq_len]);
        }
        sc.ctx[model.seq_len - 1] = best;

        const is_last = (i == out.len - 1);
        const chunk_full = (i - chunk_start + 1) == chunk_size;
        if (chunk_full or is_last) {
            if (on_chunk) |cb| {
                const ctx = callback_ctx orelse return error.MissingStreamContext;
                try cb(ctx, out[chunk_start .. i + 1], false);
            }
            chunk_start = i + 1;
        }
    }

    if (on_chunk) |cb| {
        const ctx = callback_ctx orelse return error.MissingStreamContext;
        try cb(ctx, "", true);
    }

    return out;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `zig build test -Dtest-filter="sampleStreaming"`
Expected: both new tests PASS.

- [ ] **Step 5: Run the full nn test suite to confirm no regression**

Run: `zig build test -Dtest-filter="nn"` (or `zig build test-nn` if such a step exists — check `build.zig` step names first)
Expected: all existing `nn` tests still pass alongside the two new ones.

- [ ] **Step 6: Commit**

```bash
git add src/features/nn/mod.zig
git commit -m "feat(nn): add sampleStreaming for chunked callback-based generation"
```

---

### Task 3: CLI wiring (`--neural` flag, handler, mutual exclusivity, help text)

**Files:**
- Modify: `src/cli/wiring.zig:20-29` (add flag to `complete_args`), `src/cli/wiring.zig:142-161` (`completeHandler`, add `.neural = parsed.flag("neural"),`)
- Modify: `src/cli/handlers/train.zig:24-33` (`CompleteOptions`, add `neural: bool = false,`), `src/cli/handlers/train.zig:79` area (`handleComplete`, add mutual-exclusivity check + neural branch)
- Modify: `src/cli/handlers/complete_handlers.zig` (add new `handleNeuralComplete` function, alongside the existing `handleFmComplete`/`handleLiveComplete`/`handleLocalBridgeComplete`/`handleSoulComplete`/`handleLearnComplete`)
- Modify: `src/cli/usage.zig:65-76` (the `complete` command's `.usage`/`.details`/`.examples`)
- Test: `tests/contracts/surface.zig` (check for existing string assertions on the `complete` usage table that need updating — read this file's relevant section before editing `usage.zig`)

**Interfaces:**
- Consumes: `nn.sampleStreaming`, `nn.loadModel`, `nn.StreamCallback` (Task 1 & 2), `usage_mod.usageError` (existing, used elsewhere in `train.zig` for the `--soul-alpha` validation error — same pattern to follow for the mutual-exclusivity error).
- Produces: working `abi complete --neural "<input>"` CLI path.

- [ ] **Step 1: Add the flag to the arg spec**

Edit `src/cli/wiring.zig`, in `complete_args` (around line 20-29), add this entry (position doesn't matter functionally, but group it near `model` for readability):

```zig
    .{ .name = "neural", .kind = .flag, .help = "use the in-process character-level neural demo model (small, not comparable to template/live quality); cannot combine with --model" },
```

- [ ] **Step 2: Wire the parsed flag into CompleteOptions assembly**

Edit `src/cli/wiring.zig`, in `completeHandler` (around line 142-161), add `.neural = parsed.flag("neural"),` to the struct literal passed to `handlers.handleComplete`, alongside the existing `.soul = parsed.value("soul"),` line.

- [ ] **Step 3: Add the field to CompleteOptions**

Edit `src/cli/handlers/train.zig`, in the `CompleteOptions` struct (lines 24-33), add:

```zig
    neural: bool = false,
```

- [ ] **Step 4: Write the failing test for mutual exclusivity**

This CLI-level behavior is best tested via the existing CLI contract test pattern — check `tests/contracts/surface.zig` or `tests/contracts/mcp_tools.zig`-adjacent CLI test files for how existing usage errors (e.g. the `--soul-alpha` range check) are tested, and add an equivalent test in the same style/location asserting that `abi complete --neural --model foo "x"` produces a `usageError` mentioning both flags. If no existing CLI-arg-error test pattern is found in the contract tests, add the test inline in `src/cli/handlers/train.zig`'s test block instead, calling `handleComplete` directly with `.{ .input = "x", .model = "foo", .neural = true }` and asserting it returns an error (not a success exit code).

Run the new test first to confirm it fails (handleComplete doesn't yet check this combination).

- [ ] **Step 5: Implement the mutual-exclusivity check and neural branch**

Edit `src/cli/handlers/train.zig`'s `handleComplete` function — add this check as the very first statement in the function body, before the existing `const selected_model = ...` line:

```zig
    if (opts.neural and opts.model != null) {
        return usage_mod.usageError("--neural cannot be combined with --model");
    }
```

Then add the neural branch immediately after that check (before the existing `const selected_model = ...` line, since the neural path doesn't use `selected_model`/`models.canonical` at all):

```zig
    if (opts.neural) {
        return complete.handleNeuralComplete(allocator, input, opts.stream);
    }
```

(`input` here is `opts.input`, matching the existing `const input = opts.input;` line immediately below — reorder so `input` is bound before this branch, or reference `opts.input` directly in the branch if `input` isn't yet in scope at that point in the function.)

- [ ] **Step 6: Implement handleNeuralComplete**

Add to `src/cli/handlers/complete_handlers.zig`, alongside the other `handleXComplete` functions:

```zig
const nn = @import("abi").features.nn;

const NEURAL_MAX_LEN: usize = 300;
const NEURAL_CHUNK_SIZE: usize = 6;
const NEURAL_CHECKPOINT_PATH = "assets/nn/persona-checkpoint.bin";
const NEURAL_DISCLOSURE =
    "in-process character-level demo model, trained on ABI's own docs — " ++
    "not a production LLM, not comparable in quality to the template or live completion paths.";

pub fn handleNeuralComplete(allocator: std.mem.Allocator, input: []const u8, stream: bool) !u8 {
    var model = nn.loadModel(allocator, NEURAL_CHECKPOINT_PATH) catch |err| {
        std.debug.print("error: neural checkpoint unavailable at {s}: {s}\n", .{ NEURAL_CHECKPOINT_PATH, @errorName(err) });
        return error.NeuralCheckpointMissing;
    };
    defer model.deinit();

    std.debug.print("model=neural-char-lm note=\"{s}\"\n", .{NEURAL_DISCLOSURE});

    const seed_char: u8 = if (input.len > 0) input[0] else ' ';

    if (stream) {
        const StreamCtx = struct {
            fn callback(_: *anyopaque, chunk: []const u8, done: bool) anyerror!void {
                if (!done) std.debug.print("{s}", .{chunk});
            }
        };
        var dummy: u8 = 0;
        const out = try nn.sampleStreaming(allocator, &model, seed_char, NEURAL_MAX_LEN, NEURAL_CHUNK_SIZE, StreamCtx.callback, &dummy);
        defer allocator.free(out);
        std.debug.print("\n", .{});
        return 0;
    }

    const out = try nn.sampleStreaming(allocator, &model, seed_char, NEURAL_MAX_LEN, NEURAL_CHUNK_SIZE, null, null);
    defer allocator.free(out);
    std.debug.print("{s}\n", .{out});
    return 0;
}
```

Check the top of `complete_handlers.zig` for the exact `const complete = ` or equivalent import alias used by `train.zig`'s `complete.handleFmComplete(...)` calls — `handleNeuralComplete` needs to be reachable via that same path from `train.zig`'s new `complete.handleNeuralComplete(...)` call in Step 5.

- [ ] **Step 7: Run the mutual-exclusivity test to verify it passes**

Run the test written in Step 4.
Expected: PASS.

- [ ] **Step 8: Update CLI help text**

Edit `src/cli/usage.zig`, the `complete` entry (lines 65-76):

```zig
.{
    .name = "complete",
    .usage = "abi complete [--live] [--confirm] [--learn] [--stream] [--neural] [--model <id>] <input>",
    .summary = "Run completion through local ABI agent pipeline; --model selects catalog id, --live enables explicit live transport, --neural uses the small in-process demo model",
    .category = .ai,
    .details = "--model selects a catalog id, --live enables explicit live transport, --learn runs the SEA self-learning loop, apple-fm requires --confirm, and --neural routes to a small in-process character-level demo model (cannot combine with --model). Use `--` before a prompt that starts with `-`.",
    .examples = &.{
        "abi complete \"summarize repository status\"",
        "abi complete -- --literal-leading-dash",
        "abi complete --learn --model claude-fable-5 \"plan next repair\"",
        "abi complete --neural \"hello\"",
    },
},
```

Check `tests/contracts/surface.zig` for any string-equality assertions against this table's old `.usage`/`.details`/`.examples` values (the file's top comment flagged this table as "frozen" and contract-tested) — update any matching assertions to the new text.

- [ ] **Step 9: Run the full CLI build and check-parity**

Run: `zig build check-parity` then `./build.sh test-cli` (or `./build.sh test-contracts` if that's where `surface.zig` lives — check `AGENTS.md`'s command table).
Expected: both green.

- [ ] **Step 10: Commit**

```bash
git add src/cli/wiring.zig src/cli/handlers/train.zig src/cli/handlers/complete_handlers.zig src/cli/usage.zig tests/contracts/surface.zig
git commit -m "feat(cli): add --neural flag to abi complete"
```

---

### Task 4: Checkpoint generation and bundling

**Files:**
- Create: `tools/generate_nn_checkpoint.zig` (a one-off Zig executable, following the existing convention of `tools/*.zig` build-time helper scripts referenced in `AGENTS.md`, e.g. `gen_plugin_registry`)
- Modify: `build.zig` (add a build step to compile/run the generator, matching the pattern used for `gen_plugin_registry` — read that step's `build.zig` definition first and mirror it)
- Create (generated, then committed): `assets/nn/persona-checkpoint.bin`
- Test: CLI smoke test for `abi complete --neural`

**Interfaces:**
- Consumes: `nn.trainModel`, `nn.saveModel` (Task 1), corpus text assembled from `AGENTS.md` + `README.md`.
- Produces: the committed checkpoint binary that Task 3's `handleNeuralComplete` loads at `assets/nn/persona-checkpoint.bin`.

- [ ] **Step 1: Write the generator tool**

Create `tools/generate_nn_checkpoint.zig`:

```zig
//! One-off dev tool: trains the nn char-LM on ABI's own docs and writes the
//! bundled checkpoint consumed by `abi complete --neural`. Not part of
//! `./build.sh check` — run manually when the corpus or model shape changes.

const std = @import("std");
const nn = @import("abi").features.nn;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const agents_md = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, "AGENTS.md", allocator, .limited(1024 * 1024));
    defer allocator.free(agents_md);
    const readme_md = try std.Io.Dir.cwd().readFileAlloc(std.Options.debug_io, "README.md", allocator, .limited(1024 * 1024));
    defer allocator.free(readme_md);

    const corpus = try std.mem.concat(allocator, u8, &.{ agents_md, "\n\n", readme_md });
    defer allocator.free(corpus);

    var model = try nn.trainModel(allocator, corpus, .{
        .seq_len = 4,
        .hidden = 64,
        .embed_dim = 16,
        .epochs = 500,
        .lr = 0.1,
        .seed = 0x5EED,
    });
    defer model.deinit();

    std.debug.print("trained: initial_loss={d:.4} final_loss={d:.4} improved={}\n", .{
        model.report.initial_loss,
        model.report.final_loss,
        model.report.improved,
    });

    try std.Io.Dir.cwd().makePath(std.Options.debug_io, "assets/nn");
    try nn.saveModel(allocator, &model, "assets/nn/persona-checkpoint.bin");
    std.debug.print("wrote assets/nn/persona-checkpoint.bin\n", .{});
}
```

(If `std.Io.Dir.makePath` isn't the exact Zig 0.17 API name, check `src/foundation/credentials.zig`'s `ensureCredentialsDir` for the correct directory-creation call and use that instead.)

- [ ] **Step 2: Wire a build step to compile the tool**

Edit `build.zig` — find the existing `gen_plugin_registry` executable step definition (referenced throughout this session as `run exe gen_plugin_registry` in build output) and add an analogous step for `generate_nn_checkpoint`, e.g.:

```zig
const gen_nn_checkpoint_exe = b.addExecutable(.{
    .name = "generate_nn_checkpoint",
    .root_module = b.createModule(.{
        .root_source_file = b.path("tools/generate_nn_checkpoint.zig"),
        .target = target,
        .optimize = optimize,
    }),
});
gen_nn_checkpoint_exe.root_module.addImport("abi", abi_module);
const run_gen_nn_checkpoint = b.addRunArtifact(gen_nn_checkpoint_exe);
const gen_nn_checkpoint_step = b.step("gen-nn-checkpoint", "Train and bundle the --neural demo model checkpoint (one-off, not part of check)");
gen_nn_checkpoint_step.dependOn(&run_gen_nn_checkpoint.step);
```

Match this to the exact variable names (`target`, `optimize`, `abi_module`) already in scope at the point in `build.zig` where `gen_plugin_registry`'s equivalent block lives — copy that block's structure precisely rather than inventing new variable names.

- [ ] **Step 3: Run the generator to produce the checkpoint**

Run: `zig build gen-nn-checkpoint`
Expected: prints `trained: ... improved=true` and `wrote assets/nn/persona-checkpoint.bin`. If `improved=false`, the training config needs adjustment (more epochs or a different learning rate) — do not bundle a checkpoint whose training didn't converge; that would silently ship a broken/undertrained model.

- [ ] **Step 4: Verify the checkpoint loads correctly**

Run: `zig build test -Dtest-filter="saveModel/loadModel"` (confirms the round-trip logic from Task 1 still holds), then manually smoke-test:

```bash
./build.sh cli
./zig-out/bin/abi complete --neural "hello"
```

Expected: exit 0, prints the `model=neural-char-lm note="..."` disclosure line followed by generated (likely low-coherence, since this is a small character-level model) text, not an error.

- [ ] **Step 5: Write and run the CLI smoke test**

Add a test near the existing CLI contract tests (check `tests/contracts/surface.zig` or wherever `abi complete`'s other flag combinations are smoke-tested) asserting `abi complete --neural "test"` exits 0 and produces non-empty stdout. If the contract-test harness shells out to the built binary, follow that exact pattern; if it calls `handleComplete` directly in-process, call it with `.{ .input = "test", .neural = true }` and assert the return is `0`.

Run: the new test.
Expected: PASS.

- [ ] **Step 6: Commit the tool, build step, and generated checkpoint**

```bash
git add tools/generate_nn_checkpoint.zig build.zig assets/nn/persona-checkpoint.bin tests/
git commit -m "feat(nn): add checkpoint generator tool and bundle trained --neural checkpoint"
```

- [ ] **Step 7: Full validation gate**

Run: `./build.sh check`
Expected: green — this confirms the new `assets/nn/persona-checkpoint.bin` binary asset doesn't break any existing build step (e.g. file-size-sensitive checks, if any exist) and the full test suite including all of Tasks 1-4's new tests passes together.

---

## Self-Review Notes (for the implementer, not a task)

- **Spec coverage:** Task 1 covers persistence, Task 2 covers streaming, Task 3 covers CLI wiring + honesty labeling + mutual exclusivity + help text, Task 4 covers the offline training/bundling step and the CLI smoke test. All sections of the design spec (`docs/superpowers/specs/2026-07-17-in-process-neural-voice-design.md`) are addressed.
- **Type consistency check:** `nn.StreamCallback`'s signature (`*const fn (ctx: *anyopaque, chunk: []const u8, done: bool) anyerror!void`) is used identically in Task 2's implementation and Task 3's `handleNeuralComplete` — verify these stay in sync if either is edited independently during implementation.
- **Open item carried from the spec:** checkpoint format versioning (in case `Model`'s shape changes later) was flagged as an open question in the spec and deliberately deferred — the `MAGIC` string (`"ABINN001"`) has a trailing version-like suffix precisely so a future format change can bump it and `loadModel` can reject old-format files cleanly, but no migration logic is implemented in this plan. Acceptable for this slice; note it if extending later.
