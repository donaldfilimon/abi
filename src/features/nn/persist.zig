//! Checkpoint serialize/deserialize for the char-LM `Model`.
//! Format is versioned (`ABINN001`) so stale files fail loudly.

const std = @import("std");
const types = @import("types.zig");
const model_mod = @import("model.zig");

const Model = model_mod.Model;
const Matrix = model_mod.Matrix;
const Activation = types.Activation;

pub const MAGIC = "ABINN001";
pub const MAX_OUTPUT_CHARS: usize = 300;
pub const DEFAULT_CHUNK: usize = 8;

pub const PersistError = error{
    NeuralCheckpointMissing,
    NeuralCheckpointCorrupt,
    NeuralCheckpointVersion,
    OutOfMemory,
};

/// Default relative path for the bundled persona checkpoint.
pub const DEFAULT_CHECKPOINT_PATH = "assets/nn/persona-checkpoint.bin";

/// Small honest corpus baked into the binary for offline training of the demo checkpoint.
pub const BUNDLED_CORPUS =
    \\abbey is the primary empathetic polymath profile for abi.
    \\aviva is the direct expert mode. abi orchestrates local completion.
    \\wdbx stores vectors blocks and wal segments on a single host.
    \\complete train agent backends plugin auth twilio tui dashboard scheduler nn.
    \\claim honest demos only never fake ane sharding or audited fhe.
    \\hello world abbey aviva abi wdbx complete neural char lm demo.
;

fn appendU32(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

fn appendI16(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: i16) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(i16, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

fn appendF32Slice(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, data: []const f32) !void {
    for (data) |v| {
        var buf: [4]u8 = undefined;
        const bits: u32 = @bitCast(v);
        std.mem.writeInt(u32, &buf, bits, .little);
        try list.appendSlice(allocator, &buf);
    }
}

const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Cursor, n: usize) PersistError![]const u8 {
        if (self.pos + n > self.bytes.len) return error.NeuralCheckpointCorrupt;
        const slice = self.bytes[self.pos .. self.pos + n];
        self.pos += n;
        return slice;
    }

    fn u32_(self: *Cursor) PersistError!u32 {
        const s = try self.take(4);
        return std.mem.readInt(u32, s[0..4], .little);
    }

    fn i16_(self: *Cursor) PersistError!i16 {
        const s = try self.take(2);
        return std.mem.readInt(i16, s[0..2], .little);
    }

    fn f32s(self: *Cursor, out: []f32) PersistError!void {
        for (out) |*slot| {
            const s = try self.take(4);
            const bits = std.mem.readInt(u32, s[0..4], .little);
            slot.* = @bitCast(bits);
        }
    }
};

/// Serialize a trained model into a newly allocated buffer.
pub fn saveModelAlloc(allocator: std.mem.Allocator, model: *const Model) ![]u8 {
    var list: std.ArrayListUnmanaged(u8) = .empty;
    errdefer list.deinit(allocator);
    try list.appendSlice(allocator, MAGIC);
    try appendU32(&list, allocator, @intCast(model.seq_len));
    try appendU32(&list, allocator, @intCast(model.embed_dim));
    try appendU32(&list, allocator, @intCast(model.hidden));
    try appendU32(&list, allocator, @intCast(model.vocab_size));
    try list.append(allocator, switch (model.activation) {
        .tanh => 0,
        .relu => 1,
    });
    for (model.vocab) |v| try appendI16(&list, allocator, v);
    try list.appendSlice(allocator, model.id_to_byte);
    try appendF32Slice(&list, allocator, model.embed.data);
    try appendF32Slice(&list, allocator, model.w1.data);
    try appendF32Slice(&list, allocator, model.b1);
    try appendF32Slice(&list, allocator, model.w2.data);
    try appendF32Slice(&list, allocator, model.b2);
    try appendU32(&list, allocator, @intCast(model.corpus.len));
    try list.appendSlice(allocator, model.corpus);
    try appendF32Slice(&list, allocator, &.{ model.report.initial_loss, model.report.final_loss });
    try appendU32(&list, allocator, @intCast(model.report.steps));
    try list.append(allocator, if (model.report.improved) 1 else 0);
    return try list.toOwnedSlice(allocator);
}

/// Deserialize a model from bytes. Caller owns the result (`deinit`).
pub fn loadModelBytes(allocator: std.mem.Allocator, bytes: []const u8) PersistError!Model {
    var c: Cursor = .{ .bytes = bytes };
    const magic = try c.take(MAGIC.len);
    if (!std.mem.eql(u8, magic, MAGIC)) return error.NeuralCheckpointVersion;

    const seq_len = try c.u32_();
    const embed_dim = try c.u32_();
    const hidden = try c.u32_();
    const vocab_size = try c.u32_();
    if (seq_len == 0 or embed_dim == 0 or hidden == 0 or vocab_size == 0 or vocab_size > 256)
        return error.NeuralCheckpointCorrupt;

    const act_b = try c.take(1);
    const activation: Activation = switch (act_b[0]) {
        0 => .tanh,
        1 => .relu,
        else => return error.NeuralCheckpointCorrupt,
    };

    var vocab: [256]i16 = undefined;
    for (&vocab) |*slot| slot.* = try c.i16_();

    const id_to_byte = allocator.alloc(u8, vocab_size) catch return error.OutOfMemory;
    errdefer allocator.free(id_to_byte);
    @memcpy(id_to_byte, try c.take(vocab_size));

    var embed = Matrix.alloc(allocator, vocab_size, embed_dim) catch return error.OutOfMemory;
    errdefer embed.free(allocator);
    try c.f32s(embed.data);

    const in_dim = seq_len * embed_dim;
    var w1 = Matrix.alloc(allocator, hidden, in_dim) catch return error.OutOfMemory;
    errdefer w1.free(allocator);
    try c.f32s(w1.data);

    const b1 = allocator.alloc(f32, hidden) catch return error.OutOfMemory;
    errdefer allocator.free(b1);
    try c.f32s(b1);

    var w2 = Matrix.alloc(allocator, vocab_size, hidden) catch return error.OutOfMemory;
    errdefer w2.free(allocator);
    try c.f32s(w2.data);

    const b2 = allocator.alloc(f32, vocab_size) catch return error.OutOfMemory;
    errdefer allocator.free(b2);
    try c.f32s(b2);

    const corpus_len = try c.u32_();
    const corpus = allocator.alloc(u8, corpus_len) catch return error.OutOfMemory;
    errdefer allocator.free(corpus);
    if (corpus_len > 0) @memcpy(corpus, try c.take(corpus_len));

    var losses: [2]f32 = undefined;
    try c.f32s(&losses);
    const steps = try c.u32_();
    const improved_b = try c.take(1);

    return .{
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
        .report = .{
            .initial_loss = losses[0],
            .final_loss = losses[1],
            .steps = steps,
            .improved = improved_b[0] != 0,
        },
    };
}

/// Load checkpoint bytes from a cwd-relative path via std.Io.
pub fn loadModelPath(io: std.Io, allocator: std.mem.Allocator, path: []const u8) PersistError!Model {
    var dir = std.Io.Dir.cwd();
    const file = dir.openFile(io, path, .{}) catch return error.NeuralCheckpointMissing;
    defer file.close(io);
    const stat = file.stat(io) catch return error.NeuralCheckpointCorrupt;
    if (stat.size > 32 * 1024 * 1024) return error.NeuralCheckpointCorrupt;
    const buf = allocator.alloc(u8, stat.size) catch return error.OutOfMemory;
    defer allocator.free(buf);
    const n = file.readPositionalAll(io, buf, 0) catch return error.NeuralCheckpointCorrupt;
    return loadModelBytes(allocator, buf[0..n]);
}

/// Best-effort write of a checkpoint; failures are propagated to the caller.
pub fn saveModelPath(io: std.Io, allocator: std.mem.Allocator, model: *const Model, path: []const u8) !void {
    const bytes = try saveModelAlloc(allocator, model);
    defer allocator.free(bytes);
    var dir = std.Io.Dir.cwd();
    try dir.writeFile(io, .{ .sub_path = path, .data = bytes });
}

/// Train the bundled demo corpus into a model (used when generating checkpoints / tests).
pub fn trainBundled(allocator: std.mem.Allocator) !Model {
    const train_mod = @import("train.zig");
    return train_mod.trainModel(allocator, BUNDLED_CORPUS, .{
        .seq_len = 2,
        .hidden = 16,
        .embed_dim = 8,
        .epochs = 120,
        .lr = 0.4,
        .seed = 0x4e335552,
    });
}

/// Greedy sample up to `max_chars`, invoking `on_chunk` every `chunk` characters.
pub fn sampleStreaming(
    allocator: std.mem.Allocator,
    model: *const Model,
    seed_char: u8,
    max_chars: usize,
    chunk: usize,
    on_chunk: *const fn (*anyopaque, []const u8) anyerror!void,
    ctx: *anyopaque,
) ![]u8 {
    const n = @min(max_chars, MAX_OUTPUT_CHARS);
    const out = try allocator.alloc(u8, n);
    errdefer allocator.free(out);
    if (n == 0) return out;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var sc = try model_mod.Scratch.alloc(arena.allocator(), model);

    const start = std.mem.indexOfScalar(u8, model.corpus, seed_char) orelse 0;
    _ = model.sampleAt(start, sc.ctx);

    var emitted: usize = 0;
    while (emitted < n) {
        const take = @min(if (chunk == 0) DEFAULT_CHUNK else chunk, n - emitted);
        for (0..take) |i| {
            model_mod.forwardLossNoTarget(model, &sc);
            var best: usize = 0;
            var best_p: f32 = sc.probs[0];
            for (sc.probs[1..], 1..) |pv, idx| {
                if (pv > best_p) {
                    best_p = pv;
                    best = idx;
                }
            }
            out[emitted + i] = model.id_to_byte[best];
            if (model.seq_len > 1) {
                std.mem.copyForwards(usize, sc.ctx[0 .. model.seq_len - 1], sc.ctx[1..model.seq_len]);
            }
            sc.ctx[model.seq_len - 1] = best;
        }
        try on_chunk(ctx, out[emitted .. emitted + take]);
        emitted += take;
    }
    return out;
}

test {
    std.testing.refAllDecls(@This());
}

test "checkpoint round-trip preserves sample output" {
    const a = std.testing.allocator;
    const train_mod = @import("train.zig");
    var model = try train_mod.trainModel(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 8,
        .embed_dim = 4,
        .epochs = 50,
        .lr = 0.5,
        .seed = 99,
    });
    defer model.deinit();

    const bytes = try saveModelAlloc(a, &model);
    defer a.free(bytes);

    var loaded = try loadModelBytes(a, bytes);
    defer loaded.deinit();

    try std.testing.expectEqual(model.vocab_size, loaded.vocab_size);
    try std.testing.expectEqual(model.seq_len, loaded.seq_len);
    try std.testing.expectEqualSlices(f32, model.w1.data, loaded.w1.data);

    const nn = @import("mod.zig");
    const a_out = try nn.sample(a, &model, 'h', 16);
    defer a.free(a_out);
    const b_out = try nn.sample(a, &loaded, 'h', 16);
    defer a.free(b_out);
    try std.testing.expectEqualStrings(a_out, b_out);
}

test "sampleStreaming chunks reconstruct full output" {
    const a = std.testing.allocator;
    const train_mod = @import("train.zig");
    var model = try train_mod.trainModel(a, "hello world ", .{
        .seq_len = 2,
        .hidden = 8,
        .embed_dim = 4,
        .epochs = 80,
        .lr = 0.5,
        .seed = 3,
    });
    defer model.deinit();

    const Ctx = struct {
        buf: std.ArrayListUnmanaged(u8) = .empty,
        a: std.mem.Allocator,
        fn cb(ptr: *anyopaque, chunk_bytes: []const u8) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ptr));
            try self.buf.appendSlice(self.a, chunk_bytes);
        }
    };
    var ctx: Ctx = .{ .a = a };
    defer ctx.buf.deinit(a);

    const streamed = try sampleStreaming(a, &model, 'h', 24, 4, Ctx.cb, &ctx);
    defer a.free(streamed);
    try std.testing.expectEqualStrings(streamed, ctx.buf.items);
}
