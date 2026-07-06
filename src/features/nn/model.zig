const std = @import("std");
const types = @import("types.zig");

const Activation = types.Activation;
const TrainReport = types.TrainReport;

pub const Model = struct {
    allocator: std.mem.Allocator,
    vocab: [256]i16,
    id_to_byte: []u8,
    vocab_size: usize,
    seq_len: usize,
    embed_dim: usize,
    hidden: usize,
    activation: Activation,
    embed: Matrix,
    w1: Matrix,
    b1: []f32,
    w2: Matrix,
    b2: []f32,
    corpus: []u8,
    report: TrainReport,

    pub fn inputDim(self: *const Model) usize {
        return self.seq_len * self.embed_dim;
    }

    pub fn deinit(self: *Model) void {
        const a = self.allocator;
        self.embed.free(a);
        self.w1.free(a);
        self.w2.free(a);
        a.free(self.b1);
        a.free(self.b2);
        a.free(self.id_to_byte);
        a.free(self.corpus);
        self.* = undefined;
    }

    pub fn sampleAt(self: *const Model, p: usize, ctx: []usize) usize {
        const n = self.corpus.len;
        for (0..self.seq_len) |s| {
            ctx[s] = @intCast(self.vocab[self.corpus[(p + s) % n]]);
        }
        return @intCast(self.vocab[self.corpus[(p + self.seq_len) % n]]);
    }
};

fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0;
    var i: usize = 0;
    while (i + 4 <= a.len) : (i += 4) {
        const av: @Vector(4, f32) = a[i..][0..4].*;
        const bv: @Vector(4, f32) = b[i..][0..4].*;
        sum += @reduce(.Add, av * bv);
    }
    while (i < a.len) : (i += 1) sum += a[i] * b[i];
    return sum;
}

pub const Matrix = struct {
    data: []f32,
    rows: usize,
    cols: usize,

    pub fn alloc(allocator: std.mem.Allocator, rows: usize, cols: usize) !Matrix {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{ .data = data, .rows = rows, .cols = cols };
    }

    pub fn free(self: Matrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn row(self: Matrix, r: usize) []f32 {
        return self.data[r * self.cols ..][0..self.cols];
    }

    pub fn matVec(self: Matrix, x: []const f32, out: []f32) void {
        std.debug.assert(x.len == self.cols and out.len == self.rows);
        for (0..self.rows) |r| out[r] = dot(self.row(r), x);
    }
};

pub const Scratch = struct {
    ctx: []usize,
    x: []f32,
    h: []f32,
    logits: []f32,
    probs: []f32,
    dlogits: []f32,
    dh: []f32,
    dhpre: []f32,
    dx: []f32,

    pub fn alloc(allocator: std.mem.Allocator, model: *const Model) !Scratch {
        const in_dim = model.inputDim();
        return .{
            .ctx = try allocator.alloc(usize, model.seq_len),
            .x = try allocator.alloc(f32, in_dim),
            .h = try allocator.alloc(f32, model.hidden),
            .logits = try allocator.alloc(f32, model.vocab_size),
            .probs = try allocator.alloc(f32, model.vocab_size),
            .dlogits = try allocator.alloc(f32, model.vocab_size),
            .dh = try allocator.alloc(f32, model.hidden),
            .dhpre = try allocator.alloc(f32, model.hidden),
            .dx = try allocator.alloc(f32, in_dim),
        };
    }
};

pub const Grads = struct {
    embed: Matrix,
    w1: Matrix,
    b1: []f32,
    w2: Matrix,
    b2: []f32,

    pub fn alloc(allocator: std.mem.Allocator, model: *const Model) !Grads {
        return .{
            .embed = try Matrix.alloc(allocator, model.vocab_size, model.embed_dim),
            .w1 = try Matrix.alloc(allocator, model.hidden, model.inputDim()),
            .b1 = blk: {
                const b = try allocator.alloc(f32, model.hidden);
                @memset(b, 0);
                break :blk b;
            },
            .w2 = try Matrix.alloc(allocator, model.vocab_size, model.hidden),
            .b2 = blk: {
                const b = try allocator.alloc(f32, model.vocab_size);
                @memset(b, 0);
                break :blk b;
            },
        };
    }

    pub fn zero(self: *Grads) void {
        @memset(self.embed.data, 0);
        @memset(self.w1.data, 0);
        @memset(self.b1, 0);
        @memset(self.w2.data, 0);
        @memset(self.b2, 0);
    }
};

pub fn activate(act: Activation, v: f32) f32 {
    return switch (act) {
        .tanh => std.math.tanh(v),
        .relu => @max(v, 0),
    };
}

pub fn activateDeriv(act: Activation, post: f32) f32 {
    return switch (act) {
        .tanh => 1 - post * post,
        .relu => if (post > 0) @as(f32, 1) else 0,
    };
}

pub fn softmax(logits: []const f32, out: []f32) void {
    std.debug.assert(logits.len == out.len);
    var max: f32 = logits[0];
    for (logits[1..]) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (logits, out) |v, *o| {
        const e = @exp(v - max);
        o.* = e;
        sum += e;
    }
    const inv = 1.0 / sum;
    for (out) |*o| o.* *= inv;
}

pub fn forwardLoss(model: *const Model, sc: *Scratch, target: usize) f32 {
    for (0..model.seq_len) |s| {
        const src = model.embed.row(sc.ctx[s]);
        @memcpy(sc.x[s * model.embed_dim ..][0..model.embed_dim], src);
    }
    model.w1.matVec(sc.x, sc.h);
    for (sc.h, model.b1) |*hv, bv| hv.* = activate(model.activation, hv.* + bv);
    model.w2.matVec(sc.h, sc.logits);
    for (sc.logits, model.b2) |*lv, bv| lv.* += bv;
    softmax(sc.logits, sc.probs);
    return -@log(@max(sc.probs[target], 1e-9));
}

pub fn backward(model: *const Model, g: *Grads, sc: *Scratch, target: usize) void {
    @memcpy(sc.dlogits, sc.probs);
    sc.dlogits[target] -= 1;

    for (0..model.vocab_size) |r| {
        const dl = sc.dlogits[r];
        g.b2[r] += dl;
        const gw2_row = g.w2.row(r);
        for (gw2_row, sc.h) |*gv, hv| gv.* += dl * hv;
    }

    for (0..model.hidden) |c| {
        var s: f32 = 0;
        for (0..model.vocab_size) |r| s += model.w2.row(r)[c] * sc.dlogits[r];
        sc.dh[c] = s;
    }
    for (sc.dhpre, sc.dh, sc.h) |*dp, dhv, hv| dp.* = dhv * activateDeriv(model.activation, hv);

    for (0..model.hidden) |r| {
        const dp = sc.dhpre[r];
        g.b1[r] += dp;
        const gw1_row = g.w1.row(r);
        for (gw1_row, sc.x) |*gv, xv| gv.* += dp * xv;
    }

    for (0..model.inputDim()) |c| {
        var s: f32 = 0;
        for (0..model.hidden) |r| s += model.w1.row(r)[c] * sc.dhpre[r];
        sc.dx[c] = s;
    }

    for (0..model.seq_len) |s| {
        const grow = g.embed.row(sc.ctx[s]);
        const dx_slice = sc.dx[s * model.embed_dim ..][0..model.embed_dim];
        for (grow, dx_slice) |*gv, dv| gv.* += dv;
    }
}

pub fn evalLoss(model: *const Model, sc: *Scratch) f32 {
    var total: f32 = 0;
    for (0..model.corpus.len) |p| {
        const target = model.sampleAt(p, sc.ctx);
        total += forwardLoss(model, sc, target);
    }
    return total / @as(f32, @floatFromInt(model.corpus.len));
}

pub fn forwardLossNoTarget(model: *const Model, sc: *Scratch) void {
    for (0..model.seq_len) |s| {
        const src = model.embed.row(sc.ctx[s]);
        @memcpy(sc.x[s * model.embed_dim ..][0..model.embed_dim], src);
    }
    model.w1.matVec(sc.x, sc.h);
    for (sc.h, model.b1) |*hv, bv| hv.* = activate(model.activation, hv.* + bv);
    model.w2.matVec(sc.h, sc.logits);
    for (sc.logits, model.b2) |*lv, bv| lv.* += bv;
    softmax(sc.logits, sc.probs);
}

test {
    std.testing.refAllDecls(@This());
}
