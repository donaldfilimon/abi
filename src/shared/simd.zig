const std = @import("std");
const builtin = @import("builtin");

const default_simd_width = switch (builtin.target.cpu.arch) {
    .x86, .x86_64 => 8,
    .wasm32, .wasm64 => 4,
    .aarch64 => 4,
    else => 4,
};

pub const SIMD_WIDTH = default_simd_width;
const FloatVector = @Vector(SIMD_WIDTH, f32);
const TextSimdWidth = if (SIMD_WIDTH >= 16) 16 else 8;
const ByteVector = @Vector(TextSimdWidth, u8);

pub const SIMDOpts = struct {
    /// Force scalar implementation regardless of slice length.
    force_scalar: bool = false,
    /// Minimum slice length that should trigger SIMD processing.
    min_len_for_simd: usize = SIMD_WIDTH,

    pub fn shouldUseSimd(self: SIMDOpts, len: usize) bool {
        return !self.force_scalar and len >= self.min_len_for_simd and SIMD_WIDTH > 1;
    }
};

pub const PerformanceMonitorDetails = struct {
    operation_count: u64,
    simd_usage_count: u64,
    total_time_ns: u64,
    average_time_ns: u64,
    simd_usage_ratio: f64,
    last_duration_ns: u64,
    last_used_simd: bool,
};

pub const PerformanceMonitor = struct {
    operation_count: std.atomic.Value(u64),
    simd_usage_count: std.atomic.Value(u64),
    total_time_ns: std.atomic.Value(u64),
    last_duration_ns: std.atomic.Value(u64),
    last_used_simd: std.atomic.Value(u8),

    pub fn init() PerformanceMonitor {
        return .{
            .operation_count = std.atomic.Value(u64).init(0),
            .simd_usage_count = std.atomic.Value(u64).init(0),
            .total_time_ns = std.atomic.Value(u64).init(0),
            .last_duration_ns = std.atomic.Value(u64).init(0),
            .last_used_simd = std.atomic.Value(u8).init(0),
        };
    }

    pub fn recordOperation(self: *PerformanceMonitor, duration_ns: u64, used_simd: bool) void {
        _ = self.operation_count.fetchAdd(1, .monotonic);
        if (used_simd) {
            _ = self.simd_usage_count.fetchAdd(1, .monotonic);
            self.last_used_simd.store(1, .monotonic);
        } else {
            self.last_used_simd.store(0, .monotonic);
        }
        _ = self.total_time_ns.fetchAdd(duration_ns, .monotonic);
        self.last_duration_ns.store(duration_ns, .monotonic);
    }

    pub fn reset(self: *PerformanceMonitor) void {
        self.operation_count.store(0, .monotonic);
        self.simd_usage_count.store(0, .monotonic);
        self.total_time_ns.store(0, .monotonic);
        self.last_duration_ns.store(0, .monotonic);
        self.last_used_simd.store(0, .monotonic);
    }

    pub fn details(self: *PerformanceMonitor) PerformanceMonitorDetails {
        const ops = self.operation_count.load(.monotonic);
        const simd_ops = self.simd_usage_count.load(.monotonic);
        const total_time = self.total_time_ns.load(.monotonic);
        const last_duration = self.last_duration_ns.load(.monotonic);
        const last_simd = self.last_used_simd.load(.monotonic) != 0;
        const avg_time: u64 = if (ops == 0) 0 else total_time / ops;
        const ratio: f64 = if (ops == 0) 0.0 else @as(f64, @floatFromInt(simd_ops)) / @as(f64, @floatFromInt(ops));

        return .{
            .operation_count = ops,
            .simd_usage_count = simd_ops,
            .total_time_ns = total_time,
            .average_time_ns = avg_time,
            .simd_usage_ratio = ratio,
            .last_duration_ns = last_duration,
            .last_used_simd = last_simd,
        };
    }
};

var global_monitor = PerformanceMonitor.init();

inline fn beginTiming() std.time.Instant {
    return std.time.Instant.now() catch return .{ .timestamp = 0 };
}

inline fn finishTiming(start: std.time.Instant, used_simd: bool) void {
    const end = std.time.Instant.now() catch {
        global_monitor.recordOperation(0, used_simd);
        return;
    };
    global_monitor.recordOperation(end.since(start), used_simd);
}

inline fn loadVector(slice: []const f32) FloatVector {
    std.debug.assert(slice.len >= SIMD_WIDTH);
    std.debug.assert(@alignOf(@TypeOf(slice.ptr)) >= @alignOf(FloatVector) or
        std.mem.isAligned(@intFromPtr(slice.ptr), @alignOf(FloatVector)));
    const ptr = @as(*const [SIMD_WIDTH]f32, @ptrCast(slice.ptr));
    return @as(FloatVector, ptr.*);
}

inline fn storeVector(vec: FloatVector, slice: []f32) void {
    std.debug.assert(slice.len >= SIMD_WIDTH);
    const arr = @as([SIMD_WIDTH]f32, vec);
    @memcpy(slice[0..SIMD_WIDTH], arr[0..]);
}

inline fn loadByteVector(slice: []const u8) ByteVector {
    std.debug.assert(slice.len >= TextSimdWidth);
    const ptr = @as(*ByteVector, @ptrCast(slice.ptr));
    return @as(ByteVector, ptr.*);
}

fn dotProductInternal(a: []const f32, b: []const f32, opts: SIMDOpts) f32 {
    std.debug.assert(a.len == b.len);
    const start = beginTiming();
    var used_simd = false;
    var sum_vec = @as(FloatVector, @splat(@as(f32, 0.0)));
    var scalar_sum: f32 = 0.0;

    var i: usize = 0;
    if (opts.shouldUseSimd(a.len)) {
        const simd_end = a.len - (a.len % SIMD_WIDTH);
        while (i < simd_end) : (i += SIMD_WIDTH) {
            const va = loadVector(a[i .. i + SIMD_WIDTH]);
            const vb = loadVector(b[i .. i + SIMD_WIDTH]);
            sum_vec += va * vb;
        }
        used_simd = simd_end != 0;
        scalar_sum += @reduce(.Add, sum_vec);
    }

    while (i < a.len) : (i += 1) {
        scalar_sum += a[i] * b[i];
    }

    finishTiming(start, used_simd);
    return scalar_sum;
}

fn vectorAddInternal(result: []f32, a: []const f32, b: []const f32) bool {
    std.debug.assert(a.len == b.len);
    std.debug.assert(result.len == a.len);
    var used_simd = false;

    var i: usize = 0;
    const simd_end = a.len - (a.len % SIMD_WIDTH);
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const va = loadVector(a[i .. i + SIMD_WIDTH]);
        const vb = loadVector(b[i .. i + SIMD_WIDTH]);
        const sum = va + vb;
        storeVector(sum, result[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }

    return used_simd;
}

fn vectorSubInternal(result: []f32, a: []const f32, b: []const f32) bool {
    std.debug.assert(a.len == b.len);
    std.debug.assert(result.len == a.len);
    var used_simd = false;

    var i: usize = 0;
    const simd_end = a.len - (a.len % SIMD_WIDTH);
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const va = loadVector(a[i .. i + SIMD_WIDTH]);
        const vb = loadVector(b[i .. i + SIMD_WIDTH]);
        const diff = va - vb;
        storeVector(diff, result[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < a.len) : (i += 1) {
        result[i] = a[i] - b[i];
    }

    return used_simd;
}

fn vectorMulInternal(result: []f32, a: []const f32, b: []const f32) bool {
    std.debug.assert(a.len == b.len);
    std.debug.assert(result.len == a.len);
    var used_simd = false;

    var i: usize = 0;
    const simd_end = a.len - (a.len % SIMD_WIDTH);
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const va = loadVector(a[i .. i + SIMD_WIDTH]);
        const vb = loadVector(b[i .. i + SIMD_WIDTH]);
        const prod = va * vb;
        storeVector(prod, result[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < a.len) : (i += 1) {
        result[i] = a[i] * b[i];
    }

    return used_simd;
}

fn scaleInternal(result: []f32, input: []const f32, scalar: f32) bool {
    std.debug.assert(result.len == input.len);
    var used_simd = false;

    var i: usize = 0;
    const simd_end = input.len - (input.len % SIMD_WIDTH);
    const scalar_vec = @as(FloatVector, @splat(scalar));
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const vec = loadVector(input[i .. i + SIMD_WIDTH]);
        const scaled = vec * scalar_vec;
        storeVector(scaled, result[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < input.len) : (i += 1) {
        result[i] = input[i] * scalar;
    }

    return used_simd;
}

fn vectorMaxInternal(result: []f32, a: []const f32, b: []const f32) bool {
    std.debug.assert(a.len == b.len);
    std.debug.assert(result.len == a.len);
    var used_simd = false;

    var i: usize = 0;
    const simd_end = a.len - (a.len % SIMD_WIDTH);
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const va = loadVector(a[i .. i + SIMD_WIDTH]);
        const vb = loadVector(b[i .. i + SIMD_WIDTH]);
        const vmax = @max(va, vb);
        storeVector(vmax, result[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < a.len) : (i += 1) {
        result[i] = @max(a[i], b[i]);
    }

    return used_simd;
}

fn vectorReluInternal(data: []f32) bool {
    var used_simd = false;
    var i: usize = 0;
    const simd_end = data.len - (data.len % SIMD_WIDTH);
    const zero_vec = @as(FloatVector, @splat(@as(f32, 0.0)));
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const vec = loadVector(data[i .. i + SIMD_WIDTH]);
        const relu = @max(vec, zero_vec);
        storeVector(relu, data[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < data.len) : (i += 1) {
        if (data[i] < 0.0) data[i] = 0.0;
    }

    return used_simd;
}

fn vectorLeakyReluInternal(data: []f32, slope: f32) bool {
    var used_simd = false;
    var i: usize = 0;
    const simd_end = data.len - (data.len % SIMD_WIDTH);
    const zero_vec = @as(FloatVector, @splat(@as(f32, 0.0)));
    const slope_vec = @as(FloatVector, @splat(slope));
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const vec = loadVector(data[i .. i + SIMD_WIDTH]);
        const mask = vec < zero_vec;
        const leaky = vec * slope_vec;
        const blended = @select(f32, mask, leaky, vec);
        storeVector(blended, data[i .. i + SIMD_WIDTH]);
    }
    used_simd = simd_end != 0;

    while (i < data.len) : (i += 1) {
        data[i] = if (data[i] > 0.0) data[i] else slope * data[i];
    }

    return used_simd;
}

/// Optimized vectorized dot product for high-performance vector similarity
pub fn vectorizedDotProduct(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var sum = @as(FloatVector, @splat(0.0));
    var i: usize = 0;
    const simd_end = a.len - (a.len % SIMD_WIDTH);

    // SIMD vectorized computation
    while (i < simd_end) : (i += SIMD_WIDTH) {
        const va = loadVector(a[i..][0..SIMD_WIDTH]);
        const vb = loadVector(b[i..][0..SIMD_WIDTH]);
        sum += va * vb;
    }

    // Horizontal sum of SIMD vector
    var result: f32 = 0.0;
    inline for (0..SIMD_WIDTH) |j| {
        result += sum[j];
    }

    // Handle remaining elements
    while (i < a.len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

fn normalizeInternal(result: []f32, input: []const f32) bool {
    std.debug.assert(result.len == input.len);
    const magnitude_sq = dotProductInternal(input, input, .{});
    if (magnitude_sq == 0.0) {
        @memcpy(result, input);
        return false;
    }
    const magnitude = std.math.sqrt(magnitude_sq);
    return scaleInternal(result, input, 1.0 / magnitude);
}

fn matrixVectorMultiplyInternal(result: []f32, matrix: []const f32, vector: []const f32, rows: usize, cols: usize) bool {
    std.debug.assert(rows * cols <= matrix.len);
    std.debug.assert(vector.len >= cols);
    std.debug.assert(result.len >= rows);
    var used_simd = false;

    for (0..rows) |row| {
        const start_idx = row * cols;
        const slice = matrix[start_idx .. start_idx + cols];
        result[row] = dotProductInternal(slice, vector, .{});
        used_simd = used_simd or (SIMDOpts{}).shouldUseSimd(cols);
    }

    return used_simd;
}

fn matrixMultiplyInternal(result: []f32, a: []const f32, b: []const f32, rows: usize, cols: usize, inner_dim: usize) bool {
    std.debug.assert(rows * inner_dim <= a.len);
    std.debug.assert(inner_dim * cols <= b.len);
    std.debug.assert(rows * cols <= result.len);
    var used_simd = false;

    for (0..rows) |i| {
        for (0..cols) |j| {
            var sum: f32 = 0.0;
            var k: usize = 0;
            if (inner_dim >= SIMD_WIDTH) {
                const simd_end = inner_dim - (inner_dim % SIMD_WIDTH);
                while (k < simd_end) : (k += SIMD_WIDTH) {
                    const row_slice = a[i * inner_dim + k .. i * inner_dim + k + SIMD_WIDTH];
                    const va = loadVector(row_slice);
                    var col_buf: [SIMD_WIDTH]f32 = undefined;
                    for (0..SIMD_WIDTH) |offset| {
                        col_buf[offset] = b[(k + offset) * cols + j];
                    }
                    const vb = @as(FloatVector, @bitCast(col_buf));
                    sum += @reduce(.Add, va * vb);
                }
                used_simd = used_simd or simd_end != 0;
            }

            while (k < inner_dim) : (k += 1) {
                sum += a[i * inner_dim + k] * b[k * cols + j];
            }

            result[i * cols + j] = sum;
        }
    }

    return used_simd;
}

pub const VectorOps = struct {
    pub fn shouldUseSimd(len: usize) bool {
        return (SIMDOpts{}).shouldUseSimd(len);
    }

    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        return dotProductInternal(a, b, .{});
    }

    pub fn matrixVectorMultiply(result: []f32, matrix: []const f32, vector: []const f32, rows: usize, cols: usize) void {
        _ = matrixVectorMultiplyInternal(result, matrix, vector, rows, cols);
    }

    pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, rows: usize, cols: usize, inner_dim: usize) void {
        _ = matrixMultiplyInternal(result, a, b, rows, cols, inner_dim);
    }

    pub fn vectorizedRelu(data: []f32) void {
        const start = beginTiming();
        const used_simd = vectorReluInternal(data);
        finishTiming(start, used_simd);
    }

    pub fn vectorizedLeakyRelu(data: []f32, slope: f32) void {
        const start = beginTiming();
        const used_simd = vectorLeakyReluInternal(data, slope);
        finishTiming(start, used_simd);
    }

    pub fn vectorMax(result: []f32, a: []const f32, b: []const f32) void {
        const start = beginTiming();
        const used_simd = vectorMaxInternal(result, a, b);
        finishTiming(start, used_simd);
    }

    pub fn vectorAdd(result: []f32, a: []const f32, b: []const f32) void {
        const start = beginTiming();
        const used_simd = vectorAddInternal(result, a, b);
        finishTiming(start, used_simd);
    }

    pub fn add(result: []f32, a: []const f32, b: []const f32) void {
        vectorAdd(result, a, b);
    }

    pub fn vectorMul(result: []f32, a: []const f32, b: []const f32) void {
        const start = beginTiming();
        const used_simd = vectorMulInternal(result, a, b);
        finishTiming(start, used_simd);
    }

    pub fn multiply(result: []f32, a: []const f32, b: []const f32) void {
        vectorMul(result, a, b);
    }

    pub fn scale(result: []f32, input: []const f32, scalar: f32) void {
        const start = beginTiming();
        const used_simd = scaleInternal(result, input, scalar);
        finishTiming(start, used_simd);
    }

    pub fn normalize(result: []f32, input: []const f32) void {
        const start = beginTiming();
        const used_simd = normalizeInternal(result, input);
        finishTiming(start, used_simd);
    }

    pub fn vectorNormalize(result: []f32, input: []const f32) void {
        VectorOps.normalize(result, input);
    }
};

pub fn distance(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0.0;
    for (a, b) |va, vb| {
        const diff = va - vb;
        sum += diff * diff;
    }
    return std.math.sqrt(sum);
}

pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const dot = dotProductInternal(a, b, .{});
    const norm_a = std.math.sqrt(dotProductInternal(a, a, .{}));
    const norm_b = std.math.sqrt(dotProductInternal(b, b, .{}));
    const denom = norm_a * norm_b;
    if (denom == 0.0) return 0.0;
    return dot / denom;
}

pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    return dotProductInternal(a, b, .{});
}

pub fn add(result: []f32, a: []const f32, b: []const f32) void {
    VectorOps.vectorAdd(result, a, b);
}

pub fn subtract(result: []f32, a: []const f32, b: []const f32) void {
    const start = beginTiming();
    const used_simd = vectorSubInternal(result, a, b);
    finishTiming(start, used_simd);
}

pub fn multiply(result: []f32, a: []const f32, b: []const f32) void {
    VectorOps.vectorMul(result, a, b);
}

pub fn scale(result: []f32, input: []const f32, scalar: f32) void {
    VectorOps.scale(result, input, scalar);
}

pub fn normalize(result: []f32, input: []const f32) void {
    VectorOps.normalize(result, input);
}

pub fn matrixVectorMultiply(result: []f32, matrix: []const f32, vector: []const f32, rows: usize, cols: usize) void {
    _ = matrixVectorMultiplyInternal(result, matrix, vector, rows, cols);
}

pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, rows: usize, cols: usize, inner_dim: usize) void {
    _ = matrixMultiplyInternal(result, a, b, rows, cols, inner_dim);
}

pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: SIMDOpts) f32 {
    return dotProductInternal(a, b, opts);
}

pub fn vectorAddSIMD(a: []const f32, b: []const f32, result: []f32) void {
    const start = beginTiming();
    const used_simd = vectorAddInternal(result, a, b);
    finishTiming(start, used_simd);
}

pub fn getPerformanceMonitor() *PerformanceMonitor {
    return &global_monitor;
}

pub fn getPerformanceMonitorDetails() PerformanceMonitorDetails {
    return global_monitor.details();
}

pub fn getVectorOps() VectorOps {
    return .{};
}

pub const text = struct {
    pub fn countByte(haystack: []const u8, needle: u8) usize {
        var count: usize = 0;
        var i: usize = 0;
        if (haystack.len >= TextSimdWidth) {
            const simd_end = haystack.len - (haystack.len % TextSimdWidth);
            const needle_vec = @as(ByteVector, @splat(needle));
            const ones = @as(ByteVector, @splat(@as(u8, 1)));
            const zeros = @as(ByteVector, @splat(@as(u8, 0)));
            while (i < simd_end) : (i += TextSimdWidth) {
                const chunk = loadByteVector(haystack[i .. i + TextSimdWidth]);
                const matches = chunk == needle_vec;
                const mask = @select(u8, matches, ones, zeros);
                count += @as(usize, @intCast(@reduce(.Add, mask)));
            }
        }

        while (i < haystack.len) : (i += 1) {
            if (haystack[i] == needle) count += 1;
        }
        return count;
    }

    pub fn findByte(haystack: []const u8, needle: u8) ?usize {
        var i: usize = 0;
        if (haystack.len >= TextSimdWidth) {
            const simd_end = haystack.len - (haystack.len % TextSimdWidth);
            const needle_vec = @as(ByteVector, @splat(needle));
            while (i < simd_end) : (i += TextSimdWidth) {
                const chunk = loadByteVector(haystack[i .. i + TextSimdWidth]);
                const matches = chunk == needle_vec;
                if (@reduce(.Or, matches)) {
                    const mask = @as([TextSimdWidth]bool, matches);
                    for (mask, 0..) |flag, offset| {
                        if (flag) return i + offset;
                    }
                }
            }
        }

        while (i < haystack.len) : (i += 1) {
            if (haystack[i] == needle) return i;
        }
        return null;
    }

    pub fn contains(haystack: []const u8, needle: u8) bool {
        return findByte(haystack, needle) != null;
    }
};
