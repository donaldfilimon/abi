//! Probabilistic Data Structures - Space-efficient approximate algorithms
//!
//! This module provides probabilistic data structures for:
//! - Count-Min Sketch for frequency estimation
//! - HyperLogLog for cardinality estimation
//! - Memory-efficient approximate algorithms

const std = @import("std");

/// Count-Min Sketch for frequency estimation
pub const CountMinSketch = struct {
    const Self = @This();

    /// Sketch data matrix
    data: std.ArrayList(std.ArrayList(u32)),
    /// Number of hash functions (rows)
    depth: usize,
    /// Sketch width (columns)
    width: usize,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new Count-Min Sketch
    pub fn init(allocator: std.mem.Allocator, depth: usize, width: usize) !*Self {
        const sketch = try allocator.create(Self);
        sketch.* = Self{
            .data = std.ArrayList(std.ArrayList(u32)).init(allocator),
            .depth = depth,
            .width = width,
            .allocator = allocator,
        };

        // Initialize the matrix
        try sketch.data.ensureTotalCapacity(depth);
        for (0..depth) |_| {
            var row = try std.ArrayList(u32).initCapacity(allocator, width);
            try row.appendNTimes(0, width);
            try sketch.data.append(row);
        }

        return sketch;
    }

    /// Deinitialize the sketch
    pub fn deinit(self: *Self) void {
        for (self.data.items) |*row| {
            row.deinit();
        }
        self.data.deinit();
        self.allocator.destroy(self);
    }

    /// Add an item to the sketch
    pub fn add(self: *Self, data: []const u8) void {
        const hash = std.hash.Wyhash.hash(0, data);
        for (0..self.depth) |i| {
            const col = (hash +% i) % self.width;
            self.data.items[i].items[col] += 1;
        }
    }

    /// Estimate the frequency of an item
    pub fn estimate(self: *Self, data: []const u8) u32 {
        const hash = std.hash.Wyhash.hash(0, data);
        var min_count: u32 = std.math.maxInt(u32);

        for (0..self.depth) |i| {
            const col = (hash +% i) % self.width;
            min_count = @min(min_count, self.data.items[i].items[col]);
        }

        return min_count;
    }
};

/// HyperLogLog for cardinality estimation
pub const HyperLogLog = struct {
    const Self = @This();

    /// Registers for storing maximum leading zeros
    registers: std.ArrayList(u8),
    /// Number of registers (2^b)
    m: usize,
    /// Precision parameter (b)
    b: u32,
    /// Memory allocator
    allocator: std.mem.Allocator,

    /// Initialize a new HyperLogLog
    pub fn init(allocator: std.mem.Allocator, b: u32) !*Self {
        const m = @as(usize, 1) << b;
        const hll = try allocator.create(Self);
        hll.* = Self{
            .registers = try std.ArrayList(u8).initCapacity(allocator, m),
            .m = m,
            .b = b,
            .allocator = allocator,
        };

        // Initialize all registers to 0
        try hll.registers.appendNTimes(0, m);

        return hll;
    }

    /// Deinitialize the HyperLogLog
    pub fn deinit(self: *Self) void {
        self.registers.deinit();
        self.allocator.destroy(self);
    }

    /// Add an item to the HyperLogLog
    pub fn add(self: *Self, data: []const u8) void {
        const hash = std.hash.Wyhash.hash(0, data);
        const j = hash >> (64 - self.b);
        const w = hash << self.b;
        const leading_zeros = @clz(w) + 1;
        const register_index = @as(usize, @intCast(j));
        if (register_index < self.registers.items.len) {
            self.registers.items[register_index] = @max(self.registers.items[register_index], @as(u8, @intCast(leading_zeros)));
        }
    }

    /// Estimate the cardinality
    pub fn estimate(self: *Self) usize {
        var sum: f64 = 0.0;
        var zero_count: usize = 0;

        for (self.registers.items) |register| {
            if (register == 0) zero_count += 1;
            sum += 1.0 / @as(f64, @floatFromInt(@as(u32, 1) << register));
        }

        const alpha = switch (self.m) {
            8 => 0.693,
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            else => 0.7213 / (1.0 + 1.079 / @as(f64, @floatFromInt(self.m))),
        };

        const cardinality_estimate = alpha * @as(f64, @floatFromInt(self.m * self.m)) / sum;
        // Small range correction
        if (cardinality_estimate <= 2.5 * @as(f64, @floatFromInt(self.m))) {
            if (zero_count != 0) {
                return @as(usize, @intFromFloat(@as(f64, @floatFromInt(self.m)) * @log(-@as(f64, @floatFromInt(zero_count)) / @as(f64, @floatFromInt(self.m)))));
            }
        }

        return @as(usize, @intFromFloat(cardinality_estimate));
    }
};
