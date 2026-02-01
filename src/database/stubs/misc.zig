const std = @import("std");

pub const cli = struct {
    pub fn run(_: std.mem.Allocator, _: []const [:0]const u8) !void {
        return error.DatabaseDisabled;
    }
};

pub const parallel_search = struct {};
pub const database = struct {};
pub const db_helpers = struct {};
pub const storage = struct {};
pub const http = struct {};
pub const fulltext = struct {
    pub const InvertedIndex = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const hybrid = struct {
    pub const HybridSearchEngine = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const filter = struct {
    pub const FilterBuilder = struct {
        pub fn init() @This() {
            return .{};
        }
    };
};
pub const batch = struct {
    pub const BatchProcessor = struct {
        pub fn init(_: std.mem.Allocator, _: anytype) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const clustering = struct {
    pub const KMeans = struct {
        pub fn init(_: std.mem.Allocator, _: usize, _: usize) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
    };
};
pub const quantization = struct {
    pub const ScalarQuantizer = struct {
        pub fn init(_: u8) @This() {
            return .{};
        }
    };
};
pub const formats = struct {
    pub const UnifiedFormat = struct {
        pub fn deinit(_: *@This()) void {}
    };
    pub const DataType = enum { f32, f16, bf16, i32, i16, i8, u8, q4_0, q4_1, q8_0 };
};
