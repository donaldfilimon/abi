const std = @import("std");

pub const DatabaseHandle = struct { db: ?*anyopaque = null };
pub const SearchResult = struct { id: u64 = 0, score: f32 = 0.0 };
pub const VectorView = struct { id: u64 = 0, vector: []const f32 = &.{}, metadata: ?[]const u8 = null };
pub const Stats = struct { count: usize = 0, dimension: usize = 0 };
pub const BatchItem = struct { id: u64 = 0, vector: []const f32 = &.{}, metadata: ?[]const u8 = null };
pub const DiagnosticsInfo = struct {
    name: []const u8 = "",
    vector_count: usize = 0,
    dimension: usize = 0,
    pub fn isHealthy(_: DiagnosticsInfo) bool {
        return false;
    }
};
