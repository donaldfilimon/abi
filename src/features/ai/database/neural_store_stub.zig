//! Stubbed AI-local neural ingestion adapter.
//!
//! Types are imported from the always-on core database engine so they stay
//! in sync with the real `neural_store.zig` adapter. Only the `Engine`
//! struct and persistence functions are stubbed out.

const std = @import("std");
const neural = @import("../../../core/database/neural.zig");

pub const Metadata = neural.Metadata;
pub const SearchOptions = neural.SearchOptions;
pub const SearchResult = neural.SearchResult;
pub const WritePolicy = neural.WritePolicy;
pub const EngineVector = neural.EngineVector;

pub const Engine = struct {
    pub fn init(_: std.mem.Allocator, _: anytype) !Engine {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Engine) void {}
    pub fn indexByVector(_: *Engine, _: []const u8, _: []const f32, _: Metadata) !void {
        return error.FeatureDisabled;
    }
    pub fn search(_: *Engine, _: []const u8, _: SearchOptions) ![]SearchResult {
        return error.FeatureDisabled;
    }
};

pub fn save(_: *Engine, _: []const u8) !void {
    return error.FeatureDisabled;
}

pub fn load(_: std.mem.Allocator, _: []const u8) !Engine {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
