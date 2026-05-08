//! Search State
//!
//! Internal state management for the search engine singleton.
//! Owns the index registry (StringHashMap of InvertedIndex) and lock.

const std = @import("std");
const sync = @import("../../foundation/mod.zig").sync;
const types = @import("types.zig");
const inverted_index = @import("index.zig");

pub const SearchConfig = types.SearchConfig;

pub const SearchState = struct {
    allocator: std.mem.Allocator,
    config: SearchConfig,
    indexes: std.StringHashMapUnmanaged(*inverted_index.InvertedIndex),
    rw_lock: sync.RwLock,

    pub fn create(allocator: std.mem.Allocator, config: SearchConfig) !*SearchState {
        const s = try allocator.create(SearchState);
        s.* = .{
            .allocator = allocator,
            .config = config,
            .indexes = .empty,
            .rw_lock = sync.RwLock.init(),
        };
        return s;
    }

    pub fn destroy(self: *SearchState) void {
        var iter = self.indexes.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.*.destroy();
        }
        self.indexes.deinit(self.allocator);
        self.allocator.destroy(self);
    }
};
