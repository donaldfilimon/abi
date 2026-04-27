//! Index Lifecycle
//!
//! Create/delete named indexes, and save/load index persistence I/O.

const std = @import("std");
const inverted_index = @import("index.zig");
const persistence = @import("persistence.zig");
const state_mod = @import("state.zig");
const types = @import("types.zig");

pub const SearchState = state_mod.SearchState;
pub const SearchError = types.SearchError;
pub const SearchIndex = types.SearchIndex;

/// Create a new named full-text index. Returns `IndexAlreadyExists` if
/// an index with the same name exists.
pub fn createIndex(s: *SearchState, name: []const u8) SearchError!SearchIndex {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.indexes.get(name) != null) return error.IndexAlreadyExists;

    const idx = inverted_index.InvertedIndex.create(s.allocator, name) catch return error.OutOfMemory;
    s.indexes.put(s.allocator, idx.name, idx) catch {
        idx.destroy();
        return error.OutOfMemory;
    };

    return .{ .name = idx.name };
}

/// Delete a named index and all its documents.
pub fn deleteIndex(s: *SearchState, name: []const u8) SearchError!void {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.indexes.fetchRemove(name)) |kv| {
        kv.value.destroy();
    } else {
        return error.IndexNotFound;
    }
}

/// Serialize a named inverted index to disk at the given path.
pub fn saveIndex(s: *SearchState, name: []const u8, path: []const u8) SearchError!void {
    s.rw_lock.lockShared();
    defer s.rw_lock.unlockShared();

    const idx = s.indexes.get(name) orelse return error.IndexNotFound;
    persistence.saveIndex(idx, path) catch return error.IoError;
}

/// Deserialize a named index from disk. Creates a new index with the
/// given name (must not already exist) and populates it from the file.
pub fn loadIndex(s: *SearchState, allocator: std.mem.Allocator, name: []const u8, path: []const u8) SearchError!void {
    s.rw_lock.lock();
    defer s.rw_lock.unlock();

    if (s.indexes.get(name) != null) return error.IndexAlreadyExists;

    const idx = persistence.loadIndex(allocator, name, path) catch |err| switch (err) {
        error.IoError => return error.IoError,
        error.IndexCorrupted => return error.IndexCorrupted,
        error.IndexAlreadyExists => return error.IndexAlreadyExists,
        error.OutOfMemory => return error.OutOfMemory,
        error.FeatureDisabled => return error.FeatureDisabled,
    };

    s.indexes.put(s.allocator, idx.name, idx) catch return error.OutOfMemory;
}
