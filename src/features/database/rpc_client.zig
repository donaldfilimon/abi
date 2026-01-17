//! RPC client for distributed database operations.
//! Provides thin wrappers that forward calls to local shard instances.
//! Intended to be used by the RPC server layer (not yet implemented).

const std = @import("std");
const Database = @import("database.zig");

pub const RpcClient = struct {
    allocator: std.mem.Allocator,
    // In a real deployment this would hold network connection state.
    // For now we keep a reference to the local Database instance.
    db: *Database,

    pub fn init(allocator: std.mem.Allocator, db: *Database) RpcClient {
        return .{ .allocator = allocator, .db = db };
    }

    /// Insert a vector with optional metadata.
    pub fn insert(self: *RpcClient, id: u64, vector: []const f32, metadata: ?[]const u8) !void {
        // Forward to the local Database implementation.
        try self.db.insert(id, vector, metadata);
    }

    /// Search for nearest neighbours.
    pub fn search(self: *RpcClient, query: []const f32, top_k: usize) ![]Database.SearchResult {
        // Allocate a temporary result buffer.
        var results = try self.allocator.alloc(Database.SearchResult, top_k);
        defer self.allocator.free(results);
        const count = try self.db.search(query, top_k, results);
        // Return a slice of the filled portion.
        return results[0..count];
    }

    /// Delete a vector by ID.
    pub fn delete(self: *RpcClient, id: u64) !void {
        try self.db.delete(id);
    }

    /// Update an existing vector.
    pub fn update(self: *RpcClient, id: u64, vector: []const f32) !void {
        try self.db.update(id, vector);
    }
};

