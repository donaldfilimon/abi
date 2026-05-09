//! Memory Backend
//!
//! In-memory object storage using StringHashMap. Supports capacity limits,
//! object metadata, and content-type tracking.

const std = @import("std");
const types = @import("types.zig");
const backend_mod = @import("backend.zig");

const StorageError = types.StorageError;
const StorageObject = types.StorageObject;
const ObjectMetadata = types.ObjectMetadata;
const StorageStats = types.StorageStats;
const Backend = backend_mod.Backend;

pub const MemoryObject = struct {
    data: []u8, // owned
    key: []u8, // owned
    content_type: []u8, // owned
    size: u64,
};

pub const MemoryBackend = struct {
    allocator: std.mem.Allocator,
    objects: std.StringHashMapUnmanaged(*MemoryObject),
    total_bytes: u64,
    max_bytes: u64,

    pub fn create(allocator: std.mem.Allocator, max_mb: u32) !*MemoryBackend {
        const mb = try allocator.create(MemoryBackend);
        mb.* = .{
            .allocator = allocator,
            .objects = .empty,
            .total_bytes = 0,
            .max_bytes = @as(u64, max_mb) * 1024 * 1024,
        };
        return mb;
    }

    pub fn destroy(self: *MemoryBackend) void {
        var iter = self.objects.iterator();
        while (iter.next()) |entry| {
            const obj = entry.value_ptr.*;
            self.allocator.free(obj.data);
            self.allocator.free(obj.key);
            self.allocator.free(obj.content_type);
            self.allocator.destroy(obj);
        }
        self.objects.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn putImpl(ctx: *anyopaque, key: []const u8, data: []const u8, meta: ?ObjectMetadata) StorageError!void {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));

        // Capacity check: account for freed size if overwriting existing entry
        if (self.max_bytes > 0) {
            var effective_total = self.total_bytes;
            if (self.objects.get(key)) |existing| {
                effective_total -= existing.size;
            }
            if (effective_total + data.len > self.max_bytes) {
                return error.StorageFull;
            }
        }

        const obj = self.allocator.create(MemoryObject) catch return error.OutOfMemory;
        errdefer self.allocator.destroy(obj);

        const owned_key = self.allocator.dupe(u8, key) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned_key);

        const owned_data = self.allocator.dupe(u8, data) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned_data);

        const ct = if (meta) |m| m.content_type else "application/octet-stream";
        const owned_ct = self.allocator.dupe(u8, ct) catch return error.OutOfMemory;
        errdefer self.allocator.free(owned_ct);

        obj.* = .{
            .data = owned_data,
            .key = owned_key,
            .content_type = owned_ct,
            .size = data.len,
        };

        // Ensure hashmap capacity so putAssumeCapacity can't fail below
        self.objects.ensureUnusedCapacity(self.allocator, 1) catch
            return error.OutOfMemory;

        // Past this point, nothing can fail — safe to remove old entry
        if (self.objects.fetchRemove(key)) |kv| {
            const old = kv.value;
            self.total_bytes -= old.size;
            self.allocator.free(old.data);
            self.allocator.free(old.key);
            self.allocator.free(old.content_type);
            self.allocator.destroy(old);
        }

        self.objects.putAssumeCapacity(obj.key, obj);
        self.total_bytes += data.len;
    }

    pub fn getImpl(ctx: *anyopaque, allocator: std.mem.Allocator, key: []const u8) StorageError![]u8 {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        const obj = self.objects.get(key) orelse return error.ObjectNotFound;
        return allocator.dupe(u8, obj.data) catch return error.OutOfMemory;
    }

    pub fn deleteImpl(ctx: *anyopaque, key: []const u8) StorageError!bool {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        if (self.objects.fetchRemove(key)) |kv| {
            const obj = kv.value;
            self.total_bytes -= obj.size;
            self.allocator.free(obj.data);
            self.allocator.free(obj.key);
            self.allocator.free(obj.content_type);
            self.allocator.destroy(obj);
            return true;
        }
        return false;
    }

    pub fn listImpl(ctx: *anyopaque, allocator: std.mem.Allocator, prefix: []const u8) StorageError![]StorageObject {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));

        var results: std.ArrayListUnmanaged(StorageObject) = .empty;
        errdefer results.deinit(allocator);

        var iter = self.objects.iterator();
        while (iter.next()) |entry| {
            const obj = entry.value_ptr.*;
            if (prefix.len == 0 or std.mem.startsWith(u8, obj.key, prefix)) {
                results.append(allocator, .{
                    .key = obj.key,
                    .size = obj.size,
                    .content_type = obj.content_type,
                }) catch return error.OutOfMemory;
            }
        }

        return results.toOwnedSlice(allocator) catch return error.OutOfMemory;
    }

    pub fn existsImpl(ctx: *anyopaque, key: []const u8) bool {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        return self.objects.get(key) != null;
    }

    pub fn statsImpl(ctx: *anyopaque) StorageStats {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        return .{
            .total_objects = self.objects.count(),
            .total_bytes = self.total_bytes,
            .backend = .memory,
        };
    }

    pub fn deinitImpl(ctx: *anyopaque) void {
        const self: *MemoryBackend = @ptrCast(@alignCast(ctx));
        self.destroy();
    }

    pub const vtable = Backend.VTable{
        .put = putImpl,
        .get = getImpl,
        .delete = deleteImpl,
        .list = listImpl,
        .exists = existsImpl,
        .getStats = statsImpl,
        .deinitFn = deinitImpl,
    };

    pub fn backend(self: *MemoryBackend) Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }
};
