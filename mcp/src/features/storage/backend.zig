//! Backend Interface
//!
//! Vtable-based storage backend abstraction. All storage backends
//! (memory, local, s3, gcs) implement this interface.

const std = @import("std");
const types = @import("types.zig");

const StorageError = types.StorageError;
const StorageObject = types.StorageObject;
const ObjectMetadata = types.ObjectMetadata;
const StorageStats = types.StorageStats;

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        put: *const fn (*anyopaque, []const u8, []const u8, ?ObjectMetadata) StorageError!void,
        get: *const fn (*anyopaque, std.mem.Allocator, []const u8) StorageError![]u8,
        delete: *const fn (*anyopaque, []const u8) StorageError!bool,
        list: *const fn (*anyopaque, std.mem.Allocator, []const u8) StorageError![]StorageObject,
        exists: *const fn (*anyopaque, []const u8) bool,
        getStats: *const fn (*anyopaque) StorageStats,
        deinitFn: *const fn (*anyopaque) void,
    };

    pub fn put(self: Backend, key: []const u8, data: []const u8, meta: ?ObjectMetadata) StorageError!void {
        return self.vtable.put(self.ptr, key, data, meta);
    }

    pub fn get(self: Backend, allocator: std.mem.Allocator, key: []const u8) StorageError![]u8 {
        return self.vtable.get(self.ptr, allocator, key);
    }

    pub fn delete(self: Backend, key: []const u8) StorageError!bool {
        return self.vtable.delete(self.ptr, key);
    }

    pub fn list(self: Backend, allocator: std.mem.Allocator, prefix: []const u8) StorageError![]StorageObject {
        return self.vtable.list(self.ptr, allocator, prefix);
    }

    pub fn exists(self: Backend, key: []const u8) bool {
        return self.vtable.exists(self.ptr, key);
    }

    pub fn getStats(self: Backend) StorageStats {
        return self.vtable.getStats(self.ptr);
    }

    pub fn deinitBackend(self: Backend) void {
        self.vtable.deinitFn(self.ptr);
    }
};
