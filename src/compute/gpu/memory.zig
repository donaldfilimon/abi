//! GPU memory management
//!
//! Provides GPU buffer allocation, transfer, and synchronization utilities.

const std = @import("std");

pub const GPUBuffer = struct {
    device_ptr: *anyopaque,
    host_ptr: []u8,
    size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, size: usize, flags: BufferFlags) !GPUBuffer {
        _ = flags;

        const host_ptr = try allocator.alloc(u8, size);

        return GPUBuffer{
            .device_ptr = undefined,
            .host_ptr = host_ptr,
            .size = size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GPUBuffer) void {
        self.allocator.free(self.host_ptr);
        self.* = undefined;
    }

    pub fn writeFromHost(self: *GPUBuffer, data: []const u8) !void {
        if (data.len > self.host_ptr.len) {
            return error.BufferTooSmall;
        }

        @memcpy(self.host_ptr[0..data.len], data);
    }

    pub fn readToHost(self: *const GPUBuffer, offset: usize, size: usize) ![]const u8 {
        if (offset + size > self.host_ptr.len) {
            return error.InvalidOffset;
        }

        return self.host_ptr[offset..][0..size];
    }

    pub fn copyToDevice(self: *GPUBuffer) !void {
        _ = self;
    }

    pub fn copyToHost(self: *GPUBuffer) !void {
        _ = self;
    }
};

pub const BufferFlags = struct {
    read_only: bool = false,
    write_only: bool = false,
    host_visible: bool = true,
    device_local: bool = false,
};

pub const GPUMemoryPool = struct {
    buffers: std.ArrayList(GPUBuffer),
    allocator: std.mem.Allocator,
    total_size: usize,
    max_size: usize,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) GPUMemoryPool {
        return GPUMemoryPool{
            .buffers = std.ArrayList(GPUBuffer).init(allocator),
            .allocator = allocator,
            .total_size = 0,
            .max_size = max_size,
        };
    }

    pub fn deinit(self: *GPUMemoryPool) void {
        for (self.buffers.items) |*buffer| {
            buffer.deinit();
        }
        self.buffers.deinit();
    }

    pub fn allocate(self: *GPUMemoryPool, size: usize, flags: BufferFlags) !*GPUBuffer {
        if (self.total_size + size > self.max_size) {
            return error.OutOfMemory;
        }

        const buffer = try GPUBuffer.init(self.allocator, size, flags);
        try self.buffers.append(buffer);

        self.total_size += size;
        return &self.buffers.items[self.buffers.items.len - 1];
    }

    pub fn free(self: *GPUMemoryPool, buffer: *GPUBuffer) void {
        var i: usize = 0;
        while (i < self.buffers.items.len) : (i += 1) {
            if (&self.buffers.items[i] == buffer) {
                std.debug.assert(self.total_size >= buffer.size);
                self.total_size -= buffer.size;
                buffer.deinit();
                _ = self.buffers.orderedRemove(i);
                return;
            }
        }
    }

    pub fn getUsage(self: *const GPUMemoryPool) f64 {
        if (self.max_size == 0) return 0;
        return @as(f64, @floatFromInt(self.total_size)) / @as(f64, @floatFromInt(self.max_size));
    }
};

pub const AsyncTransfer = struct {
    source: *const GPUBuffer,
    destination: *GPUBuffer,
    size: usize,
    offset: usize,
    completed: std.atomic.Value(bool),

    pub fn init(source: *const GPUBuffer, destination: *GPUBuffer, size: usize, offset: usize) AsyncTransfer {
        return AsyncTransfer{
            .source = source,
            .destination = destination,
            .size = size,
            .offset = offset,
            .completed = std.atomic.Value(bool).init(false),
        };
    }

    pub fn start(self: *AsyncTransfer) !void {
        _ = self;
    }

    pub fn wait(self: *AsyncTransfer) void {
        while (!self.completed.load(.acquire)) {
            std.atomic.spinLoopHint();
        }
    }

    pub fn isComplete(self: *const AsyncTransfer) bool {
        return self.completed.load(.acquire);
    }
};
