//! Gradient Bucket Manager for Efficient AllReduce.
//!
//! Provides gradient bucketing to fuse small gradient tensors
//! for more efficient AllReduce operations in distributed training.

const std = @import("std");

/// Gradient bucket for fusing small gradients.
pub const GradientBucket = struct {
    allocator: std.mem.Allocator,
    buffer: []f32,
    capacity: usize,
    used: usize,
    offsets: std.ArrayListUnmanaged(GradientOffset),
    ready: bool,

    const GradientOffset = struct {
        param_id: usize,
        offset: usize,
        size: usize,
    };

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !GradientBucket {
        return .{
            .allocator = allocator,
            .buffer = try allocator.alloc(f32, capacity),
            .capacity = capacity,
            .used = 0,
            .offsets = .{},
            .ready = false,
        };
    }

    pub fn deinit(self: *GradientBucket) void {
        self.allocator.free(self.buffer);
        self.offsets.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add gradient to bucket.
    pub fn add(self: *GradientBucket, param_id: usize, gradient: []const f32) !bool {
        if (self.used + gradient.len > self.capacity) {
            return false; // Bucket full
        }

        @memcpy(self.buffer[self.used..][0..gradient.len], gradient);

        try self.offsets.append(self.allocator, .{
            .param_id = param_id,
            .offset = self.used,
            .size = gradient.len,
        });

        self.used += gradient.len;
        return true;
    }

    /// Mark bucket as ready for AllReduce.
    pub fn markReady(self: *GradientBucket) void {
        self.ready = true;
    }

    /// Get gradient data for AllReduce.
    pub fn getData(self: *const GradientBucket) []f32 {
        return self.buffer[0..self.used];
    }

    /// Extract gradient after AllReduce.
    pub fn extractGradient(self: *const GradientBucket, param_id: usize) ?[]const f32 {
        for (self.offsets.items) |offset| {
            if (offset.param_id == param_id) {
                return self.buffer[offset.offset..][0..offset.size];
            }
        }
        return null;
    }

    /// Reset bucket for reuse.
    pub fn reset(self: *GradientBucket) void {
        self.used = 0;
        self.offsets.clearRetainingCapacity();
        self.ready = false;
    }
};

/// Gradient bucket manager for efficient AllReduce.
pub const GradientBucketManager = struct {
    allocator: std.mem.Allocator,
    buckets: std.ArrayListUnmanaged(GradientBucket),
    bucket_size: usize,
    current_bucket: usize,

    pub fn init(allocator: std.mem.Allocator, num_buckets: usize, bucket_size: usize) !GradientBucketManager {
        var manager = GradientBucketManager{
            .allocator = allocator,
            .buckets = .{},
            .bucket_size = bucket_size,
            .current_bucket = 0,
        };

        for (0..num_buckets) |_| {
            const bucket = try GradientBucket.init(allocator, bucket_size);
            try manager.buckets.append(allocator, bucket);
        }

        return manager;
    }

    pub fn deinit(self: *GradientBucketManager) void {
        for (self.buckets.items) |*bucket| {
            bucket.deinit();
        }
        self.buckets.deinit(self.allocator);
        self.* = undefined;
    }

    /// Add gradient to appropriate bucket.
    pub fn addGradient(self: *GradientBucketManager, param_id: usize, gradient: []const f32) !void {
        // Try current bucket first
        if (self.current_bucket < self.buckets.items.len) {
            if (try self.buckets.items[self.current_bucket].add(param_id, gradient)) {
                return;
            }
            // Bucket full, mark ready and move to next
            self.buckets.items[self.current_bucket].markReady();
            self.current_bucket += 1;
        }

        // Try to add to new bucket
        if (self.current_bucket < self.buckets.items.len) {
            _ = try self.buckets.items[self.current_bucket].add(param_id, gradient);
        }
    }

    /// Get all ready buckets for AllReduce.
    pub fn getReadyBuckets(self: *GradientBucketManager) []GradientBucket {
        var ready_count: usize = 0;
        for (self.buckets.items) |bucket| {
            if (bucket.ready) ready_count += 1;
        }

        // Return slice of ready buckets
        return self.buckets.items[0..ready_count];
    }

    /// Reset all buckets for next iteration.
    pub fn reset(self: *GradientBucketManager) void {
        for (self.buckets.items) |*bucket| {
            bucket.reset();
        }
        self.current_bucket = 0;
    }
};
