//! Parallel processing utilities for LLM workloads.
//! Provides helpers to run inference across multiple threads.
//! This is a simple wrapper around std.Thread and std.atomic.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ParallelExecutor = struct {
    allocator: Allocator,
    thread_count: usize,
    threads: []std.Thread,

    pub fn init(allocator: Allocator, thread_count: usize) !ParallelExecutor {
        const threads = try allocator.alloc(std.Thread, thread_count);
        var exec = ParallelExecutor{ .allocator = allocator, .thread_count = thread_count, .threads = threads };
        return exec;
    }

    pub fn deinit(self: *ParallelExecutor) void {
        self.allocator.free(self.threads);
    }

    /// Execute a function in parallel on a slice of inputs.
    /// The function receives the slice index and a pointer to the element.
    pub fn parallelFor(self: *ParallelExecutor, items: anytype, func: fn (usize, @typeInfo(@TypeOf(items)).Pointer.child) void) void {
        const len = items.len;
        const per_thread = (len + self.thread_count - 1) / self.thread_count;
        var start: usize = 0;
        var i: usize = 0;
        while (i < self.thread_count) : (i += 1) {
            const end = @min(start + per_thread, len);
            const slice = items[start..end];
            self.threads[i] = std.Thread.spawn(.{}, struct {
                fn run(slice: @TypeOf(slice), func: fn (usize, @typeInfo(@TypeOf(slice)).Pointer.child) void) void {
                    var idx: usize = 0;
                    while (idx < slice.len) : (idx += 1) {
                        func(idx, slice[idx]);
                    }
                }
            }.run, .{ slice, func }) catch {};
            start = end;
        }
        // Join threads
        var j: usize = 0;
        while (j < self.thread_count) : (j += 1) {
            self.threads[j].join();
        }
    }
};

