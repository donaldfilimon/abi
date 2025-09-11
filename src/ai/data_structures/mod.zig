//! Data Structures Module - High-performance concurrent data structures
//!
//! This module provides thread-safe and high-performance data structures
//! optimized for AI and machine learning workloads:
//! - Lock-free data structures for concurrent access
//! - Memory-efficient implementations
//! - SIMD-accelerated operations where applicable

const std = @import("std");

// Re-export lock-free data structures
pub const LockFreeQueue = @import("lockfree.zig").LockFreeQueue;
pub const LockFreeStack = @import("lockfree.zig").LockFreeStack;
pub const AtomicList = @import("lockfree.zig").AtomicList;
pub const ConcurrentHashMap = @import("lockfree.zig").ConcurrentHashMap;

// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Initialize a lock-free queue with the specified capacity
pub fn createLockFreeQueue(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeQueue(T) {
    return LockFreeQueue(T).init(allocator, capacity);
}

test "Data structures module imports" {
    // Test that all main types are accessible
    _ = LockFreeQueue;
    _ = LockFreeStack;
    _ = AtomicList;
    _ = ConcurrentHashMap;
}
