//! Micro-utilities that support the WDBX database implementation, primarily
//! focused on safe memory management helpers.

const std = @import("std");

/// Utility helpers for the WDBX module.
///
/// The primary purpose of this file is to provide small, reusable
/// functions that are used across the WDBX codebase.  Currently we
/// expose a single helper for freeing optional string slices.
///
/// This module deliberately stays tiny; if more shared helpers are
/// required in the future, they can be added here while keeping the
/// implementation consistent and testable.
pub const utils = struct {
    /// Frees an optional slice if it is allocated.
    ///
    /// Zig's optional types can be `null` or a value.  In the
    /// WDBX code many structs own optional `[]const u8` strings that
    /// need to be freed when the struct is dropped.  This helper
    /// abstracts the pattern
    ///
    /// ```zig
    /// if (opt) |ptr| allocator.free(ptr);
    /// ```
    ///
    /// Usage:
    ///
    /// ```zig
    /// utils.freeOptional(allocator, self.db_path);
    /// ```
    ///
    /// # Parameters
    ///
    /// - `allocator`: the allocator that owns the memory.
    /// - `opt`: the optional slice to free, or `null`.
    ///
    /// # Panics
    ///
    /// Does not panic. If `opt` is not `null`, `allocator.free`
    /// is called with the slice; the slice must have been allocated
    /// from that allocator.
    pub inline fn freeOptional(allocator: std.mem.Allocator, opt: ?[]const u8) void {
        if (opt) |ptr| allocator.free(ptr);
    }
};
