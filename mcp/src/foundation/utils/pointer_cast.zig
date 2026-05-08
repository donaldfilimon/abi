// Centralized pointer-cast helper for Zig 0.17 upgrade path
pub fn implCast(comptime Impl: type, ptr: *anyopaque) *Impl {
    return @ptrCast(@alignCast(ptr));
}
