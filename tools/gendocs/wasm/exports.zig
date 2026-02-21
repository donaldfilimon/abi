const std = @import("std");
const engine = @import("engine.zig");

var bump_memory: [512 * 1024]u8 align(16) = undefined;
var bump_cursor: u32 = 8;

pub export fn reset_alloc() void {
    bump_cursor = 8;
}

pub export fn alloc(len: u32) u32 {
    const aligned = (len + 7) & ~@as(u32, 7);
    const next = bump_cursor + aligned;
    if (next >= bump_memory.len) {
        return 0;
    }
    const out = bump_cursor;
    bump_cursor = next;
    return out;
}

pub export fn score_query(q_ptr: u32, q_len: u32, t_ptr: u32, t_len: u32) i32 {
    const q = readSlice(q_ptr, q_len) orelse return 0;
    const t = readSlice(t_ptr, t_len) orelse return 0;
    return engine.score(q, t);
}

pub export fn find_match_start(q_ptr: u32, q_len: u32, t_ptr: u32, t_len: u32) i32 {
    const q = readSlice(q_ptr, q_len) orelse return -1;
    const t = readSlice(t_ptr, t_len) orelse return -1;
    return engine.findMatchStart(q, t);
}

fn readSlice(ptr: u32, len: u32) ?[]const u8 {
    const start: usize = @intCast(ptr);
    const size: usize = @intCast(len);
    const end = start + size;
    if (start >= bump_memory.len or end > bump_memory.len) return null;
    return bump_memory[start..end];
}

pub fn main() void {}

test "exports score_query uses engine" {
    reset_alloc();

    const q = "gpu";
    const t = "gpu-dashboard";

    const q_ptr = alloc(@intCast(q.len));
    const t_ptr = alloc(@intCast(t.len));

    @memcpy(bump_memory[@intCast(q_ptr)..][0..q.len], q);
    @memcpy(bump_memory[@intCast(t_ptr)..][0..t.len], t);

    try std.testing.expect(score_query(q_ptr, @intCast(q.len), t_ptr, @intCast(t.len)) > 0);
}
