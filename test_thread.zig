const std = @import("std");
pub fn main() void {
    const thread = std.Thread;
    @compileLog(@typeInfo(thread).Struct.decls);
}
