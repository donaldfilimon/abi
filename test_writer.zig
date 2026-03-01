const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var buf = try std.ArrayList(u8).initCapacity(allocator, 1024);
    defer buf.deinit();

    const writer = buf.writer();
    try writer.writeInt(u32, 1234, .little);

    std.debug.print("Success! {}\n", .{buf.items.len});
}
