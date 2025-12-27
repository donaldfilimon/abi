const std = @import("std");
const abi = @import("abi");

fn sampleTask(_: std.mem.Allocator) !u32 {
    return 42;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var engine = try abi.compute.createDefaultEngine(allocator);
    defer engine.deinit();

    const result = try abi.compute.runTask(&engine, u32, sampleTask, 1000);
    std.debug.print("Task result: {d}\n", .{result});
}
