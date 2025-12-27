const std = @import("std");
const abi = @import("abi");

fn dotProduct(a: *const [4]f32, b: *const [4]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, abi.FrameworkOptions{
        .enable_gpu = true,
    });
    defer abi.shutdown(&framework);

    const vec_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vec_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };

    const result = dotProduct(&vec_a, &vec_b);
    std.debug.print("Dot product: {d}\n", .{result});
    std.debug.print("GPU status: {s}\n", .{if (abi.gpu.isAvailable()) "available" else "not available"});
}
