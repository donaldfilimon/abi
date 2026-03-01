const std = @import("std");
const builtin = @import("builtin");

pub fn main(_: std.process.Init) anyerror!void {
    @disableInstrumentation();

    const test_fn_list = builtin.test_functions;
    var passed: u64 = 0;
    var skipped: u64 = 0;
    var failed: u64 = 0;

    for (test_fn_list) |test_fn| {
        std.debug.print("{s}... ", .{test_fn.name});

        if (test_fn.func()) |_| {
            std.debug.print("PASS\n", .{});
            passed += 1;
        } else |err| {
            if (err != error.SkipZigTest) {
                std.debug.print("FAIL\n", .{});
                failed += 1;
                return err;
            }
            if (err == error.SkipZigTest) {
                std.debug.print("SKIP\n", .{});
                skipped += 1;
            }
        }
    }

    std.debug.print("{} passed, {} skipped, {} failed\n", .{ passed, skipped, failed });

    if (failed != 0) std.process.exit(1);
}
