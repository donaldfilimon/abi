const std = @import("std");
const abi = @import("abi");
const testing = std.testing;

test "Framework: initialization and shutdown" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    try testing.expect(framework.state == .initialized);
}

test "Framework: multiple init/shutdown cycles" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // First cycle
    {
        var framework = try abi.init(allocator, .{});
        defer abi.shutdown(&framework);
        try testing.expect(framework.state == .initialized);
    }

    // Second cycle
    {
        var framework = try abi.init(allocator, .{});
        defer abi.shutdown(&framework);
        try testing.expect(framework.state == .initialized);
    }

    // Third cycle
    {
        var framework = try abi.init(allocator, .{});
        defer abi.shutdown(&framework);
        try testing.expect(framework.state == .initialized);
    }
}

test "Framework: feature availability" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var framework = try abi.init(allocator, .{});
    defer abi.shutdown(&framework);

    // Check that features are available
    const has_ai = @hasDecl(abi, "ai");
    const has_database = @hasDecl(abi, "database");
    const has_gpu = @hasDecl(abi, "gpu");
    const has_monitoring = @hasDecl(abi, "monitoring");

    try testing.expect(has_ai);
    try testing.expect(has_database);
    try testing.expect(has_gpu);
    try testing.expect(has_monitoring);
}
