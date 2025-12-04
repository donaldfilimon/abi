const std = @import("std");

/// Generate native Zig documentation using built-in tools
pub fn generateZigNativeDocs(_: std.mem.Allocator) !void {
    const docs_dir = "docs/zig-docs";
    try std.fs.cwd().makePath(docs_dir);

    const run_result = std.process.Child.run(.{
        .allocator = std.heap.page_allocator,
        .argv = &[_][]const u8{
            "zig",
            "doc",
            "src/mod.zig",
            "--output-dir",
            docs_dir,
        },
    }) catch |err| {
        std.log.warn("failed to invoke 'zig doc': {s}", .{@errorName(err)});
        return writeZigDocFallback();
    };
    defer std.heap.page_allocator.free(run_result.stdout);
    defer std.heap.page_allocator.free(run_result.stderr);

    if (run_result.term == .Exited and run_result.term.Exited == 0) {
        std.log.info("Generated Zig native docs at {s}", .{docs_dir});
        return;
    }

    const trimmed_stderr = std.mem.trim(u8, run_result.stderr, " \n\r\t");
    if (trimmed_stderr.len != 0) {
        std.log.warn("zig doc stderr: {s}", .{trimmed_stderr});
    }

    try writeZigDocFallback();
}

fn writeZigDocFallback() !void {
    std.log.warn("writing placeholder Zig native docs because generation failed", .{});
    var out = try std.fs.cwd().createFile("docs/zig-docs/index.html", .{ .truncate = true });
    defer out.close();
    try out.writeAll(
        "<html><body><h1>Zig Native Docs</h1><p>Documentation generation failed; placeholder created.</p></body></html>",
    );
}
