const std = @import("std");
const planner = @import("planner.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📚 Generating ABI API Documentation with GitHub Pages optimization", .{});

    try cleanDocsOutput();
    try ensureDirectoryLayout();

    const plan = planner.buildDefaultPlan(allocator);
    try plan.execute();

    std.log.info("✅ GitHub Pages documentation generation completed!", .{});
    std.log.info("📝 To deploy: Enable GitHub Pages in repository settings (source: docs folder)", .{});
    std.log.info("🚀 GitHub Actions workflow created for automated deployment", .{});
}

fn ensureDirectoryLayout() !void {
    try std.fs.cwd().makePath("docs/generated");
    try std.fs.cwd().makePath("docs/assets/css");
    try std.fs.cwd().makePath("docs/assets/js");
    try std.fs.cwd().makePath("docs/_layouts");
    try std.fs.cwd().makePath("docs/_data");
    try std.fs.cwd().makePath(".github/workflows");
}

fn cleanDocsOutput() !void {
    const paths = [_][]const u8{
        "docs/generated",
        "docs/assets/css",
        "docs/assets/js",
        "docs/zig-docs",
    };

    const cwd = std.fs.cwd();
    for (paths) |path| {
        const exists = blk: {
            cwd.access(path, .{}) catch |err| switch (err) {
                error.FileNotFound => break :blk false,
                else => return err,
            };
            break :blk true;
        };
        if (!exists) continue;

        cwd.deleteTree(path) catch |err| switch (err) {
            error.NotDir => {
                cwd.deleteFile(path) catch |delete_err| switch (delete_err) {
                    error.FileNotFound => continue,
                    else => return delete_err,
                };
            },
            else => return err,
        };
    }
}
