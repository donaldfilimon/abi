const std = @import("std");
const planner = @import("planner.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📚 Generating ABI API Documentation with GitHub Pages optimization", .{});

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
