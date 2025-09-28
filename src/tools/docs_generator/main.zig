const std = @import("std");
const planner = @import("planner.zig");
const config = @import("config.zig");

const module_docs = @import("generators/module_docs.zig");
const api_reference = @import("generators/api_reference.zig");
const examples = @import("generators/examples.zig");
const performance_guide = @import("generators/performance_guide.zig");
const definitions_reference = @import("generators/definitions_reference.zig");
const code_index = @import("generators/code_index.zig");
const search_index = @import("generators/search_index.zig");
const docs_index = @import("generators/docs_index.zig");
const readme_redirect = @import("generators/readme_redirect.zig");
const native_docs = @import("generators/native_docs.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("📚 Generating ABI API Documentation with GitHub Pages optimization", .{});

    try ensureDirectoryLayout();

    const steps = [_]planner.GenerationStep{
        .{ .name = "Disable Jekyll", .run = config.generateNoJekyll },
        .{ .name = "Write Jekyll Config", .run = config.generateJekyllConfig },
        .{ .name = "Write Documentation Layout", .run = config.generateGitHubPagesLayout },
        .{ .name = "Write Navigation Data", .run = config.generateNavigationData },
        .{ .name = "Write SEO Metadata", .run = config.generateSEOMetadata },
        .{ .name = "Generate Module Docs", .run = module_docs.generateModuleDocs },
        .{ .name = "Generate API Reference", .run = api_reference.generateApiReference },
        .{ .name = "Generate Examples", .run = examples.generateExamples },
        .{ .name = "Generate Performance Guide", .run = performance_guide.generatePerformanceGuide },
        .{ .name = "Generate Definitions Reference", .run = definitions_reference.generateDefinitionsReference },
        .{ .name = "Generate Code API Index", .run = code_index.generateCodeApiIndex },
        .{ .name = "Generate Search Index", .run = search_index.generateSearchIndex },
        .{ .name = "Write GitHub Pages Assets", .run = config.generateGitHubPagesAssets },
        .{ .name = "Write Docs Index", .run = docs_index.generateDocsIndexHtml },
        .{ .name = "Write README Redirect", .run = readme_redirect.generateReadmeRedirect },
        .{ .name = "Write GitHub Actions Workflow", .run = config.generateGitHubActionsWorkflow },
        .{ .name = "Generate Zig Native Docs", .run = native_docs.generateZigNativeDocs },
    };

    const plan = planner.GenerationPlan.init(allocator, &steps);
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
