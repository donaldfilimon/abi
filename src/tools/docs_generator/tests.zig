const std = @import("std");
const testing = std.testing;

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
const planner = @import("planner.zig");

const TempEnv = struct {
    tmp: testing.TmpDir,
    original_path: [std.fs.max_path_bytes]u8,
    original_len: usize,

    fn cleanup(self: *TempEnv) void {
        std.process.changeCurDir(self.original_path[0..self.original_len]) catch {};
        self.tmp.cleanup();
    }
};

fn setupTempEnv() !TempEnv {
    const tmp = testing.tmpDir(.{});
    var original_buf: [std.fs.max_path_bytes]u8 = undefined;
    const original = try std.fs.cwd().realpath(".", &original_buf);

    var env = TempEnv{
        .tmp = tmp,
        .original_path = undefined,
        .original_len = original.len,
    };
    std.mem.copyForwards(u8, env.original_path[0..original.len], original);

    var tmp_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path = try env.tmp.dir.realpath(".", &tmp_buf);
    try std.process.changeCurDir(tmp_path);

    return env;
}

fn ensureBaseLayout() !void {
    try std.fs.cwd().makePath("docs/generated");
    try std.fs.cwd().makePath("docs/assets/css");
    try std.fs.cwd().makePath("docs/assets/js");
    try std.fs.cwd().makePath("docs/_layouts");
    try std.fs.cwd().makePath("docs/_data");
    try std.fs.cwd().makePath(".github/workflows");
}

fn writeFile(path: []const u8, contents: []const u8) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(contents);
}

fn readFileAlloc(path: []const u8) ![]u8 {
    return try std.fs.cwd().readFileAlloc(testing.allocator, path, 1 << 20);
}

test "module docs generator emits header" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try module_docs.generateModuleDocs(testing.allocator);
    const data = try readFileAlloc("docs/generated/MODULE_REFERENCE.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "# ABI Module Reference"));
}

test "api reference generator emits section" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try api_reference.generateApiReference(testing.allocator);
    const data = try readFileAlloc("docs/generated/API_REFERENCE.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "## 🗄️ Database API"));
}

test "examples generator emits quick start" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try examples.generateExamples(testing.allocator);
    const data = try readFileAlloc("docs/generated/EXAMPLES.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "Quick Start"));
}

test "performance guide generator emits optimization section" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try performance_guide.generatePerformanceGuide(testing.allocator);
    const data = try readFileAlloc("docs/generated/PERFORMANCE_GUIDE.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "## ⚡ Optimization Strategies"));
}

test "definitions reference generator emits glossary" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try definitions_reference.generateDefinitionsReference(testing.allocator);
    const data = try readFileAlloc("docs/generated/DEFINITIONS_REFERENCE.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "## 📊 Quick Reference Index"));
}

test "code index generator scans sample source" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();
    try std.fs.cwd().makePath("src");
    try writeFile("src/sample.zig", "pub fn sample() void {}\n");

    try code_index.generateCodeApiIndex(testing.allocator);
    const data = try readFileAlloc("docs/generated/CODE_API_INDEX.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "# Code API Index"));
}

test "search index generator emits json" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();
    try writeFile("docs/generated/SAMPLE.md", "# Sample\n\nSome docs content.\n");

    try search_index.generateSearchIndex(testing.allocator);
    const data = try readFileAlloc("docs/generated/search_index.json");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "\"file\""));
}

test "docs index generator emits html" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try docs_index.generateDocsIndexHtml(testing.allocator);
    const data = try readFileAlloc("docs/index.html");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "<!DOCTYPE html>"));
}

test "readme redirect generator emits landing page" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try readme_redirect.generateReadmeRedirect(testing.allocator);
    const data = try readFileAlloc("docs/README.md");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "# ABI Documentation"));
}

test "native docs generator writes output" {
    var env = try setupTempEnv();
    defer env.cleanup();
    try ensureBaseLayout();

    try native_docs.generateZigNativeDocs(testing.allocator);
    const data = try readFileAlloc("docs/zig-docs/index.html");
    defer testing.allocator.free(data);
    try testing.expect(std.mem.containsAtLeast(u8, data, 1, "Zig Native Docs"));
}

test "planner default plan orders steps by category" {
    const plan = planner.buildDefaultPlan(testing.allocator);
    const expected_names = [_][]const u8{
        "Disable Jekyll",
        "Write Jekyll Config",
        "Write Documentation Layout",
        "Write Navigation Data",
        "Write SEO Metadata",
        "Generate Module Docs",
        "Generate API Reference",
        "Generate Examples",
        "Generate Performance Guide",
        "Generate Definitions Reference",
        "Generate Code API Index",
        "Generate Search Index",
        "Write GitHub Pages Assets",
        "Write Docs Index",
        "Write README Redirect",
        "Write GitHub Actions Workflow",
        "Generate Zig Native Docs",
    };

    try testing.expectEqual(@as(usize, expected_names.len), plan.steps.len);

    for (expected_names, plan.steps) |expected_name, step| {
        try testing.expect(std.mem.eql(u8, expected_name, step.name));
    }

    try testing.expect(plan.steps[0].category == planner.StepCategory.configuration);
    try testing.expect(plan.steps[5].category == planner.StepCategory.content);
    try testing.expect(plan.steps[12].category == planner.StepCategory.assets);
    try testing.expect(plan.steps[15].category == planner.StepCategory.workflow);
    try testing.expect(plan.steps[16].category == planner.StepCategory.native_docs);
}
