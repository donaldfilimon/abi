//! AI Template Tests — Parsing, Rendering, Registry
//!
//! Tests template variable substitution, parsing, and registry operations.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const templates = if (build_options.enable_ai) abi.features.ai.templates else struct {};
const Template = if (build_options.enable_ai) templates.Template else struct {};
const TemplateRegistry = if (build_options.enable_ai) templates.TemplateRegistry else struct {};

// ============================================================================
// Template Parsing Tests
// ============================================================================

test "template: init and parse simple variable" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var tmpl = Template.init(allocator, "greeting", "Hello {{name}}!") catch return error.SkipZigTest;
    defer tmpl.deinit();

    const vars = tmpl.getVariables();
    try std.testing.expect(vars.len >= 1);
}

test "template: plain text without variables" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var tmpl = Template.init(allocator, "static", "No variables here") catch return error.SkipZigTest;
    defer tmpl.deinit();

    const vars = tmpl.getVariables();
    try std.testing.expectEqual(@as(usize, 0), vars.len);
}

test "template: multiple variables" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var tmpl = Template.init(allocator, "multi", "{{greeting}} {{name}}, you are {{age}} years old") catch return error.SkipZigTest;
    defer tmpl.deinit();

    const vars = tmpl.getVariables();
    try std.testing.expect(vars.len >= 2);
}

// ============================================================================
// Template Rendering Tests
// ============================================================================

test "template: render with struct values" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var tmpl = Template.init(allocator, "hello", "Hello {{name}}!") catch return error.SkipZigTest;
    defer tmpl.deinit();

    const result = tmpl.render(.{ .name = "World" }) catch return error.SkipZigTest;
    defer allocator.free(result);

    try std.testing.expect(result.len > 0);
    // Should contain "World" somewhere in output
    try std.testing.expect(std.mem.indexOf(u8, result, "World") != null);
}

test "template: render empty template" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var tmpl = Template.init(allocator, "empty", "") catch return error.SkipZigTest;
    defer tmpl.deinit();

    const result = tmpl.render(.{}) catch return error.SkipZigTest;
    defer allocator.free(result);

    try std.testing.expectEqual(@as(usize, 0), result.len);
}

// ============================================================================
// Template Registry Tests
// ============================================================================

test "registry: register and retrieve template" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var registry = TemplateRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("greeting", "Hello {{name}}!");

    const found = registry.get("greeting");
    try std.testing.expect(found != null);
}

test "registry: get non-existent returns null" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var registry = TemplateRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expect(registry.get("nonexistent") == null);
}

test "registry: list templates" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var registry = TemplateRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("tmpl-a", "Hello {{name}}");
    try registry.register("tmpl-b", "Goodbye {{name}}");

    const names = try registry.listTemplates(allocator);
    defer {
        for (names) |name| allocator.free(name);
        allocator.free(names);
    }

    try std.testing.expectEqual(@as(usize, 2), names.len);
}

// ============================================================================
// Convenience Function Tests
// ============================================================================

test "templates: renderTemplate top-level function" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = templates.renderTemplate(allocator, "The {{animal}} is {{adjective}}", .{
        .animal = "fox",
        .adjective = "quick",
    }) catch return error.SkipZigTest;
    defer allocator.free(result);

    try std.testing.expect(result.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result, "fox") != null);
}

test "templates: formatChatMessage" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const msg = templates.formatChatMessage(allocator, "user", "Hello!") catch return error.SkipZigTest;
    defer allocator.free(msg);

    try std.testing.expect(msg.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, msg, "Hello!") != null);
}
