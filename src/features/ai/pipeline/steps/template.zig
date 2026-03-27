//! Template Step — Render prompt with {variable} interpolation.
//!
//! Replaces {context} with joined fragments, {input} with user input,
//! and {key} with metadata values from PipelineContext.

const std = @import("std");
const types = @import("../types.zig");
const ctx_mod = @import("../context.zig");
const PipelineContext = ctx_mod.PipelineContext;

pub fn execute(pctx: *PipelineContext, cfg: types.TemplateConfig) !void {
    const template_str = cfg.template_str;
    var result = std.ArrayListUnmanaged(u8).empty;
    defer result.deinit(pctx.allocator);

    var i: usize = 0;
    while (i < template_str.len) {
        if (template_str[i] == '{') {
            // Find closing brace
            const end = std.mem.indexOfScalarPos(u8, template_str, i + 1, '}') orelse {
                try result.append(pctx.allocator, template_str[i]);
                i += 1;
                continue;
            };
            const var_name = template_str[i + 1 .. end];

            if (std.mem.eql(u8, var_name, "context")) {
                const joined = try pctx.joinFragments("\n");
                defer pctx.allocator.free(joined);
                try result.appendSlice(pctx.allocator, joined);
            } else if (std.mem.eql(u8, var_name, "input")) {
                try result.appendSlice(pctx.allocator, pctx.input);
            } else if (pctx.metadata.get(var_name)) |value| {
                try result.appendSlice(pctx.allocator, value);
            } else {
                // Keep unresolved variables as-is
                try result.appendSlice(pctx.allocator, template_str[i .. end + 1]);
            }
            i = end + 1;
        } else {
            try result.append(pctx.allocator, template_str[i]);
            i += 1;
        }
    }

    const rendered = try pctx.allocator.dupe(u8, result.items);
    if (pctx.rendered_prompt) |old| pctx.allocator.free(old);
    pctx.rendered_prompt = rendered;
}

test "template interpolation replaces input variable" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "Hello Abbey", "test-session", 1);
    defer pctx.deinit();

    try execute(&pctx, .{ .template_str = "You said: {input}" });

    try std.testing.expectEqualStrings("You said: Hello Abbey", pctx.rendered_prompt.?);
}

test "template interpolation replaces context variable" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "test-session", 2);
    defer pctx.deinit();

    try pctx.addFragment("fragment one");
    try pctx.addFragment("fragment two");

    try execute(&pctx, .{ .template_str = "Context: {context}" });

    try std.testing.expectEqualStrings("Context: fragment one\nfragment two", pctx.rendered_prompt.?);
}

test "template interpolation replaces metadata variable" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "test-session", 3);
    defer pctx.deinit();

    try pctx.setMetadata("name", "Abbey");

    try execute(&pctx, .{ .template_str = "Hello {name}!" });

    try std.testing.expectEqualStrings("Hello Abbey!", pctx.rendered_prompt.?);
}

test "template keeps unresolved variables as-is" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "test-session", 4);
    defer pctx.deinit();

    try execute(&pctx, .{ .template_str = "Hello {unknown}!" });

    try std.testing.expectEqualStrings("Hello {unknown}!", pctx.rendered_prompt.?);
}

test "template unclosed brace treated as literal" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "test-session", 5);
    defer pctx.deinit();

    try execute(&pctx, .{ .template_str = "broken {variable" });

    // Unclosed brace: '{' is emitted as literal, rest follows
    try std.testing.expectEqualStrings("broken {variable", pctx.rendered_prompt.?);
}

test "template empty string produces empty result" {
    const allocator = std.testing.allocator;
    var pctx = try PipelineContext.init(allocator, "hi", "test-session", 6);
    defer pctx.deinit();

    try execute(&pctx, .{ .template_str = "" });

    try std.testing.expectEqualStrings("", pctx.rendered_prompt.?);
}
