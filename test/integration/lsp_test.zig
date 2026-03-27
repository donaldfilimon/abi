//! Integration Tests: LSP Feature
//!
//! Tests the LSP module exports, configuration, client type,
//! and protocol type accessibility through the public `abi.lsp` surface.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const lsp = abi.lsp;

// ============================================================================
// Feature gate
// ============================================================================

test "lsp: isEnabled reflects feature flag" {
    if (build_options.feat_lsp) {
        try std.testing.expect(lsp.isEnabled());
    } else {
        try std.testing.expect(!lsp.isEnabled());
    }
}

// ============================================================================
// Types
// ============================================================================

test "lsp: Position type is accessible" {
    const pos = lsp.types.Position{ .line = 10, .character = 5 };
    try std.testing.expectEqual(@as(u32, 10), pos.line);
    try std.testing.expectEqual(@as(u32, 5), pos.character);
}

test "lsp: Range type is accessible" {
    const range = lsp.types.Range{
        .start = .{ .line = 1, .character = 0 },
        .end = .{ .line = 1, .character = 10 },
    };
    try std.testing.expectEqual(@as(u32, 1), range.start.line);
    try std.testing.expectEqual(@as(u32, 10), range.end.character);
}

test "lsp: TextDocumentItem type is accessible" {
    const item = lsp.types.TextDocumentItem{
        .uri = "file:///test.zig",
        .languageId = "zig",
        .version = 1,
        .text = "const x = 42;",
    };
    try std.testing.expectEqualStrings("file:///test.zig", item.uri);
    try std.testing.expectEqualStrings("zig", item.languageId);
}

test "lsp: FormattingOptions type is accessible" {
    const opts = lsp.types.FormattingOptions{};
    try std.testing.expectEqual(@as(u32, 4), opts.tabSize);
    try std.testing.expect(opts.insertSpaces);
}

test "lsp: TextDocumentIdentifier type is accessible" {
    const tdi = lsp.types.TextDocumentIdentifier{ .uri = "file:///foo.zig" };
    try std.testing.expectEqualStrings("file:///foo.zig", tdi.uri);
}

// ============================================================================
// Config
// ============================================================================

test "lsp: Config defaults" {
    const config = lsp.Config.defaults();
    try std.testing.expectEqualStrings("zls", config.zls_path);
    try std.testing.expect(config.zig_exe_path == null);
    try std.testing.expect(config.workspace_root == null);
    try std.testing.expectEqualStrings("info", config.log_level);
    try std.testing.expect(config.enable_snippets);
}

test "lsp: Config with custom values" {
    const config = lsp.Config{
        .zls_path = "/usr/bin/zls",
        .log_level = "debug",
        .enable_snippets = false,
    };
    try std.testing.expectEqualStrings("/usr/bin/zls", config.zls_path);
    try std.testing.expectEqualStrings("debug", config.log_level);
    try std.testing.expect(!config.enable_snippets);
}

// ============================================================================
// Response type
// ============================================================================

test "lsp: Response type is accessible" {
    const R = lsp.Response;
    _ = R;
}

// ============================================================================
// Client type
// ============================================================================

test "lsp: Client type is accessible" {
    const C = lsp.Client;
    _ = C;
}

// ============================================================================
// Env config
// ============================================================================

test "lsp: loadConfigFromEnv returns EnvConfig" {
    var env = try lsp.loadConfigFromEnv(std.testing.allocator);
    defer env.deinit();
    try std.testing.expectEqualStrings("zls", env.config.zls_path);
}

// ============================================================================
// Sub-modules
// ============================================================================

test "lsp: jsonrpc sub-module is accessible" {
    const j = lsp.jsonrpc;
    _ = j;
}

test "lsp: types sub-module is accessible" {
    const t = lsp.types;
    _ = t;
}

// Sibling test modules (pulled in via refAllDecls)
const _protocol = @import("lsp_protocol_test.zig");

test {
    std.testing.refAllDecls(@This());
}
