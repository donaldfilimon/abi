//! Integration Tests: LSP Protocol Logic
//!
//! Tests LSP protocol behavior beyond type availability: client lifecycle,
//! stub error handling, document operations, JSON-RPC framing, and
//! conditional behavior based on build_options.feat_lsp.
//!
//! Type availability and basic config tests live in lsp_test.zig.
//! This file focuses on protocol logic and error paths.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const lsp = abi.lsp;

// ============================================================================
// Feature gate consistency
// ============================================================================

test "lsp protocol: isEnabled and Context.isEnabled are consistent" {
    try std.testing.expectEqual(lsp.isEnabled(), lsp.Context.isEnabled());
}

// ============================================================================
// Compound LSP types (not covered in lsp_test.zig)
// ============================================================================

test "lsp protocol: DidOpenTextDocumentParams wraps TextDocumentItem" {
    const params = lsp.types.DidOpenTextDocumentParams{
        .textDocument = .{
            .uri = "file:///foo.zig",
            .text = "const x = 1;",
        },
    };
    try std.testing.expectEqualStrings("file:///foo.zig", params.textDocument.uri);
    try std.testing.expectEqualStrings("zig", params.textDocument.languageId);
}

test "lsp protocol: TextDocumentPositionParams combines doc and position" {
    const params = lsp.types.TextDocumentPositionParams{
        .textDocument = .{ .uri = "file:///bar.zig" },
        .position = .{ .line = 3, .character = 12 },
    };
    try std.testing.expectEqual(@as(u32, 3), params.position.line);
    try std.testing.expectEqualStrings("file:///bar.zig", params.textDocument.uri);
}

test "lsp protocol: ReferencesParams with includeDeclaration false" {
    const params = lsp.types.ReferencesParams{
        .textDocument = .{ .uri = "file:///ref.zig" },
        .position = .{ .line = 0, .character = 5 },
        .context = .{ .includeDeclaration = false },
    };
    try std.testing.expect(!params.context.includeDeclaration);
}

test "lsp protocol: ReferencesContext defaults includeDeclaration to true" {
    const ctx = lsp.types.ReferencesContext{};
    try std.testing.expect(ctx.includeDeclaration);
}

test "lsp protocol: RenameParams carries newName" {
    const params = lsp.types.RenameParams{
        .textDocument = .{ .uri = "file:///rename.zig" },
        .position = .{ .line = 1, .character = 4 },
        .newName = "better_name",
    };
    try std.testing.expectEqualStrings("better_name", params.newName);
}

test "lsp protocol: DocumentFormattingParams combines doc and options" {
    const params = lsp.types.DocumentFormattingParams{
        .textDocument = .{ .uri = "file:///fmt.zig" },
        .options = .{ .tabSize = 8 },
    };
    try std.testing.expectEqual(@as(u32, 8), params.options.tabSize);
}

test "lsp protocol: FormattingOptions custom values" {
    const opts = lsp.types.FormattingOptions{ .tabSize = 2, .insertSpaces = false };
    try std.testing.expectEqual(@as(u32, 2), opts.tabSize);
    try std.testing.expect(!opts.insertSpaces);
}

// ============================================================================
// Client init error handling (stub returns FeatureDisabled)
// ============================================================================

test "lsp protocol: Client.init returns error when feature disabled" {
    if (build_options.feat_lsp) return;
    const result = lsp.Client.init(std.testing.allocator, .{}, lsp.Config.defaults());
    try std.testing.expectError(error.FeatureDisabled, result);
}

// ============================================================================
// Stub client: all operations return FeatureDisabled
// ============================================================================

test "lsp protocol: stub didOpen returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.didOpen(.{
        .uri = "file:///test.zig",
        .text = "const x = 1;",
    }));
}

test "lsp protocol: stub hover returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.hover("file:///t.zig", .{ .line = 0, .character = 0 }));
}

test "lsp protocol: stub completion returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.completion("file:///t.zig", .{ .line = 0, .character = 0 }));
}

test "lsp protocol: stub definition returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.definition("file:///t.zig", .{ .line = 0, .character = 0 }));
}

test "lsp protocol: stub references returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.references("file:///t.zig", .{ .line = 0, .character = 0 }, true));
}

test "lsp protocol: stub rename returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.rename("file:///t.zig", .{ .line = 0, .character = 0 }, "new"));
}

test "lsp protocol: stub formatting returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.formatting("file:///t.zig", .{}));
}

test "lsp protocol: stub diagnostics returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.diagnostics("file:///t.zig"));
}

test "lsp protocol: stub requestRaw returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.requestRaw("textDocument/hover", null));
}

test "lsp protocol: stub notifyRaw returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.notifyRaw("initialized", null));
}

test "lsp protocol: stub waitForNotification returns FeatureDisabled" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, client.waitForNotification("textDocument/publishDiagnostics", 1000));
}

test "lsp protocol: stub workspaceRoot returns dot" {
    if (build_options.feat_lsp) return;
    var client: lsp.Client = undefined;
    client.allocator = std.testing.allocator;
    try std.testing.expectEqualStrings(".", client.workspaceRoot());
}

// ============================================================================
// Utility functions: resolveWorkspaceRoot, resolvePath, pathToUri
// ============================================================================

test "lsp protocol: resolveWorkspaceRoot returns error when disabled" {
    if (build_options.feat_lsp) return;
    try std.testing.expectError(
        error.FeatureDisabled,
        lsp.resolveWorkspaceRoot(std.testing.allocator, .{}, null),
    );
}

test "lsp protocol: resolvePath returns error when disabled" {
    if (build_options.feat_lsp) return;
    try std.testing.expectError(
        error.FeatureDisabled,
        lsp.resolvePath(std.testing.allocator, .{}, null, "src/main.zig"),
    );
}

test "lsp protocol: pathToUri returns error when disabled" {
    if (build_options.feat_lsp) return;
    try std.testing.expectError(
        error.FeatureDisabled,
        lsp.pathToUri(std.testing.allocator, "/home/user/project/src/main.zig"),
    );
}

// ============================================================================
// EnvConfig lifecycle
// ============================================================================

test "lsp protocol: EnvConfig deinit is safe to call immediately" {
    var env = try lsp.loadConfigFromEnv(std.testing.allocator);
    env.deinit();
}

// ============================================================================
// JSON-RPC framing
// ============================================================================

test "lsp protocol: jsonrpc.HeaderError variants exist" {
    const e1: lsp.jsonrpc.HeaderError = error.MissingContentLength;
    const e2: lsp.jsonrpc.HeaderError = error.InvalidContentLength;
    const e3: lsp.jsonrpc.HeaderError = error.PayloadTooLarge;
    _ = e1;
    _ = e2;
    _ = e3;
}

test "lsp protocol: jsonrpc writeMessage and readMessageAlloc round-trip" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const payload = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\"}";

    var buf: [512]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try lsp.jsonrpc.writeMessage(&writer, payload);

    var reader = std.Io.Reader.fixed(writer.buffered());
    const msg = try lsp.jsonrpc.readMessageAlloc(arena.allocator(), &reader, 4096);
    try std.testing.expect(msg != null);
    try std.testing.expectEqualStrings(payload, msg.?);
}

test "lsp protocol: jsonrpc round-trip with empty payload" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var buf: [128]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try lsp.jsonrpc.writeMessage(&writer, "");

    var reader = std.Io.Reader.fixed(writer.buffered());
    const msg = try lsp.jsonrpc.readMessageAlloc(arena.allocator(), &reader, 4096);
    try std.testing.expect(msg != null);
    try std.testing.expectEqual(@as(usize, 0), msg.?.len);
}

// ============================================================================
// Config customization (non-default values not tested in lsp_test.zig)
// ============================================================================

test "lsp protocol: Config with all optional fields set" {
    const config = lsp.Config{
        .zls_path = "/opt/zls/bin/zls",
        .zig_exe_path = "/opt/zig/zig",
        .workspace_root = "/home/user/project",
        .log_level = "debug",
        .enable_snippets = false,
    };
    try std.testing.expectEqualStrings("/opt/zls/bin/zls", config.zls_path);
    try std.testing.expectEqualStrings("/opt/zig/zig", config.zig_exe_path.?);
    try std.testing.expectEqualStrings("/home/user/project", config.workspace_root.?);
    try std.testing.expectEqualStrings("debug", config.log_level);
    try std.testing.expect(!config.enable_snippets);
}

test {
    std.testing.refAllDecls(@This());
}
