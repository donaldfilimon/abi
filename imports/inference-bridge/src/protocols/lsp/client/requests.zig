//! LSP request methods for the ZLS client.
//!
//! Provides hover, completion, definition, references, rename,
//! formatting, and diagnostics requests as free functions that
//! operate on any type with `allocator` and `requestRaw` fields.

const std = @import("std");
const types = @import("../types.zig");
const transport = @import("transport.zig");

pub const Response = transport.Response;

pub fn hover(client: anytype, uri: []const u8, pos: types.Position) !Response {
    return positionRequest(client, "textDocument/hover", uri, pos);
}

pub fn completion(client: anytype, uri: []const u8, pos: types.Position) !Response {
    return positionRequest(client, "textDocument/completion", uri, pos);
}

pub fn definition(client: anytype, uri: []const u8, pos: types.Position) !Response {
    return positionRequest(client, "textDocument/definition", uri, pos);
}

pub fn references(client: anytype, uri: []const u8, pos: types.Position, include_decl: bool) !Response {
    const params = types.ReferencesParams{
        .textDocument = .{ .uri = uri },
        .position = pos,
        .context = .{ .includeDeclaration = include_decl },
    };
    const params_json = try std.json.Stringify.valueAlloc(client.allocator, params, .{});
    defer client.allocator.free(params_json);
    return client.requestRaw("textDocument/references", params_json);
}

pub fn rename(
    client: anytype,
    uri: []const u8,
    pos: types.Position,
    new_name: []const u8,
) !Response {
    const params = types.RenameParams{
        .textDocument = .{ .uri = uri },
        .position = pos,
        .newName = new_name,
    };
    const params_json = try std.json.Stringify.valueAlloc(client.allocator, params, .{});
    defer client.allocator.free(params_json);
    return client.requestRaw("textDocument/rename", params_json);
}

pub fn formatting(
    client: anytype,
    uri: []const u8,
    options: types.FormattingOptions,
) !Response {
    const params = types.DocumentFormattingParams{
        .textDocument = .{ .uri = uri },
        .options = options,
    };
    const params_json = try std.json.Stringify.valueAlloc(client.allocator, params, .{});
    defer client.allocator.free(params_json);
    return client.requestRaw("textDocument/formatting", params_json);
}

pub fn diagnostics(client: anytype, uri: []const u8) !Response {
    const params = struct {
        textDocument: types.TextDocumentIdentifier,
    }{
        .textDocument = .{ .uri = uri },
    };
    const params_json = try std.json.Stringify.valueAlloc(client.allocator, params, .{});
    defer client.allocator.free(params_json);
    return client.requestRaw("textDocument/diagnostic", params_json);
}

fn positionRequest(client: anytype, method: []const u8, uri: []const u8, pos: types.Position) !Response {
    const params = types.TextDocumentPositionParams{
        .textDocument = .{ .uri = uri },
        .position = pos,
    };
    const params_json = try std.json.Stringify.valueAlloc(client.allocator, params, .{});
    defer client.allocator.free(params_json);
    return client.requestRaw(method, params_json);
}

test {
    std.testing.refAllDecls(@This());
}
