//! LSP request methods for the ZLS client.
//!
//! Provides hover, completion, definition, references, rename,
//! formatting, and diagnostics requests.

const std = @import("std");
const types = @import("../types.zig");
const transport = @import("transport.zig");

pub const Response = transport.Response;

/// Mixin providing LSP request methods. Expects the parent type to have:
///   - `allocator: std.mem.Allocator`
///   - `fn requestRaw(self, method, params_json) !Response`
pub fn RequestsMixin(comptime Self: type) type {
    return struct {
        pub fn hover(self: *Self, uri: []const u8, pos: types.Position) !Response {
            return self.positionRequest("textDocument/hover", uri, pos);
        }

        pub fn completion(self: *Self, uri: []const u8, pos: types.Position) !Response {
            return self.positionRequest("textDocument/completion", uri, pos);
        }

        pub fn definition(self: *Self, uri: []const u8, pos: types.Position) !Response {
            return self.positionRequest("textDocument/definition", uri, pos);
        }

        pub fn references(self: *Self, uri: []const u8, pos: types.Position, include_decl: bool) !Response {
            const params = types.ReferencesParams{
                .textDocument = .{ .uri = uri },
                .position = pos,
                .context = .{ .includeDeclaration = include_decl },
            };
            const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
            defer self.allocator.free(params_json);
            return self.requestRaw("textDocument/references", params_json);
        }

        pub fn rename(
            self: *Self,
            uri: []const u8,
            pos: types.Position,
            new_name: []const u8,
        ) !Response {
            const params = types.RenameParams{
                .textDocument = .{ .uri = uri },
                .position = pos,
                .newName = new_name,
            };
            const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
            defer self.allocator.free(params_json);
            return self.requestRaw("textDocument/rename", params_json);
        }

        pub fn formatting(
            self: *Self,
            uri: []const u8,
            options: types.FormattingOptions,
        ) !Response {
            const params = types.DocumentFormattingParams{
                .textDocument = .{ .uri = uri },
                .options = options,
            };
            const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
            defer self.allocator.free(params_json);
            return self.requestRaw("textDocument/formatting", params_json);
        }

        pub fn diagnostics(self: *Self, uri: []const u8) !Response {
            const params = struct {
                textDocument: types.TextDocumentIdentifier,
            }{
                .textDocument = .{ .uri = uri },
            };
            const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
            defer self.allocator.free(params_json);
            return self.requestRaw("textDocument/diagnostic", params_json);
        }

        fn positionRequest(self: *Self, method: []const u8, uri: []const u8, pos: types.Position) !Response {
            const params = types.TextDocumentPositionParams{
                .textDocument = .{ .uri = uri },
                .position = pos,
            };
            const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
            defer self.allocator.free(params_json);
            return self.requestRaw(method, params_json);
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
