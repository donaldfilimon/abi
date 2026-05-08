//! LSP notification methods for the ZLS client.
//!
//! Provides didOpen and other notification helpers as free functions.

const std = @import("std");
const types = @import("../types.zig");

pub fn didOpen(client: anytype, doc: types.TextDocumentItem) !void {
    const params = types.DidOpenTextDocumentParams{ .textDocument = doc };
    const params_json = try std.json.Stringify.valueAlloc(client.allocator, params, .{});
    defer client.allocator.free(params_json);
    try client.notifyRaw("textDocument/didOpen", params_json);
}

test {
    std.testing.refAllDecls(@This());
}
