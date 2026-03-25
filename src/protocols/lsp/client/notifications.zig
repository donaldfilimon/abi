//! LSP notification methods for the ZLS client.
//!
//! Provides didOpen, didClose, and didChange notifications.

const std = @import("std");
const types = @import("../types.zig");
const transport = @import("transport.zig");

/// Mixin providing LSP notification methods. Expects the parent type to have:
///   - `allocator: std.mem.Allocator`
///   - `fn notifyRaw(self, method, params_json) !void`
pub fn NotificationsMixin(comptime Self: type) type {
    return struct {
        pub fn didOpen(self: *Self, doc: types.TextDocumentItem) !void {
            const params = types.DidOpenTextDocumentParams{ .textDocument = doc };
            const params_json = try std.json.Stringify.valueAlloc(self.allocator, params, .{});
            defer self.allocator.free(params_json);
            try self.notifyRaw("textDocument/didOpen", params_json);
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
