const std = @import("std");

pub const AuthHeader = struct {
    value: []u8,

    pub fn header(self: *const AuthHeader) std.http.Header {
        return .{ .name = "authorization", .value = self.value };
    }

    pub fn deinit(self: *AuthHeader, allocator: std.mem.Allocator) void {
        // Securely wipe auth token before freeing to prevent memory forensics.
        std.crypto.secureZero(u8, self.value);
        allocator.free(self.value);
        self.* = undefined;
    }
};

pub fn buildBearerHeader(allocator: std.mem.Allocator, token: []const u8) !AuthHeader {
    const value = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
    return .{ .value = value };
}

test "buildBearerHeader formats correctly" {
    var auth = try buildBearerHeader(std.testing.allocator, "test-token-123");
    defer auth.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("Bearer test-token-123", auth.value);
}

test "AuthHeader.header returns HTTP header" {
    var auth = try buildBearerHeader(std.testing.allocator, "sk-abc");
    defer auth.deinit(std.testing.allocator);
    const hdr = auth.header();
    try std.testing.expectEqualStrings("authorization", hdr.name);
    try std.testing.expectEqualStrings("Bearer sk-abc", hdr.value);
}

test {
    std.testing.refAllDecls(@This());
}
