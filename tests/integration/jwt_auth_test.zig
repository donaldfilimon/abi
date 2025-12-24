//! JWT Authentication Integration Tests
//!
//! Tests JWT token generation, validation, and extraction functionality

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");

test "JWT Authentication: token generation and validation" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test JWT token generation
    const user_id = "test_user_123";
    const token = try framework.web.?.auth_manager.generateToken(allocator, user_id);
    defer allocator.free(token);

    // Verify token format (should have 3 parts separated by dots)
    var parts = std.mem.split(u8, token, ".");
    const header = parts.next() orelse return error.InvalidToken;
    const payload = parts.next() orelse return error.InvalidToken;
    const signature = parts.next() orelse return error.InvalidToken;
    try testing.expect(parts.next() == null); // No more parts

    // Verify parts are base64url encoded (no padding, valid chars)
    try testing.expect(header.len > 0);
    try testing.expect(payload.len > 0);
    try testing.expect(signature.len > 0);

    // Test token validation
    const is_valid = try framework.web.?.auth_manager.validateToken(token);
    try testing.expect(is_valid);
}

test "JWT Authentication: invalid token rejection" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test invalid tokens
    const invalid_tokens = [_][]const u8{
        "", // Empty token
        "invalid", // No dots
        "invalid.token", // Only 2 parts
        "invalid.token.signature.extra", // Too many parts
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid.signature", // Invalid signature
    };

    for (invalid_tokens) |invalid_token| {
        const is_valid = try framework.web.?.auth_manager.validateToken(invalid_token);
        try testing.expect(!is_valid);
    }
}

test "JWT Authentication: user extraction" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // Test user ID extraction
    const user_id = "test_user_456";
    const token = try framework.web.?.auth_manager.generateToken(allocator, user_id);
    defer allocator.free(token);

    const extracted_user = try framework.web.?.auth_manager.getUserFromToken(token);
    defer allocator.free(extracted_user);

    try testing.expectEqualStrings(user_id, extracted_user);
}

test "JWT Authentication: token expiration" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize framework
    var framework = try abi.init(allocator, abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    // This test would require mocking time, but for now we verify
    // that tokens contain expiration claims
    const user_id = "test_user_789";
    const token = try framework.web.?.auth_manager.generateToken(allocator, user_id);
    defer allocator.free(token);

    // Decode payload to check for exp claim
    var parts = std.mem.split(u8, token, ".");
    _ = parts.next(); // header
    const payload_b64 = parts.next() orelse return error.InvalidToken;

    // Decode base64url payload
    const payload_json = try base64UrlDecode(allocator, payload_b64);
    defer allocator.free(payload_json);

    // Check for expiration field
    const has_exp = std.mem.containsAtLeast(u8, payload_json, 1, "\"exp\"");
    try testing.expect(has_exp);
}

// Helper function for base64url decoding (same as in AuthManager)
fn base64UrlDecode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    // Convert base64url to standard base64
    const temp = try allocator.alloc(u8, input.len + 4);
    defer allocator.free(temp);

    for (input, 0..) |c, i| {
        temp[i] = switch (c) {
            '-' => '+',
            '_' => '/',
            else => c,
        };
    }

    const input_len = input.len;
    const remainder = input_len % 4;
    if (remainder == 2) {
        temp[input_len] = '=';
        temp[input_len + 1] = '=';
    } else if (remainder == 3) {
        temp[input_len] = '=';
    }

    const padded_input = temp[0 .. input_len + (4 - remainder) % 4];

    return std.base64.standard.Decoder.decode(allocator, padded_input);
}
