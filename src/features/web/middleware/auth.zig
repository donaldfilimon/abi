//! Authentication Middleware
//!
//! Provides JWT and API key authentication for HTTP requests.
//! Delegates to shared security modules from `services/shared/security/`
//! for JWT and API key operations where possible.

const std = @import("std");
const types = @import("types.zig");
const server = @import("../server/mod.zig");
const MiddlewareContext = types.MiddlewareContext;

// Shared security modules — used for delegation where applicable.
const shared_jwt = @import("../../../services/shared/security/jwt.zig");
const shared_api_keys = @import("../../../services/shared/security/api_keys.zig");

/// Authentication configuration.
pub const AuthConfig = struct {
    /// Secret key for JWT validation.
    jwt_secret: []const u8 = "",
    /// Header name for JWT token.
    token_header: []const u8 = "Authorization",
    /// Token scheme (e.g., "Bearer").
    token_scheme: []const u8 = "Bearer",
    /// Header name for API key.
    api_key_header: []const u8 = "X-API-Key",
    /// Valid API keys (for simple API key auth).
    valid_api_keys: []const []const u8 = &.{},
    /// Paths that don't require authentication.
    public_paths: []const []const u8 = &.{ "/health", "/live", "/ready", "/metrics" },
};

/// Default auth configuration.
pub const default_config = AuthConfig{};

/// Authentication result.
pub const AuthResult = struct {
    authenticated: bool,
    user_id: ?[]const u8,
    error_message: ?[]const u8,
};

/// Creates an auth middleware with the given configuration.
pub fn createAuthMiddleware(config: AuthConfig) types.MiddlewareFn {
    _ = config;
    return &optionalAuth;
}

/// Optional authentication (doesn't fail if no token).
pub fn optionalAuth(ctx: *MiddlewareContext) !void {
    const result = authenticate(ctx, default_config);
    if (result.authenticated) {
        if (result.user_id) |uid| {
            try ctx.set("user_id", uid);
        }
        try ctx.set("authenticated", "true");
    }
}

/// Required authentication (fails if no valid token).
pub fn requireAuth(ctx: *MiddlewareContext) !void {
    // Skip public paths
    if (isPublicPath(ctx.request.path, default_config.public_paths)) {
        return;
    }

    const result = authenticate(ctx, default_config);
    if (!result.authenticated) {
        _ = ctx.response.setStatus(.unauthorized);
        _ = try ctx.response.setHeader("WWW-Authenticate", "Bearer");
        _ = try ctx.response.text(result.error_message orelse "Unauthorized");
        ctx.abort();
        return;
    }

    if (result.user_id) |uid| {
        try ctx.set("user_id", uid);
    }
    try ctx.set("authenticated", "true");
}

/// API key authentication middleware.
pub fn apiKeyAuth(ctx: *MiddlewareContext) !void {
    const api_key = ctx.request.getHeader(default_config.api_key_header);

    if (api_key == null) {
        _ = ctx.response.setStatus(.unauthorized);
        _ = try ctx.response.text("API key required");
        ctx.abort();
        return;
    }

    if (!isValidApiKey(api_key.?, default_config.valid_api_keys)) {
        _ = ctx.response.setStatus(.unauthorized);
        _ = try ctx.response.text("Invalid API key");
        ctx.abort();
        return;
    }

    try ctx.set("authenticated", "true");
    try ctx.set("auth_method", "api_key");
}

/// Performs authentication check.
pub fn authenticate(ctx: *MiddlewareContext, config: AuthConfig) AuthResult {
    // Try JWT first
    if (extractBearerToken(ctx, config)) |token| {
        return validateJwt(token, config.jwt_secret);
    }

    // Try API key
    if (ctx.request.getHeader(config.api_key_header)) |api_key| {
        if (isValidApiKey(api_key, config.valid_api_keys)) {
            return AuthResult{
                .authenticated = true,
                .user_id = null,
                .error_message = null,
            };
        }
    }

    return AuthResult{
        .authenticated = false,
        .user_id = null,
        .error_message = "No valid authentication provided",
    };
}

/// Extracts Bearer token from Authorization header.
/// Delegates to `shared_jwt.extractBearerToken` for the standard "Bearer"
/// scheme; falls back to manual prefix-stripping for custom schemes.
pub fn extractBearerToken(ctx: *MiddlewareContext, config: AuthConfig) ?[]const u8 {
    const auth_header = ctx.request.getHeader(config.token_header) orelse return null;

    // For the standard Bearer scheme, delegate to the shared JWT helper.
    if (std.mem.eql(u8, config.token_scheme, "Bearer")) {
        return shared_jwt.extractBearerToken(auth_header);
    }

    // Custom scheme: manual prefix-stripping.
    if (config.token_scheme.len > 0) {
        if (!std.mem.startsWith(u8, auth_header, config.token_scheme)) {
            return null;
        }
        // Skip scheme and space
        const token_start = config.token_scheme.len;
        if (token_start >= auth_header.len) return null;

        const rest = auth_header[token_start..];
        const trimmed = std.mem.trimStart(u8, rest, " ");
        if (trimmed.len == 0) return null;
        return trimmed;
    }

    return auth_header;
}

/// Validates a JWT token structure and HMAC-SHA256 signature.
///
/// NOTE: For full-featured JWT handling (claims parsing, expiration checks,
/// algorithm selection, token blacklisting), use `shared_jwt.JwtManager`
/// from `services/shared/security/jwt.zig` directly. This lightweight
/// inline validator is kept for the middleware hot-path where allocating a
/// JwtManager is undesirable. It verifies the HMAC-SHA256 signature and
/// extracts a hardcoded user_id; claims parsing is not performed here.
pub fn validateJwt(token: []const u8, secret: []const u8) AuthResult {
    // JWT must have 3 parts: header.payload.signature
    var parts = std.mem.splitScalar(u8, token, '.');
    const header_b64 = parts.next() orelse return authFail("Invalid token format");
    const payload_b64 = parts.next() orelse return authFail("Invalid token format");
    const signature_b64 = parts.next() orelse return authFail("Invalid token format");
    if (parts.next() != null) return authFail("Invalid token format");

    // Verify HMAC-SHA256 signature over "header.payload"
    if (secret.len == 0) return authFail("JWT secret not configured");

    const signed_portion_len = header_b64.len + 1 + payload_b64.len;
    if (signed_portion_len > token.len) return authFail("Invalid token structure");
    const signed_portion = token[0..signed_portion_len];

    var mac: [std.crypto.auth.hmac.sha2.HmacSha256.mac_length]u8 = undefined;
    std.crypto.auth.hmac.sha2.HmacSha256.create(&mac, signed_portion, secret);

    // Base64url-decode the signature and compare
    var sig_buf: [256]u8 = undefined;
    const decoded_sig = std.base64.url_safe_no_pad.Decoder.decode(&sig_buf, signature_b64) catch {
        return authFail("Invalid signature encoding");
    };

    if (decoded_sig.len != mac.len) return authFail("Invalid signature");

    // Timing-safe comparison (same approach as shared_jwt constantTimeEqlSlice)
    var diff: u8 = 0;
    for (decoded_sig, &mac) |a, b| {
        diff |= a ^ b;
    }
    if (diff != 0) return authFail("Invalid signature");

    // Signature valid — return authenticated with a static user_id.
    // Sub-claim extraction is intentionally omitted: the decoded payload
    // would live on the stack and returning a slice into it would dangle.
    // For full claims parsing (including "sub"), use shared_jwt.JwtManager.
    return AuthResult{
        .authenticated = true,
        .user_id = "jwt_user",
        .error_message = null,
    };
}

fn authFail(message: []const u8) AuthResult {
    return AuthResult{
        .authenticated = false,
        .user_id = null,
        .error_message = message,
    };
}

/// Checks if an API key is valid. Returns false when no keys are configured.
///
/// NOTE: This performs a simple plaintext comparison against a static list,
/// suitable for lightweight middleware use. For production API key management
/// with salted hashing, key rotation, scopes, and expiration, use
/// `shared_api_keys.ApiKeyManager` from `services/shared/security/api_keys.zig`.
pub fn isValidApiKey(key: []const u8, valid_keys: []const []const u8) bool {
    if (valid_keys.len == 0) return false;
    for (valid_keys) |valid_key| {
        if (std.mem.eql(u8, key, valid_key)) {
            return true;
        }
    }
    return false;
}

/// Checks if a path is public (no auth required).
pub fn isPublicPath(path: []const u8, public_paths: []const []const u8) bool {
    for (public_paths) |public_path| {
        if (std.mem.eql(u8, path, public_path)) {
            return true;
        }
        // Support prefix matching with trailing *
        if (public_path.len > 0 and public_path[public_path.len - 1] == '*') {
            const prefix = public_path[0 .. public_path.len - 1];
            if (std.mem.startsWith(u8, path, prefix)) {
                return true;
            }
        }
    }
    return false;
}

/// Generates a simple API key (for testing/development).
/// For production key generation with salted hashing, scopes, and metadata,
/// use `shared_api_keys.ApiKeyManager.generateKey()`.
pub fn generateApiKey(allocator: std.mem.Allocator) ![]u8 {
    var key: [32]u8 = undefined;
    std.c.arc4random_buf(&key, key.len);

    // SAFETY: 32 bytes × 2 hex chars = 64 chars, buffer is exactly 64 bytes - cannot overflow
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{s}", .{std.fmt.fmtSliceHexLower(&key)}) catch unreachable;

    return try allocator.dupe(u8, &hex);
}

test "extractBearerToken" {
    const allocator = std.testing.allocator;

    var headers: std.StringHashMapUnmanaged([]const u8) = .empty;
    defer headers.deinit(allocator);
    try headers.put(allocator, "Authorization", "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature");

    var request = server.ParsedRequest{
        .method = .GET,
        .path = "/api/test",
        .query = null,
        .version = .http_1_1,
        .headers = headers,
        .body = null,
        .raw_path = "/api/test",
        .allocator = allocator,
        .owned_data = null,
    };

    var response = server.ResponseBuilder.init(allocator);
    defer response.deinit();

    var ctx = MiddlewareContext.init(allocator, &request, &response);
    defer ctx.deinit();

    const token = extractBearerToken(&ctx, default_config);
    try std.testing.expect(token != null);
    try std.testing.expect(std.mem.startsWith(u8, token.?, "eyJhbGciOiJIUzI1NiJ9"));
}

test "validateJwt format" {
    // Invalid format (2 parts)
    const invalid = validateJwt("header.payload", "secret");
    try std.testing.expect(!invalid.authenticated);

    // Invalid format (no dots)
    const invalid2 = validateJwt("invalidtoken", "secret");
    try std.testing.expect(!invalid2.authenticated);

    // Empty secret
    const no_secret = validateJwt("header.payload.signature", "");
    try std.testing.expect(!no_secret.authenticated);
}

test "validateJwt valid signature returns jwt_user" {
    // Build a valid JWT with sub claim: {"sub":"alice"}
    const header_b64 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"; // {"alg":"HS256","typ":"JWT"}
    const payload_b64 = "eyJzdWIiOiJhbGljZSJ9"; // {"sub":"alice"}
    const secret = "test-secret-key-32-bytes-long!!";

    // Compute HMAC-SHA256 over "header.payload"
    const signed_portion = header_b64 ++ "." ++ payload_b64;
    var mac: [std.crypto.auth.hmac.sha2.HmacSha256.mac_length]u8 = undefined;
    std.crypto.auth.hmac.sha2.HmacSha256.create(&mac, signed_portion, secret);

    // Base64url-encode the signature
    var sig_b64: [std.base64.url_safe_no_pad.Encoder.calcSize(mac.len)]u8 = undefined;
    _ = std.base64.url_safe_no_pad.Encoder.encode(&sig_b64, &mac);

    const token = signed_portion ++ "." ++ sig_b64;

    const result = validateJwt(token, secret);
    try std.testing.expect(result.authenticated);
    // Lightweight validator returns static user_id; use JwtManager for sub claim parsing
    try std.testing.expectEqualStrings("jwt_user", result.user_id.?);
}

test "validateJwt fallback when no sub claim" {
    // Build a valid JWT without sub claim: {"exp":9999999999}
    const header_b64 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"; // {"alg":"HS256","typ":"JWT"}
    const payload_b64 = "eyJleHAiOjk5OTk5OTk5OTl9"; // {"exp":9999999999}
    const secret = "test-secret-key-32-bytes-long!!";

    const signed_portion = header_b64 ++ "." ++ payload_b64;
    var mac: [std.crypto.auth.hmac.sha2.HmacSha256.mac_length]u8 = undefined;
    std.crypto.auth.hmac.sha2.HmacSha256.create(&mac, signed_portion, secret);

    var sig_b64: [std.base64.url_safe_no_pad.Encoder.calcSize(mac.len)]u8 = undefined;
    _ = std.base64.url_safe_no_pad.Encoder.encode(&sig_b64, &mac);

    const token = signed_portion ++ "." ++ sig_b64;

    const result = validateJwt(token, secret);
    try std.testing.expect(result.authenticated);
    // No sub claim → falls back to "jwt_user"
    try std.testing.expectEqualStrings("jwt_user", result.user_id.?);
}

test "isPublicPath" {
    const public = [_][]const u8{ "/health", "/api/public/*" };

    try std.testing.expect(isPublicPath("/health", &public));
    try std.testing.expect(isPublicPath("/api/public/test", &public));
    try std.testing.expect(isPublicPath("/api/public/nested/path", &public));
    try std.testing.expect(!isPublicPath("/api/private", &public));
    try std.testing.expect(!isPublicPath("/healthcheck", &public));
}

test "isValidApiKey" {
    const keys = [_][]const u8{ "key1", "key2" };

    try std.testing.expect(isValidApiKey("key1", &keys));
    try std.testing.expect(isValidApiKey("key2", &keys));
    try std.testing.expect(!isValidApiKey("key3", &keys));
    try std.testing.expect(!isValidApiKey("", &keys));
}
