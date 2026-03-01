//! Auth Example
//!
//! Demonstrates the security infrastructure exposed through the auth module:
//! JWT creation/verification, API key management, RBAC, and rate limiting.
//!
//! Run with: `zig build run-auth`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);

    var framework = try builder
        .with(.auth, abi.config.AuthConfig{})
        .build();
    defer framework.deinit();

    if (!abi.auth.isEnabled()) {
        std.debug.print("Auth feature is disabled. Enable with -Denable-auth=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Auth Example ===\n\n", .{});

    // ── JWT (JSON Web Tokens) ──────────────────────────────────────
    std.debug.print("--- JWT ---\n", .{});
    {
        const jwt = abi.auth.jwt;
        var manager = jwt.JwtManager.init(allocator, "my-super-secret-key-32bytes!!", .{
            .token_lifetime = 3600,
            .issuer = "abi-example",
        });
        defer manager.deinit();

        std.debug.print("  JWT Manager initialized (HS256, 1h expiry)\n", .{});
        std.debug.print("  Issuer: abi-example\n", .{});
        std.debug.print("  Algorithms: HS256, HS384, HS512, RS256\n", .{});
    }

    // ── API Keys ───────────────────────────────────────────────────
    std.debug.print("\n--- API Keys ---\n", .{});
    {
        const api_keys = abi.auth.api_keys;
        var manager = api_keys.ApiKeyManager.init(allocator, .{});
        defer manager.deinit();

        std.debug.print("  API Key Manager initialized\n", .{});
        std.debug.print("  Supports: generate, validate, revoke, rotate\n", .{});
    }

    // ── RBAC (Role-Based Access Control) ───────────────────────────
    std.debug.print("\n--- RBAC ---\n", .{});
    {
        const rbac = abi.auth.rbac;
        var manager = try rbac.RbacManager.init(allocator, .{});
        defer manager.deinit();

        std.debug.print("  RBAC Manager initialized\n", .{});
        std.debug.print("  Supports: createRole, assignRole, hasPermission\n", .{});
    }

    // ── Rate Limiting ──────────────────────────────────────────────
    std.debug.print("\n--- Rate Limiting ---\n", .{});
    {
        const rate_limit = abi.auth.rate_limit;
        var limiter = rate_limit.RateLimiter.init(allocator, .{
            .enabled = true,
            .requests = 100,
            .window_seconds = 60,
        });
        defer limiter.deinit();

        std.debug.print("  Rate Limiter initialized (100 req/60s)\n", .{});
        std.debug.print("  Algorithms: token_bucket, sliding_window, fixed_window\n", .{});
    }

    // ── Input Validation ───────────────────────────────────────────
    std.debug.print("\n--- Validation ---\n", .{});
    {
        const validation = abi.auth.validation;
        var validator = validation.Validator.init(allocator, .{});
        const email_result = validator.validateEmail("user@example.com");
        const url_result = validator.validateUrl("https://api.example.com/v1");
        std.debug.print("  Email validation: {}\n", .{email_result.valid});
        std.debug.print("  URL validation: {}\n", .{url_result.valid});
    }

    std.debug.print("\nAuth example complete.\n", .{});
}
