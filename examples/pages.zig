//! Pages Example
//!
//! Demonstrates the dashboard/UI pages module with URL routing,
//! template rendering, and path parameter extraction.
//!
//! Run with: `zig build run-pages`

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withPagesDefaults()
        .build();
    defer framework.deinit();

    if (!abi.pages.isEnabled()) {
        std.debug.print("Pages feature is disabled. Enable with -Denable-pages=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Pages Example ===\n\n", .{});

    // Register a static page
    abi.pages.addPage(.{
        .path = "/",
        .title = "Home",
        .content = .{ .static = "<h1>Welcome to ABI Framework</h1><p>Dashboard home page</p>" },
        .method = .GET,
    }) catch |err| {
        std.debug.print("Failed to add page: {t}\n", .{err});
        return;
    };

    // Register a template page with {{variable}} substitution
    abi.pages.addPage(.{
        .path = "/users/{id}",
        .title = "User Profile",
        .content = .{
            .template = .{
                .source = "<h1>Profile: {{username}}</h1><p>Role: {{role}}</p>",
                .default_vars = blk: {
                    var vars: [8]abi.pages.TemplateVar = [_]abi.pages.TemplateVar{.{}} ** 8;
                    vars[0] = .{ .key = "username", .value = "alice" };
                    vars[1] = .{ .key = "role", .value = "admin" };
                    break :blk vars;
                },
                .var_count = 2,
            },
        },
        .method = .GET,
        .require_auth = true,
    }) catch |err| {
        std.debug.print("Failed to add template page: {t}\n", .{err});
        return;
    };

    abi.pages.addPage(.{
        .path = "/about",
        .title = "About",
        .content = .{ .static = "<h1>About</h1><p>ABI Framework v0.4.0</p>" },
        .method = .GET,
        .cache_ttl_ms = 60000,
    }) catch |err| {
        std.debug.print("Failed to add page: {t}\n", .{err});
        return;
    };

    std.debug.print("Registered 3 pages\n\n", .{});

    // Match and render pages
    std.debug.print("Rendering pages:\n", .{});

    const test_urls = [_][]const u8{ "/", "/about", "/missing" };
    for (test_urls) |url| {
        const match = abi.pages.matchPage(url) catch null;
        if (match) |m| {
            var result = abi.pages.renderPage(allocator, m.page.path, &.{}) catch |err| {
                std.debug.print("  {s} -> render error: {t}\n", .{ url, err });
                continue;
            };
            defer result.deinit(allocator);

            std.debug.print("  {s} -> {s} ({} bytes", .{
                url, result.title, result.body.len,
            });
            if (m.param_count > 0) {
                std.debug.print(", params={{", .{});
                for (m.params[0..m.param_count], 0..) |p, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}={s}", .{ p.name, p.value });
                }
                std.debug.print("}}", .{});
            }
            std.debug.print(")\n", .{});
        } else {
            std.debug.print("  {s} -> 404 not found\n", .{url});
        }
    }

    // Stats
    const s = abi.pages.stats();
    std.debug.print("\nPages stats: {} registered, {} renders\n", .{
        s.total_pages, s.total_renders,
    });
}
