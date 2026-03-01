//! API Gateway Example
//!
//! Demonstrates the API gateway with radix-tree routing,
//! rate limiting, and circuit breaker patterns.
//!
//! Run with: `zig build run-gateway`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);

    var framework = try builder
        .with(.gateway, .{})
        .build();
    defer framework.deinit();

    if (!abi.gateway.isEnabled()) {
        std.debug.print("Gateway feature is disabled. Enable with -Denable-gateway=true\n", .{});
        return;
    }

    std.debug.print("=== ABI API Gateway Example ===\n\n", .{});

    // Register routes
    const routes = [_]abi.gateway.Route{
        .{ .path = "/api/users", .method = .GET, .upstream = "user-service" },
        .{ .path = "/api/users/{id}", .method = .GET, .upstream = "user-service" },
        .{ .path = "/api/search", .method = .POST, .upstream = "search-service" },
        .{ .path = "/health", .method = .GET, .upstream = "health-check" },
    };

    for (routes) |route| {
        abi.gateway.addRoute(route) catch |err| {
            std.debug.print("Failed to add route {s}: {t}\n", .{ route.path, err });
            continue;
        };
        std.debug.print("Registered: {t} {s} -> {s}\n", .{
            route.method, route.path, route.upstream,
        });
    }

    // Match routes (simulating incoming requests)
    std.debug.print("\nRoute matching:\n", .{});
    const test_paths = [_]struct { path: []const u8, method: abi.gateway.HttpMethod }{
        .{ .path = "/api/users", .method = .GET },
        .{ .path = "/api/users/42", .method = .GET },
        .{ .path = "/api/search", .method = .POST },
        .{ .path = "/health", .method = .GET },
        .{ .path = "/api/unknown", .method = .GET },
    };

    for (test_paths) |t| {
        const match = abi.gateway.matchRoute(t.path, t.method) catch null;
        if (match) |m| {
            std.debug.print("  {s} -> upstream={s}", .{
                t.path, m.route.upstream,
            });
            if (m.param_count > 0) {
                std.debug.print(" params={{", .{});
                for (m.params[0..m.param_count], 0..) |p, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}={s}", .{ p.name, p.value });
                }
                std.debug.print("}}", .{});
            }
            std.debug.print("\n", .{});
        } else {
            std.debug.print("  {s} -> no match\n", .{t.path});
        }
    }

    // Check circuit breaker state
    const state = abi.gateway.getCircuitState("upstream-api");
    std.debug.print("\nCircuit breaker state: {t}\n", .{state});

    // Stats
    const s = abi.gateway.stats();
    std.debug.print("Gateway stats: {} routes, {} total requests\n", .{
        s.active_routes, s.total_requests,
    });
}
