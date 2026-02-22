//! Web Example
//!
//! Demonstrates the web module: HTTP client, persona routing,
//! chat handling, and JSON utilities.
//!
//! Run with: `zig build run-web`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withWebDefaults()
        .build();
    defer framework.deinit();

    if (!abi.web.isEnabled()) {
        std.debug.print("Web feature is disabled. Enable with -Denable-web=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Web Example ===\n\n", .{});

    // Chat request/response types
    std.debug.print("--- Chat Handler Types ---\n", .{});
    const request = abi.web.ChatRequest{
        .content = "Hello from the ABI web module!",
    };
    std.debug.print("  Content: {s}\n", .{request.content});
    std.debug.print("  Session: {s}\n", .{request.session_id orelse "(none)"});
    std.debug.print("  Persona: {s}\n", .{request.persona orelse "(auto-route)"});

    // Persona routing
    std.debug.print("\n--- Persona Router ---\n", .{});
    std.debug.print("  PersonaRouter and Route types available\n", .{});
    std.debug.print("  Routes map URL paths to AI persona handlers\n", .{});

    // HTTP Client type
    std.debug.print("\n--- HTTP Client ---\n", .{});
    std.debug.print("  HttpClient type available for outbound requests\n", .{});
    std.debug.print("  RequestOptions configures timeout and retry behavior\n", .{});

    // Error types
    std.debug.print("\n--- Web Errors ---\n", .{});
    std.debug.print("  WebError type covers HTTP, routing, and handler failures\n", .{});

    std.debug.print("\nWeb example complete.\n", .{});
}
