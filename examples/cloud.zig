//! Cloud Example
//!
//! Demonstrates the cloud module: provider configuration, event handling,
//! and serverless function wrappers for AWS Lambda, GCP Functions, and Azure.
//!
//! Run with: `zig build run-cloud`

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var builder = abi.Framework.builder(allocator);
    var framework = try builder
        .withCloudDefaults()
        .build();
    defer framework.deinit();

    if (!abi.cloud.isEnabled()) {
        std.debug.print("Cloud feature is disabled. Enable with -Denable-cloud=true\n", .{});
        return;
    }

    std.debug.print("=== ABI Cloud Example ===\n\n", .{});

    // Cloud provider types
    std.debug.print("--- Cloud Providers ---\n", .{});
    const providers = [_]abi.cloud.CloudProvider{ .aws_lambda, .gcp_functions, .azure_functions };
    for (providers) |p| {
        std.debug.print("  Provider: {s}\n", .{p.name()});
    }

    // Cloud event structure
    std.debug.print("\n--- Cloud Events ---\n", .{});
    const event = abi.cloud.CloudEvent{
        .request_id = "example-req-001",
        .provider = .aws_lambda,
        .allocator = allocator,
    };
    std.debug.print("  CloudEvent type available\n", .{});
    _ = event;

    // Cloud config
    std.debug.print("\n--- Cloud Config ---\n", .{});
    const config = abi.cloud.CloudConfig{
        .memory_mb = 512,
        .timeout_seconds = 60,
        .tracing_enabled = true,
    };
    std.debug.print("  Memory: {} MB, Timeout: {}s\n", .{ config.memory_mb, config.timeout_seconds });

    // HTTP method handling
    std.debug.print("\n--- HTTP Methods ---\n", .{});
    const methods = [_]abi.cloud.HttpMethod{ .GET, .POST, .PUT, .DELETE };
    for (methods) |m| {
        std.debug.print("  {t}\n", .{m});
    }

    std.debug.print("\nCloud example complete.\n", .{});
}
