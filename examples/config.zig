//! Configuration Example
//!
//! Demonstrates the ABI framework configuration system with GPU,
//! AI, and database settings using the Builder pattern.

const std = @import("std");
const abi = @import("abi");

pub fn main(_: std.process.Init) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create configuration using the Builder pattern
    var builder = abi.config.Builder.init(allocator);

    // Configure GPU with specific settings
    const config = builder
        .with(.gpu, .{
            .backend = .auto,
            .device_index = 0, // First available GPU
            .memory_limit = 4 * 1024 * 1024 * 1024, // 4GB limit
            .async_enabled = true,
            .cache_kernels = true,
            .recovery = .{
                .enabled = true,
                .max_retries = 3,
                .fallback_to_cpu = true,
            },
        })
        .withDefault(.ai)
        .withDefault(.database)
        .build();

    // Initialize framework with configuration
    var app = try abi.Framework.init(allocator, config);
    defer app.deinit();

    std.debug.print("ABI Framework v{s}\n", .{abi.version()});
    std.debug.print("Configuration:\n", .{});

    // Display GPU configuration
    if (config.gpu) |gpu_config| {
        std.debug.print("  GPU backend: {t}\n", .{gpu_config.backend});
        std.debug.print("  Device index: {d}\n", .{gpu_config.device_index});
        std.debug.print("  Async enabled: {}\n", .{gpu_config.async_enabled});
        std.debug.print("  Kernel caching: {}\n", .{gpu_config.cache_kernels});
        if (gpu_config.memory_limit) |limit| {
            std.debug.print("  Memory limit: {d} MB\n", .{limit / 1024 / 1024});
        }
    }

    // Display AI configuration status
    if (config.ai) |_| {
        std.debug.print("  AI: enabled\n", .{});
    }

    // Display database configuration status
    if (config.database) |_| {
        std.debug.print("  Database: enabled\n", .{});
    }

    std.debug.print("\nFramework initialized successfully!\n", .{});
}
