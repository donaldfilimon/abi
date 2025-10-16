//! Interactive CLI for ABI Framework - Refactored Version
//!
//! This module provides a comprehensive interactive command-line interface
//! with proper I/O boundary abstraction and modern error handling.

const std = @import("std");
const builtin = @import("builtin");
const patterns = @import("../shared/patterns/common.zig");

/// Re-export commonly used types
pub const Allocator = std.mem.Allocator;
pub const Writer = patterns.Writer;
pub const Logger = patterns.Logger;
pub const ErrorContext = patterns.ErrorContext;

/// CLI Command structure with I/O abstraction
pub const Command = struct {
    name: []const u8,
    aliases: []const []const u8 = &.{},
    summary: []const u8,
    usage: []const u8,
    details: ?[]const u8 = null,
    run: *const fn (ctx: *Context, args: [][:0]u8) anyerror!void,
};

/// CLI Context with injected I/O
pub const Context = struct {
    allocator: Allocator,
    writer: Writer,
    logger: Logger,
    gpu_available: bool = false,
    interactive_mode: bool = false,

    pub fn init(allocator: Allocator, writer: Writer) Context {
        return Context{
            .allocator = allocator,
            .writer = writer,
            .logger = Logger.init(writer, .info),
            .gpu_available = detectGPU(),
        };
    }

    fn detectGPU() bool {
        // Simple GPU detection - would be more sophisticated in real implementation
        return switch (builtin.os.tag) {
            .windows => std.process.hasEnvVar(std.heap.page_allocator, "NVIDIA_GPU") catch false,
            .linux => std.fs.accessAbsolute("/dev/dri", .{}) != error.FileNotFound,
            .macos => true, // Assume Metal is available on macOS
            else => false,
        };
    }

    /// Display information using injected writer instead of std.debug.print
    pub fn displayInfo(self: *Context, comptime fmt: []const u8, args: anytype) !void {
        try self.logger.info(fmt, args);
    }

    /// Display warning using injected writer
    pub fn displayWarning(self: *Context, comptime fmt: []const u8, args: anytype) !void {
        try self.logger.warn(fmt, args);
    }

    /// Display error using injected writer
    pub fn displayError(self: *Context, comptime fmt: []const u8, args: anytype) !void {
        try self.logger.err(fmt, args);
    }
};

/// GPU Backend types
pub const GPUBackend = enum {
    auto,
    vulkan,
    metal,
    cuda,
    opencl,
    webgpu,
    cpu_fallback,

    pub fn toString(self: GPUBackend) []const u8 {
        return switch (self) {
            .auto => "Auto-detect",
            .vulkan => "Vulkan",
            .metal => "Metal",
            .cuda => "CUDA",
            .opencl => "OpenCL",
            .webgpu => "WebGPU",
            .cpu_fallback => "CPU Fallback",
        };
    }
};

/// Performance metrics with proper error handling
pub const PerformanceMetrics = struct {
    gpu_memory_mb: f64 = 0.0,
    cpu_usage_percent: f64 = 0.0,
    memory_usage_mb: f64 = 0.0,
    frame_time_ms: f64 = 0.0,
    throughput_ops_sec: f64 = 0.0,

    pub fn collect(allocator: Allocator) !PerformanceMetrics {
        _ = allocator;
        // Placeholder implementation - would collect real metrics
        return PerformanceMetrics{
            .gpu_memory_mb = 512.0,
            .cpu_usage_percent = 25.5,
            .memory_usage_mb = 128.0,
            .frame_time_ms = 16.67,
            .throughput_ops_sec = 1000.0,
        };
    }

    pub fn display(self: PerformanceMetrics, ctx: *Context) !void {
        try ctx.displayInfo("Performance Metrics:", .{});
        try ctx.displayInfo("  GPU Memory: {d:.1} MB", .{self.gpu_memory_mb});
        try ctx.displayInfo("  CPU Usage: {d:.1}%", .{self.cpu_usage_percent});
        try ctx.displayInfo("  Memory Usage: {d:.1} MB", .{self.memory_usage_mb});
        try ctx.displayInfo("  Frame Time: {d:.2} ms", .{self.frame_time_ms});
        try ctx.displayInfo("  Throughput: {d:.0} ops/sec", .{self.throughput_ops_sec});
    }
};

/// Command implementations with proper I/O
const Commands = struct {
    pub fn helpCommand(ctx: *Context, args: [][:0]u8) !void {
        _ = args;
        try ctx.displayInfo("ABI Framework Interactive CLI", .{});
        try ctx.displayInfo("Available commands:", .{});
        try ctx.displayInfo("  help     - Show this help message", .{});
        try ctx.displayInfo("  status   - Show framework status", .{});
        try ctx.displayInfo("  gpu      - GPU operations", .{});
        try ctx.displayInfo("  perf     - Performance monitoring", .{});
        try ctx.displayInfo("  exit     - Exit the CLI", .{});
    }

    pub fn statusCommand(ctx: *Context, args: [][:0]u8) !void {
        _ = args;
        try ctx.displayInfo("Framework Status:", .{});
        try ctx.displayInfo("  Version: 0.2.0", .{});
        try ctx.displayInfo("  GPU Available: {}", .{ctx.gpu_available});
        try ctx.displayInfo("  Interactive Mode: {}", .{ctx.interactive_mode});
        try ctx.displayInfo("  Platform: {s}", .{@tagName(builtin.os.tag)});
    }

    pub fn gpuCommand(ctx: *Context, args: [][:0]u8) !void {
        if (args.len == 0) {
            try ctx.displayInfo("GPU subsystem status:", .{});
            try ctx.displayInfo("  Available: {}", .{ctx.gpu_available});
            if (ctx.gpu_available) {
                try ctx.displayInfo("  Backend: Auto-detect", .{});
                try ctx.displayInfo("  Status: Ready", .{});
            } else {
                try ctx.displayWarning("  No GPU detected, using CPU fallback", .{});
            }
            return;
        }

        const subcommand = args[0];
        if (std.mem.eql(u8, subcommand, "info")) {
            try ctx.displayInfo("GPU Information:", .{});
            try ctx.displayInfo("  Vendor: Unknown", .{});
            try ctx.displayInfo("  Memory: Unknown", .{});
            try ctx.displayInfo("  Compute Units: Unknown", .{});
        } else {
            try ctx.displayError("Unknown GPU subcommand: {s}", .{subcommand});
        }
    }

    pub fn perfCommand(ctx: *Context, args: [][:0]u8) !void {
        _ = args;
        const metrics = PerformanceMetrics.collect(ctx.allocator) catch |err| {
            const error_ctx = ErrorContext.init("Failed to collect performance metrics")
                .withLocation(@src())
                .withCause(err);
            try ctx.displayError("{}", .{error_ctx});
            return;
        };

        try metrics.display(ctx);
    }

    pub fn exitCommand(ctx: *Context, args: [][:0]u8) !void {
        _ = args;
        try ctx.displayInfo("Goodbye!", .{});
        std.process.exit(0);
    }
};

/// Available commands registry
const available_commands = [_]Command{
    .{ .name = "help", .aliases = &.{ "h", "?" }, .summary = "Show help information", .usage = "help", .run = Commands.helpCommand },
    .{ .name = "status", .aliases = &.{"stat"}, .summary = "Show framework status", .usage = "status", .run = Commands.statusCommand },
    .{ .name = "gpu", .aliases = &.{}, .summary = "GPU operations and information", .usage = "gpu [info]", .run = Commands.gpuCommand },
    .{ .name = "perf", .aliases = &.{"performance"}, .summary = "Performance monitoring", .usage = "perf", .run = Commands.perfCommand },
    .{ .name = "exit", .aliases = &.{ "quit", "q" }, .summary = "Exit the CLI", .usage = "exit", .run = Commands.exitCommand },
};

/// Interactive CLI runner with proper error handling
pub fn runInteractive(allocator: Allocator, writer: Writer) !void {
    var ctx = Context.init(allocator, writer);
    ctx.interactive_mode = true;

    try ctx.displayInfo("🚀 ABI Framework Interactive CLI v0.2.0", .{});
    try ctx.displayInfo("Type 'help' for available commands, 'exit' to quit.", .{});

    const stdin = std.io.getStdIn().reader();
    var input_buffer: [256]u8 = undefined;

    while (true) {
        try ctx.writer.print("> ");

        if (try stdin.readUntilDelimiterOrEof(input_buffer[0..], '\n')) |input| {
            const trimmed = std.mem.trim(u8, input, " \t\r\n");
            if (trimmed.len == 0) continue;

            // Parse command and arguments
            var args = std.ArrayList([:0]u8).init(allocator);
            defer args.deinit();

            var iter = std.mem.split(u8, trimmed, " ");
            while (iter.next()) |arg| {
                if (arg.len > 0) {
                    const owned_arg = try allocator.dupeZ(u8, arg);
                    try args.append(owned_arg);
                }
            }

            if (args.items.len == 0) continue;

            const command_name = args.items[0];
            const command_args = args.items[1..];

            // Find and execute command
            var found = false;
            for (available_commands) |cmd| {
                if (std.mem.eql(u8, cmd.name, command_name)) {
                    cmd.run(&ctx, command_args) catch |err| {
                        const error_ctx = ErrorContext.init("Command execution failed")
                            .withLocation(@src())
                            .withCause(err);
                        try ctx.displayError("{}", .{error_ctx});
                    };
                    found = true;
                    break;
                }

                // Check aliases
                for (cmd.aliases) |alias| {
                    if (std.mem.eql(u8, alias, command_name)) {
                        cmd.run(&ctx, command_args) catch |err| {
                            const error_ctx = ErrorContext.init("Command execution failed")
                                .withLocation(@src())
                                .withCause(err);
                            try ctx.displayError("{}", .{error_ctx});
                        };
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            if (!found) {
                try ctx.displayError("Unknown command: {s}. Type 'help' for available commands.", .{command_name});
            }

            // Clean up allocated arguments
            for (args.items) |arg| {
                allocator.free(arg);
            }
        } else {
            break; // EOF
        }
    }
}

/// Non-interactive CLI runner
pub fn runCommand(allocator: Allocator, writer: Writer, args: [][:0]u8) !void {
    var ctx = Context.init(allocator, writer);

    if (args.len == 0) {
        try Commands.helpCommand(&ctx, &.{});
        return;
    }

    const command_name = args[0];
    const command_args = args[1..];

    // Find and execute command
    for (available_commands) |cmd| {
        if (std.mem.eql(u8, cmd.name, command_name)) {
            try cmd.run(&ctx, command_args);
            return;
        }

        // Check aliases
        for (cmd.aliases) |alias| {
            if (std.mem.eql(u8, alias, command_name)) {
                try cmd.run(&ctx, command_args);
                return;
            }
        }
    }

    try ctx.displayError("Unknown command: {s}. Use 'help' for available commands.", .{command_name});
}

test "Context initialization with injected writer" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();

    const writer = buffer.writer().any();
    var ctx = Context.init(std.testing.allocator, writer);

    try ctx.displayInfo("Test message", .{});
    const output = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "Test message") != null);
}

test "Performance metrics collection and display" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();

    const writer = buffer.writer().any();
    var ctx = Context.init(std.testing.allocator, writer);

    const metrics = try PerformanceMetrics.collect(std.testing.allocator);
    try metrics.display(&ctx);

    const output = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "Performance Metrics") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "GPU Memory") != null);
}

test "Command execution with proper error handling" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();

    const writer = buffer.writer().any();
    
    try runCommand(std.testing.allocator, writer, &.{"status"});
    
    const output = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, output, "Framework Status") != null);
}