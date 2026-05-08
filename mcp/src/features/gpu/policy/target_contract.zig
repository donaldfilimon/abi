//! GPU Backend Policy Contract Validator
//!
//! Compile-time contract that validates GPU backend resolution for the build
//! target. Uses @compileError to catch misconfigurations before runtime.
//! Included in the `typecheck` build step as a standalone object build.

const std = @import("std");
const builtin = @import("builtin");
const policy = @import("mod.zig");

fn expectedPlatformForTarget() policy.PlatformClass {
    return switch (builtin.target.os.tag) {
        .macos => .macos,
        .linux => if (builtin.abi == .android) .android else .linux,
        .windows => .windows,
        else => @compileError(std.fmt.comptimePrint(
            "GPU policy contract supports only macOS, Linux, and Windows targets; got {s}-{s}",
            .{ @tagName(builtin.target.cpu.arch), @tagName(builtin.target.os.tag) },
        )),
    };
}

fn expectPlatform(actual: policy.PlatformClass, expected: policy.PlatformClass) void {
    if (actual != expected) {
        @compileError(std.fmt.comptimePrint(
            "GPU policy contract classification mismatch for {s}-{s}: expected {s}, got {s}",
            .{
                @tagName(builtin.target.cpu.arch),
                @tagName(builtin.target.os.tag),
                @tagName(expected),
                @tagName(actual),
            },
        ));
    }
}

fn expectAutoOrder(actual: []const []const u8, comptime expected: []const []const u8) void {
    if (actual.len != expected.len) {
        @compileError(std.fmt.comptimePrint(
            "GPU policy contract auto backend count mismatch for {s}: expected {}, got {}",
            .{ @tagName(policy.classifyBuiltin()), expected.len, actual.len },
        ));
    }

    inline for (expected, 0..) |expected_name, index| {
        if (!std.mem.eql(u8, actual[index], expected_name)) {
            @compileError(std.fmt.comptimePrint(
                "GPU policy contract auto backend mismatch for {s} at index {}: expected {s}, got {s}",
                .{ @tagName(policy.classifyBuiltin()), index, expected_name, actual[index] },
            ));
        }
    }
}

fn expectHints(actual: policy.OptimizationHints, comptime expected: policy.OptimizationHints) void {
    inline for (@typeInfo(policy.OptimizationHints).@"struct".fields) |field| {
        const actual_value = @field(actual, field.name);
        const expected_value = @field(expected, field.name);
        if (actual_value != expected_value) {
            @compileError(std.fmt.comptimePrint(
                "GPU policy contract hint mismatch for {s}.{s}: expected {}, got {}",
                .{
                    @tagName(policy.classifyBuiltin()),
                    field.name,
                    expected_value,
                    actual_value,
                },
            ));
        }
    }
}

comptime {
    const expected_platform = expectedPlatformForTarget();
    const actual_platform = policy.classifyBuiltin();
    expectPlatform(actual_platform, expected_platform);

    const auto_names = policy.resolveAutoBackendNames(.{
        .platform = actual_platform,
        .enable_gpu = true,
        .enable_web = false,
        .can_link_metal = true,
        .allow_simulated = false,
    });
    const hints = policy.optimizationHintsForPlatform(actual_platform);

    switch (actual_platform) {
        .macos => {
            const expected_order = [_][]const u8{ "metal", "vulkan", "opengl", "stdgpu" };
            expectAutoOrder(auto_names.slice(), expected_order[0..]);
            expectHints(hints, .{
                .default_local_size = 256,
                .default_queue_depth = 8,
                .prefer_unified_memory = true,
                .prefer_pinned_staging = false,
                .transfer_chunk_bytes = 8 * 1024 * 1024,
            });
        },
        .linux => {
            const expected_order = [_][]const u8{ "cuda", "vulkan", "opengl", "stdgpu" };
            expectAutoOrder(auto_names.slice(), expected_order[0..]);
            expectHints(hints, .{
                .default_local_size = 256,
                .default_queue_depth = 8,
                .prefer_unified_memory = false,
                .prefer_pinned_staging = true,
                .transfer_chunk_bytes = 16 * 1024 * 1024,
            });
        },
        .windows => {
            const expected_order = [_][]const u8{ "cuda", "vulkan", "opengl", "stdgpu" };
            expectAutoOrder(auto_names.slice(), expected_order[0..]);
            expectHints(hints, .{
                .default_local_size = 128,
                .default_queue_depth = 4,
                .prefer_unified_memory = false,
                .prefer_pinned_staging = true,
                .transfer_chunk_bytes = 8 * 1024 * 1024,
            });
        },
        else => @compileError(std.fmt.comptimePrint(
            "GPU policy contract has no expectation set for platform class {s}",
            .{@tagName(actual_platform)},
        )),
    }
}
