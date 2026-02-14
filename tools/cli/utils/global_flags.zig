//! Global CLI Flags for Runtime Feature Control
//!
//! This module provides command-line flag parsing for runtime feature
//! enable/disable and feature listing.
//!
//! ## Usage
//!
//! ```bash
//! # List available features
//! abi --list-features
//!
//! # Disable GPU for this run
//! abi --disable-gpu db stats
//!
//! # Enable specific features
//! abi --enable-ai --disable-training llm chat
//! ```
//!
//! ## Flags
//!
//! - `--list-features` - List all available features and their status
//! - `--enable-<feature>` - Enable a feature at runtime
//! - `--disable-<feature>` - Disable a feature at runtime
//!
//! Features: gpu, ai, llm, embeddings, agents, training, database, network, observability, web

const std = @import("std");
const config_module = @import("abi").config;
const registry_mod = @import("abi").registry;
const output = @import("output.zig");

pub const Feature = config_module.Feature;

/// Errors that can occur during flag parsing or application.
pub const FlagError = error{
    /// Feature name not recognized
    UnknownFeature,
    /// Attempted to enable a feature not compiled in
    FeatureNotCompiled,
    /// Invalid flag format
    InvalidFlagFormat,
    /// Out of memory
    OutOfMemory,
};

/// Parsed global flags result.
pub const GlobalFlags = struct {
    /// Feature overrides (enable/disable)
    feature_overrides: std.AutoHashMapUnmanaged(Feature, bool),

    /// Whether to show features and exit
    show_features: bool,

    /// Remaining arguments after flag parsing
    remaining_args: []const [:0]const u8,

    /// Allocator used for internal storage
    allocator: std.mem.Allocator,

    pub fn deinit(self: *GlobalFlags) void {
        self.feature_overrides.deinit(self.allocator);
    }

    /// Check if a feature should be enabled based on overrides.
    pub fn isFeatureEnabled(self: *const GlobalFlags, feature: Feature, default: bool) bool {
        return self.feature_overrides.get(feature) orelse default;
    }

    /// Apply overrides to a registry.
    pub fn applyToRegistry(self: *const GlobalFlags, reg: *registry_mod.Registry) !void {
        var iter = self.feature_overrides.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.*) {
                try reg.enableFeature(entry.key_ptr.*);
            } else {
                reg.disableFeature(entry.key_ptr.*) catch |err| {
                    // Ignore InvalidMode for comptime_only features
                    if (err != registry_mod.Registry.Error.InvalidMode) {
                        return err;
                    }
                };
            }
        }
    }

    /// Validate that all enabled features are compiled in.
    /// Returns error details if validation fails.
    pub fn validate(self: *const GlobalFlags, comptime comptime_status: type) ?ValidationError {
        // Use inline for to check each feature at comptime
        const features = std.meta.fields(Feature);
        inline for (features) |field| {
            const feature: Feature = @enumFromInt(field.value);
            if (self.feature_overrides.get(feature)) |want_enabled| {
                if (want_enabled) {
                    const compiled = comptime comptime_status.isCompiledIn(feature);
                    if (!compiled) {
                        return ValidationError{
                            .feature = feature,
                            .reason = .not_compiled,
                        };
                    }
                }
            }
        }
        return null;
    }
};

/// Validation error details.
pub const ValidationError = struct {
    feature: Feature,
    reason: Reason,

    pub const Reason = enum {
        not_compiled,
        dependency_missing,
        conflict,
    };

    /// Print a user-friendly error message.
    pub fn print(self: ValidationError) void {
        std.debug.print("\n", .{});
        std.debug.print("Error: Cannot enable feature '{t}'\n", .{self.feature});
        std.debug.print("\n", .{});

        switch (self.reason) {
            .not_compiled => {
                std.debug.print("Reason: Feature not compiled into this build.\n", .{});
                std.debug.print("\n", .{});
                std.debug.print("Solution: Rebuild with:\n", .{});
                std.debug.print("  zig build -Denable-{t}=true\n", .{self.feature});
            },
            .dependency_missing => {
                std.debug.print("Reason: Required dependency not available.\n", .{});
            },
            .conflict => {
                std.debug.print("Reason: Conflicts with another enabled feature.\n", .{});
            },
        }
        std.debug.print("\n", .{});
    }
};

/// Options for flag parsing behavior.
pub const ParseOptions = struct {
    /// If true, unknown features will cause an error instead of a warning
    strict: bool = false,
};

/// Parse global flags from command-line arguments.
/// Returns the parsed flags and remaining arguments.
pub fn parseGlobalFlags(
    allocator: std.mem.Allocator,
    args: []const [:0]const u8,
) !GlobalFlags {
    return parseGlobalFlagsWithOptions(allocator, args, .{});
}

/// Parse global flags with configurable options.
pub fn parseGlobalFlagsWithOptions(
    allocator: std.mem.Allocator,
    args: []const [:0]const u8,
    options: ParseOptions,
) !GlobalFlags {
    var flags = GlobalFlags{
        .feature_overrides = .{},
        .show_features = false,
        .remaining_args = &.{},
        .allocator = allocator,
    };
    errdefer flags.feature_overrides.deinit(allocator);

    var remaining = std.ArrayListUnmanaged([:0]const u8){};
    defer remaining.deinit(allocator);

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);

        if (std.mem.eql(u8, arg, "--no-color")) {
            output.disableColor();
            continue;
        }

        if (std.mem.eql(u8, arg, "--list-features")) {
            flags.show_features = true;
            continue;
        }

        if (std.mem.eql(u8, arg, "--help-features")) {
            flags.show_features = true;
            continue;
        }

        if (std.mem.startsWith(u8, arg, "--enable-")) {
            const feature_name = arg["--enable-".len..];
            try applyFeatureOverride(allocator, &flags, feature_name, true, options.strict);
            continue;
        }

        if (std.mem.startsWith(u8, arg, "--disable-")) {
            const feature_name = arg["--disable-".len..];
            try applyFeatureOverride(allocator, &flags, feature_name, false, options.strict);
            continue;
        }

        if (std.mem.startsWith(u8, arg, "--enable=")) {
            const feature_name = arg["--enable=".len..];
            try applyFeatureOverride(allocator, &flags, feature_name, true, options.strict);
            continue;
        }

        if (std.mem.startsWith(u8, arg, "--disable=")) {
            const feature_name = arg["--disable=".len..];
            try applyFeatureOverride(allocator, &flags, feature_name, false, options.strict);
            continue;
        }

        if (std.mem.eql(u8, arg, "--enable") or std.mem.eql(u8, arg, "--disable")) {
            if (i + 1 >= args.len) {
                printInvalidFlagFormat(arg);
                if (options.strict) {
                    return FlagError.InvalidFlagFormat;
                }
                continue;
            }
            i += 1;
            const feature_name = std.mem.sliceTo(args[i], 0);
            if (std.mem.eql(u8, arg, "--enable")) {
                try applyFeatureOverride(allocator, &flags, feature_name, true, options.strict);
            } else {
                try applyFeatureOverride(allocator, &flags, feature_name, false, options.strict);
            }
            continue;
        }

        // Not a global flag, keep it
        try remaining.append(allocator, args[i]);
    }

    flags.remaining_args = try remaining.toOwnedSlice(allocator);
    return flags;
}

fn applyFeatureOverride(
    allocator: std.mem.Allocator,
    flags: *GlobalFlags,
    feature_name: []const u8,
    enabled: bool,
    strict: bool,
) !void {
    if (feature_name.len == 0) {
        printInvalidFeatureName();
        if (strict) {
            return FlagError.InvalidFlagFormat;
        }
        return;
    }

    if (parseFeature(feature_name)) |feature| {
        try flags.feature_overrides.put(allocator, feature, enabled);
        return;
    }

    printUnknownFeatureError(feature_name);
    if (strict) {
        return FlagError.UnknownFeature;
    }
}

fn printInvalidFeatureName() void {
    std.debug.print("\nError: Missing feature name. Use --enable-<feature> or --disable-<feature>.\n\n", .{});
}

fn printInvalidFlagFormat(flag: []const u8) void {
    std.debug.print("\nError: Invalid flag format '{s}'. Expected a feature value after it.\n\n", .{flag});
}

/// Print error message for unknown feature.
fn printUnknownFeatureError(feature_name: []const u8) void {
    std.debug.print("\nError: Unknown feature '{s}'\n\n", .{feature_name});
    std.debug.print("Available features:\n", .{});

    const features = std.meta.fields(Feature);
    inline for (features) |field| {
        std.debug.print("  - {s}\n", .{field.name});
    }
    std.debug.print("\nUse --list-features to see status of each feature.\n\n", .{});
}

/// Parse a feature name string to Feature enum.
fn parseFeature(name: []const u8) ?Feature {
    const features = std.meta.fields(Feature);
    inline for (features) |field| {
        if (std.mem.eql(u8, name, field.name)) {
            return @enumFromInt(field.value);
        }
    }
    return null;
}

/// Print available features and their status to stderr (using std.debug.print).
pub fn printFeaturesToStderr(comptime comptime_status: type) void {
    std.debug.print("\nAvailable Features:\n", .{});
    std.debug.print("--------------------------------------------------\n", .{});

    const features = std.meta.fields(Feature);
    inline for (features) |field| {
        const feature: Feature = @enumFromInt(field.value);
        const compiled = comptime comptime_status.isCompiledIn(feature);
        const status_icon = if (compiled) "[x]" else "[ ]";
        const status_text = if (compiled) "COMPILED" else "DISABLED";

        std.debug.print("  {s} {s: <15} {s}\n", .{ status_icon, field.name, status_text });
    }

    std.debug.print("\nUsage:\n", .{});
    std.debug.print("  --enable-<feature>   Enable a feature at runtime\n", .{});
    std.debug.print("  --disable-<feature>  Disable a feature at runtime\n", .{});
    std.debug.print("  --list-features      Show this list\n", .{});
    std.debug.print("\nNote: Features must be compiled in to be enabled at runtime.\n", .{});
    std.debug.print("      Rebuild with -Denable-<feature>=true to compile in.\n", .{});
}

/// Print available features and their status to a writer.
pub fn printFeatures(writer: anytype, comptime_status: anytype) !void {
    try writer.print("\nAvailable Features:\n", .{});
    try writer.print("--------------------------------------------------\n", .{});

    const features = std.meta.fields(Feature);
    inline for (features) |field| {
        const feature: Feature = @enumFromInt(field.value);
        const compiled = comptime_status.isCompiledIn(feature);
        const status_icon = if (compiled) "[x]" else "[ ]";
        const status_text = if (compiled) "COMPILED" else "DISABLED";

        try writer.print("  {s} {s: <15} {s}\n", .{ status_icon, field.name, status_text });
    }

    try writer.print("\nUsage:\n", .{});
    try writer.print("  --enable-<feature>   Enable a feature at runtime\n", .{});
    try writer.print("  --disable-<feature>  Disable a feature at runtime\n", .{});
    try writer.print("  --list-features      Show this list\n", .{});
    try writer.print("\nNote: Features must be compiled in to be enabled at runtime.\n", .{});
    try writer.print("      Rebuild with -Denable-<feature>=true to compile in.\n", .{});
}

/// Simple compile-time status checker using abi's registry module.
pub const ComptimeStatus = struct {
    pub fn isCompiledIn(comptime feature: Feature) bool {
        return comptime registry_mod.isFeatureCompiledIn(feature);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parseGlobalFlags with no flags" {
    const args = [_][:0]const u8{ "abi", "db", "stats" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    try std.testing.expect(!flags.show_features);
    try std.testing.expectEqual(@as(usize, 3), flags.remaining_args.len);
}

test "parseGlobalFlags with --list-features" {
    const args = [_][:0]const u8{ "abi", "--list-features" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    try std.testing.expect(flags.show_features);
    try std.testing.expectEqual(@as(usize, 1), flags.remaining_args.len);
}

test "parseGlobalFlags with enable/disable" {
    const args = [_][:0]const u8{ "abi", "--enable-gpu", "--disable-ai", "db" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    try std.testing.expect(flags.isFeatureEnabled(.gpu, false));
    try std.testing.expect(!flags.isFeatureEnabled(.ai, true));
    try std.testing.expectEqual(@as(usize, 2), flags.remaining_args.len);
}

test "parseGlobalFlags supports equals style flags" {
    const args = [_][:0]const u8{ "abi", "--enable=gpu", "--disable=ai", "llm" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    try std.testing.expect(flags.isFeatureEnabled(.gpu, false));
    try std.testing.expect(!flags.isFeatureEnabled(.ai, true));
    try std.testing.expectEqual(@as(usize, 2), flags.remaining_args.len);
}

test "parseGlobalFlags supports --enable/--disable value args" {
    const args = [_][:0]const u8{ "abi", "--enable", "gpu", "--disable", "ai", "llm" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    try std.testing.expect(flags.isFeatureEnabled(.gpu, false));
    try std.testing.expect(!flags.isFeatureEnabled(.ai, true));
    try std.testing.expectEqual(@as(usize, 2), flags.remaining_args.len);
}

test "parseFeature recognizes valid features" {
    try std.testing.expectEqual(Feature.gpu, parseFeature("gpu").?);
    try std.testing.expectEqual(Feature.ai, parseFeature("ai").?);
    try std.testing.expectEqual(Feature.database, parseFeature("database").?);
    try std.testing.expect(parseFeature("invalid") == null);
}

test "validate catches uncompiled features" {
    const args = [_][:0]const u8{ "abi", "--enable-gpu" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    // Mock comptime status that says GPU is not compiled
    const MockStatus = struct {
        pub fn isCompiledIn(feature: Feature) bool {
            return feature != .gpu;
        }
    };

    const validation_result = flags.validate(MockStatus);
    try std.testing.expect(validation_result != null);
    try std.testing.expectEqual(Feature.gpu, validation_result.?.feature);
    try std.testing.expectEqual(ValidationError.Reason.not_compiled, validation_result.?.reason);
}

test "validate passes for compiled features" {
    const args = [_][:0]const u8{ "abi", "--enable-gpu" };
    var flags = try parseGlobalFlags(std.testing.allocator, &args);
    defer flags.deinit();
    defer std.testing.allocator.free(flags.remaining_args);

    // Mock comptime status that says GPU is compiled
    const MockStatus = struct {
        pub fn isCompiledIn(_: Feature) bool {
            return true;
        }
    };

    const validation_result = flags.validate(MockStatus);
    try std.testing.expect(validation_result == null);
}

test "strict mode returns error for unknown feature" {
    const args = [_][:0]const u8{ "abi", "--enable-foobar" };
    const result = parseGlobalFlagsWithOptions(std.testing.allocator, &args, .{ .strict = true });
    try std.testing.expectError(FlagError.UnknownFeature, result);
}

test "strict mode returns error for equals style unknown feature" {
    const args = [_][:0]const u8{ "abi", "--enable=foobar" };
    const result = parseGlobalFlagsWithOptions(std.testing.allocator, &args, .{ .strict = true });
    try std.testing.expectError(FlagError.UnknownFeature, result);
}
