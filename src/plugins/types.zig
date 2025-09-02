//! Plugin System Types
//!
//! This module defines all the core types, errors, and constants used throughout
//! the plugin system.

const std = @import("std");

/// Plugin system errors
pub const PluginError = error{
    // Loading errors
    PluginNotFound,
    InvalidPlugin,
    LoadFailed,
    SymbolNotFound,
    DependencyMissing,

    // Version errors
    IncompatibleVersion,
    UnsupportedABI,

    // Runtime errors
    InitializationFailed,
    ExecutionFailed,
    InvalidParameters,
    PluginCrashed,

    // System errors
    OutOfMemory,
    PermissionDenied,
    InvalidPath,

    // Registration errors
    AlreadyRegistered,
    NotRegistered,
    ConflictingPlugin,
};

/// Plugin types supported by the system
pub const PluginType = enum {
    // Database plugins
    vector_database,
    indexing_algorithm,
    compression_algorithm,

    // AI/ML plugins
    neural_network,
    embedding_generator,
    training_algorithm,
    inference_engine,

    // Processing plugins
    text_processor,
    image_processor,
    audio_processor,
    data_transformer,

    // I/O plugins
    data_loader,
    data_exporter,
    protocol_handler,

    // Utility plugins
    logger,
    metrics_collector,
    security_provider,
    configuration_provider,

    // Custom plugins
    custom,

    pub fn toString(self: PluginType) []const u8 {
        return switch (self) {
            .vector_database => "vector_database",
            .indexing_algorithm => "indexing_algorithm",
            .compression_algorithm => "compression_algorithm",
            .neural_network => "neural_network",
            .embedding_generator => "embedding_generator",
            .training_algorithm => "training_algorithm",
            .inference_engine => "inference_engine",
            .text_processor => "text_processor",
            .image_processor => "image_processor",
            .audio_processor => "audio_processor",
            .data_transformer => "data_transformer",
            .data_loader => "data_loader",
            .data_exporter => "data_exporter",
            .protocol_handler => "protocol_handler",
            .logger => "logger",
            .metrics_collector => "metrics_collector",
            .security_provider => "security_provider",
            .configuration_provider => "configuration_provider",
            .custom => "custom",
        };
    }

    pub fn fromString(s: []const u8) ?PluginType {
        if (std.mem.eql(u8, s, "vector_database")) return .vector_database;
        if (std.mem.eql(u8, s, "indexing_algorithm")) return .indexing_algorithm;
        if (std.mem.eql(u8, s, "compression_algorithm")) return .compression_algorithm;
        if (std.mem.eql(u8, s, "neural_network")) return .neural_network;
        if (std.mem.eql(u8, s, "embedding_generator")) return .embedding_generator;
        if (std.mem.eql(u8, s, "training_algorithm")) return .training_algorithm;
        if (std.mem.eql(u8, s, "inference_engine")) return .inference_engine;
        if (std.mem.eql(u8, s, "text_processor")) return .text_processor;
        if (std.mem.eql(u8, s, "image_processor")) return .image_processor;
        if (std.mem.eql(u8, s, "audio_processor")) return .audio_processor;
        if (std.mem.eql(u8, s, "data_transformer")) return .data_transformer;
        if (std.mem.eql(u8, s, "data_loader")) return .data_loader;
        if (std.mem.eql(u8, s, "data_exporter")) return .data_exporter;
        if (std.mem.eql(u8, s, "protocol_handler")) return .protocol_handler;
        if (std.mem.eql(u8, s, "logger")) return .logger;
        if (std.mem.eql(u8, s, "metrics_collector")) return .metrics_collector;
        if (std.mem.eql(u8, s, "security_provider")) return .security_provider;
        if (std.mem.eql(u8, s, "configuration_provider")) return .configuration_provider;
        if (std.mem.eql(u8, s, "custom")) return .custom;
        return null;
    }
};

/// Plugin version information
pub const PluginVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,
    pre_release: ?[]const u8 = null,

    pub fn init(major: u32, minor: u32, patch: u32) PluginVersion {
        return .{ .major = major, .minor = minor, .patch = patch };
    }

    pub fn isCompatible(self: PluginVersion, required: PluginVersion) bool {
        // Semantic versioning compatibility check
        if (self.major != required.major) return false;
        if (self.minor < required.minor) return false;
        return true;
    }

    pub fn format(
        self: PluginVersion,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        if (self.pre_release) |pre| {
            try writer.print("{d}.{d}.{d}-{s}", .{ self.major, self.minor, self.patch, pre });
        } else {
            try writer.print("{d}.{d}.{d}", .{ self.major, self.minor, self.patch });
        }
    }
};

/// Plugin metadata and information
pub const PluginInfo = struct {
    // Identity
    name: []const u8,
    version: PluginVersion,
    author: []const u8,
    description: []const u8,

    // Type and compatibility
    plugin_type: PluginType,
    abi_version: PluginVersion,

    // Dependencies
    dependencies: []const []const u8 = &.{},

    // Capabilities
    provides: []const []const u8 = &.{},
    requires: []const []const u8 = &.{},

    // Optional metadata
    license: ?[]const u8 = null,
    homepage: ?[]const u8 = null,
    repository: ?[]const u8 = null,

    pub fn isCompatible(self: PluginInfo, framework_abi: PluginVersion) bool {
        return self.abi_version.isCompatible(framework_abi);
    }
};

/// Plugin configuration
pub const PluginConfig = struct {
    // Basic settings
    enabled: bool = true,
    auto_load: bool = true,
    priority: i32 = 0, // Higher priority loads first

    // Resource limits
    max_memory_mb: ?u32 = null,
    max_cpu_time_ms: ?u32 = null,

    // Security settings
    sandboxed: bool = false,
    permissions: []const []const u8 = &.{},

    // Plugin-specific configuration
    parameters: std.StringHashMap([]const u8),

    pub fn init(allocator: std.mem.Allocator) PluginConfig {
        return .{
            .parameters = std.StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *PluginConfig) void {
        self.parameters.deinit();
    }

    pub fn setParameter(self: *PluginConfig, key: []const u8, value: []const u8) !void {
        try self.parameters.put(key, value);
    }

    pub fn getParameter(self: *PluginConfig, key: []const u8) ?[]const u8 {
        return self.parameters.get(key);
    }
};

/// Plugin state tracking
pub const PluginState = enum {
    unloaded,
    loading,
    loaded,
    initializing,
    initialized,
    running,
    paused,
    stopping,
    stopped,
    error_state,

    pub fn toString(self: PluginState) []const u8 {
        return switch (self) {
            .unloaded => "unloaded",
            .loading => "loading",
            .loaded => "loaded",
            .initializing => "initializing",
            .initialized => "initialized",
            .running => "running",
            .paused => "paused",
            .stopping => "stopping",
            .stopped => "stopped",
            .error_state => "error",
        };
    }

    pub fn canTransitionTo(self: PluginState, new_state: PluginState) bool {
        return switch (self) {
            .unloaded => new_state == .loading,
            .loading => new_state == .loaded or new_state == .error_state,
            .loaded => new_state == .initializing or new_state == .unloaded,
            .initializing => new_state == .initialized or new_state == .error_state,
            .initialized => new_state == .running or new_state == .stopping,
            .running => new_state == .paused or new_state == .stopping,
            .paused => new_state == .running or new_state == .stopping,
            .stopping => new_state == .stopped or new_state == .error_state,
            .stopped => new_state == .unloaded,
            .error_state => new_state == .unloaded,
        };
    }
};

/// Plugin execution context
pub const PluginContext = struct {
    allocator: std.mem.Allocator,
    logger: ?*anyopaque = null,
    config: *PluginConfig,

    // Framework callbacks
    log_fn: ?*const fn (level: u8, message: []const u8) void = null,
    get_service_fn: ?*const fn (service_name: []const u8) ?*anyopaque = null,

    pub fn log(self: *PluginContext, level: u8, message: []const u8) void {
        if (self.log_fn) |log_func| {
            log_func(level, message);
        }
    }

    pub fn getService(self: *PluginContext, service_name: []const u8) ?*anyopaque {
        if (self.get_service_fn) |get_service_func| {
            return get_service_func(service_name);
        }
        return null;
    }
};

// =============================================================================
// TESTS
// =============================================================================

test "PluginType conversion" {
    try std.testing.expectEqual(PluginType.neural_network, PluginType.fromString("neural_network").?);
    try std.testing.expectEqualStrings("neural_network", PluginType.neural_network.toString());
    try std.testing.expectEqual(@as(?PluginType, null), PluginType.fromString("invalid"));
}

test "PluginVersion compatibility" {
    const v1_0_0 = PluginVersion.init(1, 0, 0);
    const v1_1_0 = PluginVersion.init(1, 1, 0);
    const v2_0_0 = PluginVersion.init(2, 0, 0);

    try std.testing.expect(v1_1_0.isCompatible(v1_0_0));
    try std.testing.expect(!v1_0_0.isCompatible(v1_1_0));
    try std.testing.expect(!v2_0_0.isCompatible(v1_0_0));
}

test "PluginState transitions" {
    try std.testing.expect(PluginState.unloaded.canTransitionTo(.loading));
    try std.testing.expect(!PluginState.unloaded.canTransitionTo(.running));
    try std.testing.expect(PluginState.running.canTransitionTo(.paused));
    try std.testing.expect(PluginState.running.canTransitionTo(.stopping));
}

test "PluginConfig parameters" {
    var config = PluginConfig.init(std.testing.allocator);
    defer config.deinit();

    try config.setParameter("test_key", "test_value");
    try std.testing.expectEqualStrings("test_value", config.getParameter("test_key").?);
    try std.testing.expectEqual(@as(?[]const u8, null), config.getParameter("nonexistent"));
}
