//! Framework orchestration layer that coordinates features and plugins.
//!
//! Manages the lifecycle of the entire framework, including feature initialization,
//! configuration management, and runtime state coordination.

const std = @import("std");
const build_options = @import("build_options");
const features = @import("../features/mod.zig");

pub const Feature = features.FeatureTag;

pub const FrameworkOptions = struct {
    enable_ai: bool = build_options.enable_ai,
    enable_gpu: bool = build_options.enable_gpu,
    enable_web: bool = build_options.enable_web,
    enable_database: bool = build_options.enable_database,
    enable_network: bool = build_options.enable_network,
    enable_profiling: bool = build_options.enable_profiling,
    disabled_features: []const Feature = &.{},
    plugin_paths: []const []const u8 = &.{},
    auto_discover_plugins: bool = false,
};

pub const FrameworkConfiguration = struct {
    enabled_features: []const Feature,
    disabled_features: []const Feature = &.{},
    plugin_paths: []const []const u8 = &.{},
    auto_discover_plugins: bool = false,

    pub fn toRuntimeConfig(
        self: FrameworkConfiguration,
        allocator: std.mem.Allocator,
    ) !RuntimeConfig {
        return RuntimeConfig.init(
            allocator,
            self.enabled_features,
            self.disabled_features,
            self.plugin_paths,
            self.auto_discover_plugins,
        );
    }
};

pub const RuntimeConfig = struct {
    enabled_features: []const Feature,
    disabled_features: []const Feature,
    plugin_paths: []const []const u8,
    auto_discover_plugins: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        enabled_features: []const Feature,
        disabled_features: []const Feature,
        plugin_paths: []const []const u8,
        auto_discover_plugins: bool,
    ) !RuntimeConfig {
        const enabled_copy = try cloneFeatures(allocator, enabled_features);
        errdefer allocator.free(enabled_copy);
        const disabled_copy = try cloneFeatures(allocator, disabled_features);
        errdefer allocator.free(disabled_copy);
        const plugin_copy = try cloneStringList(allocator, plugin_paths);
        errdefer freeStringList(allocator, plugin_copy);
        return RuntimeConfig{
            .enabled_features = enabled_copy,
            .disabled_features = disabled_copy,
            .plugin_paths = plugin_copy,
            .auto_discover_plugins = auto_discover_plugins,
        };
    }

    pub fn deinit(self: *RuntimeConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.enabled_features);
        allocator.free(self.disabled_features);
        freeStringList(allocator, self.plugin_paths);
        self.* = undefined;
    }
};

pub const Framework = struct {
    allocator: std.mem.Allocator,
    config: RuntimeConfig,
    running: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: RuntimeConfig) !Framework {
        var instance = Framework{
            .allocator = allocator,
            .config = config,
            .running = false,
        };
        try instance.start();
        return instance;
    }

    pub fn deinit(self: *Framework) void {
        if (self.running) {
            self.stop();
        }
        self.config.deinit(self.allocator);
    }

    pub fn start(self: *Framework) !void {
        if (self.running) return;
        try features.lifecycle.initFeatures(self.allocator, self.config.enabled_features);
        self.running = true;
    }

    pub fn stop(self: *Framework) void {
        if (!self.running) return;
        features.lifecycle.deinitFeatures(self.config.enabled_features);
        self.running = false;
    }

    pub fn isRunning(self: *const Framework) bool {
        return self.running;
    }

    pub fn isFeatureEnabled(self: *const Framework, feature: Feature) bool {
        return std.mem.indexOfScalar(Feature, self.config.enabled_features, feature) != null;
    }
};

pub fn createFramework(allocator: std.mem.Allocator, config: RuntimeConfig) !Framework {
    return Framework.init(allocator, config);
}

pub fn runtimeConfigFromOptions(
    allocator: std.mem.Allocator,
    options: FrameworkOptions,
) !RuntimeConfig {
    var enabled = try buildEnabledFeatures(allocator, options);
    defer enabled.deinit(allocator);

    const disabled_slice = try cloneFeatures(allocator, options.disabled_features);
    errdefer allocator.free(disabled_slice);

    var filtered = try filterDisabledFeatures(allocator, &enabled, disabled_slice);

    const enabled_slice = try filtered.toOwnedSlice(allocator);
    const plugin_copy = try cloneStringList(allocator, options.plugin_paths);

    return RuntimeConfig{
        .enabled_features = enabled_slice,
        .disabled_features = disabled_slice,
        .plugin_paths = plugin_copy,
        .auto_discover_plugins = options.auto_discover_plugins,
    };
}

fn buildEnabledFeatures(allocator: std.mem.Allocator, options: FrameworkOptions) !std.ArrayList(Feature) {
    var enabled = std.ArrayList(Feature).empty;
    if (options.enable_ai) try enabled.append(allocator, .ai);
    if (options.enable_gpu) try enabled.append(allocator, .gpu);
    if (options.enable_database) try enabled.append(allocator, .database);
    if (options.enable_web) try enabled.append(allocator, .web);
    if (options.enable_network) try enabled.append(allocator, .network);
    if (options.enable_profiling) try enabled.append(allocator, .monitoring);
    try enabled.append(allocator, .connectors);
    try enabled.append(allocator, .compute);
    try enabled.append(allocator, .simd);
    return enabled;
}

fn filterDisabledFeatures(
    allocator: std.mem.Allocator,
    enabled: *std.ArrayList(Feature),
    disabled: []const Feature,
) !std.ArrayList(Feature) {
    var filtered = std.ArrayList(Feature).empty;
    for (enabled.items) |feature| {
        if (std.mem.indexOfScalar(Feature, disabled, feature) == null) {
            try filtered.append(allocator, feature);
        }
    }
    return filtered;
}

fn cloneFeatures(allocator: std.mem.Allocator, items: []const Feature) ![]Feature {
    const copy = try allocator.alloc(Feature, items.len);
    std.mem.copyForwards(Feature, copy, items);
    return copy;
}

fn cloneStringList(
    allocator: std.mem.Allocator,
    items: []const []const u8,
) ![]const []const u8 {
    const copy = try allocator.alloc([]const u8, items.len);
    errdefer allocator.free(copy);
    for (items, 0..) |item, i| {
        copy[i] = try allocator.dupe(u8, item);
    }
    return copy;
}

fn freeStringList(allocator: std.mem.Allocator, items: []const []const u8) void {
    for (items) |item| {
        allocator.free(item);
    }
    allocator.free(items);
}
