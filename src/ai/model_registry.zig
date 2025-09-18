//! Advanced Model Registry Module
//!
//! This module provides comprehensive model management capabilities including:
//! - Model versioning and lineage tracking
//! - Model metadata management
//! - Model comparison and analysis
//! - Model deployment tracking
//! - Performance metrics storage
//! - Model lifecycle management

const std = @import("std");
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const Allocator = std.mem.Allocator;

/// Model registry entry
pub const ModelEntry = struct {
    id: []const u8,
    name: []const u8,
    version: []const u8,
    architecture: []const u8,
    framework: []const u8,
    created_at: u64,
    updated_at: u64,
    author: []const u8,
    description: []const u8,

    // Model metadata
    input_shape: []const usize,
    output_shape: []const usize,
    num_parameters: usize,
    model_size_bytes: usize,

    // Training metadata
    training_config: TrainingConfig,
    training_metrics: TrainingMetrics,

    // Deployment info
    deployment_status: DeploymentStatus,
    deployment_version: ?[]const u8,

    // File paths
    model_path: []const u8,
    checkpoint_path: ?[]const u8,

    // Tags and categories
    tags: ArrayList([]const u8),
    categories: ArrayList([]const u8),

    pub const TrainingConfig = struct {
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
        optimizer: []const u8,
        loss_function: []const u8,
        dataset: []const u8,
        total_samples: usize,
    };

    pub const TrainingMetrics = struct {
        final_loss: f32,
        final_accuracy: f32,
        training_time_seconds: u64,
        best_epoch: usize,
        convergence_epoch: ?usize,
    };

    pub const DeploymentStatus = enum {
        development,
        staging,
        production,
        archived,
        deprecated,
    };

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8, version: []const u8) !ModelEntry {
        const id_copy = try allocator.dupe(u8, id);
        errdefer allocator.free(id_copy);
        const name_copy = try allocator.dupe(u8, name);
        errdefer allocator.free(name_copy);
        const version_copy = try allocator.dupe(u8, version);
        errdefer allocator.free(version_copy);

        return ModelEntry{
            .id = id_copy,
            .name = name_copy,
            .version = version_copy,
            .architecture = "",
            .framework = "ABI",
            .created_at = @as(u64, @intCast(std.time.nanoTimestamp())),
            .updated_at = @as(u64, @intCast(std.time.nanoTimestamp())),
            .author = "",
            .description = "",
            .input_shape = &[_]usize{},
            .output_shape = &[_]usize{},
            .num_parameters = 0,
            .model_size_bytes = 0,
            .training_config = undefined,
            .training_metrics = undefined,
            .deployment_status = .development,
            .deployment_version = null,
            .model_path = "",
            .checkpoint_path = null,
            .tags = ArrayList([]const u8).init(allocator),
            .categories = ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *ModelEntry, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        allocator.free(self.version);
        allocator.free(self.architecture);
        allocator.free(self.framework);
        allocator.free(self.author);
        allocator.free(self.description);
        allocator.free(self.input_shape);
        allocator.free(self.output_shape);
        if (self.deployment_version) |dv| allocator.free(dv);
        allocator.free(self.model_path);
        if (self.checkpoint_path) |cp| allocator.free(cp);

        for (self.tags.items) |tag| {
            allocator.free(tag);
        }
        self.tags.deinit();

        for (self.categories.items) |category| {
            allocator.free(category);
        }
        self.categories.deinit();
    }
};

/// Advanced model registry
pub const ModelRegistry = struct {
    allocator: Allocator,
    models: StringHashMap(*ModelEntry),
    versions: StringHashMap(ArrayList(*ModelEntry)), // model_name -> versions
    metrics_history: StringHashMap(ArrayList(PerformanceMetrics)), // model_id -> metrics history

    pub const PerformanceMetrics = struct {
        timestamp: u64,
        accuracy: f32,
        precision: f32,
        recall: f32,
        f1_score: f32,
        latency_ms: f32,
        throughput_samples_per_sec: f32,
        memory_usage_mb: f32,
        gpu_utilization: ?f32,
    };

    pub fn init(allocator: Allocator) ModelRegistry {
        return ModelRegistry{
            .allocator = allocator,
            .models = StringHashMap(*ModelEntry).init(allocator),
            .versions = StringHashMap(ArrayList(*ModelEntry)).init(allocator),
            .metrics_history = StringHashMap(ArrayList(PerformanceMetrics)).init(allocator),
        };
    }

    pub fn deinit(self: *ModelRegistry) void {
        var model_it = self.models.iterator();
        while (model_it.next()) |entry| {
            entry.value_ptr.*.deinit(self.allocator);
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.models.deinit();

        var version_it = self.versions.iterator();
        while (version_it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.versions.deinit();

        var metrics_it = self.metrics_history.iterator();
        while (metrics_it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }
        self.metrics_history.deinit();
    }

    /// Register a new model
    pub fn registerModel(self: *ModelRegistry, entry: *ModelEntry) !void {
        const id_copy = try self.allocator.dupe(u8, entry.id);
        defer self.allocator.free(id_copy);

        try self.models.put(id_copy, entry);

        // Add to versions map
        const name_copy = try self.allocator.dupe(u8, entry.name);
        defer self.allocator.free(name_copy);

        const versions_entry = try self.versions.getOrPut(name_copy);
        if (!versions_entry.found_existing) {
            versions_entry.value_ptr.* = ArrayList(*ModelEntry).init(self.allocator);
        }
        try versions_entry.value_ptr.*.append(entry);

        // Initialize metrics history
        const metrics_id_copy = try self.allocator.dupe(u8, entry.id);
        defer self.allocator.free(metrics_id_copy);

        try self.metrics_history.put(metrics_id_copy, ArrayList(PerformanceMetrics).init(self.allocator));

        std.debug.print("Registered model: {} v{}\n", .{ entry.name, entry.version });
    }

    /// Get model by ID
    pub fn getModel(self: *ModelRegistry, id: []const u8) ?*ModelEntry {
        return self.models.get(id);
    }

    /// Get all versions of a model
    pub fn getModelVersions(self: *ModelRegistry, name: []const u8) ?[]*ModelEntry {
        const versions = self.versions.get(name) orelse return null;
        return versions.items;
    }

    /// Get latest version of a model
    pub fn getLatestVersion(self: *ModelRegistry, name: []const u8) ?*ModelEntry {
        const versions = self.getModelVersions(name) orelse return null;
        if (versions.len == 0) return null;

        var latest = versions[0];
        for (versions[1..]) |version| {
            if (std.mem.order(u8, version.version, latest.version) == .gt) {
                latest = version;
            }
        }
        return latest;
    }

    /// Compare two models
    pub fn compareModels(self: *ModelRegistry, id1: []const u8, id2: []const u8) !ModelComparison {
        const model1 = self.getModel(id1) orelse return error.ModelNotFound;
        const model2 = self.getModel(id2) orelse return error.ModelNotFound;

        return ModelComparison{
            .model1_id = model1.id,
            .model2_id = model2.id,
            .parameter_difference = @as(isize, @intCast(model2.num_parameters)) - @as(isize, @intCast(model1.num_parameters)),
            .accuracy_difference = model2.training_metrics.final_accuracy - model1.training_metrics.final_accuracy,
            .size_difference_bytes = @as(isize, @intCast(model2.model_size_bytes)) - @as(isize, @intCast(model1.model_size_bytes)),
            .architecture_changed = !std.mem.eql(u8, model1.architecture, model2.architecture),
        };
    }

    /// Record performance metrics
    pub fn recordMetrics(self: *ModelRegistry, model_id: []const u8, metrics: PerformanceMetrics) !void {
        const history = self.metrics_history.getPtr(model_id) orelse return error.ModelNotFound;
        try history.append(metrics);
    }

    /// Get performance history
    pub fn getPerformanceHistory(self: *ModelRegistry, model_id: []const u8) ?[]PerformanceMetrics {
        const history = self.metrics_history.get(model_id) orelse return null;
        return history.items;
    }

    /// Promote model to production
    pub fn promoteToProduction(self: *ModelRegistry, model_id: []const u8) !void {
        const model = self.getModel(model_id) orelse return error.ModelNotFound;
        model.deployment_status = .production;
        model.updated_at = @as(u64, @intCast(std.time.nanoTimestamp()));

        // Set deployment version
        if (model.deployment_version) |dv| {
            self.allocator.free(dv);
        }
        model.deployment_version = try self.allocator.dupe(u8, model.version);

        std.debug.print("Promoted model {} to production\n", .{model.name});
    }

    /// Archive old model versions
    pub fn archiveOldVersions(self: *ModelRegistry, model_name: []const u8, keep_versions: usize) !void {
        const versions = self.getModelVersions(model_name) orelse return;

        if (versions.len <= keep_versions) return;

        // Sort by version (newest first)
        // Archive older versions
        for (versions[keep_versions..]) |model| {
            model.deployment_status = .archived;
            model.updated_at = @as(u64, @intCast(std.time.nanoTimestamp()));
        }

        std.debug.print("Archived {} old versions of model {}\n", .{ versions.len - keep_versions, model_name });
    }

    /// Search models by tags
    pub fn searchByTags(self: *ModelRegistry, tags: []const []const u8) ![]*ModelEntry {
        var results = ArrayList(*ModelEntry).init(self.allocator);
        errdefer results.deinit();

        var model_it = self.models.iterator();
        while (model_it.next()) |entry| {
            const model = entry.value_ptr.*;
            var has_all_tags = true;

            for (tags) |search_tag| {
                var found_tag = false;
                for (model.tags.items) |model_tag| {
                    if (std.mem.eql(u8, search_tag, model_tag)) {
                        found_tag = true;
                        break;
                    }
                }
                if (!found_tag) {
                    has_all_tags = false;
                    break;
                }
            }

            if (has_all_tags) {
                try results.append(model);
            }
        }

        return results.toOwnedSlice();
    }
};

/// Model comparison result
pub const ModelComparison = struct {
    model1_id: []const u8,
    model2_id: []const u8,
    parameter_difference: isize,
    accuracy_difference: f32,
    size_difference_bytes: isize,
    architecture_changed: bool,

    pub fn format(
        self: ModelComparison,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Model Comparison:\n", .{});
        try writer.print("  Model 1: {s}\n", .{self.model1_id});
        try writer.print("  Model 2: {s}\n", .{self.model2_id});
        try writer.print("  Parameter difference: {}\n", .{self.parameter_difference});
        try writer.print("  Accuracy difference: {d:.4}\n", .{self.accuracy_difference});
        try writer.print("  Size difference: {} bytes\n", .{self.size_difference_bytes});
        try writer.print("  Architecture changed: {}\n", .{self.architecture_changed});
    }
};

/// Model registry CLI interface
pub const RegistryCLI = struct {
    registry: *ModelRegistry,

    pub fn init(registry: *ModelRegistry) RegistryCLI {
        return RegistryCLI{ .registry = registry };
    }

    pub fn listModels(self: RegistryCLI) !void {
        std.debug.print("Registered Models:\n", .{});
        std.debug.print("==================\n", .{});

        var it = self.registry.models.iterator();
        while (it.next()) |entry| {
            const model = entry.value_ptr.*;
            std.debug.print("ID: {s}\n", .{model.id});
            std.debug.print("  Name: {s} v{s}\n", .{ model.name, model.version });
            std.debug.print("  Architecture: {s}\n", .{model.architecture});
            std.debug.print("  Parameters: {}\n", .{model.num_parameters});
            std.debug.print("  Status: {}\n", .{@tagName(model.deployment_status)});
            std.debug.print("  Created: {}\n", .{model.created_at});
            std.debug.print("\n", .{});
        }
    }

    pub fn showModelDetails(self: RegistryCLI, model_id: []const u8) !void {
        const model = self.registry.getModel(model_id) orelse {
            std.debug.print("Model not found: {s}\n", .{model_id});
            return error.ModelNotFound;
        };

        std.debug.print("Model Details:\n", .{});
        std.debug.print("==============\n", .{});
        std.debug.print("ID: {s}\n", .{model.id});
        std.debug.print("Name: {s}\n", .{model.name});
        std.debug.print("Version: {s}\n", .{model.version});
        std.debug.print("Architecture: {s}\n", .{model.architecture});
        std.debug.print("Author: {s}\n", .{model.author});
        std.debug.print("Description: {s}\n", .{model.description});
        std.debug.print("Parameters: {}\n", .{model.num_parameters});
        std.debug.print("Model Size: {} bytes\n", .{model.model_size_bytes});
        std.debug.print("Status: {}\n", .{@tagName(model.deployment_status)});
        std.debug.print("Created: {}\n", .{model.created_at});
        std.debug.print("Updated: {}\n", .{model.updated_at});

        if (model.tags.items.len > 0) {
            std.debug.print("Tags: ", .{});
            for (model.tags.items, 0..) |tag, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{s}", .{tag});
            }
            std.debug.print("\n", .{});
        }
    }

    pub fn compareModelsCLI(self: RegistryCLI, id1: []const u8, id2: []const u8) !void {
        const comparison = try self.registry.compareModels(id1, id2);
        try comparison.format("", .{}, std.io.getStdOut().writer());
    }
};

test "Model registry basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    // Create a test model entry
    var model = try ModelEntry.init(allocator, "test-model-1", "TestModel", "1.0.0");
    defer model.deinit(allocator);

    // Set some basic properties
    model.architecture = try allocator.dupe(u8, "Transformer");
    model.num_parameters = 1000000;
    model.model_size_bytes = 4000000;

    // Register the model
    try registry.registerModel(&model);

    // Test retrieval
    const retrieved = registry.getModel("test-model-1");
    try testing.expect(retrieved != null);
    try testing.expect(std.mem.eql(u8, retrieved.?.name, "TestModel"));
    try testing.expect(std.mem.eql(u8, retrieved.?.version, "1.0.0"));
    try testing.expect(retrieved.?.num_parameters == 1000000);
}

test "Model version management" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var registry = ModelRegistry.init(allocator);
    defer registry.deinit();

    // Create multiple versions
    var model_v1 = try ModelEntry.init(allocator, "test-model-v1", "TestModel", "1.0.0");
    defer model_v1.deinit(allocator);
    model_v1.architecture = try allocator.dupe(u8, "MLP");

    var model_v2 = try ModelEntry.init(allocator, "test-model-v2", "TestModel", "2.0.0");
    defer model_v2.deinit(allocator);
    model_v2.architecture = try allocator.dupe(u8, "Transformer");

    try registry.registerModel(&model_v1);
    try registry.registerModel(&model_v2);

    // Test version retrieval
    const versions = registry.getModelVersions("TestModel");
    try testing.expect(versions != null);
    try testing.expect(versions.?.len == 2);

    // Test latest version
    const latest = registry.getLatestVersion("TestModel");
    try testing.expect(latest != null);
    try testing.expect(std.mem.eql(u8, latest.?.version, "2.0.0"));
}
