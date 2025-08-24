//! MLAI Module - Machine Learning and AI Utilities
//!
//! This module provides comprehensive machine learning and AI capabilities:
//! - Vector database operations (WDBX format)
//! - Machine learning algorithms and models
//! - Data preprocessing and feature engineering
//! - Model training and evaluation
//! - AI agent integration
//! - Performance monitoring and optimization

const std = @import("std");
const core = @import("../core/mod.zig");

/// Re-export commonly used types
pub const Allocator = core.Allocator;

/// Core MLAI components
pub const mlai = @import("mlai.zig");
pub const wdbx = @import("wdbx/db.zig");

/// Re-export main types for convenience
pub const MLData = mlai.MLData;
pub const LinearRegression = mlai.LinearRegression;

/// MLAI configuration
pub const MLAIConfig = struct {
    /// Enable GPU acceleration for ML operations
    enable_gpu: bool = true,

    /// Enable SIMD optimizations
    enable_simd: bool = true,

    /// Default batch size for operations
    default_batch_size: u32 = 32,

    /// Enable model caching
    enable_caching: bool = true,

    /// Maximum memory usage for ML operations (bytes)
    max_memory: usize = 2 * 1024 * 1024 * 1024, // 2GB

    /// Enable automatic model saving
    auto_save_models: bool = true,

    /// Model save directory
    model_dir: []const u8 = "./models",
};

/// Model types supported by the framework
pub const ModelType = enum {
    linear_regression,
    logistic_regression,
    neural_network,
    decision_tree,
    random_forest,
    svm,
    kmeans,
    pca,
    embedding,
    custom,
};

/// Training configuration
pub const TrainingConfig = struct {
    /// Learning rate
    learning_rate: f32 = 0.001,

    /// Number of epochs
    epochs: u32 = 100,

    /// Batch size
    batch_size: u32 = 32,

    /// Validation split ratio
    validation_split: f32 = 0.2,

    /// Early stopping patience
    patience: u32 = 10,

    /// Enable verbose logging
    verbose: bool = false,
};

/// Model evaluation metrics
pub const Metrics = struct {
    /// Mean Absolute Error
    mae: f32 = 0.0,

    /// Mean Squared Error
    mse: f32 = 0.0,

    /// Root Mean Squared Error
    rmse: f32 = 0.0,

    /// R-squared score
    r2_score: f32 = 0.0,

    /// Accuracy (for classification)
    accuracy: f32 = 0.0,

    /// Precision (for classification)
    precision: f32 = 0.0,

    /// Recall (for classification)
    recall: f32 = 0.0,

    /// F1 score (for classification)
    f1_score: f32 = 0.0,
};

/// Dataset structure for ML operations
pub const Dataset = struct {
    /// Feature matrix (n_samples x n_features)
    features: []const []const f32,

    /// Target values
    targets: []const f32,

    /// Feature names
    feature_names: ?[]const []const u8 = null,

    /// Target name
    target_name: ?[]const u8 = null,

    /// Number of samples
    pub fn nSamples(self: *const Dataset) usize {
        return self.features.len;
    }

    /// Number of features
    pub fn nFeatures(self: *const Dataset) usize {
        if (self.features.len == 0) return 0;
        return self.features[0].len;
    }
};

/// Data preprocessing utilities
pub const preprocessing = struct {
    /// Standardize features (zero mean, unit variance)
    pub fn standardize(data: []f32) void {
        if (data.len == 0) return;

        // Calculate mean
        var mean: f32 = 0.0;
        for (data) |val| {
            mean += val;
        }
        mean /= @as(f32, @floatFromInt(data.len));

        // Calculate standard deviation
        var variance: f32 = 0.0;
        for (data) |val| {
            const diff = val - mean;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(data.len));
        const std_dev = @sqrt(variance);

        // Standardize
        if (std_dev > 0.0) {
            for (data) |*val| {
                val.* = (val.* - mean) / std_dev;
            }
        }
    }

    /// Normalize features to [0, 1] range
    pub fn normalize(data: []f32) void {
        if (data.len == 0) return;

        // Find min and max
        var min_val = data[0];
        var max_val = data[0];
        for (data[1..]) |val| {
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        // Normalize
        const range = max_val - min_val;
        if (range > 0.0) {
            for (data) |*val| {
                val.* = (val.* - min_val) / range;
            }
        }
    }
};

/// Model registry for managing trained models
pub const ModelRegistry = struct {
    allocator: Allocator,
    models: std.StringHashMap(*anyopaque),
    model_types: std.StringHashMap(ModelType),

    /// Initialize a new model registry
    pub fn init(allocator: Allocator) ModelRegistry {
        return .{
            .allocator = allocator,
            .models = std.StringHashMap(*anyopaque).init(allocator),
            .model_types = std.StringHashMap(ModelType).init(allocator),
        };
    }

    /// Deinitialize the registry
    pub fn deinit(self: *ModelRegistry) void {
        self.models.deinit();
        self.model_types.deinit();
    }

    /// Register a model
    pub fn registerModel(
        self: *ModelRegistry,
        name: []const u8,
        model_type: ModelType,
        model: *anyopaque,
    ) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        try self.models.put(name_copy, model);
        try self.model_types.put(name_copy, model_type);
    }

    /// Get a model by name
    pub fn getModel(self: *const ModelRegistry, name: []const u8) ?*anyopaque {
        return self.models.get(name);
    }

    /// Get model type by name
    pub fn getModelType(self: *const ModelRegistry, name: []const u8) ?ModelType {
        return self.model_types.get(name);
    }

    /// List all registered models
    pub fn listModels(self: *const ModelRegistry) []const []const u8 {
        return self.models.keys();
    }
};

/// Initialize the MLAI system
pub fn init(config: MLAIConfig) !void {
    _ = config;
    // TODO: Implement MLAI system initialization
}

/// Deinitialize the MLAI system
pub fn deinit() void {
    // TODO: Implement MLAI system cleanup
}

/// Get the global model registry
pub fn getRegistry() *ModelRegistry {
    // TODO: Implement global registry
    return undefined;
}
