//! ABI Root Module - Modernized for Zig 0.16
//!
//! This is the main entry point for the ABI framework, providing a clean
//! interface to all modernized components with proper initialization patterns

const std = @import("std");

// Core framework exports
pub const framework = @import("framework/runtime_modern.zig");
pub const ml = @import("ml/ml_modern.zig");
pub const utils = @import("shared/utils_modern.zig");
pub const collections = @import("core/collections.zig");
pub const metrics = @import("metrics.zig");
pub const simd = @import("simd.zig");

// Feature exports
pub const ai = @import("features/ai/mod.zig");
pub const gpu = @import("features/gpu/mod.zig");
pub const database = @import("features/database/mod.zig");
pub const web = @import("features/web/mod.zig");
pub const connectors = @import("features/connectors/mod.zig");
pub const memory = utils.memory;

// Legacy plugin system (maintained for compatibility)
pub const plugins = @import("shared/mod.zig");

// Additional modernized modules
pub const abi = struct {
    /// Framework version information
    pub const VERSION = struct {
        pub const MAJOR = 2;
        pub const MINOR = 0;
        pub const PATCH = 0;
        pub const BUILD = "zig-0.16-dev";

        pub fn string() []const u8 {
            return "2.0.0-zig-0.16-dev";
        }

        pub fn isCompatible(major: u32, minor: u32) bool {
            return major == MAJOR and minor <= MINOR;
        }
    };

    pub const ai = @import("features/ai/mod.zig");

    /// Framework initialization
    pub fn init(allocator: std.mem.Allocator, config: framework.RuntimeConfig) !framework.Runtime {
        return try framework.createRuntime(allocator, config);
    }

    /// Default configuration
    pub fn defaultConfig() framework.RuntimeConfig {
        return framework.defaultConfig();
    }

    /// Create a new neural network
    pub fn createNeuralNetwork(allocator: std.mem.Allocator) ml.NeuralNetwork {
        return ml.NeuralNetwork.init(allocator);
    }

    /// Create a memory pool for specific type
    pub fn createMemoryPool(comptime T: type, allocator: std.mem.Allocator, initial_capacity: usize) !memory.MemoryPool(T) {
        return try utils.memory.MemoryPool(T).create(allocator, initial_capacity);
    }

    /// Create configuration manager
    pub fn createConfig(allocator: std.mem.Allocator) utils.config.Config {
        return utils.config.Config.init(allocator);
    }
};

/// Convenience re-exports for common types
pub const ArrayList = collections.ArrayList;
pub const StringHashMap = collections.StringHashMap;
pub const AutoHashMap = collections.AutoHashMap;
pub const ArenaAllocator = collections.ArenaAllocator;

pub const Runtime = framework.Runtime;
pub const RuntimeConfig = framework.RuntimeConfig;
pub const Component = framework.Component;

pub const NeuralNetwork = ml.NeuralNetwork;
pub const Layer = ml.Layer;
pub const LayerConfig = ml.LayerConfig;
pub const Activation = ml.Activation;
pub const VectorOps = ml.VectorOps;

/// Global framework initialization function
pub fn initFramework(allocator: std.mem.Allocator, config: ?RuntimeConfig) !Runtime {
    const runtime_config = config orelse abi.defaultConfig();
    return try abi.init(allocator, runtime_config);
}

test "abi root - version info" {
    const testing = std.testing;

    try testing.expectEqualStrings("2.0.0-zig-0.16-dev", abi.VERSION.string());
    try testing.expect(abi.VERSION.isCompatible(2, 0));
    try testing.expect(!abi.VERSION.isCompatible(1, 0));
    try testing.expect(!abi.VERSION.isCompatible(3, 0));
}

test "abi root - framework initialization" {
    const testing = std.testing;

    var runtime_instance = try initFramework(testing.allocator, null);
    defer runtime_instance.deinit();

    try testing.expect(!runtime_instance.isRunning());

    try runtime_instance.start();
    try testing.expect(runtime_instance.isRunning());

    runtime_instance.stop();
    try testing.expect(!runtime_instance.isRunning());
}

test "abi root - neural network creation" {
    const testing = std.testing;

    var network = abi.createNeuralNetwork(testing.allocator);
    defer network.deinit();

    try network.addLayer(.{
        .layer_type = .dense,
        .input_size = 2,
        .output_size = 1,
        .activation = .sigmoid,
    });

    try testing.expectEqual(@as(?u32, 2), network.getInputSize());
    try testing.expectEqual(@as(?u32, 1), network.getOutputSize());
}

test "abi root - memory pool creation" {
    const testing = std.testing;

    var pool = try abi.createMemoryPool(u32, testing.allocator, 10);
    defer pool.deinit();

    const item = try pool.acquire();
    item.* = 42;

    pool.release(item);
}
test "root module - modern collections aliases" {
    const testing = std.testing;
    var list = std.ArrayList(i32){};
    defer list.deinit(testing.allocator);

    try list.append(testing.allocator, 42);
    try testing.expectEqual(@as(usize, 1), list.items.len);
    try testing.expectEqual(@as(i32, 42), list.items[0]);

    var map = std.StringHashMap(u32).init(testing.allocator);
    defer map.deinit();

    try map.put("answer", 42);
    const value = map.get("answer");
    try testing.expect(value != null);
    try testing.expectEqual(@as(u32, 42), value.?);
}
