//! Virtual Table Interface for Pluggable Inference Providers
//!
//! This module defines the common interface for local CPU, GPU, and cloud inference
//! backends. It enables runtime backend selection and makes the system extensible.

const std = @import("std");
const build_options = @import("build_options");

/// Unified error set for all inference providers.
pub const Error = error{
    ProviderNotAvailable,
    InitializationFailed,
    OutOfMemory,
    InvalidModel,
    UnsupportedOperation,
    ExecutionFailed,
};

/// Common response type for all inference operations.
pub const InferenceResult = struct {
    /// The generated text or output
    text: []const u8,
    /// Number of tokens generated
    completion_tokens: usize,
    /// Number of prompt tokens
    prompt_tokens: usize,
    /// Time taken in milliseconds
    latency_ms: f64,
};

/// Configuration for an inference request.
pub const RequestConfig = struct {
    /// Temperature for sampling (0.0 - 2.0)
    temperature: f32 = 0.7,
    /// Top-p sampling
    top_p: f32 = 0.9,
    /// Maximum tokens to generate
    max_tokens: u64 = 1024,
    /// Seed for deterministic output (optional)
    seed: ?u64 = null,
    /// Stop sequences
    stop_sequences: []const []const u8 = &.{},
};

/// Provider type enumeration for type-safe provider selection.
pub const ProviderType = enum {
    cpu,
    gpu,
    connector,
};

/// Health status of a provider.
pub const HealthStatus = struct {
    available: bool,
    latency_ms: f64,
    load: f32,
};

/// VTable for inference providers.
/// Each provider implements this interface and registers itself.
pub const ProviderVTable = struct {
    /// Provider metadata
    name: []const u8,
    provider_type: ProviderType,

    /// Initialize the provider with configuration.
    init_fn: *const fn (
        allocator: std.mem.Allocator,
        model_path: []const u8,
    ) Error!*anyopaque,

    /// Deinitialize and cleanup the provider.
    deinit_fn: *const fn (ctx: *anyopaque) void,

    /// Run inference on the provider.
    infer_fn: *const fn (
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        config: RequestConfig,
    ) Error!InferenceResult,

    /// Check provider health/status.
    health_fn: *const fn (ctx: *anyopaque) HealthStatus,

    /// Check if provider is available without full init.
    check_available_fn: *const fn () bool,
};

/// Generic inference provider wrapper.
/// This provides a type-safe wrapper around any inference backend.
pub fn InferenceProvider(comptime Provider: type) type {
    return struct {
        const Self = @This();

        ctx: *anyopaque,
        vtable: *const ProviderVTable,
        allocator: std.mem.Allocator,

        pub fn init(vtable_ptr: *const ProviderVTable, allocator: std.mem.Allocator, model_path: []const u8) Error!Self {
            const ctx = try vtable_ptr.init_fn(allocator, model_path);
            return Self{
                .ctx = ctx,
                .vtable = vtable_ptr,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.vtable.deinit_fn(self.ctx);
        }

        pub fn infer(self: *Self, prompt: []const u8, config: RequestConfig) Error!InferenceResult {
            return try self.vtable.infer_fn(self.ctx, self.allocator, prompt, config);
        }

        pub fn health(self: *Self) HealthStatus {
            return self.vtable.health_fn(self.ctx);
        }

        pub fn isAvailable(self: *Self) bool {
            return self.vtable.check_available_fn();
        }
    };
}

/// Registry of available providers.
/// This allows dynamic provider registration and selection.
pub const ProviderRegistry = struct {
    var providers: std.ArrayListUnmanaged(ProviderEntry) = .{};
    var initialized = std.atomic.Value(bool).init(false);

    const ProviderEntry = struct {
        name: []const u8,
        vtable: *const ProviderVTable,
    };

    pub fn init(allocator: std.mem.Allocator) void {
        if (initialized.load(.acquire)) return;

        providers = std.ArrayListUnmanaged(ProviderEntry).initCapacity(
            allocator,
            4,
        ) catch return;

        initialized.store(true, .release);
    }

    pub fn deinit(allocator: std.mem.Allocator) void {
        if (!initialized.load(.acquire)) return;
        providers.deinit(allocator);
        initialized.store(false, .release);
    }

    pub fn register(name: []const u8, vtable: *const ProviderVTable) void {
        const allocator = std.heap.page_allocator;
        providers.append(allocator, .{
            .name = name,
            .vtable = vtable,
        }) catch return;
    }

    pub fn getProvider(name: []const u8) ?*const ProviderVTable {
        for (providers.items) |entry| {
            if (std.mem.eql(u8, entry.name, name)) {
                return entry.vtable;
            }
        }
        return null;
    }

    pub fn getAvailableProvider() ?*const ProviderVTable {
        for (providers.items) |entry| {
            if (entry.vtable.check_available_fn()) {
                return entry.vtable;
            }
        }
        return null;
    }

    pub fn listProviders(allocator: std.mem.Allocator) ![][2][]const u8 {
        var result = try allocator.alloc([2][]const u8, providers.items.len);
        for (providers.items, 0..) |entry, i| {
            result[i][0] = entry.name;
            result[i][1] = @tagName(entry.vtable.provider_type);
        }
        return result;
    }
};

// ============================================================================
// Built-in Provider Implementations
// ============================================================================

/// CPU Provider stub implementation.
pub const CpuProvider = struct {
    const Self = @This();

    allocator: std.mem.Allocator,

    fn cpuInit(allocator: std.mem.Allocator, _: []const u8) Error!*anyopaque {
        const ctx = try allocator.create(Self);
        ctx.* = .{ .allocator = allocator };
        return ctx;
    }

    fn cpuDeinit(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.allocator.destroy(self);
    }

    fn cpuInfer(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        config: RequestConfig,
    ) Error!InferenceResult {
        _ = ctx;
        _ = config;

        // In a real implementation, this would inference using CPU
        const output = try std.fmt.allocPrint(allocator, "CPU: {s}", .{prompt});

        return InferenceResult{
            .text = output,
            .completion_tokens = 10,
            .prompt_tokens = prompt.len / 4,
            .latency_ms = 100.0,
        };
    }

    fn cpuHealth(ctx: *anyopaque) HealthStatus {
        _ = ctx;
        return .{ .available = true, .latency_ms = 50.0, .load = 0.5 };
    }

    fn cpuAvailable() bool {
        // CPU is always available
        return true;
    }

    pub const vtable: ProviderVTable = .{
        .name = "cpu",
        .provider_type = .cpu,
        .init_fn = cpuInit,
        .deinit_fn = cpuDeinit,
        .infer_fn = cpuInfer,
        .health_fn = cpuHealth,
        .check_available_fn = cpuAvailable,
    };
};

/// GPU Provider stub implementation.
pub const GpuProvider = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    device_id: u32,

    fn gpuInit(allocator: std.mem.Allocator, model_path: []const u8) Error!*anyopaque {
        if (!build_options.feat_gpu) return Error.ProviderNotAvailable;

        const ctx = try allocator.create(Self);
        ctx.* = .{
            .allocator = allocator,
            .device_id = 0,
        };
        _ = model_path;
        return ctx;
    }

    fn gpuDeinit(ctx: *anyopaque) void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        self.allocator.destroy(self);
    }

    fn gpuInfer(
        ctx: *anyopaque,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        config: RequestConfig,
    ) Error!InferenceResult {
        _ = ctx;
        _ = config;
        const output = try std.fmt.allocPrint(allocator, "GPU: {s}", .{prompt});
        return InferenceResult{
            .text = output,
            .completion_tokens = 10,
            .prompt_tokens = prompt.len / 4,
            .latency_ms = 20.0,
        };
    }

    fn gpuHealth(ctx: *anyopaque) HealthStatus {
        _ = ctx;
        return .{ .available = true, .latency_ms = 10.0, .load = 0.3 };
    }

    fn gpuAvailable() bool {
        return build_options.feat_gpu;
    }

    pub const vtable: ProviderVTable = .{
        .name = "gpu",
        .provider_type = .gpu,
        .init_fn = gpuInit,
        .deinit_fn = gpuDeinit,
        .infer_fn = gpuInfer,
        .health_fn = gpuHealth,
        .check_available_fn = gpuAvailable,
    };
};

test {
    std.testing.refAllDecls(@This());
}
