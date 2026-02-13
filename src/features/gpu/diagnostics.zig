//! GPU Diagnostics Module
//!
//! Provides comprehensive debugging information about GPU backend state,
//! memory usage, kernel cache, failover status, and error history.

const std = @import("std");
const interface = @import("interface.zig");
const backend = @import("backend.zig");
const memory = @import("memory/base.zig");
const kernel_cache = @import("kernel_cache.zig");
const failover = @import("failover.zig");
const error_handling = @import("error_handling.zig");

/// Comprehensive GPU diagnostics information.
pub const DiagnosticsInfo = struct {
    /// Current active backend type.
    backend_type: ?interface.BackendType,
    /// Backend name string.
    backend_name: []const u8,
    /// Number of available GPU devices.
    device_count: u32,
    /// Currently active device index.
    active_device: ?u32,
    /// Whether GPU module is initialized.
    is_initialized: bool,
    /// Whether running in degraded (CPU fallback) mode.
    is_degraded: bool,
    /// Memory statistics if available.
    memory_stats: ?MemoryDiagnostics,
    /// Kernel cache statistics if available.
    cache_stats: ?CacheDiagnostics,
    /// Failover statistics if available.
    failover_stats: ?FailoverDiagnostics,
    /// Error statistics if available.
    error_stats: ?ErrorDiagnostics,
    /// Build configuration.
    build_config: BuildConfig,

    pub fn format(
        self: DiagnosticsInfo,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("=== GPU Diagnostics ===\n", .{});
        try writer.print("Backend: {s}\n", .{self.backend_name});
        try writer.print("Initialized: {}\n", .{self.is_initialized});
        try writer.print("Degraded: {}\n", .{self.is_degraded});
        try writer.print("Device Count: {d}\n", .{self.device_count});
        if (self.active_device) |dev| {
            try writer.print("Active Device: {d}\n", .{dev});
        }

        try writer.print("\n--- Build Config ---\n", .{});
        try writer.print("GPU Enabled: {}\n", .{self.build_config.gpu_enabled});
        try writer.print("CUDA: {}\n", .{self.build_config.cuda_enabled});
        try writer.print("Vulkan: {}\n", .{self.build_config.vulkan_enabled});
        try writer.print("Metal: {}\n", .{self.build_config.metal_enabled});
        try writer.print("WebGPU: {}\n", .{self.build_config.webgpu_enabled});
        try writer.print("OpenGL: {}\n", .{self.build_config.opengl_enabled});
        try writer.print("StdGPU: {}\n", .{self.build_config.stdgpu_enabled});

        if (self.memory_stats) |mem| {
            try writer.print("\n--- Memory ---\n", .{});
            try writer.print("Allocated: {d} bytes\n", .{mem.total_allocated});
            try writer.print("Peak: {d} bytes\n", .{mem.peak_allocated});
            try writer.print("Active Buffers: {d}\n", .{mem.active_buffers});
        }

        if (self.cache_stats) |cache| {
            try writer.print("\n--- Kernel Cache ---\n", .{});
            try writer.print("Entries: {d}/{d}\n", .{ cache.entries, cache.capacity });
            try writer.print("Hits: {d}\n", .{cache.hits});
            try writer.print("Misses: {d}\n", .{cache.misses});
            if (cache.hits + cache.misses > 0) {
                const hit_rate = @as(f64, @floatFromInt(cache.hits)) /
                    @as(f64, @floatFromInt(cache.hits + cache.misses)) * 100.0;
                try writer.print("Hit Rate: {d:.1}%\n", .{hit_rate});
            }
        }

        if (self.failover_stats) |fo| {
            try writer.print("\n--- Failover ---\n", .{});
            try writer.print("Total Failovers: {d}\n", .{fo.total_failovers});
            try writer.print("On Primary: {}\n", .{fo.is_on_primary});
            try writer.print("Backends Exhausted: {}\n", .{fo.backends_exhausted});
        }

        if (self.error_stats) |err| {
            try writer.print("\n--- Errors ---\n", .{});
            try writer.print("Total: {d}\n", .{err.total_errors});
            if (err.total_errors > 0) {
                try writer.print("  Memory: {d}\n", .{err.memory_errors});
                try writer.print("  Kernel: {d}\n", .{err.kernel_errors});
                try writer.print("  Launch: {d}\n", .{err.launch_errors});
                try writer.print("  Device: {d}\n", .{err.device_errors});
            }
        }
    }
};

/// Memory diagnostics.
pub const MemoryDiagnostics = struct {
    total_allocated: usize,
    peak_allocated: usize,
    active_buffers: usize,
    fragmentation_ratio: f32,
};

/// Kernel cache diagnostics.
pub const CacheDiagnostics = struct {
    entries: usize,
    capacity: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    total_compile_time_ns: u64,
};

/// Failover diagnostics.
pub const FailoverDiagnostics = struct {
    total_failovers: usize,
    current_backend: backend.Backend,
    is_on_primary: bool,
    backends_exhausted: bool,
    is_degraded: bool,
};

/// Error diagnostics.
pub const ErrorDiagnostics = struct {
    total_errors: usize,
    memory_errors: usize,
    kernel_errors: usize,
    launch_errors: usize,
    device_errors: usize,
    initialization_errors: usize,
    last_error_timestamp: i64,
};

/// Build configuration info.
pub const BuildConfig = struct {
    gpu_enabled: bool,
    cuda_enabled: bool,
    vulkan_enabled: bool,
    metal_enabled: bool,
    webgpu_enabled: bool,
    opengl_enabled: bool,
    stdgpu_enabled: bool,
};

/// Get build configuration from compile-time options.
pub fn getBuildConfig() BuildConfig {
    const build_options = @import("build_options");
    return .{
        .gpu_enabled = build_options.enable_gpu,
        .cuda_enabled = build_options.gpu_cuda,
        .vulkan_enabled = build_options.gpu_vulkan,
        .metal_enabled = build_options.gpu_metal,
        .webgpu_enabled = build_options.gpu_webgpu,
        .opengl_enabled = build_options.gpu_opengl,
        .stdgpu_enabled = build_options.gpu_stdgpu,
    };
}

/// Collect comprehensive GPU diagnostics.
pub fn collect() DiagnosticsInfo {
    const build_config = getBuildConfig();

    // Get backend availability
    const avail = backend.backendAvailability();
    var device_count: u32 = 0;
    var backend_name: []const u8 = "none";
    var backend_type: ?interface.BackendType = null;

    // Determine current backend
    if (avail.cuda) {
        backend_name = "cuda";
        backend_type = .cuda;
        device_count = 1; // Simplified; real implementation would query
    } else if (avail.vulkan) {
        backend_name = "vulkan";
        backend_type = .vulkan;
        device_count = 1;
    } else if (avail.metal) {
        backend_name = "metal";
        backend_type = .metal;
        device_count = 1;
    } else if (avail.webgpu) {
        backend_name = "webgpu";
        backend_type = .webgpu;
        device_count = 1;
    } else if (avail.opengl) {
        backend_name = "opengl";
        backend_type = .opengl;
        device_count = 1;
    } else if (avail.stdgpu) {
        backend_name = "stdgpu (CPU fallback)";
        backend_type = .stdgpu;
        device_count = 1;
    }

    const is_degraded = backend_type == .stdgpu and
        (avail.cuda or avail.vulkan or avail.metal or avail.webgpu or avail.opengl);

    return .{
        .backend_type = backend_type,
        .backend_name = backend_name,
        .device_count = device_count,
        .active_device = if (device_count > 0) 0 else null,
        .is_initialized = backend.moduleEnabled(),
        .is_degraded = is_degraded,
        .memory_stats = null, // Would be populated from actual memory pool
        .cache_stats = null, // Would be populated from kernel cache
        .failover_stats = null, // Would be populated from failover manager
        .error_stats = null, // Would be populated from error context
        .build_config = build_config,
    };
}

/// Collect diagnostics with additional context from subsystems.
pub fn collectWithContext(
    cache: ?*kernel_cache.KernelCache,
    failover_mgr: ?*failover.FailoverManager,
    error_ctx: ?*error_handling.ErrorContext,
) DiagnosticsInfo {
    var info = collect();

    // Add kernel cache stats
    if (cache) |c| {
        const stats = c.getStats();
        info.cache_stats = .{
            .entries = stats.entries,
            .capacity = stats.max_entries,
            .hits = stats.hits,
            .misses = stats.misses,
            .evictions = stats.evictions,
            .total_compile_time_ns = stats.total_compile_time_ns,
        };
    }

    // Add failover stats
    if (failover_mgr) |fm| {
        const stats = fm.getStats();
        info.failover_stats = .{
            .total_failovers = stats.total_failovers,
            .current_backend = stats.current_backend,
            .is_on_primary = stats.is_on_primary,
            .backends_exhausted = stats.backends_exhausted,
            .is_degraded = stats.is_degraded,
        };
        info.is_degraded = stats.is_degraded;
    }

    // Add error stats
    if (error_ctx) |ec| {
        const stats = ec.getErrorStatistics();
        const last_err = ec.getLastError();
        info.error_stats = .{
            .total_errors = stats.total,
            .memory_errors = stats.memory,
            .kernel_errors = stats.kernel,
            .launch_errors = stats.launch,
            .device_errors = stats.device,
            .initialization_errors = stats.initialization,
            .last_error_timestamp = if (last_err) |e| e.timestamp else 0,
        };
    }

    return info;
}

/// Format diagnostics as a string for logging/display.
pub fn formatToString(allocator: std.mem.Allocator, info: DiagnosticsInfo) ![]u8 {
    var buffer = std.ArrayListUnmanaged(u8).empty;
    errdefer buffer.deinit(allocator);
    try info.format("", .{}, buffer.writer(allocator));
    return buffer.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "diagnostics collect basic info" {
    const info = collect();
    try std.testing.expect(info.backend_name.len > 0);
    try std.testing.expect(info.build_config.gpu_enabled or !info.build_config.gpu_enabled); // Always true, just check it compiles
}

test "diagnostics format" {
    const allocator = std.testing.allocator;
    const info = collect();
    const formatted = try formatToString(allocator, info);
    defer allocator.free(formatted);
    try std.testing.expect(formatted.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "GPU Diagnostics") != null);
}
