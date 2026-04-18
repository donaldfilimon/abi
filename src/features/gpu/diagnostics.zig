//! GPU Diagnostics Module
//!
//! Provides comprehensive debugging information about GPU backend state,
//! memory usage, kernel cache, failover status, and error history.

const std = @import("std");
const interface = @import("interface.zig");
const backend = @import("backend.zig");
const Backend = backend.Backend;
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
        try writer.writeAll("GPU Diagnostics\n");
        try writer.print("Backend: {s}\n", .{self.backend_name});
        try writer.print("Device Count: {d}\n", .{self.device_count});
        try writer.print("Initialized: {s}\n", .{if (self.is_initialized) "yes" else "no"});
        try writer.print("Degraded Mode: {s}\n", .{if (self.is_degraded) "yes" else "no"});

        try writer.writeAll("\nBuild Configuration:\n");
        try writer.print("  GPU: {s}\n", .{if (self.build_config.gpu_enabled) "enabled" else "disabled"});
        try writer.print("  CUDA: {s}\n", .{if (self.build_config.cuda_enabled) "enabled" else "disabled"});
        try writer.print("  Vulkan: {s}\n", .{if (self.build_config.vulkan_enabled) "enabled" else "disabled"});
        try writer.print("  Metal: {s}\n", .{if (self.build_config.metal_enabled) "enabled" else "disabled"});
        try writer.print("  WebGPU: {s}\n", .{if (self.build_config.webgpu_enabled) "enabled" else "disabled"});
        try writer.print("  OpenGL: {s}\n", .{if (self.build_config.opengl_enabled) "enabled" else "disabled"});
        try writer.print("  stdGPU: {s}\n", .{if (self.build_config.stdgpu_enabled) "enabled" else "disabled"});
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
        .gpu_enabled = build_options.feat_gpu,
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

    // Get backend availability - check each backend
    var device_count: u32 = 0;
    var backend_name: []const u8 = "none";
    var backend_type: ?interface.BackendType = null;

    inline for (std.enums.values(backend.Backend)) |b| {
        const avail = backend.backendAvailability(b);
        if (avail.available and device_count == 0) {
            device_count = if (avail.device_count > 0) avail.device_count else 1;
            backend_name = backend.backendName(b);
            backend_type = b;
            break;
        }
    }

    // Determine if running in degraded mode (CPU fallback when better backends available)
    var is_degraded = false;
    if (backend_type == .stdgpu) {
        // stdgpu is degraded only if better backends are build-enabled (not necessarily available at runtime)
        inline for (.{ Backend.cuda, Backend.vulkan, Backend.metal, Backend.webgpu, Backend.opengl }) |preferred| {
            const avail = backend.backendAvailability(preferred);
            if (avail.enabled and avail.available) {
                is_degraded = true;
                break;
            }
        }
    }

    return .{
        .backend_type = backend_type,
        .backend_name = backend_name,
        .device_count = device_count,
        .active_device = if (device_count > 0) 0 else null,
        .is_initialized = backend.moduleEnabled(),
        .is_degraded = is_degraded,
        .memory_stats = null,
        .cache_stats = null,
        .failover_stats = null,
        .error_stats = null,
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
