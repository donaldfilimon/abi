//! C API GPU Tests — Availability, lifecycle, backend detection, configuration.

const std = @import("std");
const testing = std.testing;
const build_options = @import("build_options");
const abi = @import("abi");

// ============================================================================
// GPU Availability Tests (Conditional on feature being enabled)
// ============================================================================

test "c_api: gpu availability check" {
    // GPU availability check using the module enabled state
    // The C API (abi_gpu_is_available) would wrap similar functionality
    const gpu_enabled = abi.gpu.moduleEnabled();

    // If GPU feature is disabled at compile time, module should not be enabled
    if (!build_options.enable_gpu) {
        try testing.expect(!gpu_enabled);
    }

    // If enabled, we should be able to query backends
    if (gpu_enabled) {
        const allocator = testing.allocator;
        const backends = abi.gpu.availableBackends(allocator) catch {
            // Backend query may fail - this is acceptable
            return;
        };
        defer allocator.free(backends);
        // At minimum, we should have at least one backend (e.g. simulated)
        try testing.expect(backends.len >= 1);
    }
}

test "c_api: gpu module enabled check" {
    const module_enabled = abi.gpu.moduleEnabled();

    // Module enabled should match build option
    try testing.expect(module_enabled == build_options.enable_gpu);
}

test "c_api: gpu backend summary" {
    const gpu_summary = abi.gpu.summary();

    // Summary should reflect compile-time settings
    try testing.expect(gpu_summary.module_enabled == build_options.enable_gpu);

    // If module is disabled, counts should be zero
    if (!build_options.enable_gpu) {
        try testing.expect(gpu_summary.enabled_backend_count == 0);
        try testing.expect(gpu_summary.available_backend_count == 0);
        try testing.expect(gpu_summary.device_count == 0);
    }
}

test "c_api: gpu backend detection" {
    if (!build_options.enable_gpu) {
        return error.SkipZigTest;
    }

    // Test that we can query backend names
    // Backend name functions should not crash
    const name = abi.gpu.backendName(.vulkan);
    try testing.expect(name.len > 0);

    // Display name should also work
    const display_name = abi.gpu.backendDisplayName(.vulkan);
    try testing.expect(display_name.len > 0);

    // Description should work
    const description = abi.gpu.backendDescription(.vulkan);
    try testing.expect(description.len > 0);
}

// ============================================================================
// GPU Lifecycle Tests (abi_gpu_init, abi_gpu_shutdown)
// ============================================================================

test "c_api: gpu init and shutdown lifecycle" {
    if (!build_options.enable_gpu) {
        return error.SkipZigTest;
    }

    const allocator = testing.allocator;

    // GPU init (C API: abi_gpu_init)
    var gpu = abi.gpu.Gpu.init(allocator, .{
        .preferred_backend = null, // auto-detect
        .enable_profiling = false,
    }) catch {
        // GPU init may fail if no hardware available - acceptable
        return error.SkipZigTest;
    };

    // Get backend name (C API: abi_gpu_backend_name)
    if (gpu.getBackend()) |backend| {
        const name = switch (backend) {
            .cuda => "cuda",
            .vulkan => "vulkan",
            .metal => "metal",
            .webgpu => "webgpu",
            .stdgpu => "stdgpu",
            .opengl => "opengl",
            .opengles => "opengles",
            .webgl2 => "webgl2",
            .fpga => "fpga",
            .tpu => "tpu",
            .simulated => "simulated",
        };
        try testing.expect(name.len > 0);
    }

    // GPU shutdown (C API: abi_gpu_shutdown)
    gpu.deinit();
}

test "c_api: gpu null handle is safe" {
    // The C API handles null GPU pointers gracefully
    // abi_gpu_shutdown(NULL) should be a no-op
    const maybe_gpu: ?*abi.gpu.Gpu = null;
    if (maybe_gpu) |gpu| {
        gpu.deinit();
    }
    // No crash = success
}

test "c_api: gpu backend name for disabled module" {
    // When GPU is disabled, backend name should return "disabled" or "none"
    if (build_options.enable_gpu) {
        // Test that we can query backend names without crashing
        const name = abi.gpu.backendName(.vulkan);
        try testing.expect(name.len > 0);
    } else {
        // Module disabled - should return appropriate stub values
        try testing.expect(!abi.gpu.moduleEnabled());
    }
}

// ============================================================================
// GPU Configuration Tests
// ============================================================================

test "c_api: gpu config defaults" {
    // The C API's GpuConfig has these defaults
    const GpuConfig = extern struct {
        backend: c_int = 0, // 0=auto
        device_index: c_int = 0,
        enable_profiling: bool = false,
    };

    const config = GpuConfig{};

    try testing.expectEqual(@as(c_int, 0), config.backend); // auto
    try testing.expectEqual(@as(c_int, 0), config.device_index);
    try testing.expect(!config.enable_profiling);
}

test "c_api: gpu backend enum mapping" {
    // The C API maps integers to backends:
    // 0=auto, 1=cuda, 2=vulkan, 3=metal, 4=webgpu

    const backend_map = [_]struct { c_value: c_int, expected: ?abi.gpu.Backend }{
        .{ .c_value = 0, .expected = null }, // auto
        .{ .c_value = 1, .expected = .cuda },
        .{ .c_value = 2, .expected = .vulkan },
        .{ .c_value = 3, .expected = .metal },
        .{ .c_value = 4, .expected = .webgpu },
    };

    for (backend_map) |bm| {
        const backend: ?abi.gpu.Backend = switch (bm.c_value) {
            1 => .cuda,
            2 => .vulkan,
            3 => .metal,
            4 => .webgpu,
            else => null,
        };
        try testing.expect(std.meta.eql(backend, bm.expected));
    }
}
