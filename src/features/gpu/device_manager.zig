//! Device discovery, selection, and multi-GPU management.
//!
//! Extracted from `unified.zig` to separate device lifecycle concerns.

const std = @import("std");
const device_mod = @import("device");
const multi_device = @import("multi_device");
const stream_mod = @import("stream");
const dispatcher_mod = @import("dispatch/coordinator");
const policy_mod = @import("policy");

pub const Device = device_mod.Device;
pub const DeviceSelector = device_mod.DeviceSelector;
pub const DeviceType = device_mod.DeviceType;
pub const DeviceFeature = device_mod.DeviceFeature;
pub const DeviceManager = device_mod.DeviceManager;
pub const DeviceGroup = multi_device.DeviceGroup;
pub const WorkDistribution = multi_device.WorkDistribution;
pub const DeviceBarrier = multi_device.DeviceBarrier;
pub const PeerTransfer = multi_device.PeerTransfer;
pub const KernelDispatcher = dispatcher_mod.KernelDispatcher;
pub const Stream = stream_mod.Stream;
pub const StreamManager = stream_mod.StreamManager;

/// Load balance strategy for multi-GPU.
pub const LoadBalanceStrategy = enum {
    /// Round-robin distribution.
    round_robin,
    /// Memory-aware distribution.
    memory_aware,
    /// Compute-aware distribution.
    compute_aware,
    /// Manual assignment.
    manual,
};

/// Multi-GPU configuration.
pub const MultiGpuConfig = struct {
    /// Devices to use (empty = use all).
    devices: []const u32 = &.{},
    /// Load balance strategy.
    strategy: LoadBalanceStrategy = .memory_aware,
};

/// Health status.
pub const HealthStatus = enum {
    healthy,
    degraded,
    unhealthy,
    unknown,
};

/// Convert a LoadBalanceStrategy to the multi_device strategy enum.
pub fn toMultiDeviceStrategy(strategy: LoadBalanceStrategy) multi_device.LoadBalanceStrategy {
    return switch (strategy) {
        .round_robin => .round_robin,
        .memory_aware => .memory_aware,
        .compute_aware => .capability_weighted,
        .manual => .pinned,
    };
}

/// Select a device based on criteria, updating active device and creating
/// a default stream if needed.
pub fn selectDevice(
    device_manager: *DeviceManager,
    stream_manager: *StreamManager,
    active_device: *?*const Device,
    default_stream: *?*Stream,
    selector: DeviceSelector,
) !void {
    const device = try device_manager.selectDevice(selector);
    active_device.* = device;

    if (default_stream.* == null) {
        default_stream.* = try stream_manager.createStream(device, .{});
    }
}

/// Enable multi-GPU mode.
pub fn enableMultiGpu(
    allocator: std.mem.Allocator,
    device_group: *?DeviceGroup,
    config: MultiGpuConfig,
) !void {
    if (device_group.* == null) {
        const multi_config = multi_device.MultiDeviceConfig{
            .strategy = toMultiDeviceStrategy(config.strategy),
            .preferred_devices = config.devices,
        };
        device_group.* = try DeviceGroup.init(allocator, multi_config);
    }

    if (device_group.*) |*dg| {
        if (config.devices.len > 0) {
            for (dg.getAllDevices()) |device| {
                dg.disableDevice(device.id);
            }
            for (config.devices) |device_id| {
                try dg.enableDevice(device_id);
            }
        }
    }
}

/// Distribute work across multiple GPUs.
pub fn distributeWork(
    allocator: std.mem.Allocator,
    device_group: *?DeviceGroup,
    active_device: ?*const Device,
    total_work: usize,
) ![]WorkDistribution {
    if (device_group.*) |*dg| {
        return dg.distributeWork(total_work);
    }
    // Single device fallback
    var result = try allocator.alloc(WorkDistribution, 1);
    result[0] = .{
        .device_id = if (active_device) |d| d.id else 0,
        .offset = 0,
        .size = total_work,
    };
    return result;
}

/// Check GPU health.
pub fn checkHealth(
    active_device: ?*const Device,
    device_manager: *const DeviceManager,
) HealthStatus {
    if (active_device == null) {
        return .unhealthy;
    }
    if (!device_manager.hasDevices()) {
        return .unhealthy;
    }
    return .healthy;
}

/// Resolve the effective memory mode based on platform policy.
pub fn resolveMemoryMode(mode: anytype) @TypeOf(mode) {
    if (mode == .automatic) {
        const hints = policy_mod.optimizationHintsForPlatform(policy_mod.classifyBuiltin());
        if (hints.prefer_unified_memory) {
            return .unified;
        }
    }
    return mode;
}

/// Initialize device discovery and selection during Gpu.init.
/// Returns the selected active device (if any) and created default stream.
pub fn initDevices(
    allocator: std.mem.Allocator,
    device_manager: *DeviceManager,
    stream_manager: *StreamManager,
    config: anytype,
) struct { active_device: ?*const Device, default_stream: ?*Stream, dispatcher: ?KernelDispatcher } {
    var active_device: ?*const Device = null;
    var default_stream: ?*Stream = null;

    if (device_manager.hasDevices()) {
        if (config.preferred_backend) |backend| {
            active_device = device_manager.selectDevice(.{ .by_backend = backend }) catch null;
        }

        if (active_device == null and config.allow_fallback) {
            active_device = device_manager.selectBestDevice() catch null;
        }

        if (active_device) |device| {
            default_stream = stream_manager.createStream(device, .{}) catch null;
        }
    }

    var disp: ?KernelDispatcher = null;
    if (active_device) |dev| {
        disp = KernelDispatcher.init(allocator, dev.backend, dev) catch null;
    }

    return .{
        .active_device = active_device,
        .default_stream = default_stream,
        .dispatcher = disp,
    };
}
