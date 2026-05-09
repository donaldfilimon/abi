//! Device selection logic: DeviceSelector union and selectBestDevice.

const std = @import("std");
const types = @import("types.zig");
const enumeration_mod = @import("enumeration.zig");

const Device = types.Device;
const DeviceType = types.DeviceType;
const DeviceFeature = types.DeviceFeature;
const DeviceSelectionCriteria = types.DeviceSelectionCriteria;
const Backend = types.Backend;
const Vendor = types.Vendor;

/// Device selection criteria.
pub const DeviceSelector = union(enum) {
    /// Select the best device based on scoring.
    best: void,
    /// Select a specific device by ID.
    by_id: u32,
    /// Select by backend preference.
    by_backend: Backend,
    /// Select by device type preference.
    by_type: DeviceType,
    /// Select by vendor preference.
    by_vendor: Vendor,
    /// Select by minimum memory requirement.
    by_memory: u64,
    /// Select by required features.
    by_features: []const DeviceFeature,
    /// Custom selection function.
    custom: *const fn ([]const Device) ?Device,

    pub fn select(self: DeviceSelector, devices: []const Device) ?Device {
        if (devices.len == 0) return null;

        return switch (self) {
            .best => selectBest(devices),
            .by_id => |id| selectById(devices, id),
            .by_backend => |backend| selectByBackend(devices, backend),
            .by_type => |device_type| selectByType(devices, device_type),
            .by_vendor => |vendor| selectByVendor(devices, vendor),
            .by_memory => |min_memory| selectByMemory(devices, min_memory),
            .by_features => |features| selectByFeatures(devices, features),
            .custom => |func| func(devices),
        };
    }

    fn selectBest(devices: []const Device) ?Device {
        if (devices.len == 0) return null;

        var best = devices[0];
        var best_score = best.score();

        for (devices[1..]) |device| {
            const device_score = device.score();
            if (device_score > best_score) {
                best = device;
                best_score = device_score;
            }
        }

        return best;
    }

    fn selectById(devices: []const Device, id: u32) ?Device {
        for (devices) |device| {
            if (device.id == id) return device;
        }
        return null;
    }

    fn selectByBackend(devices: []const Device, backend: Backend) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.backend == backend) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }

    fn selectByType(devices: []const Device, device_type: DeviceType) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.device_type == device_type) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }

    fn selectByVendor(devices: []const Device, vendor: Vendor) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.vendor == vendor) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }

    fn selectByMemory(devices: []const Device, min_memory: u64) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            if (device.total_memory) |mem| {
                if (mem >= min_memory) {
                    const device_score = device.score();
                    if (best == null or device_score > best_score) {
                        best = device;
                        best_score = device_score;
                    }
                }
            }
        }

        return best;
    }

    fn selectByFeatures(devices: []const Device, features: []const DeviceFeature) ?Device {
        var best: ?Device = null;
        var best_score: u32 = 0;

        for (devices) |device| {
            var has_all = true;
            for (features) |feature| {
                if (!device.supportsFeature(feature)) {
                    has_all = false;
                    break;
                }
            }

            if (has_all) {
                const device_score = device.score();
                if (best == null or device_score > best_score) {
                    best = device;
                    best_score = device_score;
                }
            }
        }

        return best;
    }
};

/// Select the best device based on criteria.
pub fn selectBestDevice(
    allocator: std.mem.Allocator,
    criteria: DeviceSelectionCriteria,
) !?Device {
    const all_devices = try enumeration_mod.enumerateAllDevices(allocator);
    defer allocator.free(all_devices);

    if (all_devices.len == 0) return null;

    var best: ?Device = null;
    var best_score: u32 = 0;

    for (all_devices) |dev| {
        if (!meetsRequirements(dev, criteria)) continue;

        const score_val = dev.score();
        if (score_val > best_score) {
            best = dev;
            best_score = score_val;
        }
    }

    return best;
}

fn meetsRequirements(dev: Device, criteria: DeviceSelectionCriteria) bool {
    if (criteria.prefer_discrete and dev.device_type != .discrete) {
        if (dev.device_type != .integrated) return false;
    }

    if (criteria.min_memory_gb > 0) {
        if (dev.total_memory) |mem| {
            const gb = mem / (1024 * 1024 * 1024);
            if (gb < criteria.min_memory_gb) return false;
        } else {
            return false; // Unknown memory doesn't meet requirement
        }
    }

    for (criteria.required_features) |feature| {
        if (!dev.supportsFeature(feature)) return false;
    }

    return true;
}

test "DeviceSelector best" {
    const devices = [_]Device{
        .{
            .id = 0,
            .backend = .vulkan,
            .name = "Device 0",
            .device_type = .integrated,
            .vendor = .intel,
            .total_memory = 2 * 1024 * 1024 * 1024,
            .available_memory = null,
            .is_emulated = true,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        },
        .{
            .id = 1,
            .backend = .cuda,
            .name = "Device 1",
            .device_type = .discrete,
            .vendor = .nvidia,
            .total_memory = 8 * 1024 * 1024 * 1024,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{ .supports_fp16 = true },
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        },
    };

    const selector = DeviceSelector{ .best = {} };
    const selected = selector.select(&devices);

    try std.testing.expect(selected != null);
    try std.testing.expect(selected.?.id == 1); // CUDA device should score higher
}

test "DeviceSelector by_backend" {
    const devices = [_]Device{
        .{
            .id = 0,
            .backend = .vulkan,
            .name = "Vulkan Device",
            .device_type = .discrete,
            .vendor = .amd,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        },
        .{
            .id = 1,
            .backend = .cuda,
            .name = "CUDA Device",
            .device_type = .discrete,
            .vendor = .nvidia,
            .total_memory = null,
            .available_memory = null,
            .is_emulated = false,
            .capability = .{},
            .compute_units = null,
            .clock_mhz = null,
            .pci_bus_id = null,
            .driver_version = null,
        },
    };

    const selector = DeviceSelector{ .by_backend = .vulkan };
    const selected = selector.select(&devices);

    try std.testing.expect(selected != null);
    try std.testing.expect(selected.?.backend == .vulkan);
}

test {
    std.testing.refAllDecls(@This());
}
