//! DeviceManager: discovery, selection, and active-device tracking.

const std = @import("std");
const types = @import("types.zig");
const selector_mod = @import("selector.zig");
const enumeration_mod = @import("enumeration.zig");

const Device = types.Device;
const Backend = types.Backend;
const DeviceSelector = selector_mod.DeviceSelector;

/// Device manager for discovery and selection.
pub const DeviceManager = struct {
    allocator: std.mem.Allocator,
    devices: []Device,
    active_device: ?*const Device,

    pub fn init(allocator: std.mem.Allocator) !DeviceManager {
        const devices = try enumeration_mod.discoverDevices(allocator);
        return .{
            .allocator = allocator,
            .devices = devices,
            .active_device = null,
        };
    }

    pub fn deinit(self: *DeviceManager) void {
        self.allocator.free(self.devices);
        self.* = undefined;
    }

    /// Get all discovered devices.
    pub fn listDevices(self: *const DeviceManager) []const Device {
        return self.devices;
    }

    /// Get the currently active device.
    pub fn getActiveDevice(self: *const DeviceManager) ?*const Device {
        return self.active_device;
    }

    /// Select and activate a device based on criteria.
    pub fn selectDevice(self: *DeviceManager, selector: DeviceSelector) !*const Device {
        const selected = selector.select(self.devices);
        if (selected) |device| {
            // Find the pointer to the device in our slice
            for (self.devices) |*d| {
                if (d.id == device.id) {
                    self.active_device = d;
                    return d;
                }
            }
        }
        return error.DeviceNotFound;
    }

    /// Select the best available device.
    pub fn selectBestDevice(self: *DeviceManager) !*const Device {
        return self.selectDevice(.best);
    }

    /// Get device by ID.
    pub fn getDevice(self: *const DeviceManager, id: u32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    /// Get devices matching a backend.
    pub fn getDevicesForBackend(self: *const DeviceManager, allocator: std.mem.Allocator, backend: Backend) ![]const Device {
        var matching = std.ArrayListUnmanaged(Device).empty;
        errdefer matching.deinit(allocator);

        for (self.devices) |device| {
            if (device.backend == backend) {
                try matching.append(allocator, device);
            }
        }

        return matching.toOwnedSlice(allocator);
    }

    /// Check if any device is available.
    pub fn hasDevices(self: *const DeviceManager) bool {
        return self.devices.len > 0;
    }

    /// Get the number of devices.
    pub fn deviceCount(self: *const DeviceManager) usize {
        return self.devices.len;
    }
};

test "DeviceManager init and deinit" {
    var manager = try DeviceManager.init(std.testing.allocator);
    defer manager.deinit();

    // Should at least not crash; device count depends on system
    _ = manager.deviceCount();
    _ = manager.hasDevices();
}

test {
    std.testing.refAllDecls(@This());
}
