//! Hardware Telemetry Sensor
//!
//! Provides ABI with physical sensation, allowing it to monitor CPU pressure,
//! available memory, and system thermals. The Triad Engine uses this to
//! dynamically throttle its own execution to avoid overloading the host.

const std = @import("std");

pub const TelemetryData = struct {
    cpu_usage_pct: f32,
    available_memory_mb: u64,
    thermal_state: ThermalState,

    pub const ThermalState = enum {
        nominal,
        fair,
        serious,
        critical,
    };
};

pub const HardwareSensor = struct {
    allocator: std.mem.Allocator,
    last_reading: TelemetryData,

    pub fn init(allocator: std.mem.Allocator) HardwareSensor {
        return .{
            .allocator = allocator,
            .last_reading = .{
                .cpu_usage_pct = 0.0,
                .available_memory_mb = 16384,
                .thermal_state = .nominal,
            },
        };
    }

    pub fn deinit(self: *HardwareSensor) void {
        _ = self;
    }

    /// Read physical hardware state.
    pub fn poll(self: *HardwareSensor) !TelemetryData {
        // Stub: In reality, this queries mach_host or /proc/stat
        self.last_reading.cpu_usage_pct = 15.4;
        self.last_reading.available_memory_mb = 8192;
        self.last_reading.thermal_state = .nominal;
        return self.last_reading;
    }

    /// Determines if ABI needs to reduce cognitive load based on host stress.
    pub fn isHostStressed(self: *const HardwareSensor) bool {
        return self.last_reading.thermal_state == .critical or self.last_reading.cpu_usage_pct > 90.0;
    }
};

test {
    std.testing.refAllDecls(@This());
}
