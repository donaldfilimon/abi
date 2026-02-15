//! Mobile Example
//!
//! Demonstrates the mobile module: platform detection, lifecycle
//! management, sensor access, and notifications.
//!
//! Note: Mobile is disabled by default. Enable with -Denable-mobile=true
//!
//! Run with: `zig build run-mobile`

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== ABI Mobile Example ===\n\n", .{});

    if (!abi.mobile.isEnabled()) {
        std.debug.print("Mobile feature is disabled (default).\n", .{});
        std.debug.print("Enable with: zig build -Denable-mobile=true\n\n", .{});

        // Show available types even when disabled
        std.debug.print("--- Available Types (stub mode) ---\n", .{});
        std.debug.print("MobileConfig: platform, orientation, sensor settings\n", .{});
        std.debug.print("LifecycleState: active, background, suspended, terminated\n", .{});
        std.debug.print("SensorData: timestamp + 3-axis values\n", .{});

        const state = abi.mobile.getLifecycleState();
        std.debug.print("\nLifecycle state: {t}\n", .{state});

        // Demonstrate stub error handling
        abi.mobile.init(allocator, .{}) catch |err| {
            std.debug.print("init: {t} (expected — feature disabled)\n", .{err});
        };

        if (abi.mobile.readSensor("accelerometer")) |_| {
            std.debug.print("readSensor: unexpected success\n", .{});
        } else |err| {
            std.debug.print("readSensor: {t} (expected — feature disabled)\n", .{err});
        }

        std.debug.print("\nMobile example complete (stub mode).\n", .{});
        return;
    }

    // Real implementation path
    try abi.mobile.init(allocator, .{});
    defer abi.mobile.deinit();

    std.debug.print("Mobile initialized\n", .{});
    std.debug.print("Lifecycle: {t}\n", .{abi.mobile.getLifecycleState()});

    // Read sensor
    const data = abi.mobile.readSensor("accelerometer") catch |err| {
        std.debug.print("Sensor read: {t}\n", .{err});
        return;
    };
    std.debug.print("Accelerometer: [{d:.2}, {d:.2}, {d:.2}]\n", .{
        data.values[0], data.values[1], data.values[2],
    });

    std.debug.print("\nMobile example complete.\n", .{});
}
