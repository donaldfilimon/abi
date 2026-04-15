//! Re-export from device/android_probe

pub const ProbeScore = @import("../../device/android_probe.zig").ProbeScore;
pub const chooseAndroidPrimary = @import("../../device/android_probe.zig").chooseAndroidPrimary;
pub const resetAndroidProbeForTests = @import("../../device/android_probe.zig").resetAndroidProbeForTests;
pub const isAndroidTarget = @import("../../device/android_probe.zig").isAndroidTarget;
pub const scoreBackend = @import("../../device/android_probe.zig").scoreBackend;
