const std = @import("std");
const builtin = @import("builtin");
const backend_mod = @import("../backend.zig");

pub const ProbeScore = struct {
    backend: backend_mod.Backend,
    score: i32,
    startup_latency_ms: u32,
    available: bool,
};

const startup_latency_threshold_ms: u32 = 45;

var cached: ?backend_mod.Backend = null;
var resolved = false;

pub fn chooseAndroidPrimary() ?backend_mod.Backend {
    if (!isAndroidTarget()) return null;
    if (resolved) return cached;

    resolved = true;

    const vulkan = scoreBackend(.vulkan, 32, 85);
    const gles = scoreBackend(.opengles, 18, 70);

    if (vulkan.available and vulkan.score >= gles.score and vulkan.startup_latency_ms <= startup_latency_threshold_ms) {
        cached = .vulkan;
    } else if (gles.available) {
        cached = .opengles;
    } else if (vulkan.available) {
        cached = .vulkan;
    } else {
        cached = null;
    }

    return cached;
}

pub fn resetAndroidProbeForTests() void {
    cached = null;
    resolved = false;
}

fn scoreBackend(backend: backend_mod.Backend, estimated_latency_ms: u32, capability_bonus: i32) ProbeScore {
    const availability = backend_mod.backendAvailability(backend);
    if (!availability.available) {
        return .{
            .backend = backend,
            .score = std.math.minInt(i32),
            .startup_latency_ms = estimated_latency_ms,
            .available = false,
        };
    }

    const level_score: i32 = switch (availability.level) {
        .device_count => 120,
        .loader => 80,
        .none => 40,
    };

    const score = level_score + capability_bonus - @as(i32, @intCast(estimated_latency_ms / 2));

    return .{
        .backend = backend,
        .score = score,
        .startup_latency_ms = estimated_latency_ms,
        .available = true,
    };
}

fn isAndroidTarget() bool {
    return builtin.target.os.tag == .linux and builtin.abi == .android;
}

test "android probe reset clears cached selection" {
    resetAndroidProbeForTests();
    try std.testing.expect(cached == null);
    try std.testing.expect(!resolved);
}
