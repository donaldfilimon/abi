//! Platform detection and OS/arch mapping.
const std = @import("std");
const builtin = @import("builtin");
const Self = @This();

pub const Os = enum {
    windows,
    linux,
    macos,
    freebsd,
    netbsd,
    openbsd,
    dragonfly,
    ios,
    wasm,
    other,

    /// Get current OS at comptime
    pub fn current() Os {
        return mapOs(builtin.target.os.tag);
    }
};

pub const Arch = enum {
    x86_64,
    x86,
    aarch64,
    arm,
    riscv64,
    wasm32,
    wasm64,
    other,

    /// Get current architecture at comptime
    pub fn current() Arch {
        return mapArch(builtin.target.cpu.arch);
    }

    /// Check if SIMD is available for this architecture
    pub fn hasSimd(self: Arch) bool {
        return switch (self) {
            .x86_64, .x86 => true, // SSE/AVX
            .aarch64, .arm => true, // NEON
            .wasm32, .wasm64 => true, // WASM SIMD
            else => false,
        };
    }
};

/// Whether the current target supports threading
/// On freestanding/WASM targets, threading APIs are not available.
pub const is_threaded_target = builtin.target.os.tag != .freestanding and
    builtin.target.cpu.arch != .wasm32 and
    builtin.target.cpu.arch != .wasm64;

/// Get CPU count in a WASM-safe manner.
/// Returns 1 on freestanding/WASM targets where std.Thread is unavailable.
/// Use this instead of std.Thread.getCpuCount() for cross-platform compatibility.
pub fn getCpuCountSafe() usize {
    if (comptime !is_threaded_target) {
        return 1;
    }
    return std.Thread.getCpuCount() catch 1;
}

pub const PlatformInfo = struct {
    os: Os,
    arch: Arch,
    max_threads: u32,

    pub fn detect() PlatformInfo {
        const thread_count = getCpuCountSafe();
        const bounded_threads = @max(thread_count, 1);
        const capped_threads = @min(bounded_threads, @as(usize, std.math.maxInt(u32)));
        return .{
            .os = mapOs(builtin.target.os.tag),
            .arch = mapArch(builtin.target.cpu.arch),
            .max_threads = @intCast(capped_threads),
        };
    }
};

pub const platform = struct {
    pub const PlatformInfo = Self.PlatformInfo;
    pub const Os = Self.Os;
    pub const Arch = Self.Arch;
};

fn mapOs(os_tag: std.Target.Os.Tag) Os {
    return switch (os_tag) {
        .windows => .windows,
        .linux => .linux,
        .macos => .macos,
        .freebsd => .freebsd,
        .netbsd => .netbsd,
        .openbsd => .openbsd,
        .dragonfly => .dragonfly,
        .ios => .ios,
        .wasi => .wasm,
        else => .other,
    };
}

fn mapArch(arch: std.Target.Cpu.Arch) Arch {
    return switch (arch) {
        .x86_64 => .x86_64,
        .x86 => .x86,
        .aarch64 => .aarch64,
        .arm => .arm,
        .riscv64 => .riscv64,
        .wasm32 => .wasm32,
        .wasm64 => .wasm64,
        else => .other,
    };
}

test "platform detection reports at least one thread" {
    const info = PlatformInfo.detect();
    try std.testing.expect(info.max_threads >= 1);
}

test "Os.current returns valid OS" {
    const os = Os.current();
    // Should map to a valid enum value on any supported platform
    try std.testing.expect(@intFromEnum(os) <= @intFromEnum(Os.other));
}

test "Arch.current returns valid architecture" {
    const arch = Arch.current();
    try std.testing.expect(@intFromEnum(arch) <= @intFromEnum(Arch.other));
}

test "Arch.hasSimd returns correct values" {
    try std.testing.expect(Arch.x86_64.hasSimd());
    try std.testing.expect(Arch.aarch64.hasSimd());
    try std.testing.expect(Arch.wasm32.hasSimd());
    try std.testing.expect(!Arch.other.hasSimd());
}

test "getCpuCountSafe returns at least 1" {
    const count = getCpuCountSafe();
    try std.testing.expect(count >= 1);
}

test "PlatformInfo fields are consistent" {
    const info = PlatformInfo.detect();
    try std.testing.expectEqual(Os.current(), info.os);
    try std.testing.expectEqual(Arch.current(), info.arch);
}
