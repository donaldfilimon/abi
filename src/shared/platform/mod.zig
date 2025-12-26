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
};

pub const PlatformInfo = struct {
    os: Os,
    arch: Arch,
    max_threads: u32,

    pub fn detect() PlatformInfo {
        const thread_count = std.Thread.getCpuCount() catch 1;
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
