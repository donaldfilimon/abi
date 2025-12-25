const std = @import("std");
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
        return .{
            .os = mapOs(std.builtin.os.tag),
            .arch = mapArch(std.builtin.cpu.arch),
            .max_threads = std.Thread.getCpuCount() catch 1,
        };
    }
};

pub const platform = struct {
    pub const PlatformInfo = Self.PlatformInfo;
    pub const Os = Self.Os;
    pub const Arch = Self.Arch;
};

fn mapOs(os_tag: std.builtin.Os.Tag) Os {
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

fn mapArch(arch: std.builtin.Cpu.Arch) Arch {
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
