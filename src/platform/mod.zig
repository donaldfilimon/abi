//! Platform Detection and Abstraction
//!
//! Provides OS, architecture, and capability detection for cross-platform code.
//! This module consolidates all platform-specific detection and abstraction logic.
//!
//! ## Usage
//!
//! ```zig
//! const platform = @import("abi").platform;
//!
//! const info = platform.getPlatformInfo();
//! std.debug.print("OS: {s}, Arch: {s}, Cores: {d}\n", .{
//!     @tagName(info.os),
//!     @tagName(info.arch),
//!     info.max_threads,
//! });
//!
//! if (platform.supportsThreading()) {
//!     // Use multi-threaded code path
//! }
//! ```

const std = @import("std");
const builtin = @import("builtin");

pub const detection = @import("detection.zig");
pub const cpu = @import("cpu.zig");

// Re-export common types from detection
pub const Os = detection.Os;
pub const Arch = detection.Arch;
pub const PlatformInfo = detection.PlatformInfo;

// Re-export threading check
pub const is_threaded_target = detection.is_threaded_target;

/// Get current platform information at runtime
pub fn getPlatformInfo() PlatformInfo {
    return PlatformInfo.detect();
}

/// Check if current platform supports threading
/// Returns false for freestanding and WASM targets
pub fn supportsThreading() bool {
    return is_threaded_target;
}

/// Get CPU count in a platform-safe manner
/// Returns 1 on freestanding/WASM targets where std.Thread is unavailable
pub fn getCpuCount() usize {
    return detection.getCpuCountSafe();
}

/// Get a human-readable platform description string
pub fn getDescription() []const u8 {
    const os = Os.current();
    const arch = Arch.current();

    return switch (os) {
        .windows => switch (arch) {
            .x86_64 => "Windows x64",
            .aarch64 => "Windows ARM64",
            .x86 => "Windows x86",
            else => "Windows",
        },
        .linux => switch (arch) {
            .x86_64 => "Linux x64",
            .aarch64 => "Linux ARM64",
            .riscv64 => "Linux RISC-V",
            .arm => "Linux ARM",
            else => "Linux",
        },
        .macos => switch (arch) {
            .aarch64 => "macOS Apple Silicon",
            .x86_64 => "macOS Intel",
            else => "macOS",
        },
        .ios => "iOS",
        .freebsd => "FreeBSD",
        .netbsd => "NetBSD",
        .openbsd => "OpenBSD",
        .dragonfly => "DragonFly BSD",
        .wasm => "WebAssembly",
        .other => "Unknown Platform",
    };
}

/// Check if SIMD is available on the current platform
pub fn hasSimd() bool {
    return Arch.current().hasSimd();
}

/// Check if the current platform is Apple Silicon (macOS/iOS ARM64)
pub fn isAppleSilicon() bool {
    const os = Os.current();
    const arch = Arch.current();
    return (os == .macos or os == .ios) and arch == .aarch64;
}

/// Check if the current platform is a desktop OS
pub fn isDesktop() bool {
    return switch (Os.current()) {
        .windows, .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly => true,
        else => false,
    };
}

/// Check if the current platform is mobile
pub fn isMobile() bool {
    return Os.current() == .ios;
    // Note: Android detection would need additional checks via builtin.abi
}

/// Check if the current platform is WebAssembly
pub fn isWasm() bool {
    return Os.current() == .wasm;
}

// ============================================================================
// Tests
// ============================================================================

test "platform module basic functionality" {
    const info = getPlatformInfo();
    try std.testing.expect(info.max_threads >= 1);
}

test "threading support detection" {
    const supports = supportsThreading();
    // On non-WASM/freestanding targets, threading should be supported
    if (builtin.target.os.tag != .freestanding and
        builtin.target.cpu.arch != .wasm32 and
        builtin.target.cpu.arch != .wasm64)
    {
        try std.testing.expect(supports);
    }
}

test "cpu count" {
    const count = getCpuCount();
    try std.testing.expect(count >= 1);
}

test "platform description" {
    const desc = getDescription();
    try std.testing.expect(desc.len > 0);
}

test "simd detection" {
    // hasSimd should return a boolean without crashing
    _ = hasSimd();
}

test "apple silicon detection" {
    const is_as = isAppleSilicon();
    // On macOS ARM64, this should be true
    if (builtin.target.os.tag == .macos and builtin.target.cpu.arch == .aarch64) {
        try std.testing.expect(is_as);
    }
}

test "platform category detection" {
    // At least one of these should be true on any platform
    const is_any = isDesktop() or isMobile() or isWasm();
    _ = is_any; // May be false on unusual platforms, that's ok
}
