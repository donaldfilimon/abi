//! GPU Platform Detection
//!
//! Centralized platform detection for GPU backend selection and availability.
//! Provides compile-time and runtime detection of GPU hardware and drivers.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");

/// Operating system for GPU backend selection
pub const Os = enum {
    windows,
    linux,
    macos,
    ios,
    android,
    wasm,
    freestanding,
    other,

    /// Get current OS at comptime
    pub fn current() Os {
        return switch (builtin.os.tag) {
            .windows => .windows,
            .linux => .linux,
            .macos => .macos,
            .ios => .ios,
            .wasi => .wasm,
            .freestanding => .freestanding,
            else => if (isAndroid()) .android else .other,
        };
    }

    fn isAndroid() bool {
        return builtin.os.tag == .linux and builtin.abi == .android;
    }
};

/// CPU architecture for GPU backend optimization
pub const Arch = enum {
    x86_64,
    x86,
    aarch64,
    arm,
    wasm32,
    wasm64,
    riscv64,
    other,

    /// Get current architecture at comptime
    pub fn current() Arch {
        return switch (builtin.cpu.arch) {
            .x86_64 => .x86_64,
            .x86 => .x86,
            .aarch64 => .aarch64,
            .arm => .arm,
            .wasm32 => .wasm32,
            .wasm64 => .wasm64,
            .riscv64 => .riscv64,
            else => .other,
        };
    }

    /// Check if SIMD is available
    pub fn hasSimd(self: Arch) bool {
        return switch (self) {
            .x86_64, .x86 => true, // SSE/AVX
            .aarch64, .arm => true, // NEON
            .wasm32, .wasm64 => true, // WASM SIMD
            else => false,
        };
    }
};

/// GPU vendor detection
pub const GpuVendor = enum {
    nvidia,
    amd,
    intel,
    apple,
    qualcomm,
    arm_mali,
    software,
    unknown,
};

/// Backend compatibility flags
pub const BackendSupport = struct {
    cuda: bool = false,
    vulkan: bool = false,
    metal: bool = false,
    webgpu: bool = false,
    opengl: bool = false,
    opencl: bool = false,
    stdgpu: bool = true, // Always available as software fallback
    fpga: bool = false,

    /// Get platform-appropriate backends
    pub fn forCurrentPlatform() BackendSupport {
        var support = BackendSupport{};
        const os = Os.current();

        switch (os) {
            .windows => {
                support.cuda = true; // NVIDIA drivers available
                support.vulkan = true;
                support.opengl = true;
                support.opencl = true;
            },
            .linux => {
                support.cuda = true;
                support.vulkan = true;
                support.opengl = true;
                support.opencl = true;
            },
            .macos => {
                support.metal = true;
                support.vulkan = true; // MoltenVK
                support.opengl = true; // Deprecated but available
            },
            .ios => {
                support.metal = true;
            },
            .android => {
                support.vulkan = true;
                support.opengl = true; // OpenGL ES
            },
            .wasm => {
                support.webgpu = true;
            },
            else => {
                // Only software fallback
            },
        }

        return support;
    }

    /// Count available backends
    pub fn count(self: BackendSupport) u8 {
        var c: u8 = 0;
        if (self.cuda) c += 1;
        if (self.vulkan) c += 1;
        if (self.metal) c += 1;
        if (self.webgpu) c += 1;
        if (self.opengl) c += 1;
        if (self.opencl) c += 1;
        if (self.stdgpu) c += 1;
        if (self.fpga) c += 1;
        return c;
    }
};

/// Platform capabilities structure
pub const PlatformCapabilities = struct {
    os: Os,
    arch: Arch,
    has_discrete_gpu: bool,
    has_integrated_gpu: bool,
    has_unified_memory: bool,
    has_threading: bool,
    has_simd: bool,
    cpu_cores: u32,
    backend_support: BackendSupport,

    /// Detect capabilities for current platform
    pub fn detect() PlatformCapabilities {
        const os = Os.current();
        const arch = Arch.current();
        const has_threading = os != .freestanding and os != .wasm;

        return .{
            .os = os,
            .arch = arch,
            .has_discrete_gpu = os == .windows or os == .linux,
            .has_integrated_gpu = true, // Most systems have some GPU
            .has_unified_memory = os == .macos or os == .ios, // Apple Silicon
            .has_threading = has_threading,
            .has_simd = arch.hasSimd(),
            .cpu_cores = getCpuCount(),
            .backend_support = BackendSupport.forCurrentPlatform(),
        };
    }

    fn getCpuCount() u32 {
        if (comptime Os.current() == .freestanding or Os.current() == .wasm) {
            return 1;
        }
        const count = std.Thread.getCpuCount() catch 1;
        return @intCast(@min(count, std.math.maxInt(u32)));
    }

    /// Check if Metal with Accelerate is available
    pub fn hasMetalAccelerate(self: PlatformCapabilities) bool {
        return self.os == .macos or self.os == .ios;
    }

    /// Check if CUDA is potentially available
    pub fn hasCudaPotential(self: PlatformCapabilities) bool {
        return (self.os == .windows or self.os == .linux) and
            self.backend_support.cuda;
    }

    /// Get recommended backend order
    pub fn recommendedBackendOrder(self: PlatformCapabilities) []const u8 {
        return switch (self.os) {
            .macos, .ios => &[_]u8{ 'M', 'V', 'O', 'S' }, // Metal, Vulkan, OpenGL, Software
            .windows, .linux => &[_]u8{ 'C', 'V', 'O', 'S' }, // CUDA, Vulkan, OpenGL, Software
            .android => &[_]u8{ 'V', 'O', 'S' }, // Vulkan, OpenGL ES, Software
            .wasm => &[_]u8{ 'W', 'S' }, // WebGPU, Software
            else => &[_]u8{'S'}, // Software only
        };
    }
};

/// Check if the current platform supports CUDA
pub fn isCudaSupported() bool {
    return comptime build_options.gpu_cuda and
        (Os.current() == .windows or Os.current() == .linux);
}

/// Check if the current platform supports Metal
pub fn isMetalSupported() bool {
    return comptime build_options.gpu_metal and
        (Os.current() == .macos or Os.current() == .ios);
}

/// Check if the current platform supports Vulkan
pub fn isVulkanSupported() bool {
    return comptime build_options.gpu_vulkan and
        Os.current() != .wasm and Os.current() != .freestanding;
}

/// Check if the current platform supports WebGPU
pub fn isWebGpuSupported() bool {
    return comptime build_options.gpu_webgpu and Os.current() == .wasm;
}

/// Check if the current platform supports OpenGL
pub fn isOpenGLSupported() bool {
    return comptime build_options.gpu_opengl and
        Os.current() != .wasm and Os.current() != .freestanding;
}

/// Check if software fallback is available (always true)
pub fn isStdGpuSupported() bool {
    return true;
}

/// Get string description of current platform
pub fn platformDescription() []const u8 {
    const os = Os.current();
    const arch = Arch.current();

    return switch (os) {
        .windows => switch (arch) {
            .x86_64 => "Windows x64",
            .aarch64 => "Windows ARM64",
            else => "Windows",
        },
        .linux => switch (arch) {
            .x86_64 => "Linux x64",
            .aarch64 => "Linux ARM64",
            .riscv64 => "Linux RISC-V",
            else => "Linux",
        },
        .macos => switch (arch) {
            .aarch64 => "macOS Apple Silicon",
            .x86_64 => "macOS Intel",
            else => "macOS",
        },
        .ios => "iOS",
        .android => "Android",
        .wasm => "WebAssembly",
        .freestanding => "Freestanding",
        .other => "Unknown Platform",
    };
}

// ============================================================================
// Tests
// ============================================================================

test "platform detection" {
    const caps = PlatformCapabilities.detect();
    try std.testing.expect(caps.cpu_cores >= 1);
    try std.testing.expect(caps.backend_support.stdgpu); // Always available
}

test "backend support count" {
    const support = BackendSupport.forCurrentPlatform();
    try std.testing.expect(support.count() >= 1); // At least stdgpu
}

test "os detection" {
    const os = Os.current();
    // Should be one of the known values
    try std.testing.expect(@intFromEnum(os) <= @intFromEnum(Os.other));
}

test "arch detection" {
    const arch = Arch.current();
    // Should be one of the known values
    try std.testing.expect(@intFromEnum(arch) <= @intFromEnum(Arch.other));
}

test "platform description" {
    const desc = platformDescription();
    try std.testing.expect(desc.len > 0);
}
