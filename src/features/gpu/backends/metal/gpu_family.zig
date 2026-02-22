//! Metal GPU Family Detection
//!
//! Detects the GPU family of the current Metal device to gate feature availability.
//! Uses the [MTLDevice supportsFamily:] selector to probe from highest to lowest.
//!
//! GPU families determine which Metal features are available:
//! - Apple7+ (Metal 3): Mesh shaders, ray tracing
//! - Apple6+: 32-bit MSAA, lossy texture compression
//! - Apple5+: Non-uniform threadgroup sizes, read-write textures
//! - Mac2: Metal 3 on macOS (Apple Silicon + discrete)
//! - Common1-3: Cross-platform baseline

const std = @import("std");
const builtin = @import("builtin");

/// Metal GPU family identifiers matching MTLGPUFamily enum values.
/// These integers are passed to [MTLDevice supportsFamily:].
pub const MetalGpuFamily = enum(u32) {
    // Apple GPU families (Apple Silicon and earlier A-series)
    apple1 = 1001,
    apple2 = 1002,
    apple3 = 1003,
    apple4 = 1004,
    apple5 = 1005,
    apple6 = 1006,
    apple7 = 1007, // Metal 3 — mesh shaders, ray tracing
    apple8 = 1008, // A16 / M2
    apple9 = 1009, // A17 Pro / M3+
    apple10 = 1010, // A18 / M4+

    // Mac GPU families
    mac1 = 2001,
    mac2 = 2002, // Metal 3 on Mac

    // Common GPU families (cross-platform baseline)
    common1 = 3001,
    common2 = 3002,
    common3 = 3003,

    unknown = 0,

    /// Human-readable name for this GPU family.
    pub fn name(self: MetalGpuFamily) []const u8 {
        return switch (self) {
            .apple1 => "Apple1 (A7)",
            .apple2 => "Apple2 (A8)",
            .apple3 => "Apple3 (A9-A10)",
            .apple4 => "Apple4 (A11)",
            .apple5 => "Apple5 (A12)",
            .apple6 => "Apple6 (A13-A14, M1)",
            .apple7 => "Apple7 (A15, M2) [Metal 3]",
            .apple8 => "Apple8 (A16, M2+)",
            .apple9 => "Apple9 (A17 Pro, M3+)",
            .apple10 => "Apple10 (A18, M4+) [Metal 4]",
            .mac1 => "Mac1",
            .mac2 => "Mac2 [Metal 3]",
            .common1 => "Common1",
            .common2 => "Common2",
            .common3 => "Common3",
            .unknown => "Unknown",
        };
    }

    /// Whether this family supports Metal 3 features.
    pub fn isMetal3(self: MetalGpuFamily) bool {
        return switch (self) {
            .apple7, .apple8, .apple9, .apple10, .mac2 => true,
            else => false,
        };
    }

    /// Whether this family supports Metal 4 features.
    ///
    /// Current conservative gate:
    /// - Apple9+ family is treated as Metal 4 capable.
    pub fn isMetal4(self: MetalGpuFamily) bool {
        return switch (self) {
            .apple9, .apple10 => true,
            else => false,
        };
    }

    /// Chip variant within an Apple GPU family (parsed from device name).
    pub const ChipVariant = enum {
        base,
        pro,
        max,
        ultra,
        unknown,
    };

    /// Infer chip variant from a Metal device name string.
    /// Device names follow the pattern "Apple M4 Pro" or "Apple A18".
    pub fn chipVariant(device_name: []const u8) ChipVariant {
        // Search case-insensitively by looking for variant keywords
        if (std.mem.indexOf(u8, device_name, "Ultra") != null) return .ultra;
        if (std.mem.indexOf(u8, device_name, "Max") != null) return .max;
        if (std.mem.indexOf(u8, device_name, "Pro") != null) return .pro;
        // If it contains an M-series or A-series chip name but no suffix, it's base
        if (std.mem.indexOf(u8, device_name, "Apple") != null) return .base;
        return .unknown;
    }

    /// Numeric generation (apple families only, 0 for non-apple).
    pub fn generation(self: MetalGpuFamily) u32 {
        const val = @intFromEnum(self);
        if (val >= 1001 and val <= 1010) return val - 1000;
        return 0;
    }
};

/// Comprehensive feature set derived from GPU family detection.
pub const MetalFeatureSet = struct {
    gpu_family: MetalGpuFamily = .unknown,
    supports_mesh_shaders: bool = false,
    supports_ray_tracing: bool = false,
    supports_mps: bool = false,
    supports_mps_graph: bool = false,
    supports_bfloat16: bool = false,
    supports_indirect_command_buffers: bool = false,
    max_threadgroup_memory: u32 = 16384,
    simdgroup_size: u32 = 32,
    has_neural_engine: bool = false,
};

/// Detect the highest supported GPU family for a Metal device.
/// Probes from Apple9 down to Apple1, then Mac2/Mac1, then Common3-1.
///
/// `device` is a Metal device ID (Obj-C object pointer).
/// `supports_family_fn` should be a cast of objc_msgSend matching:
///   fn(device, sel_supportsFamily, family_int) -> bool
pub fn detectGpuFamily(
    device: ?*anyopaque,
    sel_supports_family: *anyopaque,
    msg_send: *const fn (?*anyopaque, *anyopaque, u32) callconv(.c) bool,
) MetalGpuFamily {
    if (device == null) return .unknown;

    // Probe Apple families from highest to lowest
    const apple_families = [_]MetalGpuFamily{
        .apple10, .apple9, .apple8, .apple7, .apple6, .apple5,
        .apple4,  .apple3, .apple2, .apple1,
    };
    for (apple_families) |family| {
        if (msg_send(device, sel_supports_family, @intFromEnum(family))) {
            return family;
        }
    }

    // Probe Mac families
    const mac_families = [_]MetalGpuFamily{ .mac2, .mac1 };
    for (mac_families) |family| {
        if (msg_send(device, sel_supports_family, @intFromEnum(family))) {
            return family;
        }
    }

    // Probe Common families
    const common_families = [_]MetalGpuFamily{ .common3, .common2, .common1 };
    for (common_families) |family| {
        if (msg_send(device, sel_supports_family, @intFromEnum(family))) {
            return family;
        }
    }

    return .unknown;
}

/// Build a complete feature set from the detected GPU family.
pub fn buildFeatureSet(family: MetalGpuFamily) MetalFeatureSet {
    const gen = family.generation();

    return .{
        .gpu_family = family,
        .supports_mesh_shaders = family.isMetal3(),
        .supports_ray_tracing = family.isMetal3(),
        // MPS available on Apple3+ (A9 and later) and all Mac families
        .supports_mps = gen >= 3 or family == .mac1 or family == .mac2,
        .supports_mps_graph = gen >= 5 or family == .mac2,
        .supports_bfloat16 = gen >= 7 or family == .mac2,
        .supports_indirect_command_buffers = gen >= 4 or
            family == .mac1 or family == .mac2,
        .max_threadgroup_memory = if (gen >= 10)
            @as(u32, 65536) // 64KB for Apple10+ (M4)
        else if (gen >= 4 or family == .mac2)
            @as(u32, 32768) // 32KB for Apple4+
        else
            16384,
        .simdgroup_size = 32, // All Apple GPUs use 32-wide SIMD
        .has_neural_engine = gen >= 5, // A12+ has Neural Engine
    };
}

/// Detect GPU family and build full feature set in one call.
pub fn detectFeatures(
    device: ?*anyopaque,
    sel_supports_family: *anyopaque,
    msg_send: *const fn (?*anyopaque, *anyopaque, u32) callconv(.c) bool,
) MetalFeatureSet {
    const family = detectGpuFamily(device, sel_supports_family, msg_send);
    return buildFeatureSet(family);
}

// ============================================================================
// Tests
// ============================================================================

test "CudaArchitecture-style: GPU family from generation" {
    try std.testing.expectEqual(MetalGpuFamily.apple7, @as(MetalGpuFamily, .apple7));
    try std.testing.expect(MetalGpuFamily.apple7.isMetal3());
    try std.testing.expect(!MetalGpuFamily.apple6.isMetal3());
    try std.testing.expect(MetalGpuFamily.apple9.isMetal4());
    try std.testing.expect(!MetalGpuFamily.apple8.isMetal4());
    try std.testing.expectEqual(@as(u32, 7), MetalGpuFamily.apple7.generation());
}

test "buildFeatureSet for Apple7" {
    const fs = buildFeatureSet(.apple7);
    try std.testing.expect(fs.supports_mesh_shaders);
    try std.testing.expect(fs.supports_ray_tracing);
    try std.testing.expect(fs.supports_mps);
    try std.testing.expect(fs.supports_bfloat16);
    try std.testing.expect(fs.has_neural_engine);
}

test "buildFeatureSet for Apple3" {
    const fs = buildFeatureSet(.apple3);
    try std.testing.expect(!fs.supports_mesh_shaders);
    try std.testing.expect(!fs.supports_ray_tracing);
    try std.testing.expect(fs.supports_mps);
    try std.testing.expect(!fs.supports_bfloat16);
    try std.testing.expect(!fs.has_neural_engine);
}

test "buildFeatureSet for Mac2" {
    const fs = buildFeatureSet(.mac2);
    try std.testing.expect(fs.supports_mesh_shaders);
    try std.testing.expect(fs.supports_ray_tracing);
    try std.testing.expect(fs.supports_mps);
    try std.testing.expect(fs.supports_mps_graph);
}

test "apple10 generation and Metal4" {
    try std.testing.expectEqual(@as(u32, 10), MetalGpuFamily.apple10.generation());
    try std.testing.expect(MetalGpuFamily.apple10.isMetal3());
    try std.testing.expect(MetalGpuFamily.apple10.isMetal4());
}

test "buildFeatureSet for Apple10 (M4)" {
    const fs = buildFeatureSet(.apple10);
    try std.testing.expect(fs.supports_mesh_shaders);
    try std.testing.expect(fs.supports_ray_tracing);
    try std.testing.expect(fs.supports_mps);
    try std.testing.expect(fs.supports_mps_graph);
    try std.testing.expect(fs.supports_bfloat16);
    try std.testing.expect(fs.has_neural_engine);
    try std.testing.expectEqual(@as(u32, 65536), fs.max_threadgroup_memory);
}

test "chipVariant parsing" {
    try std.testing.expectEqual(MetalGpuFamily.ChipVariant.ultra, MetalGpuFamily.chipVariant("Apple M4 Ultra"));
    try std.testing.expectEqual(MetalGpuFamily.ChipVariant.max, MetalGpuFamily.chipVariant("Apple M4 Max"));
    try std.testing.expectEqual(MetalGpuFamily.ChipVariant.pro, MetalGpuFamily.chipVariant("Apple M4 Pro"));
    try std.testing.expectEqual(MetalGpuFamily.ChipVariant.base, MetalGpuFamily.chipVariant("Apple M4"));
    try std.testing.expectEqual(MetalGpuFamily.ChipVariant.unknown, MetalGpuFamily.chipVariant("Unknown GPU"));
}

test "apple10 name" {
    try std.testing.expectEqualStrings("Apple10 (A18, M4+) [Metal 4]", MetalGpuFamily.apple10.name());
}
