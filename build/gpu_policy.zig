const std = @import("std");

/// High-level platform classification for GPU backend selection.
pub const PlatformClass = enum {
    macos,
    linux,
    windows,
    ios,
    android,
    web,
    freestanding,
    other,
};

pub const BackendName = []const u8;

/// Per-platform tuning hints consumed by the runtime GPU layer.
pub const OptimizationHints = struct {
    default_local_size: u32,
    default_queue_depth: u32,
    prefer_unified_memory: bool,
    prefer_pinned_staging: bool,
    transfer_chunk_bytes: usize,
};

// ── Platform backend orderings ──────────────────────────────────────────

pub const macos_order = [_]BackendName{ "metal", "vulkan", "opengl", "stdgpu" };
pub const linux_order = [_]BackendName{ "cuda", "vulkan", "opengl", "stdgpu" };
pub const windows_order = [_]BackendName{ "cuda", "vulkan", "opengl", "stdgpu" };
pub const ios_order = [_]BackendName{ "metal", "opengles", "stdgpu" };
pub const android_order = [_]BackendName{ "vulkan", "opengles", "stdgpu" };
pub const web_order = [_]BackendName{ "webgpu", "webgl2", "simulated" };
pub const fallback_order = [_]BackendName{ "stdgpu", "simulated" };

/// Fixed-capacity list that deduplicates backend names on insert.
pub const BackendNameList = struct {
    items: [12][]const u8 = undefined,
    len: usize = 0,

    pub fn append(self: *BackendNameList, name: []const u8) void {
        if (self.len >= self.items.len) return;
        for (self.items[0..self.len]) |existing|
            if (std.mem.eql(u8, existing, name)) return;
        self.items[self.len] = name;
        self.len += 1;
    }

    pub fn slice(self: *const BackendNameList) []const []const u8 {
        return self.items[0..self.len];
    }
};

/// Inputs for auto backend selection.
pub const SelectionContext = struct {
    platform: PlatformClass,
    enable_gpu: bool,
    enable_web: bool,
    can_link_metal: bool = true,
    warn_if_metal_skipped: bool = false,
    allow_simulated: bool = false,
    android_primary: ?[]const u8 = null,
};

/// Classify an OS tag + ABI pair into a `PlatformClass`.
pub fn classify(os_tag: std.Target.Os.Tag, abi: std.Target.Abi) PlatformClass {
    return switch (os_tag) {
        .macos => .macos,
        .windows => .windows,
        .ios => .ios,
        .wasi => .web,
        .freestanding => .freestanding,
        .linux => if (abi == .android) .android else .linux,
        else => .other,
    };
}

/// Return the default backend preference order for a platform.
pub fn defaultOrder(platform: PlatformClass) []const BackendName {
    return switch (platform) {
        .macos => macos_order[0..],
        .linux => linux_order[0..],
        .windows => windows_order[0..],
        .ios => ios_order[0..],
        .android => android_order[0..],
        .web => web_order[0..],
        .freestanding, .other => fallback_order[0..],
    };
}

/// Override Android ordering when OpenGL ES is the preferred primary.
pub fn withAndroidPrimary(primary: BackendName) []const BackendName {
    if (!std.mem.eql(u8, primary, "opengles"))
        return android_order[0..];
    return &.{ "opengles", "vulkan", "stdgpu" };
}

/// Resolve the list of backend names to enable, respecting platform rules,
/// Metal availability, and web/simulated gates.
pub fn resolveAutoBackendNames(ctx: SelectionContext) BackendNameList {
    var result = BackendNameList{};

    const base_order = blk: {
        if (ctx.platform == .android) {
            if (ctx.android_primary) |primary|
                break :blk withAndroidPrimary(primary);
        }
        break :blk defaultOrder(ctx.platform);
    };

    for (base_order) |name| {
        if (std.mem.eql(u8, name, "metal") and !ctx.can_link_metal) {
            if (ctx.warn_if_metal_skipped)
                std.log.warn("Skipping Metal backend: required Apple frameworks not found.", .{});
            continue;
        }
        if (!shouldInclude(ctx, name)) continue;
        result.append(name);
    }

    return result;
}

fn shouldInclude(ctx: SelectionContext, name: []const u8) bool {
    if (std.mem.eql(u8, name, "webgpu") or std.mem.eql(u8, name, "webgl2"))
        return ctx.enable_web and ctx.platform == .web;
    if (std.mem.eql(u8, name, "simulated"))
        return ctx.allow_simulated and (ctx.enable_gpu or ctx.enable_web);
    return ctx.enable_gpu;
}

/// Return runtime optimization hints appropriate for the given platform.
pub fn optimizationHintsForPlatform(platform: PlatformClass) OptimizationHints {
    return switch (platform) {
        .macos => .{
            .default_local_size = 256,
            .default_queue_depth = 8,
            .prefer_unified_memory = true,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 8 * 1024 * 1024,
        },
        .linux => .{
            .default_local_size = 256,
            .default_queue_depth = 8,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = true,
            .transfer_chunk_bytes = 16 * 1024 * 1024,
        },
        .windows => .{
            .default_local_size = 128,
            .default_queue_depth = 4,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = true,
            .transfer_chunk_bytes = 8 * 1024 * 1024,
        },
        .ios => .{
            .default_local_size = 128,
            .default_queue_depth = 4,
            .prefer_unified_memory = true,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 4 * 1024 * 1024,
        },
        .android => .{
            .default_local_size = 128,
            .default_queue_depth = 4,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 4 * 1024 * 1024,
        },
        .web => .{
            .default_local_size = 64,
            .default_queue_depth = 2,
            .prefer_unified_memory = false,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 1 * 1024 * 1024,
        },
        .freestanding, .other => .{
            .default_local_size = 64,
            .default_queue_depth = 2,
            .prefer_unified_memory = true,
            .prefer_pinned_staging = false,
            .transfer_chunk_bytes = 1 * 1024 * 1024,
        },
    };
}
