const std = @import("std");
const builtin = @import("builtin");

pub const PlatformClass = enum {
    macos,
    linux,
    windows,
    ios,
    android,
    web,
    freebsd,
    netbsd,
    openbsd,
    dragonfly,
    haiku,
    solaris,
    freestanding,
    other,
};

pub const BackendName = []const u8;

pub const macos_order = [_]BackendName{ "metal", "vulkan", "opengl", "stdgpu" };
pub const linux_order = [_]BackendName{ "cuda", "vulkan", "opengl", "stdgpu" };
pub const windows_order = [_]BackendName{ "cuda", "vulkan", "opengl", "stdgpu" };
pub const ios_order = [_]BackendName{ "metal", "opengles", "stdgpu" };
pub const android_order = [_]BackendName{ "vulkan", "opengles", "stdgpu" };
pub const web_order = [_]BackendName{ "webgpu", "webgl2", "simulated" };
pub const freebsd_order = [_]BackendName{ "vulkan", "opengl", "stdgpu" };
pub const netbsd_order = [_]BackendName{ "opengl", "stdgpu" };
pub const openbsd_order = [_]BackendName{ "opengl", "stdgpu" };
pub const dragonfly_order = [_]BackendName{ "opengl", "stdgpu" };
pub const haiku_order = [_]BackendName{ "opengl", "stdgpu" };
pub const solaris_order = [_]BackendName{ "opengl", "stdgpu" };
pub const fallback_order = [_]BackendName{ "stdgpu", "simulated" };

pub fn classify(os_tag: std.Target.Os.Tag, abi: std.Target.Abi) PlatformClass {
    return switch (os_tag) {
        .macos => .macos,
        .windows => .windows,
        .ios, .tvos, .watchos => .ios,
        .wasi, .emscripten => .web,
        .freestanding => .freestanding,
        .freebsd => .freebsd,
        .netbsd => .netbsd,
        .openbsd => .openbsd,
        .dragonfly => .dragonfly,
        .haiku => .haiku,
        .illumos => .solaris,
        .linux => if (abi == .android) .android else .linux,
        else => .other,
    };
}

pub fn classifyBuiltin() PlatformClass {
    return classify(builtin.target.os.tag, builtin.abi);
}

pub fn defaultOrder(platform: PlatformClass) []const BackendName {
    return switch (platform) {
        .macos => macos_order[0..],
        .linux => linux_order[0..],
        .windows => windows_order[0..],
        .ios => ios_order[0..],
        .android => android_order[0..],
        .web => web_order[0..],
        .freebsd => freebsd_order[0..],
        .netbsd => netbsd_order[0..],
        .openbsd => openbsd_order[0..],
        .dragonfly => dragonfly_order[0..],
        .haiku => haiku_order[0..],
        .solaris => solaris_order[0..],
        .freestanding, .other => fallback_order[0..],
    };
}

pub fn withAndroidPrimary(primary: BackendName) []const BackendName {
    if (!std.mem.eql(u8, primary, "opengles")) {
        return android_order[0..];
    }
    return &.{ "opengles", "vulkan", "stdgpu" };
}

pub fn defaultOrderForTarget(os_tag: std.Target.Os.Tag, abi: std.Target.Abi) []const BackendName {
    return defaultOrder(classify(os_tag, abi));
}

test {
    std.testing.refAllDecls(@This());
}
