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

test "cross-target policy: macos classification and order" {
    const class = classify(.macos, .none);
    try std.testing.expectEqual(PlatformClass.macos, class);
    const order = defaultOrder(class);
    try std.testing.expectEqualStrings("metal", order[0]);
    try std.testing.expectEqualStrings("vulkan", order[1]);
    try std.testing.expectEqualStrings("opengl", order[2]);
    try std.testing.expectEqualStrings("stdgpu", order[3]);
    try std.testing.expectEqual(@as(usize, 4), order.len);
}

test "cross-target policy: linux classification and order" {
    const class = classify(.linux, .gnu);
    try std.testing.expectEqual(PlatformClass.linux, class);
    const order = defaultOrder(class);
    try std.testing.expectEqualStrings("cuda", order[0]);
    try std.testing.expectEqualStrings("vulkan", order[1]);
    try std.testing.expectEqualStrings("opengl", order[2]);
    try std.testing.expectEqualStrings("stdgpu", order[3]);
    try std.testing.expectEqual(@as(usize, 4), order.len);
}

test "cross-target policy: windows classification and order" {
    const class = classify(.windows, .msvc);
    try std.testing.expectEqual(PlatformClass.windows, class);
    const order = defaultOrder(class);
    try std.testing.expectEqualStrings("cuda", order[0]);
    try std.testing.expectEqualStrings("vulkan", order[1]);
    try std.testing.expectEqualStrings("opengl", order[2]);
    try std.testing.expectEqualStrings("stdgpu", order[3]);
    try std.testing.expectEqual(@as(usize, 4), order.len);
}

test "cross-target policy: web classification and order" {
    const class1 = classify(.wasi, .musl);
    try std.testing.expectEqual(PlatformClass.web, class1);
    const class2 = classify(.emscripten, .none);
    try std.testing.expectEqual(PlatformClass.web, class2);
    
    const order = defaultOrder(class1);
    try std.testing.expectEqualStrings("webgpu", order[0]);
    try std.testing.expectEqualStrings("webgl2", order[1]);
    try std.testing.expectEqualStrings("simulated", order[2]);
}

test "cross-target policy: android classification and order" {
    const class = classify(.linux, .android);
    try std.testing.expectEqual(PlatformClass.android, class);
    const order = defaultOrder(class);
    try std.testing.expectEqualStrings("vulkan", order[0]);
    try std.testing.expectEqualStrings("opengles", order[1]);
    try std.testing.expectEqualStrings("stdgpu", order[2]);
    
    const modified = withAndroidPrimary("opengles");
    try std.testing.expectEqualStrings("opengles", modified[0]);
    try std.testing.expectEqualStrings("vulkan", modified[1]);
}

test {
    std.testing.refAllDecls(@This());
}
