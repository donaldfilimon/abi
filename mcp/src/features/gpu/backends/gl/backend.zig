//! Unified GL backend-family helpers for OpenGL + OpenGL ES.

const std = @import("std");
const build_options = @import("build_options");
const interface = @import("../../interface.zig");
const profile_mod = @import("profile.zig");

pub fn isProfileEnabledAtBuild(profile: profile_mod.Profile) bool {
    return switch (profile) {
        .desktop => build_options.gpu_opengl,
        .es => build_options.gpu_opengles,
    };
}

pub fn createVTableForProfile(
    allocator: std.mem.Allocator,
    profile: profile_mod.Profile,
) interface.BackendError!interface.Backend {
    if (!isProfileEnabledAtBuild(profile)) {
        return interface.BackendError.NotAvailable;
    }

    return switch (profile) {
        .desktop => blk: {
            const opengl_vtable = @import("../opengl_vtable.zig");
            break :blk opengl_vtable.createOpenGLVTable(allocator);
        },
        .es => blk: {
            const opengles_vtable = @import("../opengles_vtable.zig");
            break :blk opengles_vtable.createOpenGLESVTable(allocator);
        },
    };
}

test "build enablement follows profile flags" {
    try std.testing.expectEqual(build_options.gpu_opengl, isProfileEnabledAtBuild(.desktop));
    try std.testing.expectEqual(build_options.gpu_opengles, isProfileEnabledAtBuild(.es));
}

test {
    std.testing.refAllDecls(@This());
}
