const backend_mod = @import("../../backend.zig");
const common = @import("common.zig");
const std = @import("std");

pub const Profile = enum {
    desktop,
    es,

    pub fn api(self: Profile) common.Api {
        return switch (self) {
            .desktop => .opengl,
            .es => .opengles,
        };
    }

    pub fn backend(self: Profile) backend_mod.Backend {
        return switch (self) {
            .desktop => .opengl,
            .es => .opengles,
        };
    }

    pub fn name(self: Profile) []const u8 {
        return switch (self) {
            .desktop => "opengl",
            .es => "opengles",
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
