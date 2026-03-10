const common = @import("common");
const opengl = @import("../opengl");
const opengles = @import("../opengles");
const std = @import("std");

const GlInitError = opengl.OpenGlError || opengles.OpenGlesError;

pub fn init(api: common.Api) GlInitError!void {
    switch (api) {
        .opengl => try opengl.init(),
        .opengles => try opengles.init(),
    }
}

pub fn deinit(api: common.Api) void {
    switch (api) {
        .opengl => opengl.deinit(),
        .opengles => opengles.deinit(),
    }
}

pub fn isAvailable(api: common.Api) bool {
    return switch (api) {
        .opengl => opengl.isAvailable(),
        .opengles => opengles.isAvailable(),
    };
}

test {
    std.testing.refAllDecls(@This());
}
