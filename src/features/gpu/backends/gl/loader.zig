const common = @import("common.zig");
const opengl = @import("../opengl.zig");
const opengles = @import("../opengles.zig");

pub fn init(api: common.Api) anyerror!void {
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
