const std = @import("std");

pub const GraphicsBackend = enum {
    vulkan,
    metal,
    direct3d12,
    opengl,
    webgpu,
};

pub const GraphicsDriver = struct {
    backend: GraphicsBackend,

    pub fn init(backend: GraphicsBackend) GraphicsDriver {
        return GraphicsDriver{ .backend = backend };
    }

    pub fn renderFrame(self: *GraphicsDriver) void {
        // cross-platform rendering placeholder
        _ = self;
    }
};

test "GraphicsDriver init" {
    var driver = GraphicsDriver.init(.vulkan);
    try std.testing.expect(driver.backend == .vulkan);
}
