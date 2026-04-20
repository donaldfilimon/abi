const std = @import("std");
const Server = @import("server.zig").Server;

const ToolHandler = @import("server.zig").ToolHandler;

pub const ToolDef = struct {
    name: []const u8,
    description: []const u8,
    input_schema: []const u8,
    handler: ToolHandler,
};

pub fn registerTools(server: *Server, modules: anytype) !void {
    inline for (modules) |mod| {
        if (@hasDecl(mod, "tools")) {
            inline for (mod.tools) |t| {
                try server.addTool(.{
                    .def = .{
                        .name = t.name,
                        .description = t.description,
                        .input_schema = t.input_schema,
                    },
                    .handler = t.handler,
                });
            }
        }
    }
}
