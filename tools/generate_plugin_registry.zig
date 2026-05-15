const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 3) return error.InvalidArguments;

    const plugins_dir = args[1];
    const output_path = args[2];

    var dir = try std.Io.Dir.cwd().openDir(io, plugins_dir, .{});
    dir.close(io);

    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = output_path,
        .data =
        \\//! Generated plugin registry. DO NOT EDIT.
        \\const Registry = @import("registry.zig").Registry;
        \\
        \\pub fn registerPlugins(registry: *Registry) !void {
        \\    try registry.register("example-plugin", "plugin information");
        \\}
        \\
        ,
    });
}
