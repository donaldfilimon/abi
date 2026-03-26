const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const cli = @import("cli.zig");

// Zigly CLI in pure Zig
// Self-building, recovering version manager for Zig + ZLS

const zigly_home_dir = ".zigly";
const zigversion_file = ".zigversion";

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try init.minimal.args.toSlice(allocator);

    if (args.len <= 1) {
        cli.printUsage();
        std.process.exit(1);
    }

    var config = try core.initConfig(allocator, init.io, init.environ_map);
    defer config.deinit();

    const command = args[1];

    if (std.mem.eql(u8, command, "install") or std.mem.eql(u8, command, "--install") or std.mem.eql(u8, command, "--update")) {
        const version = if (args.len > 2) args[2] else "";
        try cli.doInstall(&config, version);
    } else if (std.mem.eql(u8, command, "use") or std.mem.eql(u8, command, "--link")) {
        const version = if (args.len > 2) args[2] else "";
        try cli.doUse(&config, version);
    } else if (std.mem.eql(u8, command, "status") or std.mem.eql(u8, command, "--status")) {
        const version = if (args.len > 2) args[2] else "";
        try cli.doStatus(&config, version);
    } else if (std.mem.eql(u8, command, "bootstrap") or std.mem.eql(u8, command, "--bootstrap")) {
        try cli.doBootstrap(&config);
    } else if (std.mem.eql(u8, command, "doctor") or std.mem.eql(u8, command, "--doctor")) {
        try cli.doDoctor(&config);
    } else if (std.mem.eql(u8, command, "current")) {
        try cli.doCurrent(&config);
    } else if (std.mem.eql(u8, command, "list") or std.mem.eql(u8, command, "ls")) {
        try cli.doList(&config);
    } else if (std.mem.eql(u8, command, "list-remote") or std.mem.eql(u8, command, "lsr")) {
        try cli.doListRemote(&config);
    } else if (std.mem.eql(u8, command, "clean")) {
        try cli.doClean(&config);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        cli.printUsage();
    } else {
        std.debug.print("ERROR: Unknown command '{s}'\n", .{command});
        cli.printUsage();
        std.process.exit(1);
    }
}
