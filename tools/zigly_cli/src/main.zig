
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core.zig");
const cli = @import("cli.zig");

// Zigly CLI in pure Zig
// Self-building, recovering version manager for Zig + ZLS

const zigly_home_dir = ".zigly";
const zigversion_file = ".zigversion";

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    // Get command line arguments using the Args iterator
    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    defer args_iter.deinit();

    // Skip program name (first arg) - the first next() returns executable path
    _ = args_iter.next();

    // Get the command (second arg)
    const command = args_iter.next() orelse {
        cli.printUsage();
        std.process.exit(1);
    };

    // Get optional version argument
    const version = args_iter.next();

    var config = try core.initConfig(allocator, io, init.environ_map);
    defer config.deinit();

    if (std.mem.eql(u8, command, "install") or std.mem.eql(u8, command, "--install") or std.mem.eql(u8, command, "--update")) {
        try cli.doInstall(&config, version orelse "");
    } else if (std.mem.eql(u8, command, "use") or std.mem.eql(u8, command, "--link")) {
        try cli.doUse(&config, version orelse "");
    } else if (std.mem.eql(u8, command, "status") or std.mem.eql(u8, command, "--status")) {
        try cli.doStatus(&config, version orelse "");
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
    } else if (std.mem.eql(u8, command, "unlink") or std.mem.eql(u8, command, "--unlink")) {
        try cli.doUnlink(&config);
    } else if (std.mem.eql(u8, command, "check") or std.mem.eql(u8, command, "--check")) {
        try cli.doCheck(&config);
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
