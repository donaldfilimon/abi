const std = @import("std");
const builtin = @import("builtin");

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
        printUsage();
        std.process.exit(1);
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "install") or std.mem.eql(u8, command, "--install") or std.mem.eql(u8, command, "--update")) {
        const version = if (args.len > 2) args[2] else try getProjectVersion(allocator, init.io);
        try doInstall(allocator, init.io, version);
    } else if (std.mem.eql(u8, command, "use") or std.mem.eql(u8, command, "--link")) {
        const version = if (args.len > 2) args[2] else try getProjectVersion(allocator, init.io);
        try doUse(allocator, init.io, version);
    } else if (std.mem.eql(u8, command, "status") or std.mem.eql(u8, command, "--status")) {
        try doStatus(allocator, init.io);
    } else if (std.mem.eql(u8, command, "bootstrap") or std.mem.eql(u8, command, "--bootstrap")) {
        try doBootstrap(allocator, init.io);
    } else if (std.mem.eql(u8, command, "doctor") or std.mem.eql(u8, command, "--doctor")) {
        try doDoctor(allocator, init.io);
    } else if (std.mem.eql(u8, command, "current")) {
        try doCurrent(allocator, init.io);
    } else if (std.mem.eql(u8, command, "list") or std.mem.eql(u8, command, "ls")) {
        try doList(allocator, init.io);
    } else if (std.mem.eql(u8, command, "list-remote") or std.mem.eql(u8, command, "lsr")) {
        try doListRemote(allocator, init.io);
    } else if (std.mem.eql(u8, command, "clean")) {
        try doClean(allocator, init.io);
    } else if (std.mem.eql(u8, command, "help") or std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage();
    } else {
        std.debug.print("ERROR: Unknown command '{s}'\n", .{command});
        printUsage();
        std.process.exit(1);
    }
}

fn printUsage() void {
    const usage =
        \\Zigly - Zig Version Manager
        \\
        \\Usage: zigly <command> [options]
        \\
        \\Commands:
        \\  install [version]    Install a specific version of Zig + ZLS (defaults to .zigversion)
        \\  use [version]        Install and set as the active global version (symlinks to ~/.local/bin)
        \\  list, ls             List installed versions
        \\  list-remote, lsr     List available versions from ziglang.org
        \\  current              Show the currently active version and project status
        \\  clean                Remove all cached versions and downloads
        \\  bootstrap            One-command project setup (install from .zigversion and link)
        \\  doctor               Report toolchain health diagnostics
        \\  status               Print path to the active zig binary (useful for scripts)
        \\
        \\Examples:
        \\  zigly install master
        \\  zigly use 0.13.0
        \\  zigly bootstrap
        \\
    ;
    std.debug.print("{s}", .{usage});
}

fn getProjectVersion(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    const file = std.Io.Dir.cwd().openFile(io, zigversion_file, .{}) catch {
        std.debug.print("ERROR: No .zigversion found in current directory.\n", .{});
        std.process.exit(1);
    };
    defer file.close(io);

    const stat = file.stat(io) catch {
        std.debug.print("ERROR: Could not stat .zigversion.\n", .{});
        std.process.exit(1);
    };

    const content = allocator.alloc(u8, stat.size) catch {
        std.debug.print("ERROR: Out of memory.\n", .{});
        std.process.exit(1);
    };
    
    _ = file.readPositionalAll(io, content, 0) catch {
        std.debug.print("ERROR: Failed to read .zigversion.\n", .{});
        std.process.exit(1);
    };

    return std.mem.trim(u8, content, " \n\r\t");
}

fn execBash(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    var self_exe_buf: [std.fs.max_path_bytes]u8 = undefined;
    const self_exe = std.fs.selfExePath(&self_exe_buf) catch {
        std.debug.print("ERROR: Could not resolve executable path\n", .{});
        std.process.exit(1);
    };
    
    const bin_dir = std.fs.path.dirname(self_exe) orelse ".";
    const project_root = std.fs.path.dirname(std.fs.path.dirname(std.fs.path.dirname(bin_dir) orelse ".") orelse ".") orelse ".";
    const zigly_script = std.fs.path.join(allocator, &[_][]const u8{ project_root, "tools", "zigly_cli", "zigly" }) catch "tools/zigly_cli/zigly";
    
    var child_args = std.ArrayListUnmanaged([]const u8).empty;
    try child_args.append(allocator, "bash");
    try child_args.append(allocator, zigly_script);
    for (args) |arg| {
        try child_args.append(allocator, arg);
    }

    var child = try std.process.spawn(io, .{
        .argv = child_args.items,
    });
    
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code != 0) std.process.exit(code),
        else => std.process.exit(1),
    }
}

fn doInstall(allocator: std.mem.Allocator, io: std.Io, version: []const u8) !void { try execBash(allocator, io, &[_][]const u8{ "install", version }); }
fn doUse(allocator: std.mem.Allocator, io: std.Io, version: []const u8) !void { try execBash(allocator, io, &[_][]const u8{ "use", version }); }
fn doStatus(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "status" }); }
fn doBootstrap(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "bootstrap" }); }
fn doDoctor(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "doctor" }); }
fn doCurrent(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "current" }); }
fn doList(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "list" }); }
fn doListRemote(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "list-remote" }); }
fn doClean(allocator: std.mem.Allocator, io: std.Io) !void { try execBash(allocator, io, &[_][]const u8{ "clean" }); }
