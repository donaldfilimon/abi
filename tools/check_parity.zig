const std = @import("std");

const DeclSet = std.StringHashMap(void);

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var debug_allocator = std.heap.DebugAllocator(.{}){};
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    var failures: usize = 0;
    failures += try checkTree(io, allocator, "src/features");
    failures += try checkTree(io, allocator, "src/plugins");

    if (failures != 0) {
        std.log.err("mod/stub parity failed for {d} pair(s)", .{failures});
        std.process.exit(1);
    }
}

fn checkTree(io: std.Io, allocator: std.mem.Allocator, root_path: []const u8) !usize {
    var root = std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer root.close(io);

    var walker = try root.walk(allocator);
    defer walker.deinit();

    var failures: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, "mod.zig")) continue;

        const dir_name = std.fs.path.dirname(entry.path) orelse ".";
        const stub_relative = try std.fs.path.join(allocator, &.{ dir_name, "stub.zig" });
        defer allocator.free(stub_relative);

        root.access(io, stub_relative, .{}) catch |err| switch (err) {
            error.FileNotFound => continue,
            else => return err,
        };

        failures += try checkPair(io, allocator, root_path, entry.path, stub_relative);
    }

    return failures;
}

fn checkPair(
    io: std.Io,
    allocator: std.mem.Allocator,
    root_path: []const u8,
    mod_relative: []const u8,
    stub_relative: []const u8,
) !usize {
    var mod_decls = try readPublicDecls(io, allocator, root_path, mod_relative);
    defer freeDeclSet(&mod_decls);

    var stub_decls = try readPublicDecls(io, allocator, root_path, stub_relative);
    defer freeDeclSet(&stub_decls);

    var failed = false;

    var mod_it = mod_decls.keyIterator();
    while (mod_it.next()) |name| {
        if (!stub_decls.contains(name.*)) {
            std.log.err("{s}/{s}: missing in stub: {s}", .{ root_path, stub_relative, name.* });
            failed = true;
        }
    }

    var stub_it = stub_decls.keyIterator();
    while (stub_it.next()) |name| {
        if (!mod_decls.contains(name.*)) {
            std.log.err("{s}/{s}: missing in mod: {s}", .{ root_path, mod_relative, name.* });
            failed = true;
        }
    }

    return @intFromBool(failed);
}

fn readPublicDecls(io: std.Io, allocator: std.mem.Allocator, root_path: []const u8, relative_path: []const u8) !DeclSet {
    const path = try std.fs.path.join(allocator, &.{ root_path, relative_path });
    defer allocator.free(path);

    const source = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(source);

    var decls = DeclSet.init(allocator);
    errdefer freeDeclSet(&decls);

    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t");
        if (trimmed.len != line.len) continue;
        if (!std.mem.startsWith(u8, trimmed, "pub const ") and !std.mem.startsWith(u8, trimmed, "pub fn ")) continue;

        const after_prefix = if (std.mem.startsWith(u8, trimmed, "pub const "))
            trimmed["pub const ".len..]
        else
            trimmed["pub fn ".len..];

        const name_end = std.mem.indexOfAny(u8, after_prefix, " (=\t") orelse continue;
        const name = std.mem.trim(u8, after_prefix[0..name_end], " \t");
        if (name.len == 0) continue;

        try decls.put(try allocator.dupe(u8, name), {});
    }

    return decls;
}

fn freeDeclSet(decls: *DeclSet) void {
    var it = decls.keyIterator();
    while (it.next()) |name| {
        decls.allocator.free(name.*);
    }
    decls.deinit();
}
