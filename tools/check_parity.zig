const std = @import("std");
const fs = @import("std").fs;
const mem = @import("std").mem;

const DeclSet = std.StringHashMap(void);

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    var debug_allocator = std.heap.DebugAllocator(.{}){};
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    // Parse command line arguments
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    var mode: []const u8 = "check"; // default mode
    var i: usize = 0;
    while (i < args.len) {
        const j = i;
        if (std.mem.eql(u8, args[j], "--mode")) {
            i += 1;
            if (i < args.len) {
                mode = args[i];
            } else {
                std.log.err("Missing value for --mode option", .{});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, args[j], "--help")) {
            printHelpAndExit();
            unreachable;
        } else {
            i += 1;
        }
    }

    if (std.mem.eql(u8, mode, "generate")) {
        return generateStubs(io, allocator);
    } else if (std.mem.eql(u8, mode, "check")) {
        var failures: usize = 0;
        failures += try checkTree(io, allocator, "src/features");
        failures += try checkTree(io, allocator, "src/plugins");

        if (failures != 0) {
            std.log.err("mod/stub parity failed for {d} pair(s)", .{failures});
            std.process.exit(1);
        }
    } else {
        std.log.err("Invalid mode: {s}. Use 'check' or 'generate'", .{mode});
        std.process.exit(1);
    }
}

fn printHelpAndExit() noreturn {
    const msg = "Usage: check_parity [--mode=check|generate] [--help]\n" ++
        "  --mode=check    Check mod/stub parity (default)\n" ++
        "  --mode=generate Generate stub files from mod files\n" ++
        "  --help          Show this help message";
    std.debug.print("{s}\n", .{msg});
    std.process.exit(0);
}

fn generateStubs(io: std.Io, allocator: std.mem.Allocator) !void {
    var generated: usize = 0;
    generated += try generateTree(io, allocator, "src/features");
    generated += try generateTree(io, allocator, "src/plugins");

    std.debug.print("Generated stubs for {d} module(s)\n", .{generated});
}

fn generateTree(io: std.Io, allocator: std.mem.Allocator, root_path: []const u8) !usize {
    var root = std.Io.Dir.cwd().openDir(io, root_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return 0,
        else => return err,
    };
    defer root.close(io);

    var walker = try root.walk(allocator);
    defer walker.deinit();

    var generated: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, "mod.zig")) continue;

        const dir_name = std.fs.path.dirname(entry.path) orelse ".";
        const stub_relative = try std.fs.path.join(allocator, &.{ dir_name, "stub.zig" });
        defer allocator.free(stub_relative);

        // Check if stub file exists, if not create it or overwrite it
        try generateStubFile(io, allocator, root_path, entry.path, stub_relative);
        generated += 1;
    }

    return generated;
}

fn generateStubFile(io: std.Io, allocator: std.mem.Allocator, root_path: []const u8, mod_relative: []const u8, stub_relative: []const u8) !void {
    const mod_path = try std.fs.path.join(allocator, &.{ root_path, mod_relative });
    defer allocator.free(mod_path);

    const stub_path = try std.fs.path.join(allocator, &.{ root_path, stub_relative });
    defer allocator.free(stub_path);

    const mod_source = try std.Io.Dir.cwd().readFileAlloc(io, mod_path, allocator, .limited(1024 * 1024));
    defer allocator.free(mod_source);

    const stub_content = try generateStubContent(allocator, mod_relative, mod_source);
    defer allocator.free(stub_content);

    // Write the stub file
    const dir = std.Io.Dir.cwd();
    var file = try dir.createFile(io, stub_path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, stub_content);

    std.debug.print("Generated stub: {s}\n", .{stub_relative});
}

fn generateStubContent(allocator: std.mem.Allocator, mod_relative: []const u8, mod_source: []const u8) ![]const u8 {
    _ = mod_source;
    // For now, let's implement a simple approach:
    // 1. Parse the mod file to extract imports and structure
    // 2. Generate a stub file with similar structure but simplified implementations

    // This is a simplified implementation that just copies the basic structure
    // A more sophisticated implementation would parse and transform the code

    // For MVP, let's create a template-based approach
    // In a real implementation, we'd want to parse the AST properly

    // Simple approach: replace function bodies with simple returns
    // Keep structs, enums, constants mostly the same

    // For now, let's just return a basic template that shows the concept
    // In a real implementation, we'd do proper code transformation

    const template = "\\n// Generated stub for {s}\\n// This is a placeholder implementation\\nconst std = @import(\"std\");\\n\\n// TODO: Implement proper stub generation\\n";
    return try std.fmt.allocPrint(allocator, template, .{mod_relative});
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
