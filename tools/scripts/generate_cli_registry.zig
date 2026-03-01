const std = @import("std");
const util = @import("util.zig");

const ModuleEntry = struct {
    ident: []u8,
    import_path: []u8,
};

fn sanitizeIdentifierAlloc(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out = try allocator.alloc(u8, raw.len);
    for (raw, 0..) |ch, i| {
        out[i] = if (std.ascii.isAlphanumeric(ch)) std.ascii.toLower(ch) else '_';
    }
    return out;
}

fn appendModuleEntries(
    allocator: std.mem.Allocator,
    modules: *std.ArrayListUnmanaged(ModuleEntry),
    listing: []const u8,
    is_dir_listing: bool,
) !void {
    var lines = std.mem.splitScalar(u8, listing, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        if (!is_dir_listing and std.mem.eql(u8, line, "editor.zig")) continue;

        const ident_raw = if (is_dir_listing)
            line
        else
            line[0 .. line.len - ".zig".len];

        const ident = try sanitizeIdentifierAlloc(allocator, ident_raw);
        errdefer allocator.free(ident);

        const import_path = if (is_dir_listing)
            try std.fmt.allocPrint(allocator, "../commands/{s}/mod.zig", .{line})
        else
            try std.fmt.allocPrint(allocator, "../commands/{s}", .{line});
        errdefer allocator.free(import_path);

        try modules.append(allocator, .{
            .ident = ident,
            .import_path = import_path,
        });
    }
}

fn collectModules(allocator: std.mem.Allocator) ![]ModuleEntry {
    var modules = std.ArrayListUnmanaged(ModuleEntry).empty;
    errdefer {
        for (modules.items) |m| {
            allocator.free(m.ident);
            allocator.free(m.import_path);
        }
        modules.deinit(allocator);
    }

    const files_cmd =
        "find tools/cli/commands -mindepth 1 -maxdepth 1 -type f -name '*.zig' ! -name 'mod.zig' | sed 's#^tools/cli/commands/##'";
    const files_result = try util.captureCommand(allocator, files_cmd);
    defer allocator.free(files_result.output);
    if (files_result.exit_code != 0) return error.CommandFailed;
    try appendModuleEntries(allocator, &modules, files_result.output, false);

    const dirs_cmd =
        "find tools/cli/commands -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/mod.zig' ';' -print | sed 's#^tools/cli/commands/##'";
    const dirs_result = try util.captureCommand(allocator, dirs_cmd);
    defer allocator.free(dirs_result.output);
    if (dirs_result.exit_code != 0) return error.CommandFailed;
    try appendModuleEntries(allocator, &modules, dirs_result.output, true);

    std.mem.sort(ModuleEntry, modules.items, {}, struct {
        fn lessThan(_: void, lhs: ModuleEntry, rhs: ModuleEntry) bool {
            return std.mem.lessThan(u8, lhs.import_path, rhs.import_path);
        }
    }.lessThan);

    return try modules.toOwnedSlice(allocator);
}

fn renderSnapshot(allocator: std.mem.Allocator, modules: []const ModuleEntry) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    try out.writer.writeAll(
        "//! Generated command registry snapshot.\n" ++
            "//! Refresh with: `zig build refresh-cli-registry`\n\n",
    );

    for (modules) |module| {
        try out.writer.print("pub const {s} = @import(\"{s}\");\n", .{
            module.ident,
            module.import_path,
        });
    }

    try out.writer.writeAll("\npub const command_modules = .{\n");
    for (modules) |module| {
        try out.writer.print("    {s},\n", .{module.ident});
    }
    try out.writer.writeAll("};\n");

    return try out.toOwnedSlice();
}

fn freeModules(allocator: std.mem.Allocator, modules: []ModuleEntry) void {
    for (modules) |m| {
        allocator.free(m.ident);
        allocator.free(m.import_path);
    }
    allocator.free(modules);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();

    var output_path: []const u8 = "tools/cli/generated/cli_registry_snapshot.zig";
    var check_only = false;

    const args = try init.minimal.args.toSlice(allocator);

    var arg_idx: usize = 1;
    while (arg_idx < args.len) : (arg_idx += 1) {
        const arg = args[arg_idx];
        if (std.mem.eql(u8, arg, "--check")) {
            check_only = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--output")) {
            arg_idx += 1;
            if (arg_idx >= args.len) return error.MissingOutputPath;
            output_path = args[arg_idx];
            continue;
        }
        if (std.mem.eql(u8, arg, "--snapshot")) {
            output_path = "tools/cli/generated/cli_registry_snapshot.zig";
            continue;
        }
    }

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.minimal.environ });
    defer io_backend.deinit();
    const io = io_backend.io();

    const modules = try collectModules(allocator);
    defer freeModules(allocator, modules);

    const snapshot_body = try renderSnapshot(allocator, modules);
    defer allocator.free(snapshot_body);

    if (check_only) {
        const existing = std.Io.Dir.cwd().readFileAlloc(io, output_path, allocator, .limited(1024 * 1024)) catch |err| switch (err) {
            error.FileNotFound => {
                std.debug.print("ERROR: registry file missing: {s}\n", .{output_path});
                std.process.exit(1);
            },
            else => return err,
        };
        defer allocator.free(existing);

        if (!std.mem.eql(u8, existing, snapshot_body)) {
            std.debug.print("ERROR: CLI registry snapshot drift: {s}\n", .{output_path});
            std.debug.print("Run: zig build refresh-cli-registry\n", .{});
            std.process.exit(1);
        }
        std.debug.print("OK: CLI registry snapshot is up to date.\n", .{});
        return;
    }

    if (std.fs.path.dirname(output_path)) |dir_name| {
        try std.Io.Dir.cwd().createDirPath(io, dir_name);
    }

    const file = try std.Io.Dir.cwd().createFile(io, output_path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, snapshot_body);
    std.debug.print("Wrote {s}\n", .{output_path});
}
