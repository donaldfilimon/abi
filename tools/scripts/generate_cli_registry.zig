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

    return modules.toOwnedSlice(allocator);
}

fn renderSnapshot(allocator: std.mem.Allocator, modules: []const ModuleEntry) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    var writer = out.writer(allocator);
    try writer.writeAll(
        "//! Generated command registry snapshot.\n" ++
            "//! Refresh with: `zig build refresh-cli-registry`\n\n",
    );

    for (modules) |module| {
        try writer.print("pub const {s} = @import(\"{s}\");\n", .{
            module.ident,
            module.import_path,
        });
    }

    try writer.writeAll("\npub const command_modules = .{\n");
    for (modules) |module| {
        try writer.print("    {s},\n", .{module.ident});
    }
    try writer.writeAll("};\n");

    return out.toOwnedSlice(allocator);
}

fn freeModules(allocator: std.mem.Allocator, modules: []ModuleEntry) void {
    for (modules) |m| {
        allocator.free(m.ident);
        allocator.free(m.import_path);
    }
    allocator.free(modules);
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var output_path: []const u8 = "tools/cli/generated/cli_registry_snapshot.zig";
    var check_only = false;

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.next();

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--check")) {
            check_only = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--output")) {
            output_path = args.next() orelse return error.MissingOutputPath;
            continue;
        }
        if (std.mem.eql(u8, arg, "--snapshot")) {
            output_path = "tools/cli/generated/cli_registry_snapshot.zig";
            continue;
        }
        return error.InvalidArgument;
    }

    const modules = try collectModules(allocator);
    defer freeModules(allocator, modules);

    const snapshot_body = try renderSnapshot(allocator, modules);
    defer allocator.free(snapshot_body);

    if (check_only) {
        const existing = std.Io.Dir.cwd().readFileAlloc(allocator, output_path, 1 << 20) catch |err| switch (err) {
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
        std.Io.Dir.cwd().makePath(dir_name) catch {};
    }

    const file = try std.Io.Dir.cwd().createFile(output_path, .{ .truncate = true });
    defer file.close();
    try file.writeAll(snapshot_body);
    std.debug.print("Wrote {s}\n", .{output_path});
}
