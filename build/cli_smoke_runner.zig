const std = @import("std");
const cli_root = @import("cli_root");
const commands_mod = cli_root.commands;

const CommandDescriptor = std.meta.Elem(@TypeOf(commands_mod.descriptors));

const Vector = struct {
    id: []const u8,
    args: []const []const u8,
    expect_success: bool = true,
};

const FixedVector = struct {
    id: []const u8,
    args: []const []const u8,
    expect_success: bool = true,
};

const skipped_help_commands = [_][]const u8{
    "chat-tui",
    "update",
};

const safe_function_vectors = [_]FixedVector{
    .{ .id = "top.help", .args = &.{"help"} },
    .{ .id = "top.version", .args = &.{"version"} },
    .{ .id = "top.flag-help", .args = &.{"--help"} },
    .{ .id = "status.show", .args = &.{"status"} },
    .{ .id = "db.stats", .args = &.{ "db", "stats" } },
    .{ .id = "gpu.summary", .args = &.{ "gpu", "summary" } },
    .{ .id = "gpu.devices", .args = &.{ "gpu", "devices" } },
    .{ .id = "gpu.backends", .args = &.{ "gpu", "backends" } },
    .{ .id = "task.list", .args = &.{ "task", "list" } },
    .{ .id = "task.stats", .args = &.{ "task", "stats" } },
    .{ .id = "config.show", .args = &.{ "config", "show" } },
    .{ .id = "config.path", .args = &.{ "config", "path" } },
    .{ .id = "llm.providers", .args = &.{ "llm", "providers" } },
    .{ .id = "llm.plugins.list", .args = &.{ "llm", "plugins", "list" } },
    .{ .id = "model.list", .args = &.{ "model", "list" } },
    .{ .id = "model.path", .args = &.{ "model", "path" } },
    .{ .id = "profile.show", .args = &.{ "profile", "show" } },
    .{ .id = "bench.list", .args = &.{ "bench", "list" } },
    .{ .id = "bench.micro.hash", .args = &.{ "bench", "micro", "hash" } },
    .{ .id = "bench.micro.noop", .args = &.{ "bench", "micro", "noop" } },
    .{ .id = "train.info", .args = &.{ "train", "info" } },
    .{ .id = "editor.help", .args = &.{ "editor", "--help" } },
    .{ .id = "ui.help", .args = &.{ "ui", "--help" } },
    .{ .id = "ui.list-themes", .args = &.{ "ui", "--list-themes" } },
    .{ .id = "ui.gpu.help", .args = &.{ "ui", "gpu", "--help" } },
    .{ .id = "ui.brain.help", .args = &.{ "ui", "brain", "--help" } },
    .{ .id = "ui.model.help", .args = &.{ "ui", "model", "--help" } },
    .{ .id = "ui.editor.help", .args = &.{ "ui", "editor", "--help" } },
    .{ .id = "ui.dashboard.alias", .args = &.{ "ui", "dashboard" } },
    .{ .id = "ui.launch.removed", .args = &.{ "ui", "launch" }, .expect_success = false },
};

fn addVector(
    allocator: std.mem.Allocator,
    vectors: *std.ArrayListUnmanaged(Vector),
    id: []const u8,
    args: []const []const u8,
    expect_success: bool,
) !void {
    const id_copy = try allocator.dupe(u8, id);
    const args_copy = try allocator.alloc([]const u8, args.len);
    @memcpy(args_copy, args);
    try vectors.append(allocator, .{
        .id = id_copy,
        .args = args_copy,
        .expect_success = expect_success,
    });
}

fn addFixedVectors(allocator: std.mem.Allocator, vectors: *std.ArrayListUnmanaged(Vector)) !void {
    for (safe_function_vectors) |vector| {
        try addVector(allocator, vectors, vector.id, vector.args, vector.expect_success);
    }
}

fn shouldGenerateHelpVector(path: []const []const u8) bool {
    if (path.len == 0) return false;
    if (path.len == 1) {
        for (skipped_help_commands) |name| {
            if (std.mem.eql(u8, path[0], name)) return false;
        }
    }
    return true;
}

fn appendHelpVectors(
    allocator: std.mem.Allocator,
    vectors: *std.ArrayListUnmanaged(Vector),
    descriptors: []const CommandDescriptor,
    path: *std.ArrayListUnmanaged([]const u8),
) !void {
    for (descriptors) |descriptor| {
        if (descriptor.visibility == .hidden) continue;

        try path.append(allocator, descriptor.name);
        defer _ = path.pop();

        if (!shouldGenerateHelpVector(path.items)) {
            if (descriptor.children.len > 0) {
                try appendHelpVectors(allocator, vectors, descriptor.children, path);
            }
            continue;
        }

        const joined = try std.mem.join(allocator, ".", path.items);
        const id = try std.fmt.allocPrint(allocator, "help.{s}", .{joined});

        const args = try allocator.alloc([]const u8, path.items.len + 1);
        args[0] = "help";
        @memcpy(args[1..], path.items);

        try vectors.append(allocator, .{
            .id = id,
            .args = args,
        });

        if (descriptor.children.len > 0) {
            try appendHelpVectors(allocator, vectors, descriptor.children, path);
        }
    }
}

fn collectVectors(allocator: std.mem.Allocator) ![]const Vector {
    var vectors = std.ArrayListUnmanaged(Vector).empty;
    var path = std.ArrayListUnmanaged([]const u8).empty;

    try addFixedVectors(allocator, &vectors);
    try appendHelpVectors(allocator, &vectors, &commands_mod.descriptors, &path);

    return vectors.items;
}

fn runVector(io: std.Io, allocator: std.mem.Allocator, abi_bin_path: []const u8, vector: Vector) !bool {
    const child_args = try allocator.alloc([]const u8, vector.args.len + 1);
    child_args[0] = abi_bin_path;
    @memcpy(child_args[1..], vector.args);

    var child = try std.process.spawn(io, .{
        .argv = child_args,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = .ignore,
    });

    const term = child.wait(io) catch |err| {
        std.debug.print("FAIL [{s}] spawn error: {t}\n", .{ vector.id, err });
        return false;
    };

    const succeeded = switch (term) {
        .exited => |code| code == 0,
        else => false,
    };

    if (succeeded != vector.expect_success) {
        std.debug.print("FAIL [{s}] args=", .{vector.id});
        for (vector.args) |arg| {
            std.debug.print(" {s}", .{arg});
        }
        std.debug.print(" result={any}\n", .{term});
        return false;
    }

    return true;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.environ });
    defer io_backend.deinit();
    const io = io_backend.io();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var args_iter = try std.process.Args.Iterator.initAllocator(init.args, arena.allocator());
    defer args_iter.deinit();

    _ = args_iter.next();

    var abi_bin_path: ?[]const u8 = null;
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--bin")) {
            abi_bin_path = args_iter.next();
        }
    }

    if (abi_bin_path == null) {
        std.debug.print("Usage: cli_smoke_runner --bin <abi-executable>\n", .{});
        std.process.exit(1);
    }

    const vectors = try collectVectors(arena.allocator());

    var passed: usize = 0;
    var failed: usize = 0;

    for (vectors) |vector| {
        if (try runVector(io, arena.allocator(), abi_bin_path.?, vector)) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    std.debug.print("CLI smoke vectors: {d} passed, {d} failed\n", .{ passed, failed });
    if (failed > 0) std.process.exit(1);
}

test {
    std.testing.refAllDecls(@This());
}
