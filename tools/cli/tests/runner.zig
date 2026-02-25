const std = @import("std");

const EntryKind = enum {
    oneshot,
    serve_probe,
    pty_session,
    long_running_probe,
};

const Entry = struct {
    id: []const u8,
    args: [][]const u8,
    kind: []const u8,
    timeout_ms: u32,
    exit_policy: []const u8,
};

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{ .environ = init.environ });
    defer io_backend.deinit();
    const io = io_backend.io();
    const cwd = std.Io.Dir.cwd();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var args_iter = try std.process.Args.Iterator.initAllocator(init.args, arena.allocator());
    defer args_iter.deinit();

    _ = args_iter.next(); // Skip executable path

    var matrix_json_path: ?[]const u8 = null;
    var abi_bin_path: ?[]const u8 = null;

    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--matrix")) {
            matrix_json_path = args_iter.next();
        } else if (std.mem.eql(u8, arg, "--bin")) {
            abi_bin_path = args_iter.next();
        }
    }

    if (matrix_json_path == null or abi_bin_path == null) {
        std.debug.print("Usage: runner --matrix <json> --bin <abi-exe>\n", .{});
        std.process.exit(1);
    }

    const json_data = try cwd.readFileAlloc(io, matrix_json_path.?, allocator, std.Io.Limit.limited(10 * 1024 * 1024));
    defer allocator.free(json_data);

    var parsed = try std.json.parseFromSlice([]Entry, allocator, json_data, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    const entries = parsed.value;
    std.debug.print("Native E2E Runner: Loaded {d} entries from {s}\n", .{ entries.len, matrix_json_path.? });

    var passed: usize = 0;
    var failed: usize = 0;

    for (entries) |entry| {
        var child_args: std.ArrayListUnmanaged([]const u8) = .empty;
        defer child_args.deinit(allocator);

        try child_args.append(allocator, abi_bin_path.?);
        for (entry.args) |a| {
            try child_args.append(allocator, a);
        }

        var child = try std.process.spawn(io, .{
            .argv = child_args.items,
            .stdin = .ignore,
            .stdout = .pipe,
            .stderr = .pipe,
        });

        const term = child.wait(io) catch |err| {
            std.debug.print("FAIL [{s}]: spawn error {t}\n", .{ entry.id, err });
            failed += 1;
            continue;
        };

        const is_ok = switch (term) {
            .exited => |code| code == 0,
            else => false,
        };

        if (is_ok) {
            passed += 1;
        } else {
            std.debug.print("FAIL [{s}]: {any}\n", .{ entry.id, term });
            failed += 1;
        }
    }

    std.debug.print("\n=== Native E2E Matrix Results ===\n", .{});
    std.debug.print("Passed: {d}\n", .{passed});
    std.debug.print("Failed: {d}\n", .{failed});

    if (failed > 0) {
        std.process.exit(1);
    }
}

test {
    std.testing.refAllDecls(@This());
}
