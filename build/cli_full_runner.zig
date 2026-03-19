const std = @import("std");
const matrix = @import("matrix_manifest");

fn runVector(io: std.Io, allocator: std.mem.Allocator, abi_bin_path: []const u8, vector: matrix.IntegrationVector) !bool {
    const child_args = try allocator.alloc([]const u8, vector.args.len + 1);
    child_args[0] = abi_bin_path;
    @memcpy(child_args[1..], vector.args);

    const deadline_ns = vector.timeout.toNs();

    // Record start time via POSIX clock_gettime (Zig 0.16 compatible)
    var start_ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &start_ts);

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

    // Check if we exceeded the timeout deadline (advisory reporting).
    var end_ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &end_ts);
    const start_ns: u64 = @intCast(start_ts.sec * std.time.ns_per_s + start_ts.nsec);
    const end_ns: u64 = @intCast(end_ts.sec * std.time.ns_per_s + end_ts.nsec);
    const elapsed_ns = end_ns - start_ns;
    if (elapsed_ns > deadline_ns) {
        std.debug.print("TIMEOUT [{s}] exceeded {d}s deadline (took {d}ms)\n", .{
            vector.id,
            deadline_ns / std.time.ns_per_s,
            elapsed_ns / std.time.ns_per_ms,
        });
        return false;
    }

    const expect_success = !vector.expectation.expect_failure;

    const succeeded = switch (term) {
        .exited => |code| code == 0,
        else => false,
    };

    if (succeeded != expect_success) {
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
    var gpa = std.heap.DebugAllocator(.{}){};
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
        std.debug.print("Usage: cli_full_runner --bin <abi-executable>\n", .{});
        std.process.exit(1);
    }

    // Use safeVectors() which excludes TUI vectors requiring a PTY.
    const vectors = matrix.safeVectors();

    var passed: usize = 0;
    var failed: usize = 0;

    std.debug.print("CLI full integration: {d} vectors from matrix manifest (TUI excluded)\n", .{vectors.len});

    for (vectors) |vector| {
        // In the full runner all features are enabled (we build with all
        // defaults), so we run every safe vector without feature filtering.
        if (try runVector(io, arena.allocator(), abi_bin_path.?, vector)) {
            passed += 1;
        } else {
            failed += 1;
        }
    }

    std.debug.print("CLI full integration: {d} passed, {d} failed, {d} total\n", .{ passed, failed, vectors.len });
    if (failed > 0) std.process.exit(1);
}

test {
    std.testing.refAllDecls(@This());
}
