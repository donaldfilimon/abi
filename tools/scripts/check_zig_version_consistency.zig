const std = @import("std");
const baseline = @import("baseline.zig");
const util = @import("util.zig");

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var errors: usize = 0;

    const zigversion_raw = util.readFileAlloc(allocator, io, ".zigversion", 1024) catch {
        std.debug.print("ERROR: .zigversion not found\n", .{});
        std.process.exit(1);
    };
    defer allocator.free(zigversion_raw);
    const zigversion = util.trimSpace(zigversion_raw);

    if (!std.mem.eql(u8, zigversion, baseline.zig_version)) {
        std.debug.print(
            "ERROR: .zigversion ({s}) does not match baseline ({s})\n",
            .{ zigversion, baseline.zig_version },
        );
        std.process.exit(1);
    }

    if (!(try util.commandExists(allocator, "zig"))) {
        std.debug.print("ERROR: no 'zig' binary found on PATH\n", .{});
        std.process.exit(1);
    }

    const active_path_res = try util.captureCommand(allocator, "command -v zig");
    defer allocator.free(active_path_res.output);
    const active_path = util.trimSpace(active_path_res.output);

    const active_ver_res = try util.captureCommand(allocator, "zig version");
    defer allocator.free(active_ver_res.output);
    const active_version = util.trimSpace(active_ver_res.output);

    if (!std.mem.eql(u8, active_version, baseline.zig_version)) {
        std.debug.print(
            "ERROR: active zig version ({s} from {s}) does not match pinned baseline ({s})\n",
            .{ active_version, active_path, baseline.zig_version },
        );
        errors += 1;
    }

    if (try util.commandExists(allocator, "zvm")) {
        const home_res = try util.captureCommand(allocator, "printf '%s' \"$HOME\"");
        defer allocator.free(home_res.output);
        const home = util.trimSpace(home_res.output);

        const zvm_zig = try std.fmt.allocPrint(allocator, "{s}/.zvm/bin/zig", .{home});
        defer allocator.free(zvm_zig);

        if (util.fileExists(io, zvm_zig) and !std.mem.eql(u8, active_path, zvm_zig)) {
            std.debug.print(
                "ERROR: PATH precedence mismatch: active zig is '{s}' but zvm-managed zig is '{s}'\n",
                .{ active_path, zvm_zig },
            );
            std.debug.print(
                "       Fix by prepending '$HOME/.zvm/bin' ahead of other zig locations in PATH.\n",
                .{},
            );
            errors += 1;
        }
    }

    const files = [_][]const u8{
        "README.md",
    };

    for (files) |file_path| {
        if (!util.fileExists(io, file_path)) {
            std.debug.print("ERROR: expected file missing: {s}\n", .{file_path});
            errors += 1;
            continue;
        }

        const content = util.readFileAlloc(allocator, io, file_path, 8 * 1024 * 1024) catch |err| {
            std.debug.print("ERROR: failed to read {s}: {t}\n", .{ file_path, err });
            errors += 1;
            continue;
        };
        defer allocator.free(content);

        if (std.mem.indexOf(u8, content, baseline.zig_version) == null) {
            std.debug.print(
                "ERROR: {s} does not mention expected Zig version: {s}\n",
                .{ file_path, baseline.zig_version },
            );
            errors += 1;
        }

        const rg_cmd = try std.fmt.allocPrint(
            allocator,
            "rg -n --with-filename -o '0\\.16\\.0-dev\\.[0-9]+\\+[A-Za-z0-9]+' {s}",
            .{file_path},
        );
        defer allocator.free(rg_cmd);

        const matches = try util.captureCommand(allocator, rg_cmd);
        defer allocator.free(matches.output);

        if (matches.exit_code != 0 and matches.exit_code != 1) {
            std.debug.print("ERROR: failed to scan Zig version strings in {s}\n", .{file_path});
            if (matches.output.len > 0) std.debug.print("{s}", .{matches.output});
            errors += 1;
            continue;
        }

        if (matches.exit_code == 1) continue;

        var lines = std.mem.splitScalar(u8, matches.output, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            const first_colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            const second_rel = std.mem.indexOfScalar(u8, line[first_colon + 1 ..], ':') orelse continue;
            const second_colon = first_colon + 1 + second_rel;

            const lineno = line[first_colon + 1 .. second_colon];
            const match = line[second_colon + 1 ..];
            if (!std.mem.eql(u8, match, baseline.zig_version)) {
                std.debug.print(
                    "ERROR: {s}:{s} has mismatched Zig version '{s}' (expected '{s}')\n",
                    .{ file_path, lineno, match, baseline.zig_version },
                );
                errors += 1;
            }
        }
    }

    if (errors > 0) {
        std.debug.print("FAILED: Zig version consistency check found {d} issue(s)\n", .{errors});
        std.debug.print("Hint: run 'zig run tools/scripts/toolchain_doctor.zig' for a full local diagnosis.\n", .{});
        std.process.exit(1);
    }

    std.debug.print("OK: Zig version consistency checks passed\n", .{});
}
