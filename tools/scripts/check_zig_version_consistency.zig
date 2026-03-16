const std = @import("std");
const baseline = @import("baseline.zig");
const toolchain = @import("toolchain_support.zig");
const util = @import("util.zig");

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
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

    var inspection = try toolchain.inspect(allocator, io);
    defer inspection.deinit(allocator);

    switch (inspection.selected_status) {
        .ok => {},
        .abi_host_zig_missing => {
            std.debug.print(
                "ERROR: ABI_HOST_ZIG points to a missing or non-executable binary ({s})\n",
                .{inspection.selected_path orelse "(unset)"},
            );
            errors += 1;
        },
        .abi_host_zig_mismatch => {
            std.debug.print(
                "ERROR: ABI_HOST_ZIG resolved to {s} ({s}), expected pinned baseline ({s})\n",
                .{
                    inspection.selected_path orelse "(unresolved)",
                    inspection.selected_version orelse "(unresolved)",
                    baseline.zig_version,
                },
            );
            errors += 1;
        },
        .zig_real_missing => {
            std.debug.print(
                "ERROR: ZIG_REAL points to a missing or non-executable binary ({s})\n",
                .{inspection.selected_path orelse "(unset)"},
            );
            errors += 1;
        },
        .zig_missing => {
            std.debug.print(
                "ERROR: ZIG points to a missing or non-executable binary ({s})\n",
                .{inspection.selected_path orelse "(unset)"},
            );
            errors += 1;
        },
        .cache_stale => {
            std.debug.print(
                "ERROR: canonical cached host-built Zig is stale ({s} from {s}, expected {s})\n",
                .{
                    inspection.selected_version orelse inspection.cache_version orelse "(unresolved)",
                    inspection.cache_path,
                    baseline.zig_version,
                },
            );
            errors += 1;
        },
        .no_zig_found, .unknown => {
            std.debug.print("ERROR: no usable zig binary was resolved\n", .{});
            errors += 1;
        },
    }

    if (inspection.selected_status == .ok and !inspection.selected_matches_expected) {
        std.debug.print(
            "ERROR: active zig version ({s} from {s}) does not match pinned baseline ({s})\n",
            .{
                inspection.selected_version orelse "(unresolved)",
                inspection.selected_path orelse "(unresolved)",
                baseline.zig_version,
            },
        );
        errors += 1;
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

        const matches = try util.captureCommand(allocator, io, rg_cmd);
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
        std.debug.print("Hint: run 'zig build toolchain-doctor' or 'abi doctor' for a full local diagnosis.\n", .{});
        std.process.exit(1);
    }

    std.debug.print("OK: Zig version consistency checks passed\n", .{});
}
