const std = @import("std");
const util = @import("util.zig");

fn addScore(value: std.json.Value, sum: *f64, count: *usize) void {
    switch (value) {
        .float => |v| {
            sum.* += v;
            count.* += 1;
        },
        .integer => |v| {
            sum.* += @as(f64, @floatFromInt(v));
            count.* += 1;
        },
        .string => |s| {
            const parsed = std.fmt.parseFloat(f64, s) catch return;
            sum.* += parsed;
            count.* += 1;
        },
        else => {},
    }
}

fn collectScores(value: std.json.Value, sum: *f64, count: *usize) void {
    switch (value) {
        .array => |arr| {
            for (arr.items) |item| {
                switch (item) {
                    .object => |obj| {
                        if (obj.get("score")) |score_value| {
                            addScore(score_value, sum, count);
                        }
                    },
                    else => addScore(item, sum, count),
                }
            }
        },
        .object => |obj| {
            if (obj.get("results")) |results| {
                collectScores(results, sum, count);
            }
        },
        else => {},
    }
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var input_json: []const u8 = "reports/ralph_upgrade_results_openai.json";
    var latest_path: ?[]u8 = null;
    defer if (latest_path) |p| allocator.free(p);

    if (util.fileExists(io, ".ralph/runs/latest.json")) {
        const latest_contents = util.readFileAlloc(allocator, io, ".ralph/runs/latest.json", 64 * 1024) catch null;
        if (latest_contents) |contents| {
            defer allocator.free(contents);
            const parsed_latest = std.json.parseFromSlice(std.json.Value, allocator, contents, .{}) catch null;
            if (parsed_latest) |parsed| {
                defer parsed.deinit();
                switch (parsed.value) {
                    .object => |obj| {
                        if (obj.get("report")) |report| {
                            switch (report) {
                                .string => |s| latest_path = allocator.dupe(u8, s) catch null,
                                else => {},
                            }
                        }
                    },
                    else => {},
                }
            }
        }
    }
    if (latest_path) |p| input_json = p;

    if (!util.fileExists(io, input_json)) {
        std.debug.print("ERROR: missing Ralph results: {s}\n", .{input_json});
        std.debug.print("Run:\n", .{});
        std.debug.print("  abi ralph improve --iterations 1\n", .{});
        std.process.exit(1);
    }

    const contents = util.readFileAlloc(allocator, io, input_json, 16 * 1024 * 1024) catch |err| {
        std.debug.print("ERROR: could not read {s}: {t}\n", .{ input_json, err });
        std.process.exit(1);
    };
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, contents, .{}) catch {
        std.debug.print("ERROR: invalid JSON in {s}\n", .{input_json});
        std.process.exit(1);
    };
    defer parsed.deinit();

    // New report format shortcut
    switch (parsed.value) {
        .object => |obj| {
            if (obj.get("last_gate_passed")) |v| {
                const passed = switch (v) {
                    .bool => |b| b,
                    else => false,
                };
                if (passed) {
                    std.debug.print("OK: Ralph gate passed (run report)\n", .{});
                    return;
                }
                std.debug.print("FAILED: Ralph run report indicates gate failure\n", .{});
                std.process.exit(1);
            }
        },
        else => {},
    }

    var sum: f64 = 0.0;
    var count: usize = 0;
    collectScores(parsed.value, &sum, &count);

    if (count == 0) {
        std.debug.print("FAILED: Ralph report present but contains no scored results\n", .{});
        std.process.exit(1);
    }

    const avg = sum / @as(f64, @floatFromInt(count));
    if (avg >= 0.75) {
        std.debug.print("OK: Ralph gate passed (average score: {d:.3})\n", .{avg});
        return;
    }

    std.debug.print("FAILED: Ralph gate score {d:.3} < 0.75 threshold\n", .{avg});
    std.process.exit(1);
}
