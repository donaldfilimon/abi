const std = @import("std");
const model = @import("model.zig");

pub const CheckError = error{DriftDetected};

pub fn verifyOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    expected: []const model.OutputFile,
) !void {
    var expected_map = std.StringHashMapUnmanaged(void).empty;
    defer expected_map.deinit(allocator);

    var missing = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (missing.items) |path| allocator.free(path);
        missing.deinit(allocator);
    }
    var changed = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (changed.items) |path| allocator.free(path);
        changed.deinit(allocator);
    }
    var extra = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (extra.items) |path| allocator.free(path);
        extra.deinit(allocator);
    }

    for (expected) |file| {
        try validateOutputPolicy(file.path, file.content);
        try expected_map.put(allocator, file.path, {});

        const current = cwd.readFileAlloc(io, file.path, allocator, .limited(16 * 1024 * 1024)) catch |err| {
            if (err == error.FileNotFound) {
                try missing.append(allocator, try allocator.dupe(u8, file.path));
                continue;
            }
            return err;
        };
        defer allocator.free(current);

        if (!std.mem.eql(u8, current, file.content)) {
            try changed.append(allocator, try allocator.dupe(u8, file.path));
        }
    }

    try collectExtraFiles(allocator, io, cwd, "docs/api", &expected_map, true, &extra);
    try collectExtraFiles(allocator, io, cwd, "docs/_docs", &expected_map, true, &extra);
    try collectExtraFiles(allocator, io, cwd, "docs/plans", &expected_map, true, &extra);
    try collectExtraFiles(allocator, io, cwd, "docs/api-app", &expected_map, false, &extra);

    if (missing.items.len == 0 and changed.items.len == 0 and extra.items.len == 0) {
        std.debug.print("OK: docs are up to date\n", .{});
        return;
    }

    std.debug.print("ERROR: docs drift detected\n", .{});
    if (missing.items.len > 0) {
        std.debug.print("  Missing files:\n", .{});
        for (missing.items) |path| std.debug.print("    - {s}\n", .{path});
    }
    if (changed.items.len > 0) {
        std.debug.print("  Changed files:\n", .{});
        for (changed.items) |path| std.debug.print("    - {s}\n", .{path});
    }
    if (extra.items.len > 0) {
        std.debug.print("  Extra files:\n", .{});
        for (extra.items) |path| std.debug.print("    - {s}\n", .{path});
    }

    return CheckError.DriftDetected;
}

fn validateOutputPolicy(path: []const u8, content: []const u8) !void {
    if (!std.mem.endsWith(u8, path, ".md")) return;

    if (std.mem.indexOf(u8, content, "](/") != null) {
        std.debug.print("ERROR: forbidden root-absolute markdown link in {s}\n", .{path});
        return CheckError.DriftDetected;
    }
    if (std.mem.indexOf(u8, content, "/Users/") != null) {
        std.debug.print("ERROR: forbidden local filesystem path in {s}\n", .{path});
        return CheckError.DriftDetected;
    }
}

fn collectExtraFiles(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    root_path: []const u8,
    expected_map: *const std.StringHashMapUnmanaged(void),
    markdown_only: bool,
    extras: *std.ArrayListUnmanaged([]const u8),
) !void {
    var root = cwd.openDir(io, root_path, .{ .iterate = true }) catch |err| {
        if (err == error.FileNotFound) return;
        return err;
    };
    defer root.close(io);

    try walkExtra(allocator, io, cwd, root, root_path, expected_map, markdown_only, extras);
}

fn walkExtra(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    dir: std.Io.Dir,
    prefix: []const u8,
    expected_map: *const std.StringHashMapUnmanaged(void),
    markdown_only: bool,
    extras: *std.ArrayListUnmanaged([]const u8),
) !void {
    var it = dir.iterate();
    while (true) {
        const maybe_entry = try it.next(io);
        const entry = maybe_entry orelse break;

        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ prefix, entry.name });
        defer allocator.free(full_path);

        switch (entry.kind) {
            .file => {
                if (markdown_only and !std.mem.endsWith(u8, entry.name, ".md")) continue;
                if (!shouldTrackExtraFile(full_path)) continue;
                if (!expected_map.contains(full_path)) {
                    try extras.append(allocator, try allocator.dupe(u8, full_path));
                }
            },
            .directory => {
                if (shouldSkipDir(full_path)) continue;
                var child = try cwd.openDir(io, full_path, .{ .iterate = true });
                defer child.close(io);
                try walkExtra(allocator, io, cwd, child, full_path, expected_map, markdown_only, extras);
            },
            else => {},
        }
    }
}

fn shouldTrackExtraFile(path: []const u8) bool {
    if (std.mem.endsWith(u8, path, "/docs_engine.wasm")) return false;
    if (std.mem.endsWith(u8, path, "/docs_engine.component.wasm")) return false;
    if (std.mem.endsWith(u8, path, "/docs_engine.wit")) return false;
    if (std.mem.startsWith(u8, path, "docs/plans/archive/")) return false;
    return true;
}

fn shouldSkipDir(path: []const u8) bool {
    return std.mem.eql(u8, path, "docs/plans/archive") or
        std.mem.startsWith(u8, path, "docs/plans/archive/");
}

pub fn writeOutputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: std.Io.Dir,
    expected: []const model.OutputFile,
) !void {
    _ = allocator;
    for (expected) |file| {
        if (std.fs.path.dirname(file.path)) |dir_path| {
            cwd.createDirPath(io, dir_path) catch |err| {
                if (err != error.PathAlreadyExists) return err;
            };
        }
        var out_file = try cwd.createFile(io, file.path, .{ .truncate = true });
        defer out_file.close(io);
        try out_file.writeStreamingAll(io, file.content);
    }
}

test "shouldTrackExtraFile ignores plans archive and wasm artifacts" {
    try std.testing.expect(!shouldTrackExtraFile("docs/plans/archive/2026-02-21-plan.md"));
    try std.testing.expect(!shouldTrackExtraFile("docs/api-app/data/docs_engine.wasm"));
    try std.testing.expect(shouldTrackExtraFile("docs/plans/index.md"));
}

test "validateOutputPolicy rejects root-absolute and local links" {
    try std.testing.expectError(
        CheckError.DriftDetected,
        validateOutputPolicy("docs/_docs/roadmap.md", "](/api/)"),
    );
    try std.testing.expectError(
        CheckError.DriftDetected,
        validateOutputPolicy("docs/_docs/roadmap.md", "/Users/donaldfilimon/abi"),
    );
    try validateOutputPolicy("docs/_docs/roadmap.md", "[OK](../plans/index.md)");
}
