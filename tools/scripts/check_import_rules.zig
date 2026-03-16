const std = @import("std");
const util = @import("util");

const allowed_bare_imports = [_][]const u8{
    "std",
    "builtin",
    "root",
    "abi",
    "build_options",
};

const ParsedLine = struct {
    file: []const u8,
    lineno: []const u8,
    content: []const u8,
};

fn captureScan(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) ![]u8 {
    const result = try util.captureCommand(allocator, io, cmd);
    defer allocator.free(result.output);

    if (result.exit_code != 0 and result.exit_code != 1) {
        std.debug.print("ERROR: import rule scan failed (exit={d})\n", .{result.exit_code});
        if (result.output.len > 0) std.debug.print("{s}", .{result.output});
        std.process.exit(1);
    }

    return allocator.dupe(u8, result.output);
}

fn parseLine(line: []const u8) ?ParsedLine {
    const first_colon = std.mem.indexOfScalar(u8, line, ':') orelse return null;
    const second_rel = std.mem.indexOfScalar(u8, line[first_colon + 1 ..], ':') orelse return null;
    const second_colon = first_colon + 1 + second_rel;

    return .{
        .file = line[0..first_colon],
        .lineno = line[first_colon + 1 .. second_colon],
        .content = std.mem.trim(u8, line[second_colon + 1 ..], " \t"),
    };
}

fn isCommentLine(trimmed: []const u8) bool {
    return std.mem.startsWith(u8, trimmed, "//") or
        std.mem.startsWith(u8, trimmed, "/*") or
        std.mem.startsWith(u8, trimmed, "*");
}

fn parseImportTarget(content: []const u8) ?[]const u8 {
    const marker = "@import(\"";
    const start = std.mem.indexOf(u8, content, marker) orelse return null;
    const after = content[start + marker.len ..];
    const end = std.mem.indexOfScalar(u8, after, '"') orelse return null;
    return after[0..end];
}

fn isAllowedBareImport(target: []const u8) bool {
    inline for (allowed_bare_imports) |allowed| {
        if (std.mem.eql(u8, target, allowed)) return true;
    }
    return false;
}

fn bareNamedModuleBase(target: []const u8) ?[]const u8 {
    if (std.mem.indexOfScalar(u8, target, '/')) |_| return null;
    if (!std.mem.endsWith(u8, target, ".zig")) return null;
    return target[0 .. target.len - ".zig".len];
}

fn scanFeatureAbiViolations(allocator: std.mem.Allocator, io: std.Io) !usize {
    const output = try captureScan(allocator, io, "rg -n --glob '*.zig' '@import\\(\"abi\"\\)' src/features");
    defer allocator.free(output);

    var violations: usize = 0;
    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        const parsed = parseLine(line) orelse continue;
        if (isCommentLine(parsed.content)) continue;

        std.debug.print("VIOLATION: {s}:{s}: {s}\n", .{ parsed.file, parsed.lineno, parsed.content });
        violations += 1;
    }

    if (violations > 0) {
        std.debug.print("\nERROR: Found {d} @import(\"abi\") violation(s) in feature modules.\n", .{violations});
        std.debug.print("Feature modules must use relative imports to avoid circular dependencies.\n", .{});
    }

    return violations;
}

fn scanNamedVsFileImports(allocator: std.mem.Allocator, io: std.Io) !usize {
    const output = try captureScan(allocator, io, "rg -n --glob '*.zig' '@import\\(\"[^\"]+\"\\)' src");
    defer allocator.free(output);

    var violations: usize = 0;
    var lines = std.mem.splitScalar(u8, output, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;

        const parsed = parseLine(line) orelse continue;
        if (isCommentLine(parsed.content)) continue;

        const target = parseImportTarget(parsed.content) orelse continue;

        if (isAllowedBareImport(target)) continue;

        if (bareNamedModuleBase(target)) |base| {
            if (isAllowedBareImport(base)) {
                std.debug.print(
                    "VIOLATION: {s}:{s}: named module import must stay bare: @import(\"{s}\") not @import(\"{s}\")\n",
                    .{ parsed.file, parsed.lineno, base, target },
                );
                violations += 1;
                continue;
            }
        }

        if (std.mem.endsWith(u8, target, ".zig")) continue;

        std.debug.print(
            "VIOLATION: {s}:{s}: non-allowlisted bare import \"{s}\"; use an explicit relative .zig path or a build-wired named module\n",
            .{ parsed.file, parsed.lineno, target },
        );
        violations += 1;
    }

    if (violations > 0) {
        std.debug.print("\nERROR: Found {d} named-vs-file import violation(s) under src/.\n", .{violations});
        std.debug.print("Only build-wired module names may stay bare; local file imports must use explicit .zig paths.\n", .{});
    }

    return violations;
}

pub fn main(_: std.process.Init) !void {
    var gpa_state = std.heap.DebugAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var violations: usize = 0;
    violations += try scanFeatureAbiViolations(allocator, io);
    violations += try scanNamedVsFileImports(allocator, io);

    if (violations > 0) {
        std.process.exit(1);
    }

    std.debug.print("OK: Import rules satisfied.\n", .{});
}
