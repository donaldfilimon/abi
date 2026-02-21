const std = @import("std");
const util = @import("util.zig");

const docs_dir = "docs/api";
const skill_block =
    \\## Zig Skill
    \\Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
    \\
;

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var dir = std.Io.Dir.cwd().openDir(io, docs_dir, .{ .iterate = true }) catch |err| {
        std.debug.print("ERROR: failed to open {s}: {t}\n", .{ docs_dir, err });
        std.process.exit(1);
    };
    defer dir.close(io);

    var updated_count: usize = 0;
    var scanned_count: usize = 0;

    var iter = dir.iterate();
    while (true) {
        const maybe_entry = iter.next(io) catch |err| {
            std.debug.print("ERROR: failed to iterate {s}: {t}\n", .{ docs_dir, err });
            std.process.exit(1);
        };
        const entry = maybe_entry orelse break;
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".md")) continue;

        scanned_count += 1;

        const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ docs_dir, entry.name });
        defer allocator.free(path);

        const content = util.readFileAlloc(allocator, io, path, 16 * 1024 * 1024) catch |err| {
            std.debug.print("ERROR: failed to read {s}: {t}\n", .{ path, err });
            std.process.exit(1);
        };
        defer allocator.free(content);

        const normalized = try normalizeFooter(allocator, content);
        defer allocator.free(normalized);

        if (std.mem.eql(u8, content, normalized)) continue;

        var out = std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true }) catch |err| {
            std.debug.print("ERROR: failed to open {s} for write: {t}\n", .{ path, err });
            std.process.exit(1);
        };
        defer out.close(io);
        out.writeStreamingAll(io, normalized) catch |err| {
            std.debug.print("ERROR: failed to write {s}: {t}\n", .{ path, err });
            std.process.exit(1);
        };

        updated_count += 1;
    }

    std.debug.print(
        "OK: postprocessed {d} API docs ({d} updated) in {s}\n",
        .{ scanned_count, updated_count, docs_dir },
    );
}

fn normalizeFooter(allocator: std.mem.Allocator, content: []const u8) ![]u8 {
    const without_existing = removeExistingSkillBlock(content);
    const trimmed = trimTrailingWhitespace(without_existing);
    return std.fmt.allocPrint(allocator, "{s}\n\n{s}", .{ trimmed, skill_block });
}

fn trimTrailingWhitespace(content: []const u8) []const u8 {
    var end = content.len;
    while (end > 0) {
        const c = content[end - 1];
        if (c == ' ' or c == '\t' or c == '\r' or c == '\n') {
            end -= 1;
            continue;
        }
        break;
    }
    return content[0..end];
}

fn removeExistingSkillBlock(content: []const u8) []const u8 {
    if (std.mem.indexOf(u8, content, "\n## Zig Skill\n")) |idx| {
        return content[0..idx];
    }
    if (std.mem.startsWith(u8, content, "## Zig Skill\n")) {
        return "";
    }
    return content;
}
