const std = @import("std");
const util = @import("util.zig");

const SectionRequirement = struct {
    label: []const u8,
    needle: []const u8,
};

const todo_sections = [_]SectionRequirement{
    .{ .label = "Objective", .needle = "## Objective" },
    .{ .label = "Scope", .needle = "## Scope" },
    .{ .label = "Verification Criteria", .needle = "## Verification Criteria" },
    .{ .label = "Checklist", .needle = "## Checklist" },
    .{ .label = "Review", .needle = "## Review" },
};

const replan_tokens = [_][]const u8{
    "Trigger:",
    "Impact:",
    "Plan change:",
    "Verification change:",
};

const Config = struct {
    strict: bool = false,
    todo_path: []const u8 = "tasks/todo.md",
    todo_owned: bool = false,
    lessons_path: []const u8 = "tasks/lessons.md",
    lessons_owned: bool = false,
    prompt_path: []const u8 = "PROMPT.md",
    prompt_owned: bool = false,
    emit_json: bool = false,

    fn deinit(self: *Config, allocator: std.mem.Allocator) void {
        if (self.todo_owned) allocator.free(self.todo_path);
        if (self.lessons_owned) allocator.free(self.lessons_path);
        if (self.prompt_owned) allocator.free(self.prompt_path);
    }
};

const Report = struct {
    strict: bool,
    todo_path: []const u8,
    lessons_path: []const u8,
    prompt_path: []const u8,
    warning_count: usize = 0,
    error_count: usize = 0,
    messages: std.ArrayListUnmanaged([]u8) = .empty,

    fn deinit(self: *Report, allocator: std.mem.Allocator) void {
        for (self.messages.items) |msg| allocator.free(msg);
        self.messages.deinit(allocator);
    }

    fn addViolation(self: *Report, allocator: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
        const msg = try std.fmt.allocPrint(allocator, fmt, args);
        try self.messages.append(allocator, msg);
        if (self.strict) {
            self.error_count += 1;
        } else {
            self.warning_count += 1;
        }
    }

    fn passed(self: Report) bool {
        if (self.strict) return self.error_count == 0;
        return true;
    }
};

fn parseArgs(init: std.process.Init, allocator: std.mem.Allocator) !Config {
    var cfg = Config{};
    errdefer cfg.deinit(allocator);

    var args_iter = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args_iter.deinit();

    var args = std.ArrayListUnmanaged([]const u8).empty;
    defer args.deinit(allocator);

    _ = args_iter.next(); // executable path
    while (args_iter.next()) |arg| {
        try args.append(allocator, arg);
    }

    var i: usize = 0;
    while (i < args.items.len) : (i += 1) {
        const arg = args.items[i];
        if (std.mem.eql(u8, arg, "--strict")) {
            cfg.strict = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            cfg.emit_json = true;
        } else if (std.mem.eql(u8, arg, "--todo")) {
            i += 1;
            if (i >= args.items.len) return error.MissingArgValue;
            const path = try allocator.dupe(u8, args.items[i]);
            if (cfg.todo_owned) allocator.free(cfg.todo_path);
            cfg.todo_path = path;
            cfg.todo_owned = true;
        } else if (std.mem.eql(u8, arg, "--lessons")) {
            i += 1;
            if (i >= args.items.len) return error.MissingArgValue;
            const path = try allocator.dupe(u8, args.items[i]);
            if (cfg.lessons_owned) allocator.free(cfg.lessons_path);
            cfg.lessons_path = path;
            cfg.lessons_owned = true;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.items.len) return error.MissingArgValue;
            const path = try allocator.dupe(u8, args.items[i]);
            if (cfg.prompt_owned) allocator.free(cfg.prompt_path);
            cfg.prompt_path = path;
            cfg.prompt_owned = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printHelp();
            std.process.exit(0);
        } else {
            return error.InvalidArg;
        }
    }

    return cfg;
}

fn printHelp() void {
    std.debug.print(
        \\Usage: zig run tools/scripts/check_workflow_orchestration.zig -- [options]
        \\
        \\Workflow contract checker for tasks/todo.md and tasks/lessons.md.
        \\
        \\Options:
        \\  --strict            Fail on contract violations
        \\  --todo <path>       Todo file (default: tasks/todo.md)
        \\  --lessons <path>    Lessons file (default: tasks/lessons.md)
        \\  --prompt <path>     Optional prompt file to validate (default: PROMPT.md)
        \\  --json              Emit machine-readable JSON summary
        \\  -h, --help          Show this help
        \\
    , .{});
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

fn looksLikeDatePrefix(text: []const u8) bool {
    if (text.len < 10) return false;
    for (0..10) |idx| {
        const ch = text[idx];
        if (idx == 4 or idx == 7) {
            if (ch != '-') return false;
        } else if (!std.ascii.isDigit(ch)) {
            return false;
        }
    }
    return true;
}

fn checkConflictMarkers(
    allocator: std.mem.Allocator,
    report: *Report,
    path: []const u8,
    content: []const u8,
) !void {
    var lines = std.mem.splitScalar(u8, content, '\n');
    var line_num: usize = 1;
    while (lines.next()) |line| : (line_num += 1) {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "<<<<<<< ") or
            std.mem.startsWith(u8, trimmed, ">>>>>>> ") or
            std.mem.eql(u8, trimmed, "======="))
        {
            try report.addViolation(allocator, "file {s} contains conflict marker at line {d}: {s}", .{ path, line_num, trimmed });
        }
    }
}

fn checkTodo(
    allocator: std.mem.Allocator,
    io: std.Io,
    report: *Report,
) !void {
    if (!util.fileExists(io, report.todo_path)) {
        try report.addViolation(allocator, "missing todo file: {s}", .{report.todo_path});
        return;
    }

    const content = util.readFileAlloc(allocator, io, report.todo_path, 4 * 1024 * 1024) catch |err| {
        try report.addViolation(allocator, "failed reading todo file {s}: {t}", .{ report.todo_path, err });
        return;
    };
    defer allocator.free(content);

    try checkConflictMarkers(allocator, report, report.todo_path, content);

    for (todo_sections) |section| {
        if (std.mem.indexOf(u8, content, section.needle) == null) {
            try report.addViolation(allocator, "todo contract missing section '{s}' in {s}", .{ section.label, report.todo_path });
        }
    }

    const has_trigger_note = containsIgnoreCase(content, "Trigger:");
    if (has_trigger_note) {
        for (replan_tokens) |token| {
            if (!containsIgnoreCase(content, token)) {
                try report.addViolation(allocator, "todo re-plan note missing token '{s}' in {s}", .{ token, report.todo_path });
            }
        }
    }
}

fn checkLessons(
    allocator: std.mem.Allocator,
    io: std.Io,
    report: *Report,
) !void {
    if (!util.fileExists(io, report.lessons_path)) {
        try report.addViolation(allocator, "missing lessons file: {s}", .{report.lessons_path});
        return;
    }

    const content = util.readFileAlloc(allocator, io, report.lessons_path, 4 * 1024 * 1024) catch |err| {
        try report.addViolation(allocator, "failed reading lessons file {s}: {t}", .{ report.lessons_path, err });
        return;
    };
    defer allocator.free(content);

    try checkConflictMarkers(allocator, report, report.lessons_path, content);

    var entry_count: usize = 0;
    var in_entry = false;
    var entry_heading: []const u8 = "";
    var entry_has_root = false;
    var entry_has_prevention = false;

    var lines = std.mem.splitScalar(u8, content, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (std.mem.startsWith(u8, trimmed, "## ")) {
            if (in_entry) {
                if (!entry_has_root) {
                    try report.addViolation(allocator, "lessons entry missing Root cause: {s}", .{entry_heading});
                }
                if (!entry_has_prevention) {
                    try report.addViolation(allocator, "lessons entry missing Prevention rule: {s}", .{entry_heading});
                }
            }

            in_entry = true;
            entry_count += 1;
            entry_heading = trimmed[3..];
            entry_has_root = false;
            entry_has_prevention = false;

            if (!looksLikeDatePrefix(entry_heading)) {
                try report.addViolation(allocator, "lessons heading should start with YYYY-MM-DD: {s}", .{entry_heading});
            }

            continue;
        }

        if (!in_entry) continue;
        if (containsIgnoreCase(trimmed, "Root cause:")) entry_has_root = true;
        if (containsIgnoreCase(trimmed, "Prevention rule:")) entry_has_prevention = true;
    }

    if (in_entry) {
        if (!entry_has_root) {
            try report.addViolation(allocator, "lessons entry missing Root cause: {s}", .{entry_heading});
        }
        if (!entry_has_prevention) {
            try report.addViolation(allocator, "lessons entry missing Prevention rule: {s}", .{entry_heading});
        }
    }

    if (entry_count == 0) {
        try report.addViolation(allocator, "lessons file has no dated entries: {s}", .{report.lessons_path});
    }
}

fn checkPrompt(
    allocator: std.mem.Allocator,
    io: std.Io,
    report: *Report,
) !void {
    if (!util.fileExists(io, report.prompt_path)) {
        // Prompt validation is optional for the markdown-reset baseline.
        // If the file exists, we still enforce section-quality checks below.
        return;
    }

    const content = util.readFileAlloc(allocator, io, report.prompt_path, 2 * 1024 * 1024) catch |err| {
        try report.addViolation(allocator, "failed reading prompt file {s}: {t}", .{ report.prompt_path, err });
        return;
    };
    defer allocator.free(content);

    if (!containsIgnoreCase(content, "## Goal") and !containsIgnoreCase(content, "## Objective")) {
        try report.addViolation(allocator, "prompt file should contain Goal or Objective section: {s}", .{report.prompt_path});
    }

    if (!containsIgnoreCase(content, "## Acceptance Criteria") and !containsIgnoreCase(content, "## Verification Criteria")) {
        try report.addViolation(allocator, "prompt file should contain Acceptance/Verification Criteria: {s}", .{report.prompt_path});
    }
}

fn printReport(allocator: std.mem.Allocator, report: Report) !void {
    if (report.strict) {
        if (report.error_count == 0) {
            std.debug.print("OK: workflow orchestration contract check passed (strict)\n", .{});
        } else {
            std.debug.print("FAILED: workflow orchestration contract check found {d} violation(s)\n", .{report.error_count});
        }
    } else {
        if (report.warning_count == 0) {
            std.debug.print("OK: workflow orchestration contract check passed (advisory)\n", .{});
        } else {
            std.debug.print("WARN: workflow orchestration advisory found {d} issue(s)\n", .{report.warning_count});
        }
    }

    for (report.messages.items) |msg| {
        std.debug.print("- {s}\n", .{msg});
    }

    _ = allocator;
}

fn printJson(allocator: std.mem.Allocator, report: Report) !void {
    const JsonPayload = struct {
        strict: bool,
        passed: bool,
        todo_path: []const u8,
        lessons_path: []const u8,
        prompt_path: []const u8,
        warning_count: usize,
        error_count: usize,
        messages: []const []const u8,
    };

    var msg_view = try allocator.alloc([]const u8, report.messages.items.len);
    defer allocator.free(msg_view);
    for (report.messages.items, 0..) |msg, idx| {
        msg_view[idx] = msg;
    }

    var writer: std.Io.Writer.Allocating = .init(allocator);
    defer writer.deinit();
    try std.json.Stringify.value(JsonPayload{
        .strict = report.strict,
        .passed = report.passed(),
        .todo_path = report.todo_path,
        .lessons_path = report.lessons_path,
        .prompt_path = report.prompt_path,
        .warning_count = report.warning_count,
        .error_count = report.error_count,
        .messages = msg_view,
    }, .{ .whitespace = .indent_2 }, &writer.writer);
    try writer.writer.writeByte('\n');
    const out = try writer.toOwnedSlice();
    defer allocator.free(out);
    std.debug.print("{s}", .{out});
}

pub fn main(init: std.process.Init) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    const io = io_backend.io();

    var cfg = parseArgs(init, allocator) catch |err| {
        if (err == error.MissingArgValue or err == error.InvalidArg) {
            printHelp();
            std.process.exit(1);
        }
        return err;
    };
    defer cfg.deinit(allocator);

    var report = Report{
        .strict = cfg.strict,
        .todo_path = cfg.todo_path,
        .lessons_path = cfg.lessons_path,
        .prompt_path = cfg.prompt_path,
    };
    defer report.deinit(allocator);

    try checkTodo(allocator, io, &report);
    try checkLessons(allocator, io, &report);
    try checkPrompt(allocator, io, &report);

    if (cfg.emit_json) {
        try printJson(allocator, report);
    } else {
        try printReport(allocator, report);
    }

    if (cfg.strict and !report.passed()) {
        std.process.exit(1);
    }
}
