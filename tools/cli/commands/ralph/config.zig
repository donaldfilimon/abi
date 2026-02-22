//! Ralph configuration, constants, and workspace helpers.

const std = @import("std");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const cli_io = utils.io_backend;

// ============================================================================
// Constants
// ============================================================================

pub const WORKSPACE_DIR = ".ralph";
pub const AGENT_DIR = ".ralph/agent";
pub const LOGS_DIR = ".ralph/diagnostics/logs";
pub const RUNS_DIR = ".ralph/runs";
pub const SKILLS_FILE = ".ralph/skills.jsonl";
pub const LATEST_RUN_FILE = ".ralph/runs/latest.json";
pub const LOCK_FILE = ".ralph/loop.lock";
pub const STATE_FILE = ".ralph/state.json";
pub const CONFIG_FILE = "ralph.yml";
pub const PROMPT_FILE = "PROMPT.md";

// ============================================================================
// RalphConfig
// ============================================================================

/// Minimal fields parsed from ralph.yml line-by-line.
pub const RalphConfig = struct {
    backend: []const u8 = "llama_cpp",
    llm_backend: []const u8 = "llama_cpp",
    llm_fallback: []const u8 = "mlx,ollama,lm_studio,vllm,plugin_http,plugin_native",
    llm_strict_backend: bool = false,
    llm_model: []const u8 = "llama3.2",
    llm_plugin: ?[]const u8 = null,
    prompt_file: []const u8 = PROMPT_FILE,
    completion_promise: []const u8 = "LOOP_COMPLETE",
    max_iterations: usize = 5,
    max_fix_attempts: usize = 2,
    require_clean_tree: bool = true,
    gate_per_iteration: []const u8 = "zig build verify-all",
};

/// Parse ralph.yml into config. Returned strings are slices into `contents`.
pub fn parseRalphYamlInto(contents: []const u8, out: *RalphConfig) void {
    const Section = enum {
        root,
        cli,
        llm,
        event_loop,
        execution,
        gates,
    };

    var section: Section = .root;
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;

        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t\"");

        if (value.len == 0 and line[line.len - 1] == ':') {
            if (std.mem.eql(u8, key, "cli")) section = .cli else if (std.mem.eql(u8, key, "llm")) section = .llm else if (std.mem.eql(u8, key, "event_loop")) section = .event_loop else if (std.mem.eql(u8, key, "execution")) section = .execution else if (std.mem.eql(u8, key, "gates")) section = .gates else section = .root;
            continue;
        }

        switch (section) {
            .root => {
                if (std.mem.eql(u8, key, "backend")) {
                    out.backend = value;
                    out.llm_backend = value;
                } else if (std.mem.eql(u8, key, "prompt_file")) {
                    out.prompt_file = value;
                } else if (std.mem.eql(u8, key, "completion_promise")) {
                    out.completion_promise = value;
                } else if (std.mem.eql(u8, key, "max_iterations")) {
                    out.max_iterations = std.fmt.parseInt(usize, value, 10) catch out.max_iterations;
                }
            },
            .cli => {
                if (std.mem.eql(u8, key, "backend")) {
                    out.backend = value;
                }
            },
            .llm => {
                if (std.mem.eql(u8, key, "backend")) {
                    out.llm_backend = value;
                } else if (std.mem.eql(u8, key, "fallback")) {
                    out.llm_fallback = value;
                } else if (std.mem.eql(u8, key, "strict_backend")) {
                    out.llm_strict_backend = parseBool(value, out.llm_strict_backend);
                } else if (std.mem.eql(u8, key, "model")) {
                    out.llm_model = value;
                } else if (std.mem.eql(u8, key, "plugin")) {
                    out.llm_plugin = if (value.len == 0) null else value;
                }
            },
            .event_loop => {
                if (std.mem.eql(u8, key, "prompt_file")) {
                    out.prompt_file = value;
                } else if (std.mem.eql(u8, key, "completion_promise")) {
                    out.completion_promise = value;
                } else if (std.mem.eql(u8, key, "max_iterations")) {
                    out.max_iterations = std.fmt.parseInt(usize, value, 10) catch out.max_iterations;
                }
            },
            .execution => {
                if (std.mem.eql(u8, key, "max_iterations")) {
                    out.max_iterations = std.fmt.parseInt(usize, value, 10) catch out.max_iterations;
                } else if (std.mem.eql(u8, key, "max_fix_attempts")) {
                    out.max_fix_attempts = std.fmt.parseInt(usize, value, 10) catch out.max_fix_attempts;
                } else if (std.mem.eql(u8, key, "require_clean_tree")) {
                    out.require_clean_tree = parseBool(value, out.require_clean_tree);
                }
            },
            .gates => {
                if (std.mem.eql(u8, key, "per_iteration")) {
                    out.gate_per_iteration = value;
                }
            },
        }
    }

    // Keep legacy backend key aligned with llm backend if only llm section was provided.
    out.backend = out.llm_backend;
}

fn parseBool(value: []const u8, default: bool) bool {
    if (std.mem.eql(u8, value, "true")) return true;
    if (std.mem.eql(u8, value, "false")) return false;
    return default;
}

// ============================================================================
// Workspace helpers
// ============================================================================

pub fn ensureDir(io: std.Io, path: []const u8) void {
    std.Io.Dir.cwd().createDirPath(io, path) catch {};
}

pub fn fileExists(io: std.Io, path: []const u8) bool {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

pub fn writeFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    content: []const u8,
) !void {
    _ = allocator;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    try file.writeStreamingAll(io, content);
}

// ============================================================================
// Loop state
// ============================================================================

/// Read and parse state.json, or return defaults on any failure.
pub const LoopState = struct {
    runs: u64 = 0,
    skills: u64 = 0,
    last_run_ts: i64 = 0,
    last_run_id: []const u8 = "",
    last_gate_passed: bool = false,
};

pub fn readState(allocator: std.mem.Allocator, io: std.Io) LoopState {
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        STATE_FILE,
        allocator,
        .limited(64 * 1024),
    ) catch return .{};
    defer allocator.free(contents);

    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        contents,
        .{},
    ) catch return .{};
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return .{},
    };

    return .{
        .runs = if (obj.get("runs")) |v| switch (v) {
            .integer => |i| @intCast(@max(i, 0)),
            else => 0,
        } else 0,
        .skills = if (obj.get("skills")) |v| switch (v) {
            .integer => |i| @intCast(@max(i, 0)),
            else => 0,
        } else 0,
        .last_run_ts = if (obj.get("last_run_ts")) |v| switch (v) {
            .integer => |i| i,
            else => 0,
        } else 0,
        .last_run_id = "",
        .last_gate_passed = if (obj.get("last_gate_passed")) |v| switch (v) {
            .bool => |b| b,
            else => false,
        } else false,
    };
}

pub fn writeState(allocator: std.mem.Allocator, io: std.Io, state: LoopState) void {
    const json = std.fmt.allocPrint(
        allocator,
        "{{\"runs\":{d},\"skills\":{d},\"last_run_ts\":{d},\"last_run_id\":\"{s}\",\"last_gate_passed\":{s}}}",
        .{
            state.runs,
            state.skills,
            state.last_run_ts,
            state.last_run_id,
            if (state.last_gate_passed) "true" else "false",
        },
    ) catch return;
    defer allocator.free(json);
    writeFile(allocator, io, STATE_FILE, json) catch {};
}

pub fn removeLockFile(io: std.Io) void {
    std.Io.Dir.cwd().deleteFile(io, LOCK_FILE) catch {};
}

/// Case-insensitive substring search.
pub fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
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
