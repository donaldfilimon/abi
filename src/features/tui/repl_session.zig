//! Session save/load for the agent TUI REPL.
//!
//! Serializes `ReplState` to `~/.abi/sessions/<name>.json` and restores it.
//! Extracted from `repl.zig` so the interactive loop stays focused on I/O
//! and command dispatch (Zig 0.17 modular boundary).
const std = @import("std");
const env = @import("../../foundation/env.zig");
const utils = @import("../../foundation/utils.zig");
const types = @import("repl_types.zig");

pub const SessionFile = struct {
    version: u32,
    session_id: i64,
    model: []const u8,
    learn_mode: bool,
    open_path: []const u8,
    open_content: []const u8,
    turn_count: usize,
    turn_history_count: usize,
    turn_history_head: usize,
    turns: []const TurnData,

    pub const TurnData = struct {
        input: []const u8,
        response: []const u8,
    };
};

/// Return `~/.abi/sessions` as an owned path. Caller frees.
pub fn sessionsDir(allocator: std.mem.Allocator) ![]const u8 {
    const home = env.get("HOME") orelse return error.HomeNotSet;
    return try utils.pathJoin(home, ".abi/sessions", allocator);
}

/// Build the on-disk JSON for the current REPL state. Caller frees the slice.
pub fn serializeSession(allocator: std.mem.Allocator, state: *const types.ReplState) ![]u8 {
    var turns = std.ArrayListUnmanaged(SessionFile.TurnData).empty;
    defer turns.deinit(allocator);
    var i: usize = 0;
    while (i < state.turn_history_count) : (i += 1) {
        const idx = (state.turn_history_head + types.MAX_TURN_HISTORY - state.turn_history_count + i) % types.MAX_TURN_HISTORY;
        try turns.append(allocator, .{
            .input = state.turn_history[idx].input,
            .response = state.turn_history[idx].response,
        });
    }

    var json_out: std.Io.Writer.Allocating = .init(allocator);
    defer json_out.deinit();
    var jw = std.json.Stringify{ .writer = &json_out.writer, .options = .{ .whitespace = .indent_2 } };
    try jw.beginObject();
    try jw.objectField("version");
    try jw.write(@as(u32, 1));
    try jw.objectField("session_id");
    try jw.write(state.session_id);
    try jw.objectField("model");
    try jw.write(state.config.model);
    try jw.objectField("learn_mode");
    try jw.write(state.config.learn_mode);
    try jw.objectField("open_path");
    try jw.write(state.open_path);
    try jw.objectField("open_content");
    try jw.write(state.open_content);
    try jw.objectField("turn_count");
    try jw.write(state.turn_count);
    try jw.objectField("turn_history_count");
    try jw.write(state.turn_history_count);
    try jw.objectField("turn_history_head");
    try jw.write(state.turn_history_head);
    try jw.objectField("turns");
    try jw.beginArray();
    for (turns.items) |t| {
        try jw.beginObject();
        try jw.objectField("input");
        try jw.write(t.input);
        try jw.objectField("response");
        try jw.write(t.response);
        try jw.endObject();
    }
    try jw.endArray();
    try jw.endObject();
    // Dupe before `defer json_out.deinit()` frees the writer buffer.
    return try allocator.dupe(u8, json_out.written());
}

/// Apply a parsed `SessionFile` onto `state`, replacing history and file context.
pub fn applySessionFile(allocator: std.mem.Allocator, state: *types.ReplState, file: SessionFile) !void {
    state.clearTurnHistory(allocator);
    state.clearFileContext(allocator);

    if (file.model.len > 0 and file.model.len <= types.MODEL_STORAGE_BYTES) {
        @memcpy(state.model_storage[0..file.model.len], file.model);
        state.config.model = state.model_storage[0..file.model.len];
    }
    state.config.learn_mode = file.learn_mode;
    state.session_id = file.session_id;
    state.turn_count = file.turn_count;
    // Clamp untrusted file values to valid ring-buffer ranges.
    state.turn_history_count = @min(file.turn_history_count, types.MAX_TURN_HISTORY);
    state.turn_history_head = file.turn_history_head % types.MAX_TURN_HISTORY;

    var ti: usize = 0;
    while (ti < state.turn_history_count and ti < file.turns.len) : (ti += 1) {
        const idx = (state.turn_history_head + types.MAX_TURN_HISTORY - state.turn_history_count + ti) % types.MAX_TURN_HISTORY;
        state.turn_history[idx] = .{
            .input = try allocator.dupe(u8, file.turns[ti].input),
            .response = try allocator.dupe(u8, file.turns[ti].response),
        };
    }

    if (file.open_path.len > 0 and file.open_content.len > 0) {
        const path_bytes = file.open_path;
        const content_bytes = file.open_content;
        const header_len = @sizeOf(usize);
        const total_len = header_len + path_bytes.len + content_bytes.len;
        const buf = try allocator.alloc(u8, total_len);
        @memcpy(buf[0..header_len], std.mem.asBytes(&path_bytes.len));
        @memcpy(buf[header_len..][0..path_bytes.len], path_bytes);
        @memcpy(buf[header_len + path_bytes.len ..], content_bytes);
        state.file_context_buf = buf;
        state.open_path = buf[header_len..][0..path_bytes.len];
        state.open_content = buf[header_len + path_bytes.len ..];
    }
}

/// `/save <name>`: write session JSON under `~/.abi/sessions/`.
pub fn saveSession(allocator: std.mem.Allocator, state: *const types.ReplState, name: []const u8, io: std.Io) !void {
    if (name.len == 0) {
        std.debug.print("usage: /save <name>\n", .{});
        return;
    }
    const sessions_dir = try sessionsDir(allocator);
    defer allocator.free(sessions_dir);

    std.Io.Dir.createDirPath(.cwd(), io, sessions_dir) catch |err| {
        std.debug.print("save: cannot create sessions dir '{s}': {s}\n", .{ sessions_dir, @errorName(err) });
        return;
    };

    const filepath = try std.fmt.allocPrint(allocator, "{s}/{s}.json", .{ sessions_dir, name });
    defer allocator.free(filepath);

    const payload = try serializeSession(allocator, state);
    defer allocator.free(payload);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = filepath, .data = payload });
    std.debug.print("session saved: {s} ({d} turns)\n", .{ filepath, state.turn_history_count });
}

/// `/load <name>`: restore session state from `~/.abi/sessions/<name>.json`.
pub fn loadSession(allocator: std.mem.Allocator, state: *types.ReplState, name: []const u8, io: std.Io) !void {
    if (name.len == 0) {
        std.debug.print("usage: /load <name>\n", .{});
        return;
    }
    const sessions_dir = try sessionsDir(allocator);
    defer allocator.free(sessions_dir);

    const filepath = try std.fmt.allocPrint(allocator, "{s}/{s}.json", .{ sessions_dir, name });
    defer allocator.free(filepath);

    const content = std.Io.Dir.cwd().readFileAlloc(io, filepath, allocator, .limited(10 * 1024 * 1024)) catch |err| {
        std.debug.print("load: cannot read '{s}': {s}\n", .{ filepath, @errorName(err) });
        return;
    };
    defer allocator.free(content);

    const parsed = std.json.parseFromSlice(SessionFile, allocator, content, .{
        .ignore_unknown_fields = true,
    }) catch |err| {
        std.debug.print("load: parse error: {s}\n", .{@errorName(err)});
        return;
    };
    defer parsed.deinit();

    try applySessionFile(allocator, state, parsed.value);
    std.debug.print("session loaded: {s} ({d} turns)\n", .{ filepath, state.turn_history_count });
}

test "serializeSession round-trips turn history" {
    const allocator = std.testing.allocator;
    var state = types.ReplState.init(.{ .model = "test-model", .learn_mode = true });
    defer state.clearTurnHistory(allocator);
    state.session_id = 99;
    state.turn_count = 3;
    state.pushTurn(allocator, "a", "b");
    state.pushTurn(allocator, "c", "d");

    const json = try serializeSession(allocator, &state);
    defer allocator.free(json);

    const parsed = try std.json.parseFromSlice(SessionFile, allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u32, 1), parsed.value.version);
    try std.testing.expectEqual(@as(i64, 99), parsed.value.session_id);
    try std.testing.expectEqualStrings("test-model", parsed.value.model);
    try std.testing.expect(parsed.value.learn_mode);
    try std.testing.expectEqual(@as(usize, 3), parsed.value.turn_count);
    try std.testing.expectEqual(@as(usize, 2), parsed.value.turns.len);
    try std.testing.expectEqualStrings("a", parsed.value.turns[0].input);
    try std.testing.expectEqualStrings("b", parsed.value.turns[0].response);
    try std.testing.expectEqualStrings("c", parsed.value.turns[1].input);
    try std.testing.expectEqualStrings("d", parsed.value.turns[1].response);
}

test "applySessionFile restores model and turns" {
    const allocator = std.testing.allocator;
    var state = types.ReplState.init(.{ .model = "old" });
    defer {
        state.clearTurnHistory(allocator);
        state.clearFileContext(allocator);
    }

    const file = SessionFile{
        .version = 1,
        .session_id = 42,
        .model = "restored-model",
        .learn_mode = true,
        .open_path = "",
        .open_content = "",
        .turn_count = 7,
        .turn_history_count = 1,
        .turn_history_head = 1,
        .turns = &.{
            .{ .input = "hi", .response = "there" },
        },
    };
    try applySessionFile(allocator, &state, file);
    try std.testing.expectEqualStrings("restored-model", state.config.model);
    try std.testing.expect(state.config.learn_mode);
    try std.testing.expectEqual(@as(i64, 42), state.session_id);
    try std.testing.expectEqual(@as(usize, 7), state.turn_count);
    try std.testing.expectEqual(@as(usize, 1), state.turn_history_count);
    const idx = (state.turn_history_head + types.MAX_TURN_HISTORY - state.turn_history_count) % types.MAX_TURN_HISTORY;
    try std.testing.expectEqualStrings("hi", state.turn_history[idx].input);
    try std.testing.expectEqualStrings("there", state.turn_history[idx].response);
}

test {
    std.testing.refAllDecls(@This());
}
