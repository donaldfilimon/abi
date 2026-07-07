//! Generic, declarative argument parser for the CLI framework.
//!
//! A command declares a `[]const Arg` spec; `parse` turns an argv slice into a
//! `Parsed` value with typed accessors. The parser is `std`-only so it can be
//! unit-tested without the handler graph.
//!
//! Semantics are a faithful generalization of the historical hand-written
//! `parseCompleteArgs`:
//!   * flags/values are order-independent and may appear before or after the
//!     positional(s);
//!   * a `value` flag consumes the next token; a missing value is a usage error;
//!   * unknown dashed tokens are usage errors; use `--` before a literal
//!     positional that begins with `-`;
//!   * non-option tokens are assigned to the next unfilled positional —
//!     overflowing the positionals is a usage error;
//!   * a missing `required` argument is a usage error.

const std = @import("std");

pub const Kind = enum { flag, value, positional };
pub const ValueKind = enum { string, uint, port };

pub const Arg = struct {
    /// Bare name without the leading dashes (e.g. "model", "input"). Flags and
    /// values are matched against `--<name>`; positionals are identified by
    /// `name` only.
    name: []const u8,
    kind: Kind,
    required: bool = false,
    value_kind: ValueKind = .string,
    choices: []const []const u8 = &.{},
    /// When true on a positional, consume the remaining argv tokens joined by
    /// spaces. This is for legacy command tails such as `plugin run <name>
    /// [input...]` where unquoted multi-word input has always been accepted.
    greedy: bool = false,
    help: []const u8 = "",
};

const Slot = struct {
    spec: Arg,
    present: bool = false,
    str: ?[]const u8 = null,
    num: u64 = 0,
    owned: bool = false,
};

pub const Parsed = struct {
    allocator: std.mem.Allocator,
    slots: []Slot,

    pub fn deinit(self: *Parsed) void {
        for (self.slots) |slot| {
            if (slot.owned) {
                if (slot.str) |owned_value| self.allocator.free(owned_value);
            }
        }
        self.allocator.free(self.slots);
        self.slots = &.{};
    }

    fn find(self: Parsed, name: []const u8) ?*const Slot {
        for (self.slots) |*slot| {
            if (std.mem.eql(u8, slot.spec.name, name)) return slot;
        }
        return null;
    }

    /// True when a boolean flag was supplied (false for unknown names).
    pub fn flag(self: Parsed, name: []const u8) bool {
        if (self.find(name)) |slot| return slot.present;
        return false;
    }

    /// The string captured for a `value` flag or positional, or null when it was
    /// not supplied (or the name is unknown).
    pub fn value(self: Parsed, name: []const u8) ?[]const u8 {
        if (self.find(name)) |slot| return slot.str;
        return null;
    }

    /// The parsed unsigned integer for a `uint`/`port` value, or null when not
    /// supplied (or the name is unknown).
    pub fn uint(self: Parsed, name: []const u8) ?u64 {
        if (self.find(name)) |slot| {
            if (!slot.present) return null;
            return slot.num;
        }
        return null;
    }
};

fn matchFlag(spec: []const Arg, token: []const u8) ?usize {
    if (token.len <= 2 or token[0] != '-' or token[1] != '-') return null;
    const name = token[2..];
    for (spec, 0..) |a, idx| {
        if (a.kind == .positional) continue;
        if (std.mem.eql(u8, a.name, name)) return idx;
    }
    return null;
}

fn parseNumber(text: []const u8, kind: ValueKind) error{Usage}!u64 {
    const n = std.fmt.parseInt(u64, text, 10) catch return error.Usage;
    if (kind == .port and (n == 0 or n > 65535)) return error.Usage;
    return n;
}

/// Parse `argv` against `spec`. `argv` is the full process argv: index 0 is the
/// executable and index 1 is the command name, so parsing starts at index 2
/// (matching the legacy parsers). The caller owns the returned `Parsed` and must
/// `deinit` it.
pub fn parse(
    allocator: std.mem.Allocator,
    spec: []const Arg,
    argv: []const []const u8,
) error{ Usage, OutOfMemory }!Parsed {
    return parseFrom(allocator, spec, argv, 2);
}

/// Parse `argv` against `spec`, starting at an explicit token index. This is
/// used by registry-owned subcommands where argv[1] is the top-level command
/// and argv[2] is the subcommand name.
pub fn parseFrom(
    allocator: std.mem.Allocator,
    spec: []const Arg,
    argv: []const []const u8,
    start_index: usize,
) error{ Usage, OutOfMemory }!Parsed {
    const slots = try allocator.alloc(Slot, spec.len);
    errdefer allocator.free(slots);
    for (spec, 0..) |a, idx| slots[idx] = .{ .spec = a };

    var i: usize = start_index;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (std.mem.eql(u8, token, "--")) {
            i += 1;
            while (i < argv.len) {
                const consumed = try assignPositional(allocator, slots, argv[i..]);
                i += consumed;
            }
            break;
        }
        if (matchFlag(spec, token)) |idx| {
            var slot = &slots[idx];
            slot.present = true;
            if (slot.spec.kind == .value) {
                i += 1;
                if (i >= argv.len) return error.Usage;
                slot.str = argv[i];
                if (slot.spec.value_kind != .string) {
                    slot.num = try parseNumber(argv[i], slot.spec.value_kind);
                }
                try validateChoices(slot.spec, argv[i]);
            }
        } else {
            const pos_idx = nextPositional(slots) orelse return error.Usage;
            if (std.mem.startsWith(u8, token, "-") and !slots[pos_idx].spec.greedy) return error.Usage;
            const consumed = try assignPositional(allocator, slots, argv[i..]);
            i += consumed - 1;
        }
    }

    for (slots) |slot| {
        if (slot.spec.required and !slot.present) return error.Usage;
    }
    return .{ .allocator = allocator, .slots = slots };
}

fn nextPositional(slots: []const Slot) ?usize {
    for (slots, 0..) |slot, idx| {
        if (slot.spec.kind == .positional and !slot.present) return idx;
    }
    return null;
}

fn assignPositional(allocator: std.mem.Allocator, slots: []Slot, tokens: []const []const u8) error{ Usage, OutOfMemory }!usize {
    if (tokens.len == 0) return error.Usage;
    const idx = nextPositional(slots) orelse return error.Usage;
    var slot = &slots[idx];
    slot.present = true;
    const token = tokens[0];
    if (slot.spec.greedy) {
        if (tokens.len == 1) {
            slot.str = token;
        } else {
            slot.str = try std.mem.join(allocator, " ", tokens);
            slot.owned = true;
        }
        if (slot.spec.value_kind != .string) {
            slot.num = try parseNumber(token, slot.spec.value_kind);
        }
        try validateChoices(slot.spec, slot.str.?);
        return tokens.len;
    }
    slot.str = token;
    if (slot.spec.value_kind != .string) {
        slot.num = try parseNumber(token, slot.spec.value_kind);
    }
    try validateChoices(slot.spec, token);
    return 1;
}

fn validateChoices(spec: Arg, token: []const u8) error{Usage}!void {
    if (spec.choices.len == 0) return;
    for (spec.choices) |choice| {
        if (std.mem.eql(u8, token, choice)) return;
    }
    return error.Usage;
}

test "parse reproduces complete semantics: order-independent flags + positional" {
    const spec = [_]Arg{
        .{ .name = "live", .kind = .flag },
        .{ .name = "confirm", .kind = .flag },
        .{ .name = "learn", .kind = .flag },
        .{ .name = "model", .kind = .value },
        .{ .name = "input", .kind = .positional, .required = true },
    };

    var p = try parse(std.testing.allocator, &spec, &.{ "abi", "complete", "--model", "fable-5", "--live", "--learn", "hi" });
    defer p.deinit();
    try std.testing.expect(p.flag("live"));
    try std.testing.expect(p.flag("learn"));
    try std.testing.expect(!p.flag("confirm"));
    try std.testing.expectEqualStrings("fable-5", p.value("model").?);
    try std.testing.expectEqualStrings("hi", p.value("input").?);
}

test "parse rejects missing value, missing positional, and overflow" {
    const spec = [_]Arg{
        .{ .name = "model", .kind = .value },
        .{ .name = "input", .kind = .positional, .required = true },
    };
    // --model with no value
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "complete", "--model" }));
    // missing required positional
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "complete", "--model", "x" }));
    // too many positionals
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "complete", "a", "b" }));
}

test "parse rejects unknown dashed tokens unless escaped with end-of-options" {
    const spec = [_]Arg{
        .{ .name = "input", .kind = .positional, .required = true },
    };
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "complete", "--bogus" }));

    var p = try parse(std.testing.allocator, &spec, &.{ "abi", "complete", "--", "--literal" });
    defer p.deinit();
    try std.testing.expectEqualStrings("--literal", p.value("input").?);
}

test "parse validates uint and port value kinds" {
    const spec = [_]Arg{
        .{ .name = "count", .kind = .value, .value_kind = .uint },
        .{ .name = "port", .kind = .value, .value_kind = .port },
    };
    var p = try parse(std.testing.allocator, &spec, &.{ "abi", "x", "--count", "42", "--port", "8080" });
    defer p.deinit();
    try std.testing.expectEqual(@as(u64, 42), p.uint("count").?);
    try std.testing.expectEqual(@as(u64, 8080), p.uint("port").?);

    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "x", "--count", "nope" }));
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "x", "--port", "0" }));
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "x", "--port", "70000" }));
}

test "parse validates allowed string choices for value and positional args" {
    const spec = [_]Arg{
        .{ .name = "mode", .kind = .value, .choices = &.{ "fast", "safe" } },
        .{ .name = "command", .kind = .positional, .required = true, .choices = &.{"status"} },
    };

    var p = try parse(std.testing.allocator, &spec, &.{ "abi", "scheduler", "--mode", "safe", "status" });
    defer p.deinit();
    try std.testing.expectEqualStrings("safe", p.value("mode").?);
    try std.testing.expectEqualStrings("status", p.value("command").?);

    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "scheduler", "--mode", "unsafe", "status" }));
    try std.testing.expectError(error.Usage, parse(std.testing.allocator, &spec, &.{ "abi", "scheduler", "--mode", "safe", "bogus" }));
}

test "parse supports optional greedy trailing positional" {
    const spec = [_]Arg{
        .{ .name = "command", .kind = .positional, .required = true, .choices = &.{"run"} },
        .{ .name = "name", .kind = .positional, .required = true },
        .{ .name = "input", .kind = .positional, .greedy = true },
    };

    var p = try parse(std.testing.allocator, &spec, &.{ "abi", "plugin", "run", "example-plugin", "hello", "world" });
    defer p.deinit();
    try std.testing.expectEqualStrings("run", p.value("command").?);
    try std.testing.expectEqualStrings("example-plugin", p.value("name").?);
    try std.testing.expectEqualStrings("hello world", p.value("input").?);

    var dashed = try parse(std.testing.allocator, &spec, &.{ "abi", "plugin", "run", "example-plugin", "--flag-like-input" });
    defer dashed.deinit();
    try std.testing.expectEqualStrings("--flag-like-input", dashed.value("input").?);

    var no_input = try parse(std.testing.allocator, &spec, &.{ "abi", "plugin", "run", "example-plugin" });
    defer no_input.deinit();
    try std.testing.expect(no_input.value("input") == null);
}

test "parseFrom starts after a registry subcommand token" {
    const spec = [_]Arg{
        .{ .name = "input", .kind = .positional, .required = true },
    };

    var p = try parseFrom(std.testing.allocator, &spec, &.{ "abi", "agent", "plan", "inspect" }, 3);
    defer p.deinit();
    try std.testing.expectEqualStrings("inspect", p.value("input").?);

    try std.testing.expectError(error.Usage, parseFrom(std.testing.allocator, &spec, &.{ "abi", "agent", "plan" }, 3));
    try std.testing.expectError(error.Usage, parseFrom(std.testing.allocator, &spec, &.{ "abi", "agent", "plan", "a", "b" }, 3));
}

test {
    std.testing.refAllDecls(@This());
}
