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
//!   * any token that does not match a declared flag/value is assigned to the
//!     next unfilled positional (so an unknown `--foo` becomes the positional,
//!     exactly as the legacy parser treated it) — overflowing the positionals
//!     is a usage error;
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
    help: []const u8 = "",
};

const Slot = struct {
    spec: Arg,
    present: bool = false,
    str: ?[]const u8 = null,
    num: u64 = 0,
};

pub const Parsed = struct {
    allocator: std.mem.Allocator,
    slots: []Slot,

    pub fn deinit(self: *Parsed) void {
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
    const slots = try allocator.alloc(Slot, spec.len);
    errdefer allocator.free(slots);
    for (spec, 0..) |a, idx| slots[idx] = .{ .spec = a };

    var i: usize = 2;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
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
            }
        } else {
            // Not a declared flag/value: assign to the next unfilled positional.
            const idx = nextPositional(slots) orelse return error.Usage;
            var slot = &slots[idx];
            slot.present = true;
            slot.str = token;
            if (slot.spec.value_kind != .string) {
                slot.num = try parseNumber(token, slot.spec.value_kind);
            }
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

test "parse treats an unknown dashed token as the positional (legacy parity)" {
    const spec = [_]Arg{
        .{ .name = "input", .kind = .positional, .required = true },
    };
    var p = try parse(std.testing.allocator, &spec, &.{ "abi", "complete", "--bogus" });
    defer p.deinit();
    try std.testing.expectEqualStrings("--bogus", p.value("input").?);
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

test {
    std.testing.refAllDecls(@This());
}
