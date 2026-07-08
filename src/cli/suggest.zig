//! CLI suggestion engine — edit-distance based "did you mean?" hints for
//! unknown commands, subcommands, flags, and choice values.
//!
//! All functions are pub so the dispatch layer and help handler can call
//! them; the module is std-only except for the `registry` and `arg` imports
//! (column-0 `pub const`/`pub fn` names only, so check-parity is a no-op).

const std = @import("std");
const registry = @import("registry.zig");
const arg = @import("arg.zig");

const MAX_SUGGESTION_LEN = 64;

pub const Suggestion = struct {
    value: []const u8,
    leading_dashes: bool = false,
};

pub fn choiceContains(choices: []const []const u8, token: []const u8) bool {
    for (choices) |choice| {
        if (std.mem.eql(u8, choice, token)) return true;
    }
    return false;
}

pub fn optionFor(spec: []const arg.Arg, name: []const u8) ?arg.Arg {
    for (spec) |a| {
        if (a.kind == .positional) continue;
        if (std.mem.eql(u8, a.name, name)) return a;
    }
    return null;
}

pub fn nextPositionalSpec(spec: []const arg.Arg, used: []const bool) ?usize {
    for (spec, 0..) |a, idx| {
        if (a.kind == .positional and !used[idx]) return idx;
    }
    return null;
}

fn editDistance(a_raw: []const u8, b_raw: []const u8) usize {
    if (a_raw.len > MAX_SUGGESTION_LEN or b_raw.len > MAX_SUGGESTION_LEN) return MAX_SUGGESTION_LEN + 1;

    var previous: [MAX_SUGGESTION_LEN + 1]usize = undefined;
    var current: [MAX_SUGGESTION_LEN + 1]usize = undefined;

    for (0..b_raw.len + 1) |idx| previous[idx] = idx;

    for (a_raw, 0..) |a_ch, a_idx| {
        current[0] = a_idx + 1;
        for (b_raw, 0..) |b_ch, b_idx| {
            const cost: usize = if (std.ascii.toLower(a_ch) == std.ascii.toLower(b_ch)) 0 else 1;
            const deletion = previous[b_idx + 1] + 1;
            const insertion = current[b_idx] + 1;
            const substitution = previous[b_idx] + cost;
            current[b_idx + 1] = @min(@min(deletion, insertion), substitution);
        }
        @memcpy(previous[0 .. b_raw.len + 1], current[0 .. b_raw.len + 1]);
    }

    return previous[b_raw.len];
}

fn suggestionThreshold(token_len: usize) usize {
    if (token_len <= 3) return 1;
    if (token_len <= 8) return 2;
    return 3;
}

fn bestSuggestion(token: []const u8, candidates: []const []const u8) ?[]const u8 {
    var best: ?[]const u8 = null;
    var best_score: usize = MAX_SUGGESTION_LEN + 1;

    for (candidates) |candidate| {
        if (candidate.len == 0) continue;
        const score = editDistance(token, candidate);
        if (score < best_score) {
            best_score = score;
            best = candidate;
        }
    }

    if (best) |candidate| {
        if (best_score <= suggestionThreshold(@max(token.len, candidate.len))) return candidate;
    }
    return null;
}

pub fn suggestCommand(name: []const u8) ?Suggestion {
    var best: ?[]const u8 = null;
    var best_score: usize = MAX_SUGGESTION_LEN + 1;
    for (registry.commands) |command| {
        const score = editDistance(name, command.name);
        if (score < best_score) {
            best_score = score;
            best = command.name;
        }
    }
    if (best) |value| {
        if (best_score <= suggestionThreshold(@max(name.len, value.len))) return .{ .value = value };
    }
    return null;
}

pub fn suggestSubcommand(command: registry.Command, name: []const u8) ?Suggestion {
    var best: ?[]const u8 = null;
    var best_score: usize = MAX_SUGGESTION_LEN + 1;
    for (command.subcommands) |subcommand| {
        const score = editDistance(name, subcommand.name);
        if (score < best_score) {
            best_score = score;
            best = subcommand.name;
        }
    }
    if (best) |value| {
        if (best_score <= suggestionThreshold(@max(name.len, value.len))) return .{ .value = value };
    }
    return null;
}

pub fn suggestForArgs(spec: []const arg.Arg, argv: []const []const u8, start_index: usize) ?Suggestion {
    var used_positionals = std.mem.zeroes([32]bool);
    if (spec.len > used_positionals.len) return null;

    var i: usize = start_index;
    while (i < argv.len) : (i += 1) {
        const token = argv[i];
        if (std.mem.eql(u8, token, "--")) break;
        if (std.mem.startsWith(u8, token, "--")) {
            const name = token[2..];
            const option = optionFor(spec, name) orelse {
                var best: ?[]const u8 = null;
                var best_score: usize = MAX_SUGGESTION_LEN + 1;
                for (spec) |a| {
                    if (a.kind == .positional) continue;
                    const score = editDistance(name, a.name);
                    if (score < best_score) {
                        best_score = score;
                        best = a.name;
                    }
                }
                if (best) |value| {
                    if (best_score <= suggestionThreshold(@max(name.len, value.len))) {
                        return .{ .value = value, .leading_dashes = true };
                    }
                }
                return null;
            };

            if (option.kind == .value) {
                i += 1;
                if (i >= argv.len) return null;
                if (option.choices.len != 0 and !choiceContains(option.choices, argv[i])) {
                    if (bestSuggestion(argv[i], option.choices)) |value| return .{ .value = value };
                }
            }
            continue;
        }

        const pos_idx = nextPositionalSpec(spec, used_positionals[0..spec.len]) orelse return null;
        used_positionals[pos_idx] = true;
        const positional = spec[pos_idx];
        if (positional.choices.len != 0 and !choiceContains(positional.choices, token)) {
            if (bestSuggestion(token, positional.choices)) |value| return .{ .value = value };
        }
        if (positional.greedy) break;
    }
    return null;
}

pub fn printHint(suggestion: Suggestion) void {
    if (suggestion.leading_dashes) {
        std.debug.print("hint: did you mean `--{s}`?\n", .{suggestion.value});
    } else {
        std.debug.print("hint: did you mean `{s}`?\n", .{suggestion.value});
    }
}

pub fn usageErrorWithSuggestion(message: []const u8, suggestion: ?Suggestion) u8 {
    std.debug.print("error: {s}\n", .{message});
    if (suggestion) |candidate| printHint(candidate);
    return 2;
}

test "suggestions choose close commands, subcommands, flags, and choices" {
    const command = for (registry.commands) |c| {
        if (std.mem.eql(u8, c.name, "agent")) break c;
    } else return error.MissingCommand;
    const dashboard = for (registry.commands) |c| {
        if (std.mem.eql(u8, c.name, "dashboard")) break c;
    } else return error.MissingCommand;

    try std.testing.expectEqualStrings("complete", (suggestCommand("complte") orelse return error.MissingSuggestion).value);
    try std.testing.expectEqualStrings("train", (suggestSubcommand(command, "trian") orelse return error.MissingSuggestion).value);

    const pane_suggestion = suggestForArgs(dashboard.args, &.{ "abi", "dashboard", "--pane", "memry" }, 2) orelse return error.MissingSuggestion;
    try std.testing.expectEqualStrings("memory", pane_suggestion.value);
    try std.testing.expect(!pane_suggestion.leading_dashes);

    const flag_suggestion = suggestForArgs(dashboard.args, &.{ "abi", "dashboard", "--plian" }, 2) orelse return error.MissingSuggestion;
    try std.testing.expectEqualStrings("plain", flag_suggestion.value);
    try std.testing.expect(flag_suggestion.leading_dashes);
}

test {
    std.testing.refAllDecls(@This());
}
