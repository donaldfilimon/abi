//! TUI completion and search utilities.
//!
//! Pure functions for case-insensitive matching, fuzzy matching,
//! and completion scoring. These do not depend on TuiState.

const std = @import("std");
const types = @import("types.zig");

const MenuItem = types.MenuItem;
const HistoryEntry = types.HistoryEntry;
const MatchType = types.MatchType;
const CompletionSuggestion = types.CompletionSuggestion;

// ═══════════════════════════════════════════════════════════════════
// String Utilities
// ═══════════════════════════════════════════════════════════════════

pub fn toLower(c: u8) u8 {
    if (c >= 'A' and c <= 'Z') return c + 32;
    return c;
}

pub fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;

    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        for (needle, 0..) |nc, j| {
            const hc = haystack[i + j];
            if (toLower(hc) != toLower(nc)) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

/// Check if haystack starts with needle (case-insensitive)
pub fn startsWithIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (needle, 0..) |nc, i| {
        if (toLower(haystack[i]) != toLower(nc)) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// Fuzzy Matching
// ═══════════════════════════════════════════════════════════════════

/// Fuzzy match: returns score if all query characters appear in order.
/// Higher score = better match (consecutive chars, early positions).
pub fn fuzzyMatch(label: []const u8, query: []const u8) ?u32 {
    if (query.len == 0) return null;
    if (query.len > label.len) return null;

    var score: u32 = 500; // Base fuzzy score
    var query_idx: usize = 0;
    var last_match_pos: usize = 0;
    var consecutive_bonus: u32 = 0;

    for (label, 0..) |lc, label_idx| {
        if (query_idx >= query.len) break;

        if (toLower(lc) == toLower(query[query_idx])) {
            // Bonus for consecutive matches
            if (label_idx == last_match_pos + 1 and label_idx > 0) {
                consecutive_bonus += 20;
            }
            // Bonus for early matches
            if (label_idx < 3) {
                score += 30 - @as(u32, @intCast(label_idx)) * 10;
            }
            last_match_pos = label_idx;
            query_idx += 1;
        }
    }

    // All query characters must be found
    if (query_idx < query.len) return null;

    return score + consecutive_bonus;
}

// ═══════════════════════════════════════════════════════════════════
// Completion Scoring
// ═══════════════════════════════════════════════════════════════════

/// Calculate completion score for a menu item.
pub fn calculateCompletionScore(
    item: *const MenuItem,
    query: []const u8,
    history_items: []const HistoryEntry,
) ?CompletionSuggestion {
    const label = item.label;

    // Check for exact prefix match (highest priority)
    if (startsWithIgnoreCase(label, query)) {
        // Check if recently used
        const is_recent = isRecentlyUsed(item, history_items);
        return CompletionSuggestion{
            .item_index = 0, // Will be set by caller
            .score = if (is_recent) 1100 else 1000,
            .match_type = if (is_recent) .history_recent else .exact_prefix,
        };
    }

    // Check for fuzzy match
    if (fuzzyMatch(label, query)) |fuzzy_score| {
        return CompletionSuggestion{
            .item_index = 0,
            .score = fuzzy_score,
            .match_type = .fuzzy,
        };
    }

    // Check for substring match in label or description
    if (containsIgnoreCase(label, query) or containsIgnoreCase(item.description, query)) {
        return CompletionSuggestion{
            .item_index = 0,
            .score = 200,
            .match_type = .substring,
        };
    }

    return null;
}

/// Check if an item was recently used.
pub fn isRecentlyUsed(item: *const MenuItem, history_items: []const HistoryEntry) bool {
    switch (item.action) {
        .command => |cmd| {
            // History is newest-first (inserted at index 0), so inspect prefix.
            const check_count = @min(history_items.len, 5);
            for (history_items[0..check_count]) |entry| {
                if (std.mem.eql(u8, entry.command_id, cmd.id)) return true;
            }
        },
        else => {},
    }
    return false;
}

/// Comparison function for sorting suggestions (higher score first).
pub fn suggestionCompare(_: void, a: CompletionSuggestion, b: CompletionSuggestion) bool {
    return a.score > b.score;
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

test "recent history prefers newest prefix entries" {
    const no_args = &[_][:0]const u8{};
    const item = MenuItem{
        .label = "Ralph",
        .description = "desc",
        .action = .{ .command = .{
            .id = "ralph",
            .command = "ralph",
            .args = no_args,
        } },
        .category = .ai,
    };

    const history = [_]HistoryEntry{
        .{ .command_id = "ralph", .timestamp = 30 },
        .{ .command_id = "db", .timestamp = 20 },
        .{ .command_id = "bench", .timestamp = 10 },
    };
    try std.testing.expect(isRecentlyUsed(&item, &history));

    const old_only = [_]HistoryEntry{
        .{ .command_id = "db", .timestamp = 30 },
        .{ .command_id = "bench", .timestamp = 20 },
        .{ .command_id = "config", .timestamp = 10 },
        .{ .command_id = "gpu", .timestamp = 9 },
        .{ .command_id = "llm", .timestamp = 8 },
        .{ .command_id = "ralph", .timestamp = 1 },
    };
    try std.testing.expect(!isRecentlyUsed(&item, &old_only));
}

test {
    std.testing.refAllDecls(@This());
}
