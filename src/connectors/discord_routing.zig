//! Discord gateway command routing (WDBX Rust `src/features/discord.rs`).
//!
//! Pure, testable routing helpers: command parsing, reply text, and truncation.
//! No transport dependency — these functions only touch `std`.

const std = @import("std");

/// Replies are truncated to this many bytes (Discord message limit is 2000).
pub const MAX_MESSAGE_CONTENT_BYTES: usize = 1900;

/// Parsed Discord command from a message body.
pub const DiscordCommand = enum {
    help,
    status,
    prompt,
    governance,
    unknown,
    not_for_abbey,
};

fn helpText() []const u8 {
    return "Available commands: !help, !status, !prompt, !governance";
}

/// Deterministic local prompt summary (reference routing reply).
pub fn promptSummary() []const u8 {
    return "prompt summary: local persona routing (Abbey/Aviva/Abi) via keyword sentiment; completions recorded to WDBX.";
}

/// Deterministic local governance summary (reference routing reply).
pub fn governanceSummary() []const u8 {
    return "governance: six-principle constitutional audit (truthfulness, safety, helpfulness, fairness, privacy, transparency) with a weighted E-score and a hard safety-class veto.";
}

fn statusText(token_configured: bool) []const u8 {
    if (token_configured) return "status: connected (token configured)";
    return "status: offline (no token configured)";
}

/// Parse a command from a message body given a command prefix. An empty prefix
/// captures nothing (so the bot never replies to every message); a message
/// without the prefix is `not_for_abbey` (no reply).
pub fn parseDiscordCommand(content: []const u8, prefix: []const u8) DiscordCommand {
    if (prefix.len == 0) return .not_for_abbey;
    if (!std.mem.startsWith(u8, content, prefix)) return .not_for_abbey;
    const rest = content[prefix.len..];
    var it = std.mem.tokenizeScalar(u8, rest, ' ');
    const cmd = it.next() orelse return .unknown;
    if (std.ascii.eqlIgnoreCase(cmd, "help")) return .help;
    if (std.ascii.eqlIgnoreCase(cmd, "status")) return .status;
    if (std.ascii.eqlIgnoreCase(cmd, "prompt")) return .prompt;
    if (std.ascii.eqlIgnoreCase(cmd, "governance")) return .governance;
    return .unknown;
}

/// Route a parsed command to a reply string (owned). Returns `null` when no
/// reply should be sent (e.g. a message that is not for Abbey).
pub fn routeDiscordMessage(
    allocator: std.mem.Allocator,
    content: []const u8,
    prefix: []const u8,
    token_configured: bool,
) !?[]u8 {
    const cmd = parseDiscordCommand(content, prefix);
    return switch (cmd) {
        .not_for_abbey => null,
        .unknown => try allocator.dupe(u8, "Unknown command. Type `!help` for available commands."),
        .help => try allocator.dupe(u8, helpText()),
        .status => try allocator.dupe(u8, statusText(token_configured)),
        .prompt => try allocator.dupe(u8, promptSummary()),
        .governance => try allocator.dupe(u8, governanceSummary()),
    };
}

/// Truncate `text` to at most `max` bytes without splitting a UTF-8 sequence.
pub fn truncate(allocator: std.mem.Allocator, text: []const u8, max: usize) ![]u8 {
    if (text.len <= max) return try allocator.dupe(u8, text);
    var end = max;
    while (end > 0 and (text[end] & 0xC0) == 0x80) end -= 1; // back up over UTF-8 continuation
    return try allocator.dupe(u8, text[0..end]);
}

test {
    std.testing.refAllDecls(@This());
}

test "parseDiscordCommand honors prefix and empty-prefix safety" {
    try std.testing.expect(parseDiscordCommand("!help", "!") == .help);
    try std.testing.expect(parseDiscordCommand("!status", "!") == .status);
    try std.testing.expect(parseDiscordCommand("!prompt", "!") == .prompt);
    try std.testing.expect(parseDiscordCommand("!governance", "!") == .governance);
    try std.testing.expect(parseDiscordCommand("!nonsense", "!") == .unknown);
    // No prefix -> never captures (avoids replying to every message).
    try std.testing.expect(parseDiscordCommand("help", "") == .not_for_abbey);
    // Prefix absent -> not for abbey.
    try std.testing.expect(parseDiscordCommand("help me", "!") == .not_for_abbey);
}

test "routeDiscordMessage produces reply strings and null for non-abbey" {
    const allocator = std.testing.allocator;
    const none = try routeDiscordMessage(allocator, "hello there", "!", true);
    try std.testing.expect(none == null);
    const status = (try routeDiscordMessage(allocator, "!status", "!", true)) orelse return error.UnexpectedNull;
    defer allocator.free(status);
    try std.testing.expect(std.mem.indexOf(u8, status, "connected") != null);
    const status_off = (try routeDiscordMessage(allocator, "!status", "!", false)) orelse return error.UnexpectedNull;
    defer allocator.free(status_off);
    try std.testing.expect(std.mem.indexOf(u8, status_off, "offline") != null);
    const gov = (try routeDiscordMessage(allocator, "!governance", "!", true)) orelse return error.UnexpectedNull;
    defer allocator.free(gov);
    try std.testing.expect(std.mem.indexOf(u8, gov, "constitutional") != null);
}

test "truncate respects byte and utf8 boundaries" {
    const allocator = std.testing.allocator;
    const short = try truncate(allocator, "hello", 1900);
    defer allocator.free(short);
    try std.testing.expectEqualStrings("hello", short);

    // A multi-byte char at the boundary must not be split.
    const t = "ééééé"; // 5 bytes * 2 = 10 bytes total; truncate to 5 -> 2 chars (4 bytes).
    const cut = try truncate(allocator, t, 5);
    defer allocator.free(cut);
    try std.testing.expectEqual(@as(usize, 4), cut.len);
    try std.testing.expectEqualStrings("éé", cut);
}
