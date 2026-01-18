//! Discord Utility Functions
//!
//! Helper functions for Discord operations including:
//! - Timestamp parsing and formatting
//! - Permission calculations
//! - Discord permission constants

const std = @import("std");

// ============================================================================
// Timestamp Utilities
// ============================================================================

/// Discord timestamp styles for formatting
pub const TimestampStyle = enum {
    SHORT_TIME, // 16:20
    LONG_TIME, // 16:20:30
    SHORT_DATE, // 20/04/2021
    LONG_DATE, // 20 April 2021
    SHORT_DATE_TIME, // 20 April 2021 16:20
    LONG_DATE_TIME, // Tuesday, 20 April 2021 16:20
    RELATIVE, // 2 months ago
};

/// Parse a Discord timestamp to Unix timestamp (seconds).
pub fn parseTimestamp(iso_timestamp: []const u8) !i64 {
    if (iso_timestamp.len == 0) return error.InvalidTimestamp;

    // Handle Discord timestamp format: <t:UNIX:STYLE>
    if (std.mem.startsWith(u8, iso_timestamp, "<t:")) {
        if (iso_timestamp[iso_timestamp.len - 1] != '>') return error.InvalidTimestamp;
        const inner = iso_timestamp[3 .. iso_timestamp.len - 1];
        const sep = std.mem.indexOfScalar(u8, inner, ':');
        const ts_slice = inner[0 .. sep orelse inner.len];
        return std.fmt.parseInt(i64, ts_slice, 10) catch return error.InvalidTimestamp;
    }

    // Handle ISO 8601 format
    if (std.mem.indexOfScalar(u8, iso_timestamp, 'T') != null) {
        return parseIso8601(iso_timestamp);
    }

    if (std.mem.indexOfScalar(u8, iso_timestamp, '-') != null) {
        return parseIso8601(iso_timestamp);
    }

    // Try parsing as plain Unix timestamp
    return std.fmt.parseInt(i64, iso_timestamp, 10) catch return error.InvalidTimestamp;
}

/// Format a Unix timestamp to Discord timestamp format.
pub fn formatTimestamp(
    allocator: std.mem.Allocator,
    unix_timestamp: i64,
    style: TimestampStyle,
) ![]u8 {
    const style_code: u8 = switch (style) {
        .SHORT_TIME => 't',
        .LONG_TIME => 'T',
        .SHORT_DATE => 'd',
        .LONG_DATE => 'D',
        .SHORT_DATE_TIME => 'f',
        .LONG_DATE_TIME => 'F',
        .RELATIVE => 'R',
    };
    return std.fmt.allocPrint(
        allocator,
        "<t:{d}:{c}>",
        .{ unix_timestamp, style_code },
    );
}

fn parseIso8601(iso: []const u8) !i64 {
    if (iso.len < 19) return error.InvalidTimestamp;
    if (iso[4] != '-' or iso[7] != '-' or
        (iso[10] != 'T' and iso[10] != 't' and iso[10] != ' ') or
        iso[13] != ':' or iso[16] != ':')
    {
        return error.InvalidTimestamp;
    }

    const year = try parseIntPart(iso[0..4]);
    const month = try parseIntPart(iso[5..7]);
    const day = try parseIntPart(iso[8..10]);
    const hour = try parseIntPart(iso[11..13]);
    const minute = try parseIntPart(iso[14..16]);
    const second = try parseIntPart(iso[17..19]);

    if (month < 1 or month > 12) return error.InvalidTimestamp;
    if (hour < 0 or hour > 23) return error.InvalidTimestamp;
    if (minute < 0 or minute > 59) return error.InvalidTimestamp;
    if (second < 0 or second > 60) return error.InvalidTimestamp;
    const max_day = daysInMonth(year, month);
    if (day < 1 or day > max_day) return error.InvalidTimestamp;

    var idx: usize = 19;
    // Skip fractional seconds
    if (idx < iso.len and iso[idx] == '.') {
        idx += 1;
        while (idx < iso.len and std.ascii.isDigit(iso[idx])) : (idx += 1) {}
    }

    var offset_seconds: i64 = 0;
    if (idx < iso.len) {
        const tz = iso[idx];
        if (tz == 'Z' or tz == 'z') {
            idx += 1;
        } else if (tz == '+' or tz == '-') {
            const sign: i64 = if (tz == '-') -1 else 1;
            idx += 1;
            if (idx + 1 >= iso.len) return error.InvalidTimestamp;
            const tz_hour = try parseIntPart(iso[idx .. idx + 2]);
            idx += 2;
            if (idx < iso.len and iso[idx] == ':') idx += 1;
            if (idx + 1 >= iso.len) return error.InvalidTimestamp;
            const tz_minute = try parseIntPart(iso[idx .. idx + 2]);
            offset_seconds = sign * ((tz_hour * 3600) + (tz_minute * 60));
        }
    }

    const days = daysFromCivil(year, month, day);
    const timestamp = (days * 86400) + (hour * 3600) + (minute * 60) + second;
    return timestamp - offset_seconds;
}

fn parseIntPart(slice: []const u8) !i64 {
    return std.fmt.parseInt(i64, slice, 10) catch return error.InvalidTimestamp;
}

fn isLeapYear(year: i64) bool {
    if (@mod(year, 4) != 0) return false;
    if (@mod(year, 100) != 0) return true;
    return @mod(year, 400) == 0;
}

fn daysInMonth(year: i64, month: i64) i64 {
    return switch (month) {
        1, 3, 5, 7, 8, 10, 12 => 31,
        4, 6, 9, 11 => 30,
        2 => if (isLeapYear(year)) 29 else 28,
        else => 0,
    };
}

fn daysFromCivil(year: i64, month: i64, day: i64) i64 {
    var y = year;
    const m = month;
    y -= if (m <= 2) @as(i64, 1) else @as(i64, 0);
    const era = @divFloor(y, 400);
    const yoe = y - era * 400;
    const mp = if (m > 2) m - 3 else m + 9;
    const doy = @divFloor(153 * mp + 2, 5) + day - 1;
    const doe = yoe * 365 + @divFloor(yoe, 4) - @divFloor(yoe, 100) + doy;
    return era * 146097 + doe - 719468;
}

// ============================================================================
// Permission Utilities
// ============================================================================

/// Calculate permissions from an array of permission flags
pub fn calculatePermissions(permissions: []const u64) u64 {
    var result: u64 = 0;
    for (permissions) |p| {
        result |= p;
    }
    return result;
}

/// Check if a permission is set
pub fn hasPermission(permissions: u64, permission: u64) bool {
    return (permissions & permission) == permission;
}

/// Discord Permissions
pub const Permission = struct {
    pub const CREATE_INSTANT_INVITE: u64 = 1 << 0;
    pub const KICK_MEMBERS: u64 = 1 << 1;
    pub const BAN_MEMBERS: u64 = 1 << 2;
    pub const ADMINISTRATOR: u64 = 1 << 3;
    pub const MANAGE_CHANNELS: u64 = 1 << 4;
    pub const MANAGE_GUILD: u64 = 1 << 5;
    pub const ADD_REACTIONS: u64 = 1 << 6;
    pub const VIEW_AUDIT_LOG: u64 = 1 << 7;
    pub const PRIORITY_SPEAKER: u64 = 1 << 8;
    pub const STREAM: u64 = 1 << 9;
    pub const VIEW_CHANNEL: u64 = 1 << 10;
    pub const SEND_MESSAGES: u64 = 1 << 11;
    pub const SEND_TTS_MESSAGES: u64 = 1 << 12;
    pub const MANAGE_MESSAGES: u64 = 1 << 13;
    pub const EMBED_LINKS: u64 = 1 << 14;
    pub const ATTACH_FILES: u64 = 1 << 15;
    pub const READ_MESSAGE_HISTORY: u64 = 1 << 16;
    pub const MENTION_EVERYONE: u64 = 1 << 17;
    pub const USE_EXTERNAL_EMOJIS: u64 = 1 << 18;
    pub const VIEW_GUILD_INSIGHTS: u64 = 1 << 19;
    pub const CONNECT: u64 = 1 << 20;
    pub const SPEAK: u64 = 1 << 21;
    pub const MUTE_MEMBERS: u64 = 1 << 22;
    pub const DEAFEN_MEMBERS: u64 = 1 << 23;
    pub const MOVE_MEMBERS: u64 = 1 << 24;
    pub const USE_VAD: u64 = 1 << 25;
    pub const CHANGE_NICKNAME: u64 = 1 << 26;
    pub const MANAGE_NICKNAMES: u64 = 1 << 27;
    pub const MANAGE_ROLES: u64 = 1 << 28;
    pub const MANAGE_WEBHOOKS: u64 = 1 << 29;
    pub const MANAGE_GUILD_EXPRESSIONS: u64 = 1 << 30;
    pub const USE_APPLICATION_COMMANDS: u64 = 1 << 31;
    pub const REQUEST_TO_SPEAK: u64 = 1 << 32;
    pub const MANAGE_EVENTS: u64 = 1 << 33;
    pub const MANAGE_THREADS: u64 = 1 << 34;
    pub const CREATE_PUBLIC_THREADS: u64 = 1 << 35;
    pub const CREATE_PRIVATE_THREADS: u64 = 1 << 36;
    pub const USE_EXTERNAL_STICKERS: u64 = 1 << 37;
    pub const SEND_MESSAGES_IN_THREADS: u64 = 1 << 38;
    pub const USE_EMBEDDED_ACTIVITIES: u64 = 1 << 39;
    pub const MODERATE_MEMBERS: u64 = 1 << 40;
    pub const VIEW_CREATOR_MONETIZATION_ANALYTICS: u64 = 1 << 41;
    pub const USE_SOUNDBOARD: u64 = 1 << 42;
    pub const CREATE_GUILD_EXPRESSIONS: u64 = 1 << 43;
    pub const CREATE_EVENTS: u64 = 1 << 44;
    pub const USE_EXTERNAL_SOUNDS: u64 = 1 << 45;
    pub const SEND_VOICE_MESSAGES: u64 = 1 << 46;
    pub const SEND_POLLS: u64 = 1 << 49;
    pub const USE_EXTERNAL_APPS: u64 = 1 << 50;
};
