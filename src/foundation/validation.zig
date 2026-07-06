const std = @import("std");

/// Validates that a slice is not empty.
/// @param src   The slice to validate
/// @return      Error.InvalidEmptySlice if src is empty, otherwise undefined
pub fn validateNonEmptySlice(src: []const u8) !void {
    if (src.len == 0) return error.InvalidEmptySlice;
}

/// Validates that a slice contains no null bytes.
/// @param src   The slice to validate
/// @return      Error.InvalidNullByte if src contains null bytes, otherwise undefined
pub fn validateNoNullBytes(src: []const u8) !void {
    if (std.mem.indexOfScalar(u8, src, 0) != null) return error.InvalidNullByte;
}

/// Validates that a string contains at least one of the specified substrings.
/// @param src         The string to validate
/// @param patterns    Array of substrings that must be present (at least one)
/// @return            Error.MissingRequiredPattern if none of the patterns are found, otherwise undefined
pub fn validateContainsOneOf(src: []const u8, patterns: [][]const u8) !void {
    for (patterns) |pattern| {
        if (std.mem.indexOf(u8, src, pattern) != null) return;
    }
    return error.MissingRequiredPattern;
}

/// Validates that a string does not contain any of the specified substrings.
/// @param src         The string to validate
/// @param forbidden   Array of substrings that must not be present
/// @return            Error.ForbiddenPatternFound if any forbidden pattern is found, otherwise undefined
pub fn validateDoesNotContain(src: []const u8, forbidden: [][]const u8) !void {
    for (forbidden) |pattern| {
        if (std.mem.indexOf(u8, src, pattern) != null) return error.ForbiddenPatternFound;
    }
}

/// Validates that a slice has a minimum length.
/// @param src     The slice to validate
/// @param minLen  The minimum required length
/// @return        Error.InvalidLength if src.len < minLen, otherwise undefined
pub fn validateMinLength(src: []const u8, minLen: usize) !void {
    if (src.len < minLen) return error.InvalidLength;
}

/// Validates that a slice has a maximum length.
/// @param src     The slice to validate
/// @param maxLen  The maximum allowed length
/// @return        Error.InvalidLength if src.len > maxLen, otherwise undefined
pub fn validateMaxLength(src: []const u8, maxLen: usize) !void {
    if (src.len > maxLen) return error.InvalidLength;
}

/// Validates that a slice has an exact length.
/// @param src     The slice to validate
/// @param exactLen The exact required length
/// @return        Error.InvalidLength if src.len != exactLen, otherwise undefined
pub fn validateExactLength(src: []const u8, exactLen: usize) !void {
    if (src.len != exactLen) return error.InvalidLength;
}

/// Returns true if `byte` is a hexadecimal digit (0-9, a-f, A-F).
pub fn isHexDigit(byte: u8) bool {
    return switch (byte) {
        '0'...'9', 'a'...'f', 'A'...'F' => true,
        else => false,
    };
}

/// Returns true if every byte in `src` is an ASCII digit (0-9).
pub fn isAllDigits(src: []const u8) bool {
    if (src.len == 0) return false;
    for (src) |byte| {
        if (byte < '0' or byte > '9') return false;
    }
    return true;
}

test "isHexDigit" {
    try std.testing.expect(isHexDigit('0'));
    try std.testing.expect(isHexDigit('9'));
    try std.testing.expect(isHexDigit('a'));
    try std.testing.expect(isHexDigit('F'));
    try std.testing.expect(!isHexDigit('g'));
    try std.testing.expect(!isHexDigit('z'));
    try std.testing.expect(!isHexDigit(' '));
}

test "isAllDigits" {
    try std.testing.expect(isAllDigits("12345"));
    try std.testing.expect(isAllDigits("0"));
    try std.testing.expect(!isAllDigits("12a45"));
    try std.testing.expect(!isAllDigits(""));
}

test "validateNonEmptySlice" {
    try std.testing.expectError(error.InvalidEmptySlice, validateNonEmptySlice(""));
    try validateNonEmptySlice("non-empty");
}

test "validateNoNullBytes" {
    try std.testing.expectError(error.InvalidNullByte, validateNoNullBytes("hello\x00world"));
    try validateNoNullBytes("hello world");
}

test "validateContainsOneOf" {
    const patterns1 = [_][]const u8{ "foo", "bar" };
    const patterns2 = [_][]const u8{ "hello", "xyz" };
    try std.testing.expectError(error.MissingRequiredPattern, validateContainsOneOf("hello world", @constCast(&patterns1)));
    try validateContainsOneOf("hello world", @constCast(&patterns2));
}

test "validateDoesNotContain" {
    const forbidden1 = [_][]const u8{"world"};
    const forbidden2 = [_][]const u8{ "world", "bar" };
    try std.testing.expectError(error.ForbiddenPatternFound, validateDoesNotContain("hello world", @constCast(&forbidden1)));
    try validateDoesNotContain("hello foo", @constCast(&forbidden2));
}

test "validateMinLength" {
    try std.testing.expectError(error.InvalidLength, validateMinLength("hi", 3));
    try validateMinLength("hello", 3);
}

test "validateMaxLength" {
    try std.testing.expectError(error.InvalidLength, validateMaxLength("hello world", 5));
    try validateMaxLength("hello", 5);
}

test "validateExactLength" {
    try std.testing.expectError(error.InvalidLength, validateExactLength("hi", 3));
    try std.testing.expectError(error.InvalidLength, validateExactLength("hello world", 5));
    try validateExactLength("hello", 5);
}

test {
    std.testing.refAllDecls(@This());
}
