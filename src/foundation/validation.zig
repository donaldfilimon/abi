const std = @import("std");

/// Validates that a slice is not empty.
/// @param name  The name of the field being validated (for error messages)
/// @param src   The slice to validate
/// @return      Error.InvalidEmptySlice if src is empty, otherwise undefined
pub fn validateNonEmptySlice(name: []const u8, src: []const u8) !void {
    _ = name;
    if (src.len == 0) return error.InvalidEmptySlice;
}

/// Validates that a slice contains no null bytes.
/// @param name  The name of the field being validated (for error messages)
/// @param src   The slice to validate
/// @return      Error.InvalidNullByte if src contains null bytes, otherwise undefined
pub fn validateNoNullBytes(name: []const u8, src: []const u8) !void {
    _ = name;
    if (std.mem.indexOfScalar(u8, src, 0) != null) return error.InvalidNullByte;
}

/// Validates that a string contains at least one of the specified substrings.
/// @param name        The name of the field being validated (for error messages)
/// @param src         The string to validate
/// @param patterns    Array of substrings that must be present (at least one)
/// @return            Error.MissingRequiredPattern if none of the patterns are found, otherwise undefined
pub fn validateContainsOneOf(name: []const u8, src: []const u8, patterns: [][]const u8) !void {
    _ = name;
    for (patterns) |pattern| {
        if (std.mem.indexOf(u8, src, pattern) != null) return;
    }
    return error.MissingRequiredPattern;
}

/// Validates that a string does not contain any of the specified substrings.
/// @param name        The name of the field being validated (for error messages)
/// @param src         The string to validate
/// @param forbidden   Array of substrings that must not be present
/// @return            Error.ForbiddenPatternFound if any forbidden pattern is found, otherwise undefined
pub fn validateDoesNotContain(name: []const u8, src: []const u8, forbidden: [][]const u8) !void {
    _ = name;
    for (forbidden) |pattern| {
        if (std.mem.indexOf(u8, src, pattern) != null) return error.ForbiddenPatternFound;
    }
}

/// Validates that a slice has a minimum length.
/// @param name    The name of the field being validated (for error messages)
/// @param src     The slice to validate
/// @param minLen  The minimum required length
/// @return        Error.InvalidLength if src.len < minLen, otherwise undefined
pub fn validateMinLength(name: []const u8, src: []const u8, minLen: usize) !void {
    _ = name;
    if (src.len < minLen) return error.InvalidLength;
}

/// Validates that a slice has a maximum length.
/// @param name    The name of the field being validated (for error messages)
/// @param src     The slice to validate
/// @param maxLen  The maximum allowed length
/// @return        Error.InvalidLength if src.len > maxLen, otherwise undefined
pub fn validateMaxLength(name: []const u8, src: []const u8, maxLen: usize) !void {
    _ = name;
    if (src.len > maxLen) return error.InvalidLength;
}

/// Validates that a slice has an exact length.
/// @param name    The name of the field being validated (for error messages)
/// @param src     The slice to validate
/// @param exactLen The exact required length
/// @return        Error.InvalidLength if src.len != exactLen, otherwise undefined
pub fn validateExactLength(name: []const u8, src: []const u8, exactLen: usize) !void {
    _ = name;
    if (src.len != exactLen) return error.InvalidLength;
}

test "validateNonEmptySlice" {
    try std.testing.expectError(error.InvalidEmptySlice, validateNonEmptySlice("test", ""));
    try validateNonEmptySlice("test", "non-empty");
}

test "validateNoNullBytes" {
    try std.testing.expectError(error.InvalidNullByte, validateNoNullBytes("test", "hello\x00world"));
    try validateNoNullBytes("test", "hello world");
}

test "validateContainsOneOf" {
    const patterns1 = [_][]const u8{ "foo", "bar" };
    const patterns2 = [_][]const u8{ "hello", "xyz" };
    try std.testing.expectError(error.MissingRequiredPattern, validateContainsOneOf("test", "hello world", @constCast(&patterns1)));
    try validateContainsOneOf("test", "hello world", @constCast(&patterns2));
}

test "validateDoesNotContain" {
    const forbidden1 = [_][]const u8{"world"};
    const forbidden2 = [_][]const u8{ "world", "bar" };
    try std.testing.expectError(error.ForbiddenPatternFound, validateDoesNotContain("test", "hello world", @constCast(&forbidden1)));
    try validateDoesNotContain("test", "hello foo", @constCast(&forbidden2));
}

test "validateMinLength" {
    try std.testing.expectError(error.InvalidLength, validateMinLength("test", "hi", 3));
    try validateMinLength("test", "hello", 3);
}

test "validateMaxLength" {
    try std.testing.expectError(error.InvalidLength, validateMaxLength("test", "hello world", 5));
    try validateMaxLength("test", "hello", 5);
}

test "validateExactLength" {
    try std.testing.expectError(error.InvalidLength, validateExactLength("test", "hi", 3));
    try std.testing.expectError(error.InvalidLength, validateExactLength("test", "hello world", 5));
    try validateExactLength("test", "hello", 5);
}
