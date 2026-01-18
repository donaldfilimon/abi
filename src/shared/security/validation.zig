//! Input validation and sanitization layer for security.
//!
//! This module provides:
//! - Input validation for common data types
//! - SQL injection prevention
//! - XSS sanitization
//! - Path traversal prevention
//! - Command injection prevention
//! - Email, URL, and other format validation
//! - Custom validation rules
//! - Sanitization utilities

const std = @import("std");

/// Validation error types
pub const ValidationError = error{
    InvalidFormat,
    TooShort,
    TooLong,
    ContainsInvalidChars,
    PathTraversal,
    SqlInjection,
    XssDetected,
    CommandInjection,
    InvalidEmail,
    InvalidUrl,
    InvalidIpAddress,
    InvalidJson,
    NullBytes,
    UnicodeOverflow,
    OutOfRange,
    PatternMismatch,
};

/// Validation result with details
pub const ValidationResult = struct {
    valid: bool,
    errors: []const ValidationErrorDetail = &.{},
    sanitized: ?[]const u8 = null,

    pub const ValidationErrorDetail = struct {
        field: []const u8,
        message: []const u8,
        code: []const u8,
    };
};

/// Validator configuration
pub const ValidatorConfig = struct {
    /// Maximum string length (0 = unlimited)
    max_length: usize = 65536,
    /// Minimum string length
    min_length: usize = 0,
    /// Allow null bytes
    allow_null_bytes: bool = false,
    /// Allow control characters
    allow_control_chars: bool = false,
    /// Allow unicode outside BMP
    allow_extended_unicode: bool = true,
    /// Trim whitespace
    trim_whitespace: bool = true,
    /// Normalize unicode (NFC)
    normalize_unicode: bool = true,
    /// Check for SQL injection patterns
    check_sql_injection: bool = true,
    /// Check for XSS patterns
    check_xss: bool = true,
    /// Check for command injection
    check_command_injection: bool = true,
    /// Check for path traversal
    check_path_traversal: bool = true,
};

/// Input validator
pub const Validator = struct {
    allocator: std.mem.Allocator,
    config: ValidatorConfig,

    pub fn init(allocator: std.mem.Allocator, config: ValidatorConfig) Validator {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Validate and sanitize a string input
    pub fn validateString(self: *Validator, input: []const u8) ValidationResult {
        // Length checks
        if (input.len < self.config.min_length) {
            return .{
                .valid = false,
                .errors = &.{.{
                    .field = "input",
                    .message = "Input too short",
                    .code = "TOO_SHORT",
                }},
            };
        }

        if (self.config.max_length > 0 and input.len > self.config.max_length) {
            return .{
                .valid = false,
                .errors = &.{.{
                    .field = "input",
                    .message = "Input too long",
                    .code = "TOO_LONG",
                }},
            };
        }

        // Null byte check
        if (!self.config.allow_null_bytes) {
            for (input) |byte| {
                if (byte == 0) {
                    return .{
                        .valid = false,
                        .errors = &.{.{
                            .field = "input",
                            .message = "Null bytes not allowed",
                            .code = "NULL_BYTES",
                        }},
                    };
                }
            }
        }

        // Control character check
        if (!self.config.allow_control_chars) {
            for (input) |byte| {
                if (byte < 32 and byte != '\t' and byte != '\n' and byte != '\r') {
                    return .{
                        .valid = false,
                        .errors = &.{.{
                            .field = "input",
                            .message = "Control characters not allowed",
                            .code = "CONTROL_CHARS",
                        }},
                    };
                }
            }
        }

        // SQL injection check
        if (self.config.check_sql_injection and containsSqlInjection(input)) {
            return .{
                .valid = false,
                .errors = &.{.{
                    .field = "input",
                    .message = "Potential SQL injection detected",
                    .code = "SQL_INJECTION",
                }},
            };
        }

        // XSS check
        if (self.config.check_xss and containsXss(input)) {
            return .{
                .valid = false,
                .errors = &.{.{
                    .field = "input",
                    .message = "Potential XSS detected",
                    .code = "XSS_DETECTED",
                }},
            };
        }

        // Command injection check
        if (self.config.check_command_injection and containsCommandInjection(input)) {
            return .{
                .valid = false,
                .errors = &.{.{
                    .field = "input",
                    .message = "Potential command injection detected",
                    .code = "COMMAND_INJECTION",
                }},
            };
        }

        // Path traversal check
        if (self.config.check_path_traversal and containsPathTraversal(input)) {
            return .{
                .valid = false,
                .errors = &.{.{
                    .field = "input",
                    .message = "Path traversal attempt detected",
                    .code = "PATH_TRAVERSAL",
                }},
            };
        }

        return .{ .valid = true };
    }

    /// Validate an email address
    pub fn validateEmail(self: *Validator, email: []const u8) ValidationResult {
        _ = self;

        if (email.len == 0 or email.len > 254) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "email", .message = "Invalid email length", .code = "INVALID_EMAIL" }},
            };
        }

        // Find @ symbol
        const at_idx = std.mem.indexOf(u8, email, "@") orelse {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "email", .message = "Missing @ symbol", .code = "INVALID_EMAIL" }},
            };
        };

        // Check local part
        const local = email[0..at_idx];
        if (local.len == 0 or local.len > 64) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "email", .message = "Invalid local part", .code = "INVALID_EMAIL" }},
            };
        }

        // Check domain part
        const domain = email[at_idx + 1 ..];
        if (domain.len == 0 or domain.len > 253) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "email", .message = "Invalid domain", .code = "INVALID_EMAIL" }},
            };
        }

        // Domain must contain at least one dot
        if (std.mem.indexOf(u8, domain, ".") == null) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "email", .message = "Invalid domain format", .code = "INVALID_EMAIL" }},
            };
        }

        // Check for invalid characters in local part
        for (local) |c| {
            if (!isValidEmailLocalChar(c)) {
                return .{
                    .valid = false,
                    .errors = &.{.{ .field = "email", .message = "Invalid character in local part", .code = "INVALID_EMAIL" }},
                };
            }
        }

        return .{ .valid = true };
    }

    /// Validate a URL
    pub fn validateUrl(self: *Validator, url: []const u8) ValidationResult {
        _ = self;

        if (url.len == 0 or url.len > 2048) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "url", .message = "Invalid URL length", .code = "INVALID_URL" }},
            };
        }

        // Check for valid scheme
        const valid_schemes = &[_][]const u8{ "http://", "https://", "ftp://", "ftps://" };
        var has_valid_scheme = false;
        for (valid_schemes) |scheme| {
            if (std.mem.startsWith(u8, url, scheme)) {
                has_valid_scheme = true;
                break;
            }
        }

        if (!has_valid_scheme) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "url", .message = "Invalid or missing URL scheme", .code = "INVALID_URL" }},
            };
        }

        // Check for dangerous protocols in data URIs
        if (std.mem.indexOf(u8, url, "javascript:") != null or
            std.mem.indexOf(u8, url, "data:") != null or
            std.mem.indexOf(u8, url, "vbscript:") != null)
        {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "url", .message = "Dangerous URL protocol", .code = "INVALID_URL" }},
            };
        }

        return .{ .valid = true };
    }

    /// Validate an IP address (v4 or v6)
    pub fn validateIpAddress(self: *Validator, ip: []const u8) ValidationResult {
        _ = self;

        if (isValidIpv4(ip) or isValidIpv6(ip)) {
            return .{ .valid = true };
        }

        return .{
            .valid = false,
            .errors = &.{.{ .field = "ip", .message = "Invalid IP address", .code = "INVALID_IP" }},
        };
    }

    /// Validate a file path (no traversal)
    pub fn validatePath(self: *Validator, path: []const u8) ValidationResult {
        if (self.config.check_path_traversal and containsPathTraversal(path)) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "path", .message = "Path traversal detected", .code = "PATH_TRAVERSAL" }},
            };
        }

        // Check for null bytes
        for (path) |byte| {
            if (byte == 0) {
                return .{
                    .valid = false,
                    .errors = &.{.{ .field = "path", .message = "Null byte in path", .code = "NULL_BYTES" }},
                };
            }
        }

        return .{ .valid = true };
    }

    /// Validate an integer within range
    pub fn validateInteger(
        self: *Validator,
        value: i64,
        min: ?i64,
        max: ?i64,
    ) ValidationResult {
        _ = self;

        if (min) |m| {
            if (value < m) {
                return .{
                    .valid = false,
                    .errors = &.{.{ .field = "value", .message = "Value below minimum", .code = "OUT_OF_RANGE" }},
                };
            }
        }

        if (max) |m| {
            if (value > m) {
                return .{
                    .valid = false,
                    .errors = &.{.{ .field = "value", .message = "Value above maximum", .code = "OUT_OF_RANGE" }},
                };
            }
        }

        return .{ .valid = true };
    }

    /// Validate JSON string
    pub fn validateJson(self: *Validator, json: []const u8) ValidationResult {
        _ = self;

        // Simple validation - try to find balanced braces/brackets
        var brace_count: i32 = 0;
        var bracket_count: i32 = 0;
        var in_string = false;
        var escape_next = false;

        for (json) |c| {
            if (escape_next) {
                escape_next = false;
                continue;
            }

            if (c == '\\' and in_string) {
                escape_next = true;
                continue;
            }

            if (c == '"') {
                in_string = !in_string;
                continue;
            }

            if (!in_string) {
                switch (c) {
                    '{' => brace_count += 1,
                    '}' => brace_count -= 1,
                    '[' => bracket_count += 1,
                    ']' => bracket_count -= 1,
                    else => {},
                }

                if (brace_count < 0 or bracket_count < 0) {
                    return .{
                        .valid = false,
                        .errors = &.{.{ .field = "json", .message = "Unbalanced JSON structure", .code = "INVALID_JSON" }},
                    };
                }
            }
        }

        if (brace_count != 0 or bracket_count != 0 or in_string) {
            return .{
                .valid = false,
                .errors = &.{.{ .field = "json", .message = "Invalid JSON structure", .code = "INVALID_JSON" }},
            };
        }

        return .{ .valid = true };
    }
};

/// Sanitizer for cleaning potentially dangerous input
pub const Sanitizer = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Sanitizer {
        return .{ .allocator = allocator };
    }

    /// Sanitize string for HTML output (prevent XSS)
    pub fn sanitizeHtml(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        for (input) |c| {
            switch (c) {
                '<' => try result.appendSlice("&lt;"),
                '>' => try result.appendSlice("&gt;"),
                '&' => try result.appendSlice("&amp;"),
                '"' => try result.appendSlice("&quot;"),
                '\'' => try result.appendSlice("&#x27;"),
                '/' => try result.appendSlice("&#x2F;"),
                '`' => try result.appendSlice("&#x60;"),
                '=' => try result.appendSlice("&#x3D;"),
                else => try result.append(c),
            }
        }

        return result.toOwnedSlice();
    }

    /// Sanitize string for SQL (escape single quotes)
    pub fn sanitizeSql(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        for (input) |c| {
            switch (c) {
                '\'' => try result.appendSlice("''"),
                '\\' => try result.appendSlice("\\\\"),
                0 => try result.appendSlice("\\0"),
                '\n' => try result.appendSlice("\\n"),
                '\r' => try result.appendSlice("\\r"),
                else => try result.append(c),
            }
        }

        return result.toOwnedSlice();
    }

    /// Sanitize string for shell commands
    pub fn sanitizeShell(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        // Wrap in single quotes and escape existing single quotes
        try result.append('\'');
        for (input) |c| {
            if (c == '\'') {
                try result.appendSlice("'\\''");
            } else {
                try result.append(c);
            }
        }
        try result.append('\'');

        return result.toOwnedSlice();
    }

    /// Sanitize filename (remove path components and dangerous chars)
    pub fn sanitizeFilename(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        // Find the actual filename (after last path separator)
        var start: usize = 0;
        for (input, 0..) |c, i| {
            if (c == '/' or c == '\\') {
                start = i + 1;
            }
        }

        const filename = if (start < input.len) input[start..] else input;

        // Remove dangerous characters
        for (filename) |c| {
            if (isValidFilenameChar(c)) {
                try result.append(c);
            } else {
                try result.append('_');
            }
        }

        // Don't allow empty result
        if (result.items.len == 0) {
            try result.appendSlice("unnamed");
        }

        // Don't allow special names
        const dangerous_names = &[_][]const u8{
            ".", "..", "CON", "PRN", "AUX", "NUL",
            "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
            "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
        };

        for (dangerous_names) |name| {
            if (std.ascii.eqlIgnoreCase(result.items, name)) {
                try result.appendSlice("_safe");
                break;
            }
        }

        return result.toOwnedSlice();
    }

    /// Sanitize URL path component
    pub fn sanitizeUrlPath(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        for (input) |c| {
            if (isValidUrlPathChar(c)) {
                try result.append(c);
            } else {
                // Percent-encode
                try result.append('%');
                const hex = "0123456789ABCDEF";
                try result.append(hex[c >> 4]);
                try result.append(hex[c & 0x0F]);
            }
        }

        return result.toOwnedSlice();
    }

    /// Remove all HTML tags from input
    pub fn stripHtmlTags(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        var in_tag = false;
        for (input) |c| {
            if (c == '<') {
                in_tag = true;
            } else if (c == '>') {
                in_tag = false;
            } else if (!in_tag) {
                try result.append(c);
            }
        }

        return result.toOwnedSlice();
    }

    /// Normalize whitespace (collapse multiple spaces, trim)
    pub fn normalizeWhitespace(self: *Sanitizer, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(self.allocator);
        errdefer result.deinit();

        var last_was_space = true; // Trim leading
        for (input) |c| {
            const is_space = std.ascii.isWhitespace(c);
            if (is_space) {
                if (!last_was_space) {
                    try result.append(' ');
                    last_was_space = true;
                }
            } else {
                try result.append(c);
                last_was_space = false;
            }
        }

        // Trim trailing
        while (result.items.len > 0 and result.items[result.items.len - 1] == ' ') {
            _ = result.pop();
        }

        return result.toOwnedSlice();
    }
};

// Helper functions

fn containsSqlInjection(input: []const u8) bool {
    const patterns = &[_][]const u8{
        "'--",
        "'; --",
        "' OR ",
        "' AND ",
        "1=1",
        "1'='1",
        "' OR '1'='1",
        "UNION SELECT",
        "UNION ALL SELECT",
        "INSERT INTO",
        "DELETE FROM",
        "DROP TABLE",
        "DROP DATABASE",
        "TRUNCATE TABLE",
        "UPDATE SET",
        "xp_cmdshell",
        "EXEC(",
        "EXECUTE(",
        "WAITFOR DELAY",
        "BENCHMARK(",
        "SLEEP(",
        "/*",
        "*/",
        "@@",
        "CHAR(",
        "CONCAT(",
        "LOAD_FILE(",
        "INTO OUTFILE",
        "INTO DUMPFILE",
    };

    const lower = std.ascii.lowerString(input) catch return false;

    for (patterns) |pattern| {
        const lower_pattern = std.ascii.lowerString(pattern) catch continue;
        if (std.mem.indexOf(u8, lower, lower_pattern) != null) {
            return true;
        }
    }

    return false;
}

fn containsXss(input: []const u8) bool {
    const patterns = &[_][]const u8{
        "<script",
        "</script",
        "javascript:",
        "vbscript:",
        "onload=",
        "onerror=",
        "onclick=",
        "onmouseover=",
        "onfocus=",
        "onblur=",
        "onsubmit=",
        "onkeydown=",
        "onkeyup=",
        "onkeypress=",
        "eval(",
        "expression(",
        "document.cookie",
        "document.domain",
        "document.write",
        "window.location",
        "innerHTML",
        "outerHTML",
        "fromCharCode",
        "String.fromCharCode",
        "&#",
        "\\u00",
        "\\x",
        "<iframe",
        "<object",
        "<embed",
        "<form",
        "<input",
        "<img src=",
        "<svg",
        "<math",
        "data:",
        "base64,",
    };

    const lower = std.ascii.lowerString(input) catch return false;

    for (patterns) |pattern| {
        const lower_pattern = std.ascii.lowerString(pattern) catch continue;
        if (std.mem.indexOf(u8, lower, lower_pattern) != null) {
            return true;
        }
    }

    return false;
}

fn containsCommandInjection(input: []const u8) bool {
    const patterns = &[_][]const u8{
        ";",
        "|",
        "&",
        "$(",
        "`",
        "$((",
        "&&",
        "||",
        "\n",
        "\r",
        "$(IFS",
        "${IFS",
        ">/dev/",
        "</dev/",
        ">>",
        "<<",
        "2>&1",
        "1>&2",
        "/etc/passwd",
        "/etc/shadow",
        "wget ",
        "curl ",
        "nc ",
        "netcat ",
        "bash ",
        "sh ",
        "python ",
        "perl ",
        "ruby ",
        "php ",
    };

    for (patterns) |pattern| {
        if (std.mem.indexOf(u8, input, pattern) != null) {
            return true;
        }
    }

    return false;
}

fn containsPathTraversal(input: []const u8) bool {
    const patterns = &[_][]const u8{
        "..",
        "..\\",
        "../",
        "....//",
        "....\\\\",
        "%2e%2e",
        "%252e%252e",
        "%c0%ae",
        "%c1%9c",
        "..%00",
        "..%0d",
        "..%0a",
        "/etc/",
        "\\windows\\",
        "\\system32\\",
        "/proc/",
        "/dev/",
    };

    const lower = std.ascii.lowerString(input) catch return false;

    for (patterns) |pattern| {
        const lower_pattern = std.ascii.lowerString(pattern) catch continue;
        if (std.mem.indexOf(u8, lower, lower_pattern) != null) {
            return true;
        }
    }

    return false;
}

fn isValidEmailLocalChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or
        c == '.' or c == '_' or c == '-' or c == '+';
}

fn isValidIpv4(ip: []const u8) bool {
    var parts: usize = 0;
    var current_value: u32 = 0;
    var has_digit = false;

    for (ip) |c| {
        if (c == '.') {
            if (!has_digit or current_value > 255) return false;
            parts += 1;
            current_value = 0;
            has_digit = false;
        } else if (c >= '0' and c <= '9') {
            current_value = current_value * 10 + (c - '0');
            has_digit = true;
        } else {
            return false;
        }
    }

    return parts == 3 and has_digit and current_value <= 255;
}

fn isValidIpv6(ip: []const u8) bool {
    // Simplified IPv6 validation
    var colon_count: usize = 0;
    var double_colon = false;

    for (ip, 0..) |c, i| {
        if (c == ':') {
            colon_count += 1;
            if (i + 1 < ip.len and ip[i + 1] == ':') {
                if (double_colon) return false; // Only one :: allowed
                double_colon = true;
            }
        } else if (!std.ascii.isHex(c)) {
            return false;
        }
    }

    // Valid IPv6 has 7 colons (8 groups) or fewer with ::
    return colon_count >= 2 and colon_count <= 7;
}

fn isValidFilenameChar(c: u8) bool {
    // Allow alphanumeric, dot, dash, underscore
    return std.ascii.isAlphanumeric(c) or c == '.' or c == '-' or c == '_' or c == ' ';
}

fn isValidUrlPathChar(c: u8) bool {
    // RFC 3986 unreserved characters plus /
    return std.ascii.isAlphanumeric(c) or
        c == '-' or c == '.' or c == '_' or c == '~' or c == '/';
}

// Tests

test "validate email" {
    const allocator = std.testing.allocator;
    var validator = Validator.init(allocator, .{});

    // Valid emails
    try std.testing.expect(validator.validateEmail("test@example.com").valid);
    try std.testing.expect(validator.validateEmail("user.name@domain.org").valid);
    try std.testing.expect(validator.validateEmail("user+tag@example.co.uk").valid);

    // Invalid emails
    try std.testing.expect(!validator.validateEmail("").valid);
    try std.testing.expect(!validator.validateEmail("notanemail").valid);
    try std.testing.expect(!validator.validateEmail("@nodomain.com").valid);
    try std.testing.expect(!validator.validateEmail("noat.com").valid);
}

test "validate url" {
    const allocator = std.testing.allocator;
    var validator = Validator.init(allocator, .{});

    // Valid URLs
    try std.testing.expect(validator.validateUrl("https://example.com").valid);
    try std.testing.expect(validator.validateUrl("http://localhost:8080/path").valid);
    try std.testing.expect(validator.validateUrl("https://sub.domain.com/path?query=1").valid);

    // Invalid URLs
    try std.testing.expect(!validator.validateUrl("").valid);
    try std.testing.expect(!validator.validateUrl("notaurl").valid);
    try std.testing.expect(!validator.validateUrl("javascript:alert(1)").valid);
}

test "detect sql injection" {
    try std.testing.expect(containsSqlInjection("'; DROP TABLE users; --"));
    try std.testing.expect(containsSqlInjection("' OR '1'='1"));
    try std.testing.expect(containsSqlInjection("1; UNION SELECT * FROM users"));
    try std.testing.expect(!containsSqlInjection("normal input text"));
    try std.testing.expect(!containsSqlInjection("user@example.com"));
}

test "detect xss" {
    try std.testing.expect(containsXss("<script>alert('xss')</script>"));
    try std.testing.expect(containsXss("javascript:alert(1)"));
    try std.testing.expect(containsXss("<img src=x onerror=alert(1)>"));
    try std.testing.expect(!containsXss("normal text content"));
    try std.testing.expect(!containsXss("Hello, World!"));
}

test "detect path traversal" {
    try std.testing.expect(containsPathTraversal("../../../etc/passwd"));
    try std.testing.expect(containsPathTraversal("..\\..\\windows\\system32"));
    try std.testing.expect(containsPathTraversal("%2e%2e%2f"));
    try std.testing.expect(!containsPathTraversal("/normal/path/file.txt"));
    try std.testing.expect(!containsPathTraversal("filename.txt"));
}

test "sanitize html" {
    const allocator = std.testing.allocator;
    var sanitizer = Sanitizer.init(allocator);

    const result = try sanitizer.sanitizeHtml("<script>alert('xss')</script>");
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "<script>") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "&lt;script&gt;") != null);
}

test "sanitize filename" {
    const allocator = std.testing.allocator;
    var sanitizer = Sanitizer.init(allocator);

    const result1 = try sanitizer.sanitizeFilename("../../../etc/passwd");
    defer allocator.free(result1);
    try std.testing.expectEqualStrings("passwd", result1);

    const result2 = try sanitizer.sanitizeFilename("file<>name.txt");
    defer allocator.free(result2);
    try std.testing.expect(std.mem.indexOf(u8, result2, "<") == null);
}

test "validate ip address" {
    const allocator = std.testing.allocator;
    var validator = Validator.init(allocator, .{});

    // Valid IPv4
    try std.testing.expect(validator.validateIpAddress("192.168.1.1").valid);
    try std.testing.expect(validator.validateIpAddress("0.0.0.0").valid);
    try std.testing.expect(validator.validateIpAddress("255.255.255.255").valid);

    // Invalid IPv4
    try std.testing.expect(!validator.validateIpAddress("256.1.1.1").valid);
    try std.testing.expect(!validator.validateIpAddress("1.2.3").valid);
    try std.testing.expect(!validator.validateIpAddress("not.an.ip").valid);
}
