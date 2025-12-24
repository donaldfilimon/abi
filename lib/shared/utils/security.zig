//! Security utilities for input validation, sanitization, and rate limiting
//! Provides comprehensive security checks for the ABI framework

const std = @import("std");

/// Maximum allowed payload size for HTTP requests (10MB)
pub const MAX_PAYLOAD_SIZE = 10 * 1024 * 1024;

/// Maximum allowed vector dimensions
pub const MAX_VECTOR_DIMENSIONS = 4096;

/// Maximum allowed array size for batch operations
pub const MAX_BATCH_SIZE = 1000;

/// Rate limiting configuration
pub const RateLimitConfig = struct {
    /// Maximum requests per minute per IP
    requests_per_minute: u32 = 60,
    /// Maximum concurrent connections per IP
    max_concurrent: u32 = 10,
    /// Time window in seconds
    window_seconds: u32 = 60,
};

/// Time provider for rate limiting (injectable for deterministic tests).
pub const TimeProvider = struct {
    context: ?*anyopaque,
    now_fn: *const fn (context: ?*anyopaque) i64,

    pub fn system() TimeProvider {
        return .{
            .context = null,
            .now_fn = systemNow,
        };
    }

    pub fn timestamp(self: TimeProvider) i64 {
        return self.now_fn(self.context);
    }

    fn systemNow(_: ?*anyopaque) i64 {
        return std.time.timestamp();
    }
};

/// Input validation errors
pub const ValidationError = error{
    PayloadTooLarge,
    InvalidJson,
    InvalidVectorDimensions,
    InvalidDataType,
    MalformedRequest,
    UnsafeCharacters,
    RateLimitExceeded,
};

/// Sanitize and validate JSON payload
pub fn validateJsonPayload(payload: []const u8) ValidationError!void {
    // Check payload size
    if (payload.len > MAX_PAYLOAD_SIZE) {
        return ValidationError.PayloadTooLarge;
    }

    // Basic JSON structure validation (more thorough parsing done elsewhere)
    if (payload.len == 0) {
        return ValidationError.MalformedRequest;
    }

    // Check for null bytes (potential security issue)
    for (payload) |byte| {
        if (byte == 0) {
            return ValidationError.UnsafeCharacters;
        }
    }

    // Check for extremely nested structures (potential DoS)
    var nesting_level: u32 = 0;
    var max_nesting: u32 = 0;

    for (payload) |byte| {
        switch (byte) {
            '{', '[' => {
                nesting_level += 1;
                if (nesting_level > max_nesting) {
                    max_nesting = nesting_level;
                }
                if (max_nesting > 32) { // Arbitrary limit to prevent DoS
                    return ValidationError.MalformedRequest;
                }
            },
            '}', ']' => {
                if (nesting_level > 0) {
                    nesting_level -= 1;
                }
            },
            else => {},
        }
    }

    if (nesting_level != 0) {
        return ValidationError.MalformedRequest;
    }
}

/// Validate vector data for security and correctness
pub fn validateVectorData(vector: []const f32) ValidationError!void {
    // Check dimensions
    if (vector.len == 0 or vector.len > MAX_VECTOR_DIMENSIONS) {
        return ValidationError.InvalidVectorDimensions;
    }

    // Check for NaN or infinite values
    for (vector) |value| {
        if (std.math.isNan(value) or std.math.isInf(value)) {
            return ValidationError.InvalidDataType;
        }
    }
}

/// Validate batch operation parameters
pub fn validateBatchOperation(count: usize) ValidationError!void {
    if (count == 0 or count > MAX_BATCH_SIZE) {
        return ValidationError.InvalidVectorDimensions; // Reuse error for batch size
    }
}

/// Sanitize string input (remove potentially dangerous characters)
pub fn sanitizeString(input: []const u8, allocator: std.mem.Allocator) ![]u8 {
    // For now, just validate and return a copy
    // In a real implementation, this would filter out dangerous characters
    if (input.len > MAX_PAYLOAD_SIZE) {
        return ValidationError.PayloadTooLarge;
    }

    // Check for null bytes
    for (input) |byte| {
        if (byte == 0) {
            return ValidationError.UnsafeCharacters;
        }
    }

    return allocator.dupe(u8, input);
}

/// Rate limiter implementation
pub const RateLimiter = struct {
    const RequestRecord = struct {
        timestamp: i64,
        count: u32,
    };

    allocator: std.mem.Allocator,
    config: RateLimitConfig,
    records: std.AutoHashMap(u32, RequestRecord), // IP -> record
    time_provider: TimeProvider,

    pub fn init(allocator: std.mem.Allocator, config: RateLimitConfig) RateLimiter {
        return initWithTimeProvider(allocator, config, TimeProvider.system());
    }

    pub fn initWithTimeProvider(
        allocator: std.mem.Allocator,
        config: RateLimitConfig,
        time_provider: TimeProvider,
    ) RateLimiter {
        return .{
            .allocator = allocator,
            .config = config,
            .records = std.AutoHashMap(u32, RequestRecord).init(allocator),
            .time_provider = time_provider,
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        self.records.deinit();
    }

    /// Check if request from IP address is allowed
    pub fn checkLimit(self: *RateLimiter, ip: u32) ValidationError!void {
        const now = self.time_provider.timestamp();
        const window_start = now - @as(i64, @intCast(self.config.window_seconds));

        var record = self.records.get(ip) orelse RequestRecord{
            .timestamp = now,
            .count = 0,
        };

        // Reset counter if window has expired
        if (record.timestamp < window_start) {
            record.timestamp = now;
            record.count = 0;
        }

        // Check limits
        if (record.count >= self.config.requests_per_minute) {
            return ValidationError.RateLimitExceeded;
        }

        // Update record
        record.count += 1;
        try self.records.put(ip, record);
    }
};

/// Cryptographically secure random number generation
pub const SecureRandom = struct {
    /// Generate cryptographically secure random bytes
    pub fn bytes(buffer: []u8) void {
        std.crypto.random.bytes(buffer);
    }

    /// Generate secure random integer in range
    pub fn int(comptime T: type, min: T, max: T) T {
        return std.crypto.random.intRangeAtMost(T, min, max - 1);
    }

    /// Generate secure random float between 0 and 1
    pub fn float() f64 {
        return std.crypto.random.float(f64);
    }

    /// Fill buffer with secure random data
    pub fn fill(buffer: []u8) void {
        std.crypto.random.bytes(buffer);
    }

    /// Generate secure token (base64 encoded)
    pub fn token(allocator: std.mem.Allocator, length: usize) ![]u8 {
        const bytes_needed = (length * 3) / 4 + 1; // Account for base64 overhead
        var random_bytes: [128]u8 = undefined;
        const actual_bytes = @min(bytes_needed, random_bytes.len);

        bytes(random_bytes[0..actual_bytes]);

        return std.base64.standard.Encoder.encode(allocator, random_bytes[0..actual_bytes]);
    }
};

/// Secure configuration defaults
pub const SecureDefaults = struct {
    /// Default rate limiting configuration
    pub const rate_limit = RateLimitConfig{
        .requests_per_minute = 60,
        .max_concurrent = 10,
        .window_seconds = 60,
    };

    /// Default timeout for operations (30 seconds)
    pub const operation_timeout_ms: u32 = 30000;

    /// Maximum database connections
    pub const max_db_connections: u32 = 100;

    /// Maximum memory usage per request (100MB)
    pub const max_memory_per_request: usize = 100 * 1024 * 1024;
};

test "security - validateJsonPayload" {
    const testing = std.testing;

    // Valid JSON
    try validateJsonPayload("{}");
    try validateJsonPayload("{\"key\": \"value\"}");
    try validateJsonPayload("[1, 2, 3]");

    // Empty payload
    try testing.expectError(ValidationError.MalformedRequest, validateJsonPayload(""));

    // Null bytes
    try testing.expectError(ValidationError.UnsafeCharacters, validateJsonPayload("{\"key\": \"value\x00\"}"));

    // Too nested
    const deeply_nested = "{" ** 40 ++ "}" ** 40;
    try testing.expectError(ValidationError.MalformedRequest, validateJsonPayload(deeply_nested));

    // Unbalanced brackets
    try testing.expectError(ValidationError.MalformedRequest, validateJsonPayload("{\"unclosed\": true"));
}

test "security - validateVectorData" {
    const testing = std.testing;

    // Valid vector
    const valid_vector = [_]f32{ 1.0, 2.0, 3.0 };
    try validateVectorData(&valid_vector);

    // Empty vector
    try testing.expectError(ValidationError.InvalidVectorDimensions, validateVectorData(&[_]f32{}));

    // Too large vector
    var large_vector: [5000]f32 = undefined;
    @memset(&large_vector, 1.0);
    try testing.expectError(ValidationError.InvalidVectorDimensions, validateVectorData(&large_vector));

    // Vector with NaN
    const nan_vector = [_]f32{ 1.0, std.math.nan(f32), 3.0 };
    try testing.expectError(ValidationError.InvalidDataType, validateVectorData(&nan_vector));

    // Vector with infinity
    const inf_vector = [_]f32{ 1.0, std.math.inf(f32), 3.0 };
    try testing.expectError(ValidationError.InvalidDataType, validateVectorData(&inf_vector));
}

test "security - validateBatchOperation" {
    const testing = std.testing;

    // Valid batch sizes
    try validateBatchOperation(1);
    try validateBatchOperation(100);
    try validateBatchOperation(1000);

    // Invalid batch sizes
    try testing.expectError(ValidationError.InvalidVectorDimensions, validateBatchOperation(0));
    try testing.expectError(ValidationError.InvalidVectorDimensions, validateBatchOperation(1001));
}

test "security - sanitizeString" {
    const testing = std.testing;

    // Valid string
    const result = try sanitizeString("hello world", testing.allocator);
    defer testing.allocator.free(result);
    try testing.expectEqualStrings("hello world", result);

    // String with null bytes
    try testing.expectError(ValidationError.UnsafeCharacters, sanitizeString("hello\x00world", testing.allocator));
}

test "security - RateLimiter" {
    const testing = std.testing;

    var limiter = RateLimiter.init(testing.allocator, .{});
    defer limiter.deinit();

    const test_ip: u32 = 0x7f000001; // 127.0.0.1

    // Should allow initial requests
    try limiter.checkLimit(test_ip);
    try limiter.checkLimit(test_ip);

    // After setting very low limit, should fail
    limiter.config.requests_per_minute = 1;
    try testing.expectError(ValidationError.RateLimitExceeded, limiter.checkLimit(test_ip));
}

test "security - SecureRandom" {
    const testing = std.testing;

    // Test token generation
    const token1 = try SecureRandom.token(testing.allocator, 16);
    defer testing.allocator.free(token1);
    try testing.expect(token1.len > 0);

    const token2 = try SecureRandom.token(testing.allocator, 16);
    defer testing.allocator.free(token2);
    try testing.expect(token2.len > 0);

    // Tokens should be different (with very high probability)
    try testing.expect(!std.mem.eql(u8, token1, token2));

    // Test random int generation
    const random_int = SecureRandom.int(u32, 0, 100);
    try testing.expect(random_int >= 0 and random_int < 100);

    // Test random float
    const random_float = SecureRandom.float();
    try testing.expect(random_float >= 0.0 and random_float < 1.0);
}
