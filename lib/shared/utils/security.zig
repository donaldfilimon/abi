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
