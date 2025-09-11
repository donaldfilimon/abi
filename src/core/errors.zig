//! Error Handling - Comprehensive error handling system for the Abi AI Framework
//!
//! This module provides a comprehensive error handling system with error codes,
//! categories, and structured error information.

const std = @import("std");

/// Main framework error type
pub const FrameworkError = error{
    // Core framework errors
    OutOfMemory,
    InvalidInput,
    InvalidConfiguration,
    OperationFailed,
    Timeout,
    ResourceExhausted,
    UnsupportedOperation,

    // Network errors
    NetworkError,
    ConnectionFailed,
    ConnectionTimeout,
    ConnectionReset,
    ConnectionRefused,

    // Database errors
    DatabaseError,
    DatabaseConnectionFailed,
    DatabaseQueryFailed,
    DatabaseTransactionFailed,
    DatabaseCorruption,

    // Agent errors
    AgentError,
    AgentNotFound,
    AgentInitializationFailed,
    AgentProcessingFailed,
    AgentTimeout,

    // Plugin errors
    PluginError,
    PluginNotFound,
    PluginLoadFailed,
    PluginInitializationFailed,
    PluginExecutionFailed,

    // Security errors
    SecurityError,
    AuthenticationFailed,
    AuthorizationFailed,
    InvalidToken,
    TokenExpired,

    // Validation errors
    ValidationError,
    InvalidFormat,
    InvalidSchema,
    InvalidData,
    InvalidParameter,

    // System errors
    SystemError,
    FileNotFound,
    PermissionDenied,
    DiskFull,
    SystemOverload,
};

/// Error categories for better error handling
pub const ErrorCategory = enum {
    core,
    network,
    database,
    agent,
    plugin,
    security,
    validation,
    system,
    unknown,
};

/// Structured error information
pub const ErrorInfo = struct {
    code: u32,
    category: ErrorCategory,
    message: []const u8,
    details: ?[]const u8 = null,
    timestamp: i64,
    source: ?[]const u8 = null,
    stack_trace: ?[]const u8 = null,

    pub fn init(code: u32, category: ErrorCategory, message: []const u8) ErrorInfo {
        return ErrorInfo{
            .code = code,
            .category = category,
            .message = message,
            .timestamp = std.time.microTimestamp(),
        };
    }

    pub fn withDetails(self: ErrorInfo, details: []const u8) ErrorInfo {
        return ErrorInfo{
            .code = self.code,
            .category = self.category,
            .message = self.message,
            .details = details,
            .timestamp = self.timestamp,
            .source = self.source,
            .stack_trace = self.stack_trace,
        };
    }

    pub fn withSource(self: ErrorInfo, source: []const u8) ErrorInfo {
        return ErrorInfo{
            .code = self.code,
            .category = self.category,
            .message = self.message,
            .details = self.details,
            .timestamp = self.timestamp,
            .source = source,
            .stack_trace = self.stack_trace,
        };
    }

    pub fn withStackTrace(self: ErrorInfo, stack_trace: []const u8) ErrorInfo {
        return ErrorInfo{
            .code = self.code,
            .category = self.category,
            .message = self.message,
            .details = self.details,
            .timestamp = self.timestamp,
            .source = self.source,
            .stack_trace = stack_trace,
        };
    }
};

/// Error code definitions
pub const ErrorCodes = struct {
    // Core framework errors (1000-1999)
    pub const OUT_OF_MEMORY = 1001;
    pub const INVALID_INPUT = 1002;
    pub const INVALID_CONFIGURATION = 1003;
    pub const OPERATION_FAILED = 1004;
    pub const TIMEOUT = 1005;
    pub const RESOURCE_EXHAUSTED = 1006;
    pub const UNSUPPORTED_OPERATION = 1007;

    // Network errors (2000-2999)
    pub const NETWORK_ERROR = 2001;
    pub const CONNECTION_FAILED = 2002;
    pub const CONNECTION_TIMEOUT = 2003;
    pub const CONNECTION_RESET = 2004;
    pub const CONNECTION_REFUSED = 2005;

    // Database errors (3000-3999)
    pub const DATABASE_ERROR = 3001;
    pub const DATABASE_CONNECTION_FAILED = 3002;
    pub const DATABASE_QUERY_FAILED = 3003;
    pub const DATABASE_TRANSACTION_FAILED = 3004;
    pub const DATABASE_CORRUPTION = 3005;

    // Agent errors (4000-4999)
    pub const AGENT_ERROR = 4001;
    pub const AGENT_NOT_FOUND = 4002;
    pub const AGENT_INITIALIZATION_FAILED = 4003;
    pub const AGENT_PROCESSING_FAILED = 4004;
    pub const AGENT_TIMEOUT = 4005;

    // Plugin errors (5000-5999)
    pub const PLUGIN_ERROR = 5001;
    pub const PLUGIN_NOT_FOUND = 5002;
    pub const PLUGIN_LOAD_FAILED = 5003;
    pub const PLUGIN_INITIALIZATION_FAILED = 5004;
    pub const PLUGIN_EXECUTION_FAILED = 5005;

    // Security errors (6000-6999)
    pub const SECURITY_ERROR = 6001;
    pub const AUTHENTICATION_FAILED = 6002;
    pub const AUTHORIZATION_FAILED = 6003;
    pub const INVALID_TOKEN = 6004;
    pub const TOKEN_EXPIRED = 6005;

    // Validation errors (7000-7999)
    pub const VALIDATION_ERROR = 7001;
    pub const INVALID_FORMAT = 7002;
    pub const INVALID_SCHEMA = 7003;
    pub const INVALID_DATA = 7004;
    pub const INVALID_PARAMETER = 7005;

    // System errors (8000-8999)
    pub const SYSTEM_ERROR = 8001;
    pub const FILE_NOT_FOUND = 8002;
    pub const PERMISSION_DENIED = 8003;
    pub const DISK_FULL = 8004;
    pub const SYSTEM_OVERLOAD = 8005;
};

/// Error handler for managing and processing errors
pub const ErrorHandler = struct {
    allocator: std.mem.Allocator,
    error_log: std.ArrayList(ErrorInfo),
    error_callbacks: std.ArrayList(ErrorCallback),

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .error_log = std.ArrayList(ErrorInfo).init(allocator),
            .error_callbacks = std.ArrayList(ErrorCallback).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.error_log.deinit();
        self.error_callbacks.deinit();
    }

    /// Handle an error
    pub fn handleError(self: *Self, error_info: ErrorInfo) void {
        // Log the error
        self.logError(error_info);

        // Call error callbacks
        for (self.error_callbacks.items) |callback| {
            callback.handler(error_info, callback.context);
        }
    }

    /// Log an error
    pub fn logError(self: *Self, error_info: ErrorInfo) void {
        self.error_log.append(error_info) catch return;

        // Log to standard error stream
        std.log.err("Error {d} ({s}): {s}", .{ error_info.code, @tagName(error_info.category), error_info.message });

        if (error_info.details) |details| {
            std.log.err("Details: {s}", .{details});
        }

        if (error_info.source) |source| {
            std.log.err("Source: {s}", .{source});
        }
    }

    /// Add error callback
    pub fn addErrorCallback(self: *Self, callback: ErrorCallback) !void {
        try self.error_callbacks.append(callback);
    }

    /// Remove error callback
    pub fn removeErrorCallback(self: *Self, index: usize) void {
        if (index < self.error_callbacks.items.len) {
            _ = self.error_callbacks.swapRemove(index);
        }
    }

    /// Get error log
    pub fn getErrorLog(self: *const Self) []const ErrorInfo {
        return self.error_log.items;
    }

    /// Clear error log
    pub fn clearErrorLog(self: *Self) void {
        self.error_log.clearRetainingCapacity();
    }

    /// Get error statistics
    pub fn getErrorStats(self: *const Self) ErrorStats {
        var stats = ErrorStats{};

        for (self.error_log.items) |error_info| {
            stats.total_errors += 1;

            switch (error_info.category) {
                .core => stats.core_errors += 1,
                .network => stats.network_errors += 1,
                .database => stats.database_errors += 1,
                .agent => stats.agent_errors += 1,
                .plugin => stats.plugin_errors += 1,
                .security => stats.security_errors += 1,
                .validation => stats.validation_errors += 1,
                .system => stats.system_errors += 1,
                .unknown => stats.unknown_errors += 1,
            }
        }

        return stats;
    }
};

/// Error callback function type
pub const ErrorCallback = struct {
    handler: *const fn (error_info: ErrorInfo, context: ?*anyopaque) void,
    context: ?*anyopaque = null,
};

/// Error statistics
pub const ErrorStats = struct {
    total_errors: u32 = 0,
    core_errors: u32 = 0,
    network_errors: u32 = 0,
    database_errors: u32 = 0,
    agent_errors: u32 = 0,
    plugin_errors: u32 = 0,
    security_errors: u32 = 0,
    validation_errors: u32 = 0,
    system_errors: u32 = 0,
    unknown_errors: u32 = 0,
};

/// Error context for providing additional error information
pub const ErrorContext = struct {
    operation: []const u8,
    component: []const u8,
    user_id: ?[]const u8 = null,
    session_id: ?[]const u8 = null,
    request_id: ?[]const u8 = null,
    additional_data: ?std.StringHashMap([]const u8) = null,

    pub fn init(operation: []const u8, component: []const u8) ErrorContext {
        return ErrorContext{
            .operation = operation,
            .component = component,
        };
    }

    pub fn withUser(self: ErrorContext, user_id: []const u8) ErrorContext {
        return ErrorContext{
            .operation = self.operation,
            .component = self.component,
            .user_id = user_id,
            .session_id = self.session_id,
            .request_id = self.request_id,
            .additional_data = self.additional_data,
        };
    }

    pub fn withSession(self: ErrorContext, session_id: []const u8) ErrorContext {
        return ErrorContext{
            .operation = self.operation,
            .component = self.component,
            .user_id = self.user_id,
            .session_id = session_id,
            .request_id = self.request_id,
            .additional_data = self.additional_data,
        };
    }

    pub fn withRequest(self: ErrorContext, request_id: []const u8) ErrorContext {
        return ErrorContext{
            .operation = self.operation,
            .component = self.component,
            .user_id = self.user_id,
            .session_id = self.session_id,
            .request_id = request_id,
            .additional_data = self.additional_data,
        };
    }
};

/// Error recovery strategies
pub const ErrorRecovery = enum {
    retry,
    fallback,
    ignore,
    abort,
    escalate,
};

/// Error recovery handler
pub const ErrorRecoveryHandler = struct {
    strategy: ErrorRecovery,
    max_retries: u32 = 3,
    retry_delay_ms: u32 = 1000,
    fallback_handler: ?*const fn () anyerror!void = null,

    pub fn init(strategy: ErrorRecovery) ErrorRecoveryHandler {
        return ErrorRecoveryHandler{
            .strategy = strategy,
        };
    }

    pub fn withRetries(self: ErrorRecoveryHandler, max_retries: u32, delay_ms: u32) ErrorRecoveryHandler {
        return ErrorRecoveryHandler{
            .strategy = self.strategy,
            .max_retries = max_retries,
            .retry_delay_ms = delay_ms,
            .fallback_handler = self.fallback_handler,
        };
    }

    pub fn withFallback(self: ErrorRecoveryHandler, fallback: *const fn () anyerror!void) ErrorRecoveryHandler {
        return ErrorRecoveryHandler{
            .strategy = self.strategy,
            .max_retries = self.max_retries,
            .retry_delay_ms = self.retry_delay_ms,
            .fallback_handler = fallback,
        };
    }
};

/// Utility functions for error handling
/// Convert framework error to error info
pub fn frameworkErrorToInfo(err: FrameworkError, message: []const u8) ErrorInfo {
    const code = switch (err) {
        .OutOfMemory => ErrorCodes.OUT_OF_MEMORY,
        .InvalidInput => ErrorCodes.INVALID_INPUT,
        .InvalidConfiguration => ErrorCodes.INVALID_CONFIGURATION,
        .OperationFailed => ErrorCodes.OPERATION_FAILED,
        .Timeout => ErrorCodes.TIMEOUT,
        .ResourceExhausted => ErrorCodes.RESOURCE_EXHAUSTED,
        .UnsupportedOperation => ErrorCodes.UNSUPPORTED_OPERATION,
        .NetworkError => ErrorCodes.NETWORK_ERROR,
        .ConnectionFailed => ErrorCodes.CONNECTION_FAILED,
        .ConnectionTimeout => ErrorCodes.CONNECTION_TIMEOUT,
        .ConnectionReset => ErrorCodes.CONNECTION_RESET,
        .ConnectionRefused => ErrorCodes.CONNECTION_REFUSED,
        .DatabaseError => ErrorCodes.DATABASE_ERROR,
        .DatabaseConnectionFailed => ErrorCodes.DATABASE_CONNECTION_FAILED,
        .DatabaseQueryFailed => ErrorCodes.DATABASE_QUERY_FAILED,
        .DatabaseTransactionFailed => ErrorCodes.DATABASE_TRANSACTION_FAILED,
        .DatabaseCorruption => ErrorCodes.DATABASE_CORRUPTION,
        .AgentError => ErrorCodes.AGENT_ERROR,
        .AgentNotFound => ErrorCodes.AGENT_NOT_FOUND,
        .AgentInitializationFailed => ErrorCodes.AGENT_INITIALIZATION_FAILED,
        .AgentProcessingFailed => ErrorCodes.AGENT_PROCESSING_FAILED,
        .AgentTimeout => ErrorCodes.AGENT_TIMEOUT,
        .PluginError => ErrorCodes.PLUGIN_ERROR,
        .PluginNotFound => ErrorCodes.PLUGIN_NOT_FOUND,
        .PluginLoadFailed => ErrorCodes.PLUGIN_LOAD_FAILED,
        .PluginInitializationFailed => ErrorCodes.PLUGIN_INITIALIZATION_FAILED,
        .PluginExecutionFailed => ErrorCodes.PLUGIN_EXECUTION_FAILED,
        .SecurityError => ErrorCodes.SECURITY_ERROR,
        .AuthenticationFailed => ErrorCodes.AUTHENTICATION_FAILED,
        .AuthorizationFailed => ErrorCodes.AUTHORIZATION_FAILED,
        .InvalidToken => ErrorCodes.INVALID_TOKEN,
        .TokenExpired => ErrorCodes.TOKEN_EXPIRED,
        .ValidationError => ErrorCodes.VALIDATION_ERROR,
        .InvalidFormat => ErrorCodes.INVALID_FORMAT,
        .InvalidSchema => ErrorCodes.INVALID_SCHEMA,
        .InvalidData => ErrorCodes.INVALID_DATA,
        .InvalidParameter => ErrorCodes.INVALID_PARAMETER,
        .SystemError => ErrorCodes.SYSTEM_ERROR,
        .FileNotFound => ErrorCodes.FILE_NOT_FOUND,
        .PermissionDenied => ErrorCodes.PERMISSION_DENIED,
        .DiskFull => ErrorCodes.DISK_FULL,
        .SystemOverload => ErrorCodes.SYSTEM_OVERLOAD,
    };

    const category = switch (err) {
        .OutOfMemory, .InvalidInput, .InvalidConfiguration, .OperationFailed, .Timeout, .ResourceExhausted, .UnsupportedOperation => .core,
        .NetworkError, .ConnectionFailed, .ConnectionTimeout, .ConnectionReset, .ConnectionRefused => .network,
        .DatabaseError, .DatabaseConnectionFailed, .DatabaseQueryFailed, .DatabaseTransactionFailed, .DatabaseCorruption => .database,
        .AgentError, .AgentNotFound, .AgentInitializationFailed, .AgentProcessingFailed, .AgentTimeout => .agent,
        .PluginError, .PluginNotFound, .PluginLoadFailed, .PluginInitializationFailed, .PluginExecutionFailed => .plugin,
        .SecurityError, .AuthenticationFailed, .AuthorizationFailed, .InvalidToken, .TokenExpired => .security,
        .ValidationError, .InvalidFormat, .InvalidSchema, .InvalidData, .InvalidParameter => .validation,
        .SystemError, .FileNotFound, .PermissionDenied, .DiskFull, .SystemOverload => .system,
    };

    return ErrorInfo.init(code, category, message);
}

/// Get error category from error code
pub fn getErrorCategory(code: u32) ErrorCategory {
    return switch (code) {
        1000...1999 => .core,
        2000...2999 => .network,
        3000...3999 => .database,
        4000...4999 => .agent,
        5000...5999 => .plugin,
        6000...6999 => .security,
        7000...7999 => .validation,
        8000...8999 => .system,
        else => .unknown,
    };
}

/// Get error message from error code
pub fn getErrorMessage(code: u32) []const u8 {
    return switch (code) {
        ErrorCodes.OUT_OF_MEMORY => "Out of memory",
        ErrorCodes.INVALID_INPUT => "Invalid input",
        ErrorCodes.INVALID_CONFIGURATION => "Invalid configuration",
        ErrorCodes.OPERATION_FAILED => "Operation failed",
        ErrorCodes.TIMEOUT => "Operation timeout",
        ErrorCodes.RESOURCE_EXHAUSTED => "Resource exhausted",
        ErrorCodes.UNSUPPORTED_OPERATION => "Unsupported operation",
        ErrorCodes.NETWORK_ERROR => "Network error",
        ErrorCodes.CONNECTION_FAILED => "Connection failed",
        ErrorCodes.CONNECTION_TIMEOUT => "Connection timeout",
        ErrorCodes.CONNECTION_RESET => "Connection reset",
        ErrorCodes.CONNECTION_REFUSED => "Connection refused",
        ErrorCodes.DATABASE_ERROR => "Database error",
        ErrorCodes.DATABASE_CONNECTION_FAILED => "Database connection failed",
        ErrorCodes.DATABASE_QUERY_FAILED => "Database query failed",
        ErrorCodes.DATABASE_TRANSACTION_FAILED => "Database transaction failed",
        ErrorCodes.DATABASE_CORRUPTION => "Database corruption",
        ErrorCodes.AGENT_ERROR => "Agent error",
        ErrorCodes.AGENT_NOT_FOUND => "Agent not found",
        ErrorCodes.AGENT_INITIALIZATION_FAILED => "Agent initialization failed",
        ErrorCodes.AGENT_PROCESSING_FAILED => "Agent processing failed",
        ErrorCodes.AGENT_TIMEOUT => "Agent timeout",
        ErrorCodes.PLUGIN_ERROR => "Plugin error",
        ErrorCodes.PLUGIN_NOT_FOUND => "Plugin not found",
        ErrorCodes.PLUGIN_LOAD_FAILED => "Plugin load failed",
        ErrorCodes.PLUGIN_INITIALIZATION_FAILED => "Plugin initialization failed",
        ErrorCodes.PLUGIN_EXECUTION_FAILED => "Plugin execution failed",
        ErrorCodes.SECURITY_ERROR => "Security error",
        ErrorCodes.AUTHENTICATION_FAILED => "Authentication failed",
        ErrorCodes.AUTHORIZATION_FAILED => "Authorization failed",
        ErrorCodes.INVALID_TOKEN => "Invalid token",
        ErrorCodes.TOKEN_EXPIRED => "Token expired",
        ErrorCodes.VALIDATION_ERROR => "Validation error",
        ErrorCodes.INVALID_FORMAT => "Invalid format",
        ErrorCodes.INVALID_SCHEMA => "Invalid schema",
        ErrorCodes.INVALID_DATA => "Invalid data",
        ErrorCodes.INVALID_PARAMETER => "Invalid parameter",
        ErrorCodes.SYSTEM_ERROR => "System error",
        ErrorCodes.FILE_NOT_FOUND => "File not found",
        ErrorCodes.PERMISSION_DENIED => "Permission denied",
        ErrorCodes.DISK_FULL => "Disk full",
        ErrorCodes.SYSTEM_OVERLOAD => "System overload",
        else => "Unknown error",
    };
}

test "error handling" {
    const testing = std.testing;

    // Test error handler initialization
    var error_handler = ErrorHandler.init(testing.allocator);
    defer error_handler.deinit();

    // Test error logging
    const error_info = ErrorInfo.init(ErrorCodes.INVALID_INPUT, .validation, "Invalid input provided");
    error_handler.handleError(error_info);

    // Test error statistics
    const stats = error_handler.getErrorStats();
    try testing.expectEqual(@as(u32, 1), stats.total_errors);
    try testing.expectEqual(@as(u32, 1), stats.validation_errors);
}

test "error context" {
    const testing = std.testing;

    // Test error context creation
    const context = ErrorContext.init("test_operation", "test_component");
    try testing.expectEqualStrings("test_operation", context.operation);
    try testing.expectEqualStrings("test_component", context.component);

    // Test error context with user
    const context_with_user = context.withUser("user123");
    try testing.expectEqualStrings("user123", context_with_user.user_id.?);
}

test "error recovery" {
    const testing = std.testing;

    // Test error recovery handler
    const recovery_handler = ErrorRecoveryHandler.init(.retry)
        .withRetries(5, 2000)
        .withFallback(testFallback);

    try testing.expectEqual(ErrorRecovery.retry, recovery_handler.strategy);
    try testing.expectEqual(@as(u32, 5), recovery_handler.max_retries);
    try testing.expectEqual(@as(u32, 2000), recovery_handler.retry_delay_ms);
    try testing.expect(recovery_handler.fallback_handler != null);
}

fn testFallback() anyerror!void {
    // Test fallback function
}
