//! Framework Error Definitions
//!
//! Centralized error definitions for the ABI framework with proper
//! categorization and context information.

const std = @import("std");
const patterns = @import("../patterns/common.zig");

/// Core framework errors
pub const FrameworkError = error{
    /// Initialization errors
    InitializationFailed,
    AlreadyInitialized,
    ConfigurationInvalid,
    DependencyMissing,
    
    /// Runtime errors
    FeatureNotAvailable,
    FeatureNotEnabled,
    ResourceExhausted,
    OperationTimeout,
    OperationCancelled,
    
    /// State errors
    InvalidState,
    StateTransitionFailed,
    ConcurrentModification,
    
    /// Resource errors
    ResourceNotFound,
    ResourceBusy,
    ResourceCorrupted,
    PermissionDenied,
    
    /// Network errors
    NetworkUnavailable,
    ConnectionFailed,
    RequestTimeout,
    InvalidResponse,
    
    /// Data errors
    InvalidData,
    DataCorrupted,
    SerializationFailed,
    DeserializationFailed,
    
    /// Generic errors
    InternalError,
    NotImplemented,
    Unsupported,
};

/// AI subsystem errors
pub const AIError = error{
    /// Model errors
    ModelNotFound,
    ModelLoadFailed,
    ModelCorrupted,
    ModelVersionMismatch,
    
    /// Training errors
    TrainingFailed,
    InsufficientData,
    ConvergenceFailed,
    GradientExplosion,
    GradientVanishing,
    
    /// Inference errors
    InferenceFailed,
    InvalidInput,
    OutputGenerationFailed,
    
    /// Data errors
    DatasetNotFound,
    DatasetCorrupted,
    PreprocessingFailed,
    
    /// Resource errors
    InsufficientMemory,
    ComputeResourceUnavailable,
};

/// GPU subsystem errors
pub const GPUError = error{
    /// Device errors
    DeviceNotFound,
    DeviceInitializationFailed,
    DeviceUnsupported,
    DriverOutdated,
    
    /// Memory errors
    OutOfVideoMemory,
    MemoryAllocationFailed,
    MemoryCorruption,
    
    /// Compute errors
    KernelCompilationFailed,
    KernelExecutionFailed,
    ShaderCompilationFailed,
    
    /// Pipeline errors
    PipelineCreationFailed,
    PipelineBindingFailed,
    
    /// Backend errors
    BackendUnavailable,
    BackendInitializationFailed,
    UnsupportedOperation,
};

/// Database subsystem errors
pub const DatabaseError = error{
    /// Connection errors
    ConnectionFailed,
    ConnectionLost,
    AuthenticationFailed,
    
    /// Query errors
    QueryFailed,
    InvalidQuery,
    QueryTimeout,
    
    /// Data errors
    RecordNotFound,
    DuplicateKey,
    ConstraintViolation,
    DataIntegrityError,
    
    /// Transaction errors
    TransactionFailed,
    DeadlockDetected,
    TransactionTimeout,
    
    /// Schema errors
    SchemaError,
    MigrationFailed,
    
    /// Vector database specific
    VectorDimensionMismatch,
    IndexCorrupted,
    SearchFailed,
};

/// Web subsystem errors
pub const WebError = error{
    /// HTTP errors
    BadRequest,
    Unauthorized,
    Forbidden,
    NotFound,
    MethodNotAllowed,
    RequestTimeout,
    InternalServerError,
    ServiceUnavailable,
    
    /// Server errors
    ServerStartFailed,
    ServerStopFailed,
    BindAddressFailed,
    
    /// Client errors
    RequestFailed,
    ResponseParsingFailed,
    InvalidUrl,
    
    /// WebSocket errors
    WebSocketUpgradeFailed,
    WebSocketConnectionClosed,
    InvalidWebSocketFrame,
};

/// Monitoring subsystem errors
pub const MonitoringError = error{
    /// Metrics errors
    MetricNotFound,
    MetricRegistrationFailed,
    MetricCollectionFailed,
    
    /// Logging errors
    LoggerInitializationFailed,
    LogWriteFailed,
    LogRotationFailed,
    
    /// Tracing errors
    TracingInitializationFailed,
    SpanCreationFailed,
    TraceExportFailed,
    
    /// Alerting errors
    AlertingConfigurationInvalid,
    AlertDeliveryFailed,
};

/// Error context with rich information
pub const ErrorInfo = struct {
    error_type: ErrorType,
    message: []const u8,
    location: ?std.builtin.SourceLocation = null,
    timestamp: i64,
    context: ?[]const u8 = null,
    cause: ?anyerror = null,
    
    pub const ErrorType = enum {
        framework,
        ai,
        gpu,
        database,
        web,
        monitoring,
        system,
        user,
    };
    
    pub fn init(error_type: ErrorType, message: []const u8) ErrorInfo {
        return .{
            .error_type = error_type,
            .message = message,
            .timestamp = std.time.milliTimestamp(),
        };
    }
    
    pub fn withLocation(self: ErrorInfo, location: std.builtin.SourceLocation) ErrorInfo {
        return .{
            .error_type = self.error_type,
            .message = self.message,
            .location = location,
            .timestamp = self.timestamp,
            .context = self.context,
            .cause = self.cause,
        };
    }
    
    pub fn withContext(self: ErrorInfo, context: []const u8) ErrorInfo {
        return .{
            .error_type = self.error_type,
            .message = self.message,
            .location = self.location,
            .timestamp = self.timestamp,
            .context = context,
            .cause = self.cause,
        };
    }
    
    pub fn withCause(self: ErrorInfo, cause: anyerror) ErrorInfo {
        return .{
            .error_type = self.error_type,
            .message = self.message,
            .location = self.location,
            .timestamp = self.timestamp,
            .context = self.context,
            .cause = cause,
        };
    }
    
    pub fn format(self: ErrorInfo, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        
        try writer.print("[{s}] {s}", .{ @tagName(self.error_type), self.message });
        
        if (self.location) |loc| {
            try writer.print(" at {s}:{d}:{d}", .{ loc.file, loc.line, loc.column });
        }
        
        if (self.context) |ctx| {
            try writer.print(" (context: {s})", .{ctx});
        }
        
        if (self.cause) |cause| {
            try writer.print(" (caused by: {s})", .{@errorName(cause)});
        }
        
        try writer.print(" [timestamp: {d}]", .{self.timestamp});
    }
};

/// Error result type for better error handling
pub fn ErrorResult(comptime T: type) type {
    return union(enum) {
        success: T,
        failure: ErrorInfo,
        
        pub fn ok(value: T) ErrorResult(T) {
            return .{ .success = value };
        }
        
        pub fn err(error_info: ErrorInfo) ErrorResult(T) {
            return .{ .failure = error_info };
        }
        
        pub fn isOk(self: ErrorResult(T)) bool {
            return switch (self) {
                .success => true,
                .failure => false,
            };
        }
        
        pub fn isErr(self: ErrorResult(T)) bool {
            return !self.isOk();
        }
        
        pub fn unwrap(self: ErrorResult(T)) T {
            return switch (self) {
                .success => |value| value,
                .failure => |err_info| std.debug.panic("Called unwrap on error: {}", .{err_info}),
            };
        }
        
        pub fn unwrapOr(self: ErrorResult(T), default: T) T {
            return switch (self) {
                .success => |value| value,
                .failure => default,
            };
        }
        
        pub fn getError(self: ErrorResult(T)) ?ErrorInfo {
            return switch (self) {
                .success => null,
                .failure => |err_info| err_info,
            };
        }
    };
}

/// Error recovery strategies
pub const RecoveryStrategy = enum {
    retry,
    fallback,
    abort,
    ignore,
    
    pub fn shouldRetry(self: RecoveryStrategy) bool {
        return self == .retry;
    }
    
    pub fn shouldFallback(self: RecoveryStrategy) bool {
        return self == .fallback;
    }
    
    pub fn shouldAbort(self: RecoveryStrategy) bool {
        return self == .abort;
    }
};

/// Error handler interface
pub const ErrorHandler = struct {
    handle_fn: *const fn (error_info: ErrorInfo) RecoveryStrategy,
    
    pub fn init(handle_fn: *const fn (error_info: ErrorInfo) RecoveryStrategy) ErrorHandler {
        return .{ .handle_fn = handle_fn };
    }
    
    pub fn handle(self: ErrorHandler, error_info: ErrorInfo) RecoveryStrategy {
        return self.handle_fn(error_info);
    }
};

/// Default error handlers
pub const DefaultHandlers = struct {
    pub fn logAndContinue(error_info: ErrorInfo) RecoveryStrategy {
        std.log.err("{}", .{error_info});
        return .ignore;
    }
    
    pub fn logAndRetry(error_info: ErrorInfo) RecoveryStrategy {
        std.log.warn("Retrying after error: {}", .{error_info});
        return .retry;
    }
    
    pub fn logAndAbort(error_info: ErrorInfo) RecoveryStrategy {
        std.log.err("Fatal error: {}", .{error_info});
        return .abort;
    }
};

/// Utility functions for error handling
pub fn wrapError(comptime error_type: ErrorInfo.ErrorType, err: anyerror, message: []const u8) ErrorInfo {
    return ErrorInfo.init(error_type, message)
        .withLocation(@src())
        .withCause(err);
}

pub fn frameworkError(message: []const u8) ErrorInfo {
    return ErrorInfo.init(.framework, message).withLocation(@src());
}

pub fn aiError(message: []const u8) ErrorInfo {
    return ErrorInfo.init(.ai, message).withLocation(@src());
}

pub fn gpuError(message: []const u8) ErrorInfo {
    return ErrorInfo.init(.gpu, message).withLocation(@src());
}

pub fn databaseError(message: []const u8) ErrorInfo {
    return ErrorInfo.init(.database, message).withLocation(@src());
}

pub fn webError(message: []const u8) ErrorInfo {
    return ErrorInfo.init(.web, message).withLocation(@src());
}

pub fn monitoringError(message: []const u8) ErrorInfo {
    return ErrorInfo.init(.monitoring, message).withLocation(@src());
}

test "ErrorInfo creation and formatting" {
    var buffer = std.ArrayList(u8).init(std.testing.allocator);
    defer buffer.deinit();
    
    const error_info = frameworkError("Test error message")
        .withContext("Test context");
    
    try buffer.writer().print("{}", .{error_info});
    const output = buffer.items;
    
    try std.testing.expect(std.mem.indexOf(u8, output, "framework") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Test error message") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "Test context") != null);
}

test "ErrorResult success and failure cases" {
    const success_result = ErrorResult(i32).ok(42);
    const failure_result = ErrorResult(i32).err(frameworkError("Test failure"));
    
    try std.testing.expect(success_result.isOk());
    try std.testing.expect(!success_result.isErr());
    try std.testing.expectEqual(@as(i32, 42), success_result.unwrap());
    
    try std.testing.expect(!failure_result.isOk());
    try std.testing.expect(failure_result.isErr());
    try std.testing.expectEqual(@as(i32, 0), failure_result.unwrapOr(0));
}

test "Error recovery strategies" {
    const retry_strategy = RecoveryStrategy.retry;
    const fallback_strategy = RecoveryStrategy.fallback;
    
    try std.testing.expect(retry_strategy.shouldRetry());
    try std.testing.expect(!retry_strategy.shouldFallback());
    
    try std.testing.expect(!fallback_strategy.shouldRetry());
    try std.testing.expect(fallback_strategy.shouldFallback());
}