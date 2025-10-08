//! Framework-wide Error Definitions
//!
//! This module defines all error types used across the Abi framework,
//! providing a unified error handling strategy.

const std = @import("std");

/// Core framework errors
pub const FrameworkError = error{
    /// Framework initialization failed
    InitializationFailed,
    /// Framework already initialized
    AlreadyInitialized,
    /// Framework not initialized
    NotInitialized,
    /// Feature not available
    FeatureNotAvailable,
    /// Feature already registered
    FeatureAlreadyRegistered,
    /// Invalid configuration
    InvalidConfiguration,
    /// Resource limit exceeded
    ResourceLimitExceeded,
};

/// AI/ML specific errors
pub const AIError = error{
    /// Model not found
    ModelNotFound,
    /// Invalid model format
    InvalidModelFormat,
    /// Model loading failed
    ModelLoadFailed,
    /// Inference failed
    InferenceFailed,
    /// Training failed
    TrainingFailed,
    /// Invalid input dimensions
    InvalidInputDimensions,
    /// Agent initialization failed
    AgentInitFailed,
};

/// Database errors
pub const DatabaseError = error{
    /// Connection failed
    ConnectionFailed,
    /// Query failed
    QueryFailed,
    /// Invalid query
    InvalidQuery,
    /// Record not found
    RecordNotFound,
    /// Duplicate key
    DuplicateKey,
    /// Transaction failed
    TransactionFailed,
    /// Index creation failed
    IndexCreationFailed,
};

/// GPU errors
pub const GPUError = error{
    /// No GPU available
    NoGPUAvailable,
    /// GPU initialization failed
    GPUInitFailed,
    /// Kernel launch failed
    KernelLaunchFailed,
    /// Memory allocation failed
    MemoryAllocationFailed,
    /// Device not supported
    DeviceNotSupported,
    /// Driver error
    DriverError,
    /// Compute capability insufficient
    InsufficientComputeCapability,
};

/// Network/Web errors
pub const NetworkError = error{
    /// Connection timeout
    ConnectionTimeout,
    /// Invalid URL
    InvalidURL,
    /// Request failed
    RequestFailed,
    /// Invalid response
    InvalidResponse,
    /// Authentication failed
    AuthenticationFailed,
    /// Rate limit exceeded
    RateLimitExceeded,
};

/// Plugin system errors
pub const PluginError = error{
    /// Plugin not found
    PluginNotFound,
    /// Plugin load failed
    PluginLoadFailed,
    /// Invalid plugin
    InvalidPlugin,
    /// Plugin initialization failed
    PluginInitFailed,
    /// Incompatible plugin version
    IncompatiblePluginVersion,
};

/// Monitoring/Observability errors
pub const MonitoringError = error{
    /// Metrics collection failed
    MetricsCollectionFailed,
    /// Invalid metric
    InvalidMetric,
    /// Trace export failed
    TraceExportFailed,
    /// Logger initialization failed
    LoggerInitFailed,
};

/// Unified error set combining all framework errors
pub const AbiError = FrameworkError ||
    AIError ||
    DatabaseError ||
    GPUError ||
    NetworkError ||
    PluginError ||
    MonitoringError ||
    std.mem.Allocator.Error ||
    std.fs.File.OpenError ||
    std.fs.File.ReadError ||
    std.fs.File.WriteError;

/// Error classification for better handling
pub const ErrorClass = enum {
    framework,
    ai,
    database,
    gpu,
    network,
    plugin,
    monitoring,
    system,
    unknown,
    
    pub fn fromError(err: anyerror) ErrorClass {
        return switch (err) {
            // Framework errors
            error.InitializationFailed,
            error.AlreadyInitialized,
            error.NotInitialized,
            error.FeatureNotAvailable,
            error.FeatureAlreadyRegistered,
            error.InvalidConfiguration,
            error.ResourceLimitExceeded,
            => .framework,
            
            // AI errors
            error.ModelNotFound,
            error.InvalidModelFormat,
            error.ModelLoadFailed,
            error.InferenceFailed,
            error.TrainingFailed,
            error.InvalidInputDimensions,
            error.AgentInitFailed,
            => .ai,
            
            // Database errors
            error.ConnectionFailed,
            error.QueryFailed,
            error.InvalidQuery,
            error.RecordNotFound,
            error.DuplicateKey,
            error.TransactionFailed,
            error.IndexCreationFailed,
            => .database,
            
            // GPU errors
            error.NoGPUAvailable,
            error.GPUInitFailed,
            error.KernelLaunchFailed,
            error.MemoryAllocationFailed,
            error.DeviceNotSupported,
            error.DriverError,
            error.InsufficientComputeCapability,
            => .gpu,
            
            // Network errors
            error.ConnectionTimeout,
            error.InvalidURL,
            error.RequestFailed,
            error.InvalidResponse,
            error.AuthenticationFailed,
            error.RateLimitExceeded,
            => .network,
            
            // Plugin errors
            error.PluginNotFound,
            error.PluginLoadFailed,
            error.InvalidPlugin,
            error.PluginInitFailed,
            error.IncompatiblePluginVersion,
            => .plugin,
            
            // Monitoring errors
            error.MetricsCollectionFailed,
            error.InvalidMetric,
            error.TraceExportFailed,
            error.LoggerInitFailed,
            => .monitoring,
            
            // System errors
            error.OutOfMemory,
            error.FileNotFound,
            error.AccessDenied,
            error.IsDir,
            error.NotDir,
            error.InvalidUtf8,
            => .system,
            
            else => .unknown,
        };
    }
    
    pub fn toString(self: ErrorClass) []const u8 {
        return switch (self) {
            .framework => "Framework",
            .ai => "AI/ML",
            .database => "Database",
            .gpu => "GPU",
            .network => "Network",
            .plugin => "Plugin",
            .monitoring => "Monitoring",
            .system => "System",
            .unknown => "Unknown",
        };
    }
};

/// Check if an error is recoverable
pub fn isRecoverable(err: anyerror) bool {
    return switch (err) {
        // Recoverable errors
        error.ConnectionTimeout,
        error.RequestFailed,
        error.RateLimitExceeded,
        error.RecordNotFound,
        error.NoGPUAvailable,
        => true,
        
        // Non-recoverable errors
        error.OutOfMemory,
        error.AlreadyInitialized,
        error.InvalidConfiguration,
        error.InsufficientComputeCapability,
        => false,
        
        else => false,
    };
}

/// Get a user-friendly error message
pub fn getMessage(err: anyerror) []const u8 {
    return switch (err) {
        // Framework
        error.InitializationFailed => "Framework initialization failed",
        error.AlreadyInitialized => "Framework is already initialized",
        error.NotInitialized => "Framework not initialized",
        error.FeatureNotAvailable => "Feature not available",
        error.FeatureAlreadyRegistered => "Feature already registered",
        error.InvalidConfiguration => "Invalid configuration",
        error.ResourceLimitExceeded => "Resource limit exceeded",
        
        // AI
        error.ModelNotFound => "Model not found",
        error.InvalidModelFormat => "Invalid model format",
        error.ModelLoadFailed => "Failed to load model",
        error.InferenceFailed => "Inference failed",
        error.TrainingFailed => "Training failed",
        error.InvalidInputDimensions => "Invalid input dimensions",
        error.AgentInitFailed => "Agent initialization failed",
        
        // Database
        error.ConnectionFailed => "Database connection failed",
        error.QueryFailed => "Query execution failed",
        error.InvalidQuery => "Invalid query",
        error.RecordNotFound => "Record not found",
        error.DuplicateKey => "Duplicate key constraint violation",
        error.TransactionFailed => "Transaction failed",
        error.IndexCreationFailed => "Index creation failed",
        
        // GPU
        error.NoGPUAvailable => "No GPU available",
        error.GPUInitFailed => "GPU initialization failed",
        error.KernelLaunchFailed => "Kernel launch failed",
        error.MemoryAllocationFailed => "GPU memory allocation failed",
        error.DeviceNotSupported => "Device not supported",
        error.DriverError => "GPU driver error",
        error.InsufficientComputeCapability => "Insufficient GPU compute capability",
        
        // Network
        error.ConnectionTimeout => "Connection timeout",
        error.InvalidURL => "Invalid URL",
        error.RequestFailed => "Request failed",
        error.InvalidResponse => "Invalid response",
        error.AuthenticationFailed => "Authentication failed",
        error.RateLimitExceeded => "Rate limit exceeded",
        
        // Plugin
        error.PluginNotFound => "Plugin not found",
        error.PluginLoadFailed => "Plugin load failed",
        error.InvalidPlugin => "Invalid plugin",
        error.PluginInitFailed => "Plugin initialization failed",
        error.IncompatiblePluginVersion => "Incompatible plugin version",
        
        // Monitoring
        error.MetricsCollectionFailed => "Metrics collection failed",
        error.InvalidMetric => "Invalid metric",
        error.TraceExportFailed => "Trace export failed",
        error.LoggerInitFailed => "Logger initialization failed",
        
        // System
        error.OutOfMemory => "Out of memory",
        error.FileNotFound => "File not found",
        error.AccessDenied => "Access denied",
        
        else => "Unknown error",
    };
}

test "ErrorClass: classification" {
    const testing = std.testing;
    
    try testing.expectEqual(ErrorClass.framework, ErrorClass.fromError(error.InitializationFailed));
    try testing.expectEqual(ErrorClass.ai, ErrorClass.fromError(error.ModelNotFound));
    try testing.expectEqual(ErrorClass.database, ErrorClass.fromError(error.QueryFailed));
    try testing.expectEqual(ErrorClass.gpu, ErrorClass.fromError(error.NoGPUAvailable));
}

test "Error: recoverability check" {
    const testing = std.testing;
    
    try testing.expect(isRecoverable(error.ConnectionTimeout));
    try testing.expect(isRecoverable(error.RecordNotFound));
    try testing.expect(!isRecoverable(error.OutOfMemory));
    try testing.expect(!isRecoverable(error.InvalidConfiguration));
}

test "Error: user-friendly messages" {
    const testing = std.testing;
    
    const msg = getMessage(error.ModelNotFound);
    try testing.expectEqualStrings("Model not found", msg);
    
    const msg2 = getMessage(error.GPUInitFailed);
    try testing.expectEqualStrings("GPU initialization failed", msg2);
}
