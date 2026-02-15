//! Composable Error Hierarchy
//!
//! Defines the framework's error taxonomy as composable error sets.
//! Feature modules can import and extend these base categories.
//! `FrameworkError` composes lifecycle, feature, config, and allocator errors.

/// Lifecycle errors for framework state transitions.
pub const LifecycleError = error{
    AlreadyInitialized,
    NotInitialized,
    InitializationFailed,
    FeatureInitFailed,
    FeatureDisabled,
    InvalidState,
    EngineCreationFailed,
};

/// GPU feature errors visible at the framework level.
pub const GpuFrameworkError = error{
    GpuDisabled,
    NoDeviceAvailable,
    InvalidConfig,
    KernelCompilationFailed,
    KernelExecutionFailed,
};

/// AI feature errors visible at the framework level.
pub const AiFrameworkError = error{
    AiDisabled,
    LlmDisabled,
    EmbeddingsDisabled,
    AgentsDisabled,
    TrainingDisabled,
    ModelNotFound,
    InferenceFailed,
};

/// Database feature errors visible at the framework level.
pub const DatabaseFrameworkError = error{
    DatabaseDisabled,
    ConnectionFailed,
    QueryFailed,
    IndexError,
    StorageError,
};

/// Network feature errors visible at the framework level.
pub const NetworkFrameworkError = error{
    NetworkDisabled,
    NodeNotFound,
    ConsensusFailed,
    Timeout,
};

/// Observability feature errors visible at the framework level.
pub const ObservabilityFrameworkError = error{
    ObservabilityDisabled,
    MetricsError,
    TracingError,
    ExportFailed,
};

/// Web feature errors visible at the framework level.
pub const WebFrameworkError = error{
    WebDisabled,
    RequestFailed,
    InvalidUrl,
};

/// Cloud feature errors visible at the framework level.
pub const CloudFrameworkError = error{
    CloudDisabled,
    UnsupportedProvider,
    InvalidEvent,
    EventParseFailed,
    ResponseSerializeFailed,
    HandlerFailed,
    TimeoutExceeded,
    ProviderError,
};

/// Analytics feature errors visible at the framework level.
pub const AnalyticsFrameworkError = error{
    AnalyticsDisabled,
    BufferFull,
    FlushFailed,
};

/// Auth feature errors visible at the framework level.
pub const AuthFrameworkError = error{
    AuthDisabled,
    InvalidCredentials,
    TokenExpired,
    Unauthorized,
};

/// Messaging feature errors visible at the framework level.
pub const MessagingFrameworkError = error{
    MessagingDisabled,
    ChannelFull,
    ChannelClosed,
};

/// Cache feature errors visible at the framework level.
pub const CacheFrameworkError = error{
    CacheDisabled,
    CacheFull,
    KeyNotFound,
};

/// Storage feature errors visible at the framework level.
pub const StorageFrameworkError = error{
    StorageDisabled,
    ObjectNotFound,
    BucketNotFound,
    StorageFull,
};

/// Search feature errors visible at the framework level.
pub const SearchFrameworkError = error{
    SearchDisabled,
    IndexNotFound,
    InvalidQuery,
    IndexCorrupted,
};

/// All feature errors combined.
pub const AllFeatureErrors = GpuFrameworkError ||
    AiFrameworkError ||
    DatabaseFrameworkError ||
    NetworkFrameworkError ||
    ObservabilityFrameworkError ||
    WebFrameworkError ||
    CloudFrameworkError ||
    AnalyticsFrameworkError ||
    AuthFrameworkError ||
    MessagingFrameworkError ||
    CacheFrameworkError ||
    StorageFrameworkError ||
    SearchFrameworkError;

/// The complete Framework error set.
/// Composes lifecycle errors, all feature errors, and infrastructure errors.
pub const FrameworkError = LifecycleError ||
    AllFeatureErrors ||
    @import("config/mod.zig").ConfigError ||
    @import("registry/mod.zig").types.Error ||
    @import("std").mem.Allocator.Error;

// ============================================================================
// Tests
// ============================================================================

const std = @import("std");

test "lifecycle errors are distinct" {
    const err: LifecycleError = error.AlreadyInitialized;
    switch (err) {
        error.AlreadyInitialized => {},
        else => unreachable,
    }
}

test "feature errors compose without overlap" {
    // Verify the combined type compiles and variants are distinct
    const err: AllFeatureErrors = error.GpuDisabled;
    const err2: AllFeatureErrors = error.DatabaseDisabled;
    const err3: AllFeatureErrors = error.SearchDisabled;
    try std.testing.expect(err != err2);
    try std.testing.expect(err2 != err3);
}

test "framework error includes all categories" {
    // Verify FrameworkError includes lifecycle + feature + config
    const err1: FrameworkError = error.AlreadyInitialized;
    const err2: FrameworkError = error.GpuDisabled;
    const err3: FrameworkError = error.OutOfMemory;
    try std.testing.expect(err1 != err2);
    try std.testing.expect(err2 != err3);
}
