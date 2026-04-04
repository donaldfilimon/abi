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

// ── Feature Framework Errors ─────────────────────────────────────────────

/// GPU feature errors visible at the framework level.
pub const GpuFrameworkError = @import("../gpu/types.zig").FrameworkError;

/// AI feature errors visible at the framework level.
pub const AiFrameworkError = @import("../ai/types.zig").FrameworkError;

/// Database feature errors visible at the framework level.
pub const DatabaseFrameworkError = @import("../database/types.zig").FrameworkError;

/// Network feature errors visible at the framework level.
pub const NetworkFrameworkError = @import("../network/types.zig").Error;

/// Observability feature errors visible at the framework level.
pub const ObservabilityFrameworkError = @import("../observability/types.zig").Error;

/// Web feature errors visible at the framework level.
pub const WebFrameworkError = @import("../web/types.zig").WebError;

/// Dashboard pages feature errors visible at the framework level.
pub const PagesFrameworkError = @import("../observability/pages/types.zig").PagesError;

/// Cloud feature errors visible at the framework level.
pub const CloudFrameworkError = @import("../cloud/types.zig").CloudError;

/// Analytics feature errors visible at the framework level.
pub const AnalyticsFrameworkError = @import("../analytics/types.zig").AnalyticsError;

/// Auth feature errors visible at the framework level.
pub const AuthFrameworkError = @import("../auth/types.zig").AuthError;

/// Messaging feature errors visible at the framework level.
pub const MessagingFrameworkError = @import("../messaging/types.zig").MessagingError;

/// Cache feature errors visible at the framework level.
pub const CacheFrameworkError = @import("../cache/types.zig").CacheError;

/// Storage feature errors visible at the framework level.
pub const StorageFrameworkError = @import("../storage/types.zig").StorageError;

/// Search feature errors visible at the framework level.
pub const SearchFrameworkError = @import("../search/types.zig").SearchError;

/// Mobile feature errors visible at the framework level.
pub const MobileFrameworkError = @import("../mobile/types.zig").MobileError;

/// API Gateway feature errors visible at the framework level.
pub const GatewayFrameworkError = @import("../gateway/types.zig").GatewayError;

/// Benchmarks feature errors visible at the framework level.
pub const BenchmarksFrameworkError = @import("../benchmarks/types.zig").BenchmarksError;

/// Compute feature errors visible at the framework level.
pub const ComputeFrameworkError = @import("../compute/types.zig").ComputeError;

/// Documents feature errors visible at the framework level.
pub const DocumentsFrameworkError = @import("../documents/types.zig").DocumentsError;

/// Desktop feature errors visible at the framework level.
pub const DesktopFrameworkError = @import("../desktop/types.zig").DesktopError;

/// All feature errors combined.
pub const AllFeatureErrors = GpuFrameworkError ||
    AiFrameworkError ||
    DatabaseFrameworkError ||
    NetworkFrameworkError ||
    ObservabilityFrameworkError ||
    WebFrameworkError ||
    PagesFrameworkError ||
    CloudFrameworkError ||
    AnalyticsFrameworkError ||
    AuthFrameworkError ||
    MessagingFrameworkError ||
    CacheFrameworkError ||
    StorageFrameworkError ||
    SearchFrameworkError ||
    MobileFrameworkError ||
    GatewayFrameworkError ||
    BenchmarksFrameworkError ||
    ComputeFrameworkError ||
    DocumentsFrameworkError ||
    DesktopFrameworkError;

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
        else => return error.TestFailed,
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

test {
    std.testing.refAllDecls(@This());
}
