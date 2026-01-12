//! Feature lifecycle management utilities.
//!
//! Provides generic lifecycle patterns for ABI feature modules to eliminate
//! duplicate initialization/deinitialization code. Supports both simple
//! single-threaded features and thread-safe features with mutex protection.
//!
//! Usage (Simple):
//! ```zig
//! const lifecycle = @import("framework/feature_lifecycle.zig");
//! var lc = lifecycle.FeatureLifecycle(build_options.enable_ai, .ai_disabled){};
//!
//! pub const init = lc.init;
//! pub const deinit = lc.deinit;
//! pub const isEnabled = lc.isEnabled;
//! pub const isInitialized = lc.isInitialized;
//! ```
//!
//! Usage (Thread-Safe):
//! ```zig
//! var lc = lifecycle.ThreadSafeLifecycle(build_options.enable_web, .web_disabled){};
//! ```

const std = @import("std");

/// Specific error set for feature disabled states.
/// Each feature has a dedicated error variant.
pub const FeatureDisabledError = error{
    AiDisabled,
    GpuDisabled,
    WebDisabled,
    DatabaseDisabled,
    NetworkDisabled,
    ProfilingDisabled,
    MonitoringDisabled,
    ExploreDisabled,
    LlmDisabled,
    TestDisabled,
};

/// Error type for lifecycle hook operations.
pub const LifecycleHookError = error{
    HookFailed,
    InitializationFailed,
    OutOfMemory,
};

/// Combined error set for lifecycle operations.
pub const LifecycleError = FeatureDisabledError || LifecycleHookError;

/// Feature identifier for selecting which disabled error to return.
pub const FeatureId = enum {
    ai_disabled,
    gpu_disabled,
    web_disabled,
    database_disabled,
    network_disabled,
    profiling_disabled,
    monitoring_disabled,
    explore_disabled,
    llm_disabled,
    test_disabled,

    /// Convert feature ID to the corresponding error.
    pub fn toError(self: FeatureId) FeatureDisabledError {
        return switch (self) {
            .ai_disabled => FeatureDisabledError.AiDisabled,
            .gpu_disabled => FeatureDisabledError.GpuDisabled,
            .web_disabled => FeatureDisabledError.WebDisabled,
            .database_disabled => FeatureDisabledError.DatabaseDisabled,
            .network_disabled => FeatureDisabledError.NetworkDisabled,
            .profiling_disabled => FeatureDisabledError.ProfilingDisabled,
            .monitoring_disabled => FeatureDisabledError.MonitoringDisabled,
            .explore_disabled => FeatureDisabledError.ExploreDisabled,
            .llm_disabled => FeatureDisabledError.LlmDisabled,
            .test_disabled => FeatureDisabledError.TestDisabled,
        };
    }
};

/// Generic feature lifecycle manager for simple (non-thread-safe) features.
///
/// This type provides standard init/deinit/status methods for feature modules
/// with compile-time feature gating. When `enabled` is false, `init()` returns
/// the specified error type.
///
/// Parameters:
/// - `enabled`: Comptime boolean flag (typically from build_options)
/// - `feature_id`: Which feature this lifecycle manages (determines disabled error)
pub fn FeatureLifecycle(comptime enabled: bool, comptime feature_id: FeatureId) type {
    return struct {
        initialized: bool = false,

        const Self = @This();

        /// Initialize the feature. Returns error if feature is disabled at compile time.
        pub fn init(self: *Self, allocator: std.mem.Allocator) LifecycleError!void {
            _ = allocator; // Reserved for future use (e.g., lifecycle hooks)

            if (!comptime enabled) {
                return feature_id.toError();
            }

            self.initialized = true;
        }

        /// Deinitialize the feature. Safe to call multiple times.
        pub fn deinit(self: *Self) void {
            self.initialized = false;
        }

        /// Returns whether this feature is enabled at compile time.
        /// This is a comptime-known value with zero runtime cost.
        pub fn isEnabled(_: *const Self) bool {
            return enabled;
        }

        /// Returns whether this feature is currently initialized.
        pub fn isInitialized(self: *const Self) bool {
            return self.initialized;
        }

        /// Initialize with lifecycle hooks for custom initialization logic.
        pub fn initWithHooks(
            self: *Self,
            allocator: std.mem.Allocator,
            hooks: LifecycleHooks,
        ) LifecycleError!void {
            if (!comptime enabled) {
                return feature_id.toError();
            }

            if (hooks.pre_init) |pre_init_fn| {
                try pre_init_fn(allocator);
            }
            errdefer if (hooks.on_init_error) |error_fn| error_fn();

            self.initialized = true;

            if (hooks.post_init) |post_init_fn| {
                try post_init_fn(allocator);
            }
        }

        /// Deinitialize with lifecycle hooks for custom cleanup logic.
        pub fn deinitWithHooks(self: *Self, hooks: LifecycleHooks) void {
            if (hooks.pre_deinit) |pre_deinit_fn| {
                pre_deinit_fn();
            }

            self.initialized = false;

            if (hooks.post_deinit) |post_deinit_fn| {
                post_deinit_fn();
            }
        }
    };
}

/// Thread-safe feature lifecycle manager with mutex protection.
///
/// Identical API to FeatureLifecycle but with mutex-protected state changes.
/// Use this for features accessed from multiple threads during initialization.
pub fn ThreadSafeLifecycle(comptime enabled: bool, comptime feature_id: FeatureId) type {
    return struct {
        initialized: bool = false,
        mutex: std.Thread.Mutex = .{},

        const Self = @This();

        /// Initialize the feature with mutex protection.
        pub fn init(self: *Self, allocator: std.mem.Allocator) LifecycleError!void {
            _ = allocator;

            if (!comptime enabled) {
                return feature_id.toError();
            }

            self.mutex.lock();
            defer self.mutex.unlock();

            self.initialized = true;
        }

        /// Deinitialize the feature with mutex protection.
        pub fn deinit(self: *Self) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.initialized = false;
        }

        /// Returns whether this feature is enabled at compile time.
        pub fn isEnabled(_: *const Self) bool {
            return enabled;
        }

        /// Returns whether this feature is currently initialized (thread-safe read).
        pub fn isInitialized(self: *Self) bool {
            self.mutex.lock();
            defer self.mutex.unlock();

            return self.initialized;
        }

        /// Initialize with lifecycle hooks and mutex protection.
        pub fn initWithHooks(
            self: *Self,
            allocator: std.mem.Allocator,
            hooks: LifecycleHooks,
        ) LifecycleError!void {
            if (!comptime enabled) {
                return feature_id.toError();
            }

            if (hooks.pre_init) |pre_init_fn| {
                try pre_init_fn(allocator);
            }
            errdefer if (hooks.on_init_error) |error_fn| error_fn();

            self.mutex.lock();
            defer self.mutex.unlock();

            self.initialized = true;

            if (hooks.post_init) |post_init_fn| {
                try post_init_fn(allocator);
            }
        }

        /// Deinitialize with lifecycle hooks and mutex protection.
        pub fn deinitWithHooks(self: *Self, hooks: LifecycleHooks) void {
            if (hooks.pre_deinit) |pre_deinit_fn| {
                pre_deinit_fn();
            }

            self.mutex.lock();
            defer self.mutex.unlock();

            self.initialized = false;

            if (hooks.post_deinit) |post_deinit_fn| {
                post_deinit_fn();
            }
        }
    };
}

/// Lifecycle hooks for custom initialization/deinitialization behavior.
///
/// All hooks are optional. Use this to inject custom logic around the
/// standard lifecycle operations.
pub const LifecycleHooks = struct {
    /// Called before initialization begins
    pre_init: ?*const fn (std.mem.Allocator) LifecycleHookError!void = null,

    /// Called after successful initialization
    post_init: ?*const fn (std.mem.Allocator) LifecycleHookError!void = null,

    /// Called if initialization fails
    on_init_error: ?*const fn () void = null,

    /// Called before deinitialization begins
    pre_deinit: ?*const fn () void = null,

    /// Called after deinitialization completes
    post_deinit: ?*const fn () void = null,
};

// ============================================================================
// Tests
// ============================================================================

test "FeatureLifecycle enabled feature" {
    var lifecycle = FeatureLifecycle(true, .test_disabled){};

    try std.testing.expect(!lifecycle.isInitialized());
    try std.testing.expect(lifecycle.isEnabled());

    try lifecycle.init(std.testing.allocator);
    try std.testing.expect(lifecycle.isInitialized());

    lifecycle.deinit();
    try std.testing.expect(!lifecycle.isInitialized());
}

test "FeatureLifecycle disabled feature" {
    var lifecycle = FeatureLifecycle(false, .test_disabled){};

    try std.testing.expect(!lifecycle.isInitialized());
    try std.testing.expect(!lifecycle.isEnabled());

    const result = lifecycle.init(std.testing.allocator);
    try std.testing.expectError(FeatureDisabledError.TestDisabled, result);
    try std.testing.expect(!lifecycle.isInitialized());

    lifecycle.deinit(); // Safe to call even when not initialized
    try std.testing.expect(!lifecycle.isInitialized());
}

test "FeatureLifecycle multiple init/deinit cycles" {
    var lifecycle = FeatureLifecycle(true, .test_disabled){};

    for (0..3) |_| {
        try lifecycle.init(std.testing.allocator);
        try std.testing.expect(lifecycle.isInitialized());
        lifecycle.deinit();
        try std.testing.expect(!lifecycle.isInitialized());
    }
}

test "ThreadSafeLifecycle enabled feature" {
    var lifecycle = ThreadSafeLifecycle(true, .test_disabled){};

    try std.testing.expect(!lifecycle.isInitialized());
    try std.testing.expect(lifecycle.isEnabled());

    try lifecycle.init(std.testing.allocator);
    try std.testing.expect(lifecycle.isInitialized());

    lifecycle.deinit();
    try std.testing.expect(!lifecycle.isInitialized());
}

test "ThreadSafeLifecycle disabled feature" {
    var lifecycle = ThreadSafeLifecycle(false, .test_disabled){};

    try std.testing.expect(!lifecycle.isInitialized());
    try std.testing.expect(!lifecycle.isEnabled());

    const result = lifecycle.init(std.testing.allocator);
    try std.testing.expectError(FeatureDisabledError.TestDisabled, result);
    try std.testing.expect(!lifecycle.isInitialized());
}
