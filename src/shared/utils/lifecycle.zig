//! Module lifecycle management utilities.
//!
//! Provides a standardized pattern for module initialization and deinitialization
//! with proper synchronization for thread-safe operations.
//!
//! # Usage Example
//!
//! ```zig
//! var lifecycle = ModuleLifecycle{};
//!
//! // Initialize the module
//! try lifecycle.init(&myModuleInitFn);
//!
//! // Check if initialized
//! if (lifecycle.isInitialized()) {
//!     // Use module
//! }
//!
//! // Deinitialize when done
//! lifecycle.deinit(&myModuleDeinitFn);
//! ```

const std = @import("std");

/// Error set for module lifecycle operations.
pub const LifecycleError = error{
    /// Memory allocation failed during initialization
    OutOfMemory,
    /// Module initialization failed
    InitFailed,
    /// Module is already initialized
    AlreadyInitialized,
    /// Module failed to acquire required resources
    ResourceUnavailable,
    /// Configuration is invalid
    InvalidConfiguration,
};

/// Function pointer type for initialization functions.
pub const InitFn = *const fn () LifecycleError!void;

/// Function pointer type for deinitialization functions.
pub const DeinitFn = *const fn () void;

/// Thread-safe module lifecycle management
pub const ModuleLifecycle = struct {
    const Self = @This();

    mutex: std.Thread.Mutex = .{},
    initialized: bool = false,

    /// Initialize the module
    /// @param init_fn Function pointer that performs module initialization (takes no parameters)
    /// @return error if initialization fails
    pub fn init(self: *Self, init_fn: InitFn) LifecycleError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.initialized) {
            return;
        }

        try init_fn();
        self.initialized = true;
    }

    /// Ensure module is initialized, calling init_fn if not
    /// @param init_fn Function pointer that performs module initialization
    /// @return error if initialization fails
    pub fn ensureInitialized(self: *Self, init_fn: InitFn) LifecycleError!void {
        if (self.isInitialized()) {
            return;
        }
        return self.init(init_fn);
    }

    /// Deinitialize the module
    /// @param deinit_fn Optional function pointer that performs cleanup
    pub fn deinit(self: *Self, deinit_fn: ?DeinitFn) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (!self.initialized) {
            return;
        }

        if (deinit_fn) |fn_ptr| {
            fn_ptr();
        }

        self.initialized = false;
    }

    /// Check if module is initialized (thread-safe)
    pub fn isInitialized(self: *Self) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.initialized;
    }
};

/// Simple module lifecycle without synchronization (for single-threaded modules)
pub const SimpleModuleLifecycle = struct {
    const Self = @This();

    initialized: bool = false,

    /// Initialize the module
    /// @param init_fn Function pointer that performs module initialization
    /// @return error if initialization fails
    pub fn init(self: *Self, init_fn: InitFn) LifecycleError!void {
        if (self.initialized) {
            return;
        }

        try init_fn();
        self.initialized = true;
    }

    /// Ensure module is initialized, calling init_fn if not
    /// @param init_fn Function pointer that performs module initialization
    /// @return error if initialization fails
    pub fn ensureInitialized(self: *Self, init_fn: InitFn) LifecycleError!void {
        if (self.isInitialized()) {
            return;
        }
        return self.init(init_fn);
    }

    /// Deinitialize the module
    /// @param deinit_fn Optional function pointer that performs cleanup
    pub fn deinit(self: *Self, deinit_fn: ?DeinitFn) void {
        if (!self.initialized) {
            return;
        }

        if (deinit_fn) |fn_ptr| {
            fn_ptr();
        }

        self.initialized = false;
    }

    /// Check if module is initialized
    pub fn isInitialized(self: *Self) bool {
        return self.initialized;
    }
};

test "simple lifecycle basic operations" {
    const TestContext = struct {
        var init_called: usize = 0;
        var deinit_called: usize = 0;

        fn init() LifecycleError!void {
            init_called += 1;
        }

        fn deinit() void {
            deinit_called += 1;
        }
    };

    var lifecycle = SimpleModuleLifecycle{};
    defer lifecycle.deinit(&TestContext.deinit);

    try std.testing.expect(!lifecycle.isInitialized());
    try lifecycle.init(&TestContext.init);
    try std.testing.expectEqual(@as(usize, 1), TestContext.init_called);
    try std.testing.expect(lifecycle.isInitialized());

    try lifecycle.ensureInitialized(&TestContext.init);
    try std.testing.expectEqual(@as(usize, 1), TestContext.init_called);

    lifecycle.deinit(&TestContext.deinit);
    try std.testing.expectEqual(@as(usize, 1), TestContext.deinit_called);
    try std.testing.expect(!lifecycle.isInitialized());
}

test "threadsafe lifecycle basic operations" {
    const TestContext = struct {
        var init_called: usize = 0;
        var deinit_called: usize = 0;

        fn init() LifecycleError!void {
            _ = @atomicRmw(usize, &init_called, .Add, 1, .monotonic);
        }

        fn deinit() void {
            _ = @atomicRmw(usize, &deinit_called, .Add, 1, .monotonic);
        }
    };

    var lifecycle = ModuleLifecycle{};
    defer lifecycle.deinit(&TestContext.deinit);

    try std.testing.expect(!lifecycle.isInitialized());
    try lifecycle.init(&TestContext.init);
    try std.testing.expectEqual(@as(usize, 1), TestContext.init_called);
    try std.testing.expect(lifecycle.isInitialized());

    try lifecycle.ensureInitialized(&TestContext.init);
    try std.testing.expectEqual(@as(usize, 1), TestContext.init_called);

    lifecycle.deinit(&TestContext.deinit);
    try std.testing.expectEqual(@as(usize, 1), TestContext.deinit_called);
    try std.testing.expect(!lifecycle.isInitialized());
}

test "lifecycle handles init errors" {
    const TestContext = struct {
        fn init() LifecycleError!void {
            return LifecycleError.InitFailed;
        }
    };

    var lifecycle = SimpleModuleLifecycle{};
    defer lifecycle.deinit(null);

    try std.testing.expectError(LifecycleError.InitFailed, lifecycle.init(&TestContext.init));
    try std.testing.expect(!lifecycle.isInitialized());
}
