//! Shared stub context type for feature modules.
//!
//! When a feature is disabled at build time, its stub module must still export
//! a Context type with init(allocator, config) and deinit(self) so the framework
//! and callers compile. This module provides a generic allocator-only context
//! for stubs that do not need to store config or any state.
//!
//! Use in stub.zig like:
//!   pub const Context = stub_context.StubContext(YourConfigType);

const std = @import("std");

/// Generic stub context: allocator only, config ignored.
/// Use for feature stubs that need a no-op Context matching the real module's API.
pub fn StubContext(comptime ConfigT: type) type {
    return struct {
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, _: ConfigT) !*@This() {
            const ctx = try allocator.create(@This());
            ctx.* = .{ .allocator = allocator };
            return ctx;
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.destroy(self);
        }
    };
}

/// Generic stub context: allocator + config storage.
/// Use for feature stubs that keep config for API parity or diagnostics.
pub fn StubContextWithConfig(comptime ConfigT: type) type {
    return struct {
        allocator: std.mem.Allocator,
        config: ConfigT,

        pub fn init(allocator: std.mem.Allocator, config: ConfigT) !*@This() {
            const ctx = try allocator.create(@This());
            ctx.* = .{ .allocator = allocator, .config = config };
            return ctx;
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.destroy(self);
        }
    };
}

/// Comptime generic that generates the common stub boilerplate functions:
/// `init`, `deinit`, `isEnabled`, `isInitialized`.
///
/// Use for feature stubs whose `init` takes `(allocator, config)`.
///
/// Example in a stub.zig:
/// ```
/// const Stub = stub_context.StubFeature(CacheConfig, CacheError);
/// pub const Context = Stub.Context;
/// pub const init = Stub.init;
/// pub const deinit = Stub.deinit;
/// pub const isEnabled = Stub.isEnabled;
/// pub const isInitialized = Stub.isInitialized;
/// ```
pub fn StubFeature(comptime ConfigType: type, comptime ErrorType: type) type {
    return struct {
        pub const Context = StubContextWithConfig(ConfigType);

        pub fn init(_: std.mem.Allocator, _: ConfigType) ErrorType!void {
            return error.FeatureDisabled;
        }

        pub fn deinit() void {}

        pub fn isEnabled() bool {
            return false;
        }

        pub fn isInitialized() bool {
            return false;
        }
    };
}

/// Same as `StubFeature` but for features whose `init` takes only an allocator
/// (no config parameter).
///
/// Example in a stub.zig:
/// ```
/// const Stub = stub_context.StubFeatureNoConfig(GpuError);
/// pub const init = Stub.init;
/// pub const deinit = Stub.deinit;
/// pub const isEnabled = Stub.isEnabled;
/// pub const isInitialized = Stub.isInitialized;
/// ```
pub fn StubFeatureNoConfig(comptime ErrorType: type) type {
    return struct {
        pub fn init(_: std.mem.Allocator) ErrorType!void {
            return error.FeatureDisabled;
        }

        pub fn deinit() void {}

        pub fn isEnabled() bool {
            return false;
        }

        pub fn isInitialized() bool {
            return false;
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
