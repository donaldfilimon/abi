//! Feature Lifecycle Management
//!
//! Initialization, deinitialization, and enable/disable operations for features.

const std = @import("std");
const types = @import("types.zig");

const Feature = types.Feature;
const RegistrationMode = types.RegistrationMode;
const FeatureRegistration = types.FeatureRegistration;
const Error = types.Error;

/// Initialize a registered feature. For runtime_toggle and dynamic modes.
pub fn initFeature(
    allocator: std.mem.Allocator,
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    feature: Feature,
) Error!void {
    const reg = registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

    if (reg.initialized) return Error.AlreadyInitialized;

    switch (reg.mode) {
        .comptime_only => {
            // Comptime features don't need explicit init via registry
            reg.initialized = true;
        },

        .runtime_toggle => {
            if (!reg.enabled) return Error.FeatureDisabled;

            const init_fn = reg.init_fn orelse return Error.InitializationFailed;
            const config_ptr = reg.config_ptr orelse return Error.InitializationFailed;
            reg.context_ptr = init_fn(allocator, config_ptr) catch return Error.InitializationFailed;
            reg.initialized = true;
        },

        .dynamic => {
            // Dynamic loading not yet implemented
            return Error.DynamicLoadingNotSupported;
        },
    }
}

/// Shutdown a feature, releasing resources.
pub fn deinitFeature(
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    feature: Feature,
) Error!void {
    const reg = registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

    if (!reg.initialized) return;

    switch (reg.mode) {
        .comptime_only => {
            reg.initialized = false;
        },

        .runtime_toggle => {
            if (reg.deinit_fn) |deinit_fn| {
                if (reg.context_ptr) |ptr| {
                    deinit_fn(ptr);
                }
            }
            reg.context_ptr = null;
            reg.initialized = false;
        },

        .dynamic => {
            return Error.DynamicLoadingNotSupported;
        },
    }
}

/// Enable a runtime-toggleable feature.
pub fn enableFeature(
    allocator: std.mem.Allocator,
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    runtime_overrides: *std.AutoHashMapUnmanaged(Feature, bool),
    feature: Feature,
) Error!void {
    const reg = registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

    if (reg.mode == .comptime_only) {
        return; // Already enabled, nothing to do
    }

    reg.enabled = true;
    try runtime_overrides.put(allocator, feature, true);
}

/// Disable a runtime-toggleable feature. Deinitializes if currently initialized.
pub fn disableFeature(
    allocator: std.mem.Allocator,
    registrations: *std.AutoHashMapUnmanaged(Feature, FeatureRegistration),
    runtime_overrides: *std.AutoHashMapUnmanaged(Feature, bool),
    feature: Feature,
) Error!void {
    const reg = registrations.getPtr(feature) orelse return Error.FeatureNotRegistered;

    if (reg.mode == .comptime_only) {
        return Error.InvalidMode;
    }

    // Deinit if initialized
    if (reg.initialized) {
        try deinitFeature(registrations, feature);
    }

    reg.enabled = false;
    try runtime_overrides.put(allocator, feature, false);
}
