const std = @import("std");

const backend_factory = @import("../backend_factory.zig");
pub const simulated = @import("simulated.zig");

pub const BackendFactory = backend_factory.BackendFactory;
pub const BackendInstance = backend_factory.BackendInstance;
pub const FactoryError = backend_factory.FactoryError;
pub const BackendFeature = backend_factory.BackendFeature;
pub const SelectionOptions = backend_factory.SelectionOptions;

pub const createBackend = backend_factory.createBackend;
pub const createBestBackend = backend_factory.createBestBackend;
pub const destroyBackend = backend_factory.destroyBackend;
pub const createVTableBackend = backend_factory.createVTableBackend;
pub const createBestVTableBackend = backend_factory.createBestVTableBackend;

pub const listAvailableBackends = backend_factory.listAvailableBackends;
pub const detectAvailableBackends = backend_factory.detectAvailableBackends;
pub const isBackendAvailable = backend_factory.isBackendAvailable;
pub const selectBestBackendWithFallback = backend_factory.selectBestBackendWithFallback;
pub const selectBackendWithFeatures = backend_factory.selectBackendWithFeatures;

test {
    std.testing.refAllDecls(@This());
}
