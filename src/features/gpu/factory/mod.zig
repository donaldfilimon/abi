pub const types = @import("types.zig");
pub const selection = @import("selection.zig");
pub const create = @import("create.zig");
pub const simulated = @import("simulated.zig");

pub const BackendFactory = types.BackendFactory;
pub const BackendInstance = types.BackendInstance;
pub const FactoryError = types.FactoryError;
pub const BackendFeature = types.BackendFeature;
pub const SelectionOptions = types.SelectionOptions;

pub const createBackend = create.createBackend;
pub const createBestBackend = create.createBestBackend;
pub const destroyBackend = create.destroyBackend;
pub const createVTableBackend = create.createVTableBackend;
pub const createBestVTableBackend = create.createBestVTableBackend;

pub const listAvailableBackends = selection.listAvailableBackends;
pub const detectAvailableBackends = selection.detectAvailableBackends;
pub const isBackendAvailable = selection.isBackendAvailable;
pub const selectBestBackendWithFallback = selection.selectBestBackendWithFallback;
pub const selectBackendWithFeatures = selection.selectBackendWithFeatures;
