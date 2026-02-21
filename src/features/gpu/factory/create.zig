const backend_factory = @import("../backend_factory.zig");

pub const createBackend = backend_factory.createBackend;
pub const createBestBackend = backend_factory.createBestBackend;
pub const destroyBackend = backend_factory.destroyBackend;
pub const createVTableBackend = backend_factory.createVTableBackend;
pub const createBestVTableBackend = backend_factory.createBestVTableBackend;
