//! Public ABI package root.
//!
//! `src/abi.zig` remains the internal composition unit for the package, but
//! `src/root.zig` is the only file wired in as the package entrypoint.

const std = @import("std");
const abi_impl = @import("abi.zig");

pub const config = abi_impl.config;
pub const Config = abi_impl.Config;
pub const Feature = abi_impl.Feature;
pub const feature_catalog = abi_impl.feature_catalog;
pub const framework = abi_impl.framework;
pub const errors = abi_impl.errors;
pub const FrameworkError = abi_impl.FrameworkError;
pub const registry = abi_impl.registry;
pub const Registry = abi_impl.Registry;
pub const services = abi_impl.services;
pub const features = abi_impl.features;
pub const meta = abi_impl.meta;
pub const App = abi_impl.App;
pub const AppBuilder = abi_impl.AppBuilder;
pub const Gpu = abi_impl.Gpu;
pub const GpuBackend = abi_impl.GpuBackend;

pub fn appBuilder(allocator: std.mem.Allocator) AppBuilder {
    return abi_impl.appBuilder(allocator);
}

pub fn version() []const u8 {
    return abi_impl.version();
}

test {
    std.testing.refAllDecls(@This());
}
