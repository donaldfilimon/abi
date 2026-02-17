//! ABI vNext public API surface.
//!
//! This namespace provides forward API types while legacy `abi.Framework` and
//! `abi.Config` remain available during the staged compatibility window.

pub const capability = @import("capability.zig");
pub const Capability = capability.Capability;

pub const config = @import("config.zig");
pub const AppConfig = config.AppConfig;

pub const app = @import("app.zig");
pub const App = app.App;
pub const AppBuilder = app.AppBuilder;
pub const StopOptions = app.StopOptions;
