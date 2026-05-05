//! Public ABI package root.
//!
//! Canonical API: `abi.<domain>` (e.g. `abi.gpu`, `abi.ai`, `abi.connectors`).

const std = @import("std");

const public_core = @import("public/core.zig");
const public_services = @import("public/services.zig");
const public_protocols = @import("public/protocols.zig");
const public_features = @import("public/features.zig");

// ── Core ─────────────────────────────────────────────────────────────────

pub const config = public_core.config;
pub const Config = public_core.Config;
pub const Feature = public_core.Feature;
pub const errors = public_core.errors;
pub const FrameworkError = public_core.FrameworkError;
pub const registry = public_core.registry;
pub const Registry = public_core.Registry;
pub const framework = public_core.framework;

// ── Services (non-feature-gated) ─────────────────────────────────────────

pub const foundation = public_services.foundation;
pub const runtime = public_services.runtime;
pub const platform = public_services.platform;
pub const connectors = public_services.connectors;
pub const cli = public_services.cli;
pub const ffi = public_services.ffi;
pub const tasks = public_services.tasks;
pub const inference = public_services.inference;

// ── Protocols (comptime-gated mod/stub) ──────────────────────────────────

pub const mcp = public_protocols.mcp;
pub const lsp = public_protocols.lsp;
pub const acp = public_protocols.acp;
pub const ha = public_protocols.ha;

// ── Features (comptime-gated mod/stub) ───────────────────────────────────

pub const gpu = public_features.gpu;
pub const ai = public_features.ai;
pub const database = public_features.database;
pub const network = public_features.network;
pub const observability = public_features.observability;
pub const web = public_features.web;
pub const pages = public_features.pages;
pub const analytics = public_features.analytics;
pub const cloud = public_features.cloud;
pub const auth = public_features.auth;
pub const messaging = public_features.messaging;
pub const cache = public_features.cache;
pub const storage = public_features.storage;
pub const search = public_features.search;
pub const mobile = public_features.mobile;
pub const gateway = public_features.gateway;
pub const benchmarks = public_features.benchmarks;
pub const compute = public_features.compute;
pub const documents = public_features.documents;
pub const desktop = public_features.desktop;
pub const tui = public_features.tui;

// ── Convenience aliases ──────────────────────────────────────────────────

/// Build metadata: package version and feature catalog.
pub const meta = @import("public/meta.zig");

/// Framework application type (shorthand for `framework.Framework`).
pub const App = public_core.App;
/// Framework builder type (shorthand for `framework.FrameworkBuilder`).
pub const AppBuilder = public_core.AppBuilder;

/// Create a framework builder (shorthand for `App.builder(allocator)`).
pub fn appBuilder(allocator: std.mem.Allocator) AppBuilder {
    return App.builder(allocator);
}

/// Return the package version string (shorthand for `meta.version()`).
pub fn version() []const u8 {
    return meta.version();
}

test {
    std.testing.refAllDecls(@This());
}
