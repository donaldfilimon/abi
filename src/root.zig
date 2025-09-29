const mod = @import("mod.zig");

/// Backwards-compatible access to the legacy aggregate namespace.
pub const abi = mod;

/// Re-export commonly used namespaces for callers that depend on direct
/// access via `@import("abi").foo`.
pub const ai = mod.ai;
pub const gpu = mod.gpu;
pub const database = mod.database;
pub const connectors = mod.connectors;
pub const monitoring = mod.monitoring;
pub const wdbx = mod.wdbx;
pub const utils = mod.utils;
pub const core = mod.core;
pub const platform = mod.platform;
pub const logging = mod.logging;
pub const observability = mod.observability;
pub const simd = mod.simd;
pub const framework = mod.framework;
pub const plugins = mod.plugins;
