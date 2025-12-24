// Compatibility shim to preserve legacy imports such as
// `@import("abi").ai.agent.Agent`.
const abi = @import("abi.zig");
pub const ai = abi.ai;
pub const database = abi.database;
pub const gpu = abi.gpu;
pub const web = abi.web;
pub const monitoring = abi.monitoring;
pub const connectors = abi.connectors;
pub const VectorOps = abi.VectorOps;
pub const framework = abi.framework;
pub const Feature = abi.Feature;
pub const Framework = abi.Framework;
pub const FrameworkOptions = abi.FrameworkOptions;
pub const FrameworkConfiguration = abi.FrameworkConfiguration;
pub const RuntimeConfig = abi.RuntimeConfig;
pub const runtimeConfigFromOptions = abi.runtimeConfigFromOptions;
pub const init = abi.init;
pub const shutdown = abi.shutdown;
pub const version = abi.version;
