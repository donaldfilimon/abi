// Compatibility shim to preserve legacy imports such as
// `@import("abi").ai.agent.Agent`.
const abi = @import("mod.zig");

pub const ai = abi.ai;
pub const database = abi.database;
pub const gpu = abi.gpu;
pub const web = abi.web;
pub const monitoring = abi.monitoring;
pub const connectors = abi.connectors;
pub const VectorOps = abi.VectorOps;
pub const framework = abi.framework;
pub const cli = abi.cli;
