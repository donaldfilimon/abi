//! `telemetry-exporter` — real implementation.
//!
//! Ported from `src/plugins/telemetry-exporter/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `telemetry-exporter` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "telemetry-exporter";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example telemetry plugin: formats a telemetry event line for the feat-telemetry observability path.";
    const TARGET_FEATURE: &'static str = "telemetry";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("telemetry-exporter event (bytes={})", input.len()))
    }
}
