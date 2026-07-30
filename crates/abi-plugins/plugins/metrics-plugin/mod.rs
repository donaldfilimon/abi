//! `metrics-plugin` — real implementation.
//!
//! Ported from `src/plugins/metrics-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `metrics-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "metrics-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-metrics gate.";
    const TARGET_FEATURE: &'static str = "metrics";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("metrics-plugin event (bytes={})", input.len()))
    }
}
