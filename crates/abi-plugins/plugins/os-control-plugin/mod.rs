//! `os-control-plugin` — real implementation.
//!
//! Ported from `src/plugins/os-control-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `os-control-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "os-control-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-os-control gate.";
    const TARGET_FEATURE: &'static str = "os-control";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("os-control-plugin event (bytes={})", input.len()))
    }
}
