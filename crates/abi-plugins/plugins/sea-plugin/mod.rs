//! `sea-plugin` — real implementation.
//!
//! Ported from `src/plugins/sea-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `sea-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "sea-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-sea gate.";
    const TARGET_FEATURE: &'static str = "sea";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("sea-plugin event (bytes={})", input.len()))
    }
}
