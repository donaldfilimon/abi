//! `mobile-plugin` — real implementation.
//!
//! Ported from `src/plugins/mobile-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `mobile-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "mobile-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-mobile gate.";
    const TARGET_FEATURE: &'static str = "mobile";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("mobile-plugin event (bytes={})", input.len()))
    }
}
