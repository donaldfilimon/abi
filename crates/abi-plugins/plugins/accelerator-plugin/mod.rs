//! `accelerator-plugin` — real implementation.
//!
//! Ported from `src/plugins/accelerator-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `accelerator-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "accelerator-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-accelerator gate.";
    const TARGET_FEATURE: &'static str = "accelerator";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("accelerator-plugin event (bytes={})", input.len()))
    }
}
