//! `foundationmodels-plugin` — real implementation.
//!
//! Ported from `src/plugins/foundationmodels-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `foundationmodels-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "foundationmodels-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-foundationmodels gate.";
    const TARGET_FEATURE: &'static str = "foundationmodels";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("foundationmodels-plugin event (bytes={})", input.len()))
    }
}
