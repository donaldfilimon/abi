//! `shader-plugin` — real implementation.
//!
//! Ported from `src/plugins/shader-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `shader-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "shader-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-shader gate.";
    const TARGET_FEATURE: &'static str = "shader";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("shader-plugin event (bytes={})", input.len()))
    }
}
