//! `mlir-plugin` — real implementation.
//!
//! Ported from `src/plugins/mlir-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `mlir-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "mlir-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-mlir gate.";
    const TARGET_FEATURE: &'static str = "mlir";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("mlir-plugin event (bytes={})", input.len()))
    }
}
