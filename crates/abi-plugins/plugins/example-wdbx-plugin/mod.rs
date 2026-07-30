//! `example-wdbx-plugin` — real implementation.
//!
//! Ported from `src/plugins/example-wdbx-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `example-wdbx-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "example-wdbx-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example WDBX plugin used by multi-plugin registry contract tests.";
    const TARGET_FEATURE: &'static str = "wdbx";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("example-wdbx-plugin executed (input len={})", input.len()))
    }
}
