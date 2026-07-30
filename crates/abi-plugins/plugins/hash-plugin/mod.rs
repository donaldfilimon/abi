//! `hash-plugin` — real implementation.
//!
//! Ported from `src/plugins/hash-plugin/mod.zig`. The output string is contract:
//! `plugin_run` over MCP and `abi plugin run` both surface it verbatim.

use crate::{Plugin, PluginError};

/// The enabled `hash-plugin` implementation.
pub struct Mod;

impl Plugin for Mod {
    const NAME: &'static str = "hash-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-hash gate.";
    const TARGET_FEATURE: &'static str = "hash";

    fn run(input: &str) -> Result<String, PluginError> {
        Ok(format!("hash-plugin event (bytes={})", input.len()))
    }
}
