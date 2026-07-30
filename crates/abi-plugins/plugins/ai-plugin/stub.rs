//! `ai-plugin` — disabled-feature stub.
//!
//! Ported from `src/plugins/ai-plugin/stub.zig`. Metadata is identical to
//! `mod.rs` and `run` always fails; `assert_plugin_parity!` checks the
//! first half at compile time, replacing Zig's `tools/check_parity.zig`.

use crate::{Plugin, PluginError};

/// The no-op `ai-plugin` stub used when its feature gate is off.
pub struct Stub;

impl Plugin for Stub {
    const NAME: &'static str = "ai-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-ai gate.";
    const TARGET_FEATURE: &'static str = "ai";

    fn run(_input: &str) -> Result<String, PluginError> {
        Err(PluginError::FeatureDisabled)
    }
}
