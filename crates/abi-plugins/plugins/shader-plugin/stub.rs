//! `shader-plugin` — disabled-feature stub.
//!
//! Ported from `src/plugins/shader-plugin/stub.zig`. Metadata is identical to
//! `mod.rs` and `run` always fails; `assert_plugin_parity!` checks the
//! first half at compile time, replacing Zig's `tools/check_parity.zig`.

use crate::{Plugin, PluginError};

/// The no-op `shader-plugin` stub used when its feature gate is off.
pub struct Stub;

impl Plugin for Stub {
    const NAME: &'static str = "shader-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example reference plugin targeting the feat-shader gate.";
    const TARGET_FEATURE: &'static str = "shader";

    fn run(_input: &str) -> Result<String, PluginError> {
        Err(PluginError::FeatureDisabled)
    }
}
