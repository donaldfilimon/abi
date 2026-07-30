//! `example-wdbx-plugin` — disabled-feature stub.
//!
//! Ported from `src/plugins/example-wdbx-plugin/stub.zig`. Metadata is identical to
//! `mod.rs` and `run` always fails; `assert_plugin_parity!` checks the
//! first half at compile time, replacing Zig's `tools/check_parity.zig`.

use crate::{Plugin, PluginError};

/// The no-op `example-wdbx-plugin` stub used when its feature gate is off.
pub struct Stub;

impl Plugin for Stub {
    const NAME: &'static str = "example-wdbx-plugin";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example WDBX plugin used by multi-plugin registry contract tests.";
    const TARGET_FEATURE: &'static str = "wdbx";

    fn run(_input: &str) -> Result<String, PluginError> {
        Err(PluginError::FeatureDisabled)
    }
}
