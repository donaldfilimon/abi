//! `telemetry-exporter` — disabled-feature stub.
//!
//! Ported from `src/plugins/telemetry-exporter/stub.zig`. Metadata is identical to
//! `mod.rs` and `run` always fails; `assert_plugin_parity!` checks the
//! first half at compile time, replacing Zig's `tools/check_parity.zig`.

use crate::{Plugin, PluginError};

/// The no-op `telemetry-exporter` stub used when its feature gate is off.
pub struct Stub;

impl Plugin for Stub {
    const NAME: &'static str = "telemetry-exporter";
    const VERSION: &'static str = "0.1.0";
    const DESCRIPTION: &'static str = "Example telemetry plugin: formats a telemetry event line for the feat-telemetry observability path.";
    const TARGET_FEATURE: &'static str = "telemetry";

    fn run(_input: &str) -> Result<String, PluginError> {
        Err(PluginError::FeatureDisabled)
    }
}
