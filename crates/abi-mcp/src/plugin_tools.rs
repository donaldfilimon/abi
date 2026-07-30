//! Backing implementations for the `plugin_list` and `plugin_run` MCP tools.
//!
//! Ported from `src/mcp/plugin_tools.zig`. Both tools construct a fresh
//! [`PluginManager`] per call and load the bundled set, exactly as the Zig
//! handlers did — no plugin mutates shared state, so there is nothing for a
//! long-lived manager to preserve between calls.

use std::fmt::Write as _;

use abi_plugins::{PluginManager, manager::PluginRunError};

/// Render the `plugin_list` response text.
///
/// The format is frozen contract: `tests/golden/mcp-tool-calls.jsonl` pins it,
/// including the trailing `;` after every entry and the fact that the order is
/// declaration order rather than alphabetical.
#[must_use]
pub fn plugin_list_text() -> String {
    let mut manager = PluginManager::new();
    manager.load_bundled();

    let mut out = format!("plugins count={}", manager.plugin_count());
    for plugin in manager.list() {
        let _ = write!(
            out,
            " name={} version={} target={} entry={} description={};",
            plugin.name,
            plugin.version,
            plugin.target_feature,
            plugin.entry_point,
            plugin.description,
        );
    }
    out
}

/// Run the bundled plugin named `name` against `input`.
///
/// `input` is `""` when the caller omits it, matching Zig's
/// `objectString(...) catch ""`.
pub fn run_plugin(name: &str, input: &str) -> Result<String, PluginRunError> {
    let mut manager = PluginManager::new();
    manager.load_bundled();
    manager.run(name, input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plugin_list_reports_all_sixteen_in_declaration_order() {
        let text = plugin_list_text();
        assert!(text.starts_with("plugins count=16 name=example-plugin"));
        // telemetry-exporter is third in declaration order but would be
        // fifteenth alphabetically; this is the cheapest guard against a sort
        // sneaking in.
        let example_wdbx = text.find("name=example-wdbx-plugin").expect("present");
        let telemetry = text.find("name=telemetry-exporter").expect("present");
        assert!(example_wdbx < telemetry);
        assert!(text.ends_with(';'));
        assert_eq!(text.matches(" name=").count(), 16);
    }

    #[test]
    fn plugin_run_dispatches_to_the_bundled_implementation() {
        assert_eq!(
            run_plugin("example-plugin", "").unwrap(),
            "example-plugin received input (len=0)"
        );
        assert_eq!(
            run_plugin("example-plugin", "hello").unwrap(),
            "example-plugin received input (len=5)"
        );
    }

    #[test]
    fn plugin_run_rejects_an_unknown_name() {
        assert_eq!(
            run_plugin("no-such-plugin", ""),
            Err(PluginRunError::PluginNotFound)
        );
    }
}
