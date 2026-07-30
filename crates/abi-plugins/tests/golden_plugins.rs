//! Golden compatibility checks for the bundled plugin set, captured from Zig.
//!
//! Two fixtures cover the plugin surfaces, and both are matched here except for
//! the single disclosed `entry_point` deviation described in the crate docs:
//! Zig wrote `mod.zig`, and the Rust plugins are `mod.rs`.
//!
//! A third test closes the loop the compiled-in [`abi_plugins::BUNDLED`] table
//! opens: it parses all sixteen on-disk `abi-plugin.json` files and asserts they
//! agree with the table field by field, so editing a manifest without editing the
//! table (or the reverse) fails the gate.

use std::fmt::Write as _;
use std::path::PathBuf;

use abi_core::registry::Registry;
use abi_plugins::{BUNDLED, PluginManager, register_all};

/// `abi plugin list` output, captured from the Zig CLI.
const GOLDEN_CLI_LIST: &str = include_str!("../../../tests/golden/plugin-list.txt");
/// Every `tools/call` with empty arguments, captured from the Zig MCP server.
const GOLDEN_MCP_CALLS: &str = include_str!("../../../tests/golden/mcp-tool-calls.jsonl");

/// The entry point Zig reported, and the one Rust reports in its place.
const ZIG_ENTRY_POINT: &str = "mod.zig";
const RUST_ENTRY_POINT: &str = "mod.rs";

/// Rewrite the Rust entry point back to Zig's so the rest of a line can be
/// compared byte for byte.
///
/// Deliberately *not* a blanket `mod.rs` -> `mod.zig` replacement over the whole
/// text: it only rewrites the two exact shapes the entry point appears in, so an
/// unrelated `mod.rs` occurrence could never be silently normalized away.
fn retarget_entry_point(text: &str) -> String {
    text.replace(
        &format!("({RUST_ENTRY_POINT})"),
        &format!("({ZIG_ENTRY_POINT})"),
    )
    .replace(
        &format!("entry={RUST_ENTRY_POINT} "),
        &format!("entry={ZIG_ENTRY_POINT} "),
    )
}

/// Render the MCP `plugin_list` text, mirroring `src/mcp/plugin_tools.zig`.
///
/// Duplicated from `abi-mcp` rather than imported, so this test fails if the two
/// ever disagree about the format instead of agreeing on a shared bug.
fn mcp_plugin_list_text(manager: &PluginManager) -> String {
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

#[test]
fn cli_plugin_list_matches_the_golden_fixture() {
    let mut registry = Registry::new();
    register_all(&mut registry).expect("bundled plugins register without collisions");

    let rendered = retarget_entry_point(&registry.format_plugin_list());
    assert_eq!(rendered, GOLDEN_CLI_LIST);
}

#[test]
fn mcp_plugin_list_matches_the_golden_fixture() {
    let expected = GOLDEN_MCP_CALLS
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("fixture line parses"))
        .find(|value| value["tool"] == "plugin_list")
        .expect("the fixture contains a plugin_list call");
    let expected_text = expected["response"]["result"]["content"][0]["text"]
        .as_str()
        .expect("the captured response carries text");

    let mut manager = PluginManager::new();
    manager.load_bundled();

    let rendered = retarget_entry_point(&mcp_plugin_list_text(&manager));
    assert_eq!(rendered, expected_text);
}

#[test]
fn mcp_plugin_run_matches_the_golden_fixture() {
    // The success path lives in the sibling args fixture; the empty-argument
    // fixture only carries `plugin_run`'s "Missing plugin name" validation error,
    // which `abi-mcp` covers.
    const GOLDEN_ARGS: &str = include_str!("../../../tests/golden/mcp-tool-calls-args.jsonl");

    let expected = GOLDEN_ARGS
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("fixture line parses"))
        .find(|value| value["tool"] == "plugin_run")
        .expect("the fixture contains a plugin_run call");

    let name = expected["arguments"]["name"]
        .as_str()
        .expect("plugin_run carries a name");
    let expected_text = expected["response"]["result"]["content"][0]["text"]
        .as_str()
        .expect("the captured response carries text");

    let mut manager = PluginManager::new();
    manager.load_bundled();
    // The captured call passes no `input`, which the server defaults to "".
    let output = manager.run(name, "").expect("bundled plugin runs");

    assert_eq!(output, expected_text);
}

#[test]
fn on_disk_manifests_match_the_compiled_in_table() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("the crate sits two levels below the repo root")
        .to_path_buf();

    for plugin in BUNDLED {
        let dir = repo_root.join(plugin.path);
        assert!(
            dir.is_dir(),
            "{} names a missing directory: {}",
            plugin.name,
            dir.display()
        );

        let manifest = abi_foundation::plugin_manifest::validate(&dir)
            .unwrap_or_else(|error| panic!("{} has an invalid manifest: {error}", plugin.name));

        assert_eq!(manifest.name, plugin.name);
        assert_eq!(manifest.version, plugin.version, "{}", plugin.name);
        assert_eq!(manifest.description, plugin.description, "{}", plugin.name);
        assert_eq!(
            manifest.target_feature, plugin.target_feature,
            "{}",
            plugin.name
        );
        assert_eq!(manifest.entry_point, plugin.entry_point, "{}", plugin.name);

        assert_eq!(
            manifest.commands.len(),
            plugin.commands.len(),
            "{} declares a different number of commands than the table",
            plugin.name
        );
        for (parsed, compiled) in manifest.commands.iter().zip(plugin.commands) {
            assert_eq!(parsed.name, compiled.name, "{}", plugin.name);
            assert_eq!(parsed.summary, compiled.summary, "{}", plugin.name);
            assert_eq!(parsed.aliases, compiled.aliases, "{}", plugin.name);
        }

        assert_eq!(
            manifest.context_providers.len(),
            plugin.context_providers.len(),
            "{} declares a different number of context providers than the table",
            plugin.name
        );
        for (parsed, compiled) in manifest
            .context_providers
            .iter()
            .zip(plugin.context_providers)
        {
            assert_eq!(parsed.name, compiled.name, "{}", plugin.name);
            assert_eq!(parsed.summary, compiled.summary, "{}", plugin.name);
        }
    }
}

#[test]
fn every_bundled_plugin_directory_holds_a_mod_and_stub_pair() {
    // The parity invariant's on-disk half. The compile-time half is
    // `assert_plugin_parity!`, which runs when the crate builds.
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("the crate sits two levels below the repo root")
        .to_path_buf();

    for plugin in BUNDLED {
        let dir = repo_root.join(plugin.path);
        for expected in ["mod.rs", "stub.rs", "abi-plugin.json"] {
            assert!(
                dir.join(expected).is_file(),
                "{} is missing {expected}",
                plugin.name
            );
        }
    }
}
