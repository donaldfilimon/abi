//! Golden compatibility checks captured from the Zig CLI.

use abi_cli::usage::{
    COMMANDS, SHORTCUTS, command_name_for_shortcut, render_command_help, render_usage,
};
use serde_json::Value;

const HELP_TEXT: &str = include_str!("../../../tests/golden/help.txt");
const HELP_JSON: &str = include_str!("../../../tests/golden/help.json");
const BACKENDS_HELP: &str = include_str!("../../../tests/golden/help-backends.txt");

#[test]
fn top_level_help_is_byte_exact() {
    assert_eq!(render_usage(), HELP_TEXT);
}

#[test]
fn command_metadata_matches_the_captured_contract() {
    let document: Value = serde_json::from_str(HELP_JSON).expect("valid captured help JSON");
    let captured = document["commands"]
        .as_array()
        .expect("commands must be an array");

    assert_eq!(captured.len(), COMMANDS.len());
    for (actual, expected) in captured.iter().zip(COMMANDS) {
        assert_eq!(actual["name"].as_str(), Some(expected.name));
        assert_eq!(actual["usage"].as_str(), Some(expected.usage));
        assert_eq!(actual["summary"].as_str(), Some(expected.summary));
        assert_eq!(actual["category"].as_str(), Some(expected.category.name()));
        assert_eq!(actual["details"].as_str(), Some(expected.details));

        let examples: Vec<&str> = actual["examples"]
            .as_array()
            .expect("examples must be an array")
            .iter()
            .map(|value| value.as_str().expect("example must be a string"))
            .collect();
        assert_eq!(examples, expected.examples);
    }
}

#[test]
fn shortcuts_match_the_captured_contract() {
    let document: Value = serde_json::from_str(HELP_JSON).expect("valid captured help JSON");
    let captured = document["shortcuts"]
        .as_array()
        .expect("shortcuts must be an array");

    assert_eq!(captured.len(), SHORTCUTS.len());
    for (actual, expected) in captured.iter().zip(SHORTCUTS) {
        assert_eq!(actual["token"].as_str(), Some(expected.token));
        assert_eq!(actual["command"].as_str(), Some(expected.command));
        assert_eq!(actual["summary"].as_str(), Some(expected.summary));
    }
    assert_eq!(command_name_for_shortcut("--tui"), Some("tui"));
}

#[test]
fn argument_free_command_help_is_byte_exact() {
    let (rendered, exit_code) = render_command_help("backends");
    assert_eq!(exit_code, 0);
    assert_eq!(rendered, BACKENDS_HELP);
}
