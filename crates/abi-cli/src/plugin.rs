//! `abi plugin list | run <name> [input]`.
//!
//! Ported from `src/cli/handlers/plugin.zig`.
//!
//! Two details are carried over deliberately:
//!
//! - Output goes to **stderr**, because Zig printed it with `std.debug.print`.
//!   `tests/golden/plugin-list.txt` was captured from that stream.
//! - `list` renders [`abi_core::registry::Registry`] (alphabetical), while the MCP
//!   `plugin_list` tool renders the manager's load order (declaration order). The
//!   two orders differ on purpose; see the `abi-plugins` crate docs.

use std::fmt::Write as _;

use abi_core::registry::Registry;
use abi_plugins::{PluginManager, register_all};

use crate::app::{Outcome, suggestion};

/// The two tokens `abi plugin` actually dispatches on. Zig derived this same
/// pair from the registry's declared `choices` for the subcommand's positional
/// arg (`arg.parse` failing against `choices` is what drove its "did you mean"
/// suggestion); this crate has no generic choices-validated arg parser yet, so
/// the pair is spelled out here instead of derived.
const SUBCOMMANDS: [&str; 2] = ["list", "run"];

const USAGE: &str = "usage: abi plugin list | run <name> [input]";
const LIST_USAGE: &str = "usage: abi plugin list";
const RUN_USAGE: &str = "usage: abi plugin run <name> [input]";

/// Dispatch `abi plugin`, excluding the top-level command token.
///
/// Help tokens are handled upstream by `app::direct_help` for `abi plugin --help`
/// and `abi plugin <sub> --help`, matching Zig's ordering: both return before any
/// registry or manager work happens.
pub(crate) fn run(args: &[String]) -> Outcome {
    let Some(subcommand) = args.first() else {
        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
    };

    match subcommand.as_str() {
        "list" => {
            if args.len() != 1 {
                return Outcome::stderr(format!("error: {LIST_USAGE}\n"), 2);
            }
            list()
        }
        "run" => {
            let Some(name) = args.get(1) else {
                return Outcome::stderr(format!("error: {RUN_USAGE}\n"), 2);
            };
            // Zig joined the trailing arguments with a single space, so
            // `abi plugin run p a b` passes "a b" rather than only "a".
            let input = args[2..].join(" ");
            run_plugin(name, &input)
        }
        other => {
            // Matches Zig's `usageErrorWithSuggestion`: the usage line, then an
            // unconditional `hint:` line when a close-enough match exists — no
            // blank line between them (unlike `unknown_command`'s top-level
            // format, which does separate the hint from its usage block).
            let mut error = format!("error: {USAGE}\n");
            if let Some(candidate) = suggestion(other, SUBCOMMANDS.into_iter()) {
                let _ = writeln!(error, "hint: did you mean `{candidate}`?");
            }
            Outcome::stderr(error, 2)
        }
    }
}

fn list() -> Outcome {
    let mut registry = Registry::new();
    if let Err(error) = register_all(&mut registry) {
        // Unreachable with the bundled set — the golden test proves it registers
        // cleanly — but a collision must not be silent if a plugin is ever added.
        return Outcome::stderr(format!("error: plugin registry conflict: {error}\n"), 1);
    }
    Outcome::stderr(registry.format_plugin_list(), 0)
}

fn run_plugin(name: &str, input: &str) -> Outcome {
    let mut manager = PluginManager::new();
    manager.load_bundled();

    match manager.run(name, input) {
        Ok(output) => Outcome::stderr(format!("{output}\n"), 0),
        // Zig propagated `error.PluginNotFound` out of the handler, which the
        // process wrapper turned into a non-zero exit. Reporting the name keeps
        // the failure diagnosable instead of a bare error trace.
        Err(error) => Outcome::stderr(format!("error: plugin `{name}`: {error}\n"), 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN_LIST: &str = include_str!("../../../tests/golden/plugin-list.txt");

    #[test]
    fn list_matches_the_golden_fixture_but_for_the_entry_point() {
        let output = list();
        assert_eq!(output.exit_code, 0);
        assert!(output.stdout.is_empty(), "Zig printed to stderr");
        // `mod.zig` -> `mod.rs` is the one disclosed deviation; every other byte
        // of the sixteen-plugin listing is asserted here.
        assert_eq!(output.stderr.replace("(mod.rs)", "(mod.zig)"), GOLDEN_LIST);
    }

    #[test]
    fn list_is_alphabetical_unlike_the_mcp_listing() {
        let output = list();
        let ai = output.stderr.find("ai-plugin").expect("present");
        let telemetry = output.stderr.find("telemetry-exporter").expect("present");
        assert!(
            ai < telemetry,
            "the CLI listing is alphabetical, so ai-plugin precedes telemetry-exporter"
        );
    }

    #[test]
    fn run_dispatches_to_a_bundled_plugin() {
        let _telemetry = crate::telemetry_lock::acquire();
        let output = run_plugin("tui-plugin", "__cmd__:ping");
        assert_eq!(output.exit_code, 0);
        assert_eq!(output.stderr, "tui-plugin: pong\n");
    }

    #[test]
    fn run_joins_trailing_arguments_with_a_space() {
        let _telemetry = crate::telemetry_lock::acquire();
        let args = ["run", "example-plugin", "__cmd__:echo\nhello", "world"].map(String::from);
        let output = run(&args);
        assert_eq!(output.exit_code, 0);
        assert_eq!(output.stderr, "example-plugin echo: hello world\n");
    }

    #[test]
    fn run_with_no_input_passes_an_empty_string() {
        let _telemetry = crate::telemetry_lock::acquire();
        let args = ["run", "example-plugin"].map(String::from);
        let output = run(&args);
        assert_eq!(output.stderr, "example-plugin received input (len=0)\n");
    }

    #[test]
    fn run_reports_an_unknown_plugin_without_panicking() {
        let _telemetry = crate::telemetry_lock::acquire();
        let output = run_plugin("no-such-plugin", "");
        assert_eq!(output.exit_code, 1);
        assert!(output.stderr.contains("no-such-plugin"));
        assert!(output.stderr.contains("plugin not found"));
    }

    #[test]
    fn malformed_grammar_exits_two_before_touching_the_registry() {
        // The three cases Zig's own handler test pinned to exit code 2.
        assert_eq!(run(&[]).exit_code, 2);
        assert_eq!(run(&["bogus".to_string()]).exit_code, 2);
        assert_eq!(run(&["run".to_string()]).exit_code, 2);
        // `list` takes no arguments.
        assert_eq!(run(&["list".to_string(), "extra".to_string()]).exit_code, 2);
    }

    #[test]
    fn a_close_typo_earns_a_did_you_mean_hint() {
        // Ported from Zig's live behavior: `arg.parse` failing a `choices`
        // check on the registry's `plugin` command (["list", "run"]) drove
        // `suggest.usageErrorWithSuggestion`, which this mirrors for the two
        // real subcommands rather than re-deriving a generic choices parser.
        let output = run(&["rn".to_string()]);
        assert_eq!(output.exit_code, 2);
        assert!(output.stderr.contains(USAGE));
        assert!(output.stderr.contains("hint: did you mean `run`?"));
    }

    #[test]
    fn a_dissimilar_token_earns_no_hint() {
        let output = run(&["bogus".to_string()]);
        assert_eq!(output.exit_code, 2);
        assert!(!output.stderr.contains("hint:"));
    }
}
