//! Process-independent CLI parsing and dispatch.
//!
//! The help surface is fully compatible with the captured Zig contract.
//! Feature handlers are deliberately rejected until their Rust ports are
//! attached, so this executable cannot accidentally claim implementation
//! parity merely because its command names exist.

use std::cmp::min;
use std::fmt::Write as _;

use serde_json::{Map, Value};

use crate::usage::{
    COMMANDS, SHORTCUTS, command_name_for_shortcut, find_command, is_help_token, render_usage,
    usage_error,
};

const HELP_USAGE: &str =
    "usage: abi help [--json|--completion <bash|zsh|fish>] [command] [subcommand]";
const HELP_JSON: &str = include_str!("../../../tests/golden/help.json");
const BASH_COMPLETION: &str = include_str!("../../../tests/golden/completion.bash");
const ZSH_COMPLETION: &str = include_str!("../../../tests/golden/completion.zsh");
const FISH_COMPLETION: &str = include_str!("../../../tests/golden/completion.fish");

fn captured_command_help(name: &str) -> Option<&'static str> {
    match name {
        "help" => Some(include_str!("../../../tests/golden/help-help.txt")),
        "complete" => Some(include_str!("../../../tests/golden/help-complete.txt")),
        "train" => Some(include_str!("../../../tests/golden/help-train.txt")),
        "agent" => Some(include_str!("../../../tests/golden/help-agent.txt")),
        "backends" => Some(include_str!("../../../tests/golden/help-backends.txt")),
        "plugin" => Some(include_str!("../../../tests/golden/help-plugin.txt")),
        "auth" => Some(include_str!("../../../tests/golden/help-auth.txt")),
        "twilio" => Some(include_str!("../../../tests/golden/help-twilio.txt")),
        "tui" => Some(include_str!("../../../tests/golden/help-tui.txt")),
        "dashboard" => Some(include_str!("../../../tests/golden/help-dashboard.txt")),
        "wdbx" => Some(include_str!("../../../tests/golden/help-wdbx.txt")),
        "scheduler" => Some(include_str!("../../../tests/golden/help-scheduler.txt")),
        "nn" => Some(include_str!("../../../tests/golden/help-nn.txt")),
        _ => None,
    }
}

/// A complete, testable process result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outcome {
    /// Bytes intended for standard output.
    pub stdout: String,
    /// Bytes intended for standard error.
    pub stderr: String,
    /// Process exit status.
    pub exit_code: u8,
}

impl Outcome {
    pub(crate) fn stderr(text: String, exit_code: u8) -> Self {
        Self {
            stdout: String::new(),
            stderr: text,
            exit_code,
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct HelpRequest<'a> {
    json: bool,
    completion: Option<&'a str>,
    names: Vec<&'a str>,
}

fn parse_help_request(args: &[String]) -> Result<HelpRequest<'_>, ()> {
    let mut request = HelpRequest::default();
    let mut index = 0;
    while index < args.len() {
        let token = args[index].as_str();
        match token {
            "--json" => {
                if request.json || request.completion.is_some() {
                    return Err(());
                }
                request.json = true;
            }
            "--completion" => {
                if request.json || request.completion.is_some() {
                    return Err(());
                }
                index += 1;
                let shell = args.get(index).ok_or(())?.as_str();
                if !matches!(shell, "bash" | "zsh" | "fish") {
                    return Err(());
                }
                request.completion = Some(shell);
            }
            _ => {
                if request.names.len() == 2 {
                    return Err(());
                }
                request.names.push(token);
            }
        }
        index += 1;
    }
    Ok(request)
}

fn completion(shell: &str) -> Option<&'static str> {
    match shell {
        "bash" => Some(BASH_COMPLETION),
        "zsh" => Some(ZSH_COMPLETION),
        "fish" => Some(FISH_COMPLETION),
        _ => None,
    }
}

fn captured_help() -> Value {
    serde_json::from_str(HELP_JSON).expect("captured help JSON is validated by contract tests")
}

fn captured_commands(document: &Value) -> &[Value] {
    document["commands"]
        .as_array()
        .expect("captured help commands are validated by contract tests")
}

fn captured_command<'a>(document: &'a Value, name: &str) -> Option<&'a Value> {
    captured_commands(document)
        .iter()
        .find(|command| command["name"].as_str() == Some(name))
}

fn captured_subcommand<'a>(command: &'a Value, name: &str) -> Option<&'a Value> {
    command["subcommands"]
        .as_array()?
        .iter()
        .find(|subcommand| subcommand["name"].as_str() == Some(name))
}

fn command_shortcuts(document: &Value, command_name: &str) -> Value {
    let shortcuts = document["shortcuts"]
        .as_array()
        .expect("captured help shortcuts are validated by contract tests")
        .iter()
        .filter(|shortcut| shortcut["command"].as_str() == Some(command_name))
        .cloned()
        .collect();
    Value::Array(shortcuts)
}

fn render_help_json(command_name: Option<&str>, subcommand_name: Option<&str>) -> Option<String> {
    if command_name.is_none() {
        return subcommand_name.is_none().then(|| HELP_JSON.to_owned());
    }

    let document = captured_help();
    let resolved_name = command_name_for_shortcut(command_name?).unwrap_or(command_name?);
    let command = captured_command(&document, resolved_name)?;

    let mut result = Map::new();
    result.insert("type".to_owned(), document["type"].clone());
    result.insert("version".to_owned(), document["version"].clone());
    result.insert("completion".to_owned(), document["completion"].clone());
    result.insert(
        "command".to_owned(),
        Value::String(resolved_name.to_owned()),
    );
    result.insert(
        "shortcuts".to_owned(),
        command_shortcuts(&document, resolved_name),
    );

    if let Some(subcommand_name) = subcommand_name {
        result.insert(
            "subcommand".to_owned(),
            captured_subcommand(command, subcommand_name)?.clone(),
        );
    } else {
        result.insert("command_detail".to_owned(), command.clone());
    }

    let mut rendered =
        serde_json::to_string(&Value::Object(result)).expect("captured help values serialize");
    rendered.push('\n');
    Some(rendered)
}

fn render_subcommand_help(command_name: &str, subcommand_name: &str) -> Option<String> {
    let document = captured_help();
    let command = captured_command(&document, command_name)?;
    let subcommand = captured_subcommand(command, subcommand_name)?;
    Some(format!(
        "usage: {}\n\n{}\n",
        subcommand["usage"].as_str()?,
        subcommand["summary"].as_str()?
    ))
}

fn render_choice_help(command_name: &str, choice_name: &str) -> Option<&'static str> {
    match (command_name, choice_name) {
        ("scheduler", "status") => Some(
            "usage: abi scheduler status\n\nRuns a one-shot scheduler status task and reports attached memory tracker counters.\n",
        ),
        ("twilio", "simulate") => Some(
            "usage: abi twilio simulate <input>\n\nOffline simulation only; live Twilio transport requires explicit connector credentials and live mode elsewhere.\n",
        ),
        _ => None,
    }
}

fn edit_distance(left: &str, right: &str) -> usize {
    let mut previous: Vec<usize> = (0..=right.chars().count()).collect();
    let mut current = vec![0; previous.len()];

    for (left_index, left_char) in left.chars().enumerate() {
        current[0] = left_index + 1;
        for (right_index, right_char) in right.chars().enumerate() {
            current[right_index + 1] = min(
                min(current[right_index] + 1, previous[right_index + 1] + 1),
                previous[right_index] + usize::from(left_char != right_char),
            );
        }
        std::mem::swap(&mut previous, &mut current);
    }
    previous[right.chars().count()]
}

/// Closest edit-distance match for `name` among `candidates`, if any is close
/// enough to be worth suggesting.
///
/// `pub(crate)` so per-command handlers with their own small, fixed choice sets
/// (see `plugin::run`) can offer the same "did you mean" hint this module
/// already gives for a mistyped top-level command or `--help` subcommand,
/// without re-deriving edit distance or its threshold.
pub(crate) fn suggestion<'a>(
    name: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> Option<&'a str> {
    candidates
        .map(|candidate| (edit_distance(name, candidate), candidate))
        .filter(|(distance, candidate)| *distance <= min(3, candidate.len() / 2 + 1))
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, candidate)| candidate)
}

fn unknown_command(name: &str) -> Outcome {
    let mut error = format!("error: unknown command '{name}'\n\n");
    if let Some(candidate) = suggestion(
        name,
        COMMANDS
            .iter()
            .map(|command| command.name)
            .chain(SHORTCUTS.iter().map(|shortcut| shortcut.token)),
    ) {
        writeln!(error, "hint: did you mean `{candidate}`?")
            .expect("writing to a String cannot fail");
    }
    error.push_str(&render_usage());
    Outcome::stderr(error, 2)
}

fn subcommand_suggestion<'a>(command: &'a Value, name: &str) -> Option<&'a str> {
    suggestion(
        name,
        command["subcommands"]
            .as_array()?
            .iter()
            .filter_map(|subcommand| subcommand["name"].as_str()),
    )
}

fn subcommand_usage_error(command_name: &str, subcommand_name: &str) -> Outcome {
    let document = captured_help();
    let Some(command) = captured_command(&document, command_name) else {
        return unknown_command(command_name);
    };
    let usage = command["usage"].as_str().unwrap_or(HELP_USAGE);
    let mut error = format!("error: {usage}\n");
    if let Some(candidate) = subcommand_suggestion(command, subcommand_name) {
        writeln!(error, "hint: did you mean `{candidate}`?")
            .expect("writing to a String cannot fail");
    }
    Outcome::stderr(error, 2)
}

fn run_help(request: &HelpRequest<'_>) -> Outcome {
    if let Some(shell) = request.completion {
        if !request.names.is_empty() {
            return Outcome::stderr(usage_error(HELP_USAGE).0, 2);
        }
        return Outcome::stderr(
            completion(shell)
                .expect("the parser accepts only supported shells")
                .to_owned(),
            0,
        );
    }

    let command_name = request
        .names
        .first()
        .map(|name| command_name_for_shortcut(name).unwrap_or(name));
    let subcommand_name = request.names.get(1).copied();

    if request.json {
        if let Some(rendered) = render_help_json(command_name, subcommand_name) {
            return Outcome::stderr(rendered, 0);
        }
        if let Some(command_name) = command_name {
            if find_command(command_name).is_none() {
                return unknown_command(request.names[0]);
            }
            if let Some(subcommand_name) = subcommand_name {
                return subcommand_usage_error(command_name, subcommand_name);
            }
        }
        return Outcome::stderr(usage_error(HELP_USAGE).0, 2);
    }

    match request.names.as_slice() {
        [] => Outcome::stderr(render_usage(), 0),
        [command] => {
            let resolved = command_name_for_shortcut(command).unwrap_or(command);
            if let Some(rendered) = captured_command_help(resolved) {
                Outcome::stderr(rendered.to_owned(), 0)
            } else {
                unknown_command(command)
            }
        }
        [command, subcommand] => {
            let resolved = command_name_for_shortcut(command).unwrap_or(command);
            if let Some(rendered) = render_subcommand_help(resolved, subcommand) {
                Outcome::stderr(rendered, 0)
            } else if find_command(resolved).is_some() {
                subcommand_usage_error(resolved, subcommand)
            } else {
                unknown_command(command)
            }
        }
        _ => Outcome::stderr(usage_error(HELP_USAGE).0, 2),
    }
}

fn direct_help(args: &[String]) -> Option<Outcome> {
    let command_name = args.first()?;
    let resolved = command_name_for_shortcut(command_name).unwrap_or(command_name);

    if args.len() == 2 && is_help_token(&args[1]) {
        return Some(if let Some(rendered) = captured_command_help(resolved) {
            Outcome::stderr(rendered.to_owned(), 0)
        } else {
            unknown_command(command_name)
        });
    }

    if args.len() == 3
        && is_help_token(&args[2])
        && let Some(rendered) = render_subcommand_help(resolved, &args[1])
            .or_else(|| render_choice_help(resolved, &args[1]).map(str::to_owned))
    {
        return Some(Outcome::stderr(rendered, 0));
    }
    None
}

/// Parse command-line arguments excluding the executable name.
///
/// Help, JSON metadata, shell completions, and help-token routing match the
/// captured Zig behavior. Commands whose handlers have not yet been ported
/// fail explicitly with exit code 1.
#[must_use]
pub fn run(args: &[String]) -> Outcome {
    let Some(command) = args.first() else {
        // Bare `abi` matches Zig: launch the diagnostics dashboard.
        return crate::dashboard::run(&[]);
    };

    if is_help_token(command) {
        return match parse_help_request(&args[1..]) {
            Ok(request) => run_help(&request),
            Err(()) => Outcome::stderr(usage_error(HELP_USAGE).0, 2),
        };
    }

    let resolved = command_name_for_shortcut(command).unwrap_or(command);
    if resolved == "wdbx" && args.get(1).is_some_and(|name| name == "simulate") {
        return crate::wdbx::run(&args[1..]);
    }

    if let Some(outcome) = direct_help(args) {
        return outcome;
    }

    if find_command(resolved).is_none() {
        return unknown_command(command);
    }

    if resolved == "wdbx" {
        return crate::wdbx::run(&args[1..]);
    }
    if resolved == "scheduler" {
        return crate::scheduler::run(&args[1..]);
    }
    if resolved == "auth" {
        return crate::auth::run(&args[1..]);
    }
    if resolved == "backends" {
        return crate::backends::run(&args[1..]);
    }
    if resolved == "plugin" {
        return crate::plugin::run(&args[1..]);
    }
    if resolved == "complete" {
        return crate::complete::run(&args[1..]);
    }
    if resolved == "train" {
        return crate::train::run(&args[1..]);
    }
    if resolved == "nn" {
        return crate::nn::run(&args[1..]);
    }
    if resolved == "dashboard" || resolved == "tui" {
        return crate::dashboard::run(&args[1..]);
    }
    if resolved == "agent" {
        return crate::agent::run(&args[1..]);
    }
    if resolved == "twilio" {
        return crate::twilio::run(&args[1..]);
    }

    // All thirteen frozen COMMANDS have handlers above. Reaching here means a
    // new name was added to `usage::COMMANDS` without a dispatch arm.
    Outcome::stderr(
        format!("error: internal: no handler registered for `{resolved}`\n"),
        1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    struct IsolatedStore {
        path: std::path::PathBuf,
    }

    impl Drop for IsolatedStore {
        fn drop(&mut self) {
            abi_foundation::env::clear_override(abi_foundation::env::WDBX_PATH);
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn with_isolated_store<T>(body: impl FnOnce(&std::path::Path) -> T) -> T {
        let _guard = abi_foundation::env::lock_for_test();
        let isolated = IsolatedStore {
            path: abi_foundation::temp_path::temp_file_path("abi-cli-app-test", "store"),
        };
        abi_foundation::env::set_override(
            abi_foundation::env::WDBX_PATH,
            &isolated.path.display().to_string(),
        );
        body(&isolated.path)
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn help_is_emitted_on_stderr() {
        let outcome = run(&args(&["help"]));
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.is_empty());
        assert_eq!(
            outcome.stderr,
            include_str!("../../../tests/golden/help.txt")
        );
    }

    #[test]
    fn global_json_and_completions_are_byte_exact() {
        assert_eq!(run(&args(&["help", "--json"])).stderr, HELP_JSON);
        for (shell, expected) in [
            ("bash", BASH_COMPLETION),
            ("zsh", ZSH_COMPLETION),
            ("fish", FISH_COMPLETION),
        ] {
            assert_eq!(
                run(&args(&["help", "--completion", shell])).stderr,
                expected
            );
        }
    }

    #[test]
    fn command_and_subcommand_help_routes_match() {
        let command = run(&args(&["complete", "--help"]));
        assert_eq!(command.exit_code, 0);
        assert_eq!(
            command.stderr,
            include_str!("../../../tests/golden/help-complete.txt")
        );

        let subcommand = run(&args(&["help", "agent", "plan"]));
        assert_eq!(subcommand.exit_code, 0);
        assert_eq!(
            subcommand.stderr,
            "usage: abi agent plan <input>\n\nRun a dry-run agent planning task through the scheduler and print memory tracker statistics.\n"
        );

        let choice_help = run(&args(&["scheduler", "status", "--help"]));
        assert_eq!(choice_help.exit_code, 0);
        assert_eq!(
            choice_help.stderr,
            "usage: abi scheduler status\n\nRuns a one-shot scheduler status task and reports attached memory tracker counters.\n"
        );
    }

    #[test]
    fn filtered_json_preserves_the_contract_shape() {
        let outcome = run(&args(&["help", "--json", "--tui"]));
        assert_eq!(outcome.exit_code, 0);
        let document: Value = serde_json::from_str(&outcome.stderr).expect("valid JSON");
        assert_eq!(document["command"], "tui");
        assert_eq!(document["command_detail"]["name"], "tui");
        assert_eq!(document["shortcuts"][0]["token"], "--tui");
    }

    #[test]
    fn malformed_help_and_unknown_commands_exit_two() {
        let malformed = run(&args(&["help", "--completion", "powershell"]));
        assert_eq!(malformed.exit_code, 2);
        assert_eq!(malformed.stderr, format!("error: {HELP_USAGE}\n"));

        let unknown = run(&args(&["complte"]));
        assert_eq!(unknown.exit_code, 2);
        assert!(unknown.stderr.contains("hint: did you mean `complete`?"));
        assert!(unknown.stderr.contains("Usage: abi <command>"));
    }

    #[test]
    fn auth_signin_without_token_fails_honestly() {
        // Without ABI_AUTH_TOKEN and with empty non-TTY stdin, signin fails loudly
        // rather than hanging or inventing a credential.
        let outcome = run(&args(&["auth", "signin", "openai"]));
        assert_eq!(outcome.exit_code, 1, "{}", outcome.stderr);
        assert!(
            outcome.stderr.contains("auth signin failed")
                || outcome.stderr.contains("empty secret")
                || outcome.stderr.contains("ABI_AUTH_TOKEN"),
            "{}",
            outcome.stderr
        );
    }

    #[test]
    fn twilio_simulate_is_attached() {
        let outcome = run(&args(&["twilio", "simulate", "hello"]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(
            outcome
                .stdout
                .contains("Twilio ConversationRelay simulation")
        );
    }

    #[test]
    fn agent_plan_is_attached() {
        let outcome = run(&args(&["agent", "plan", "inspect WDBX"]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("agent=cli-agent"));
        assert!(outcome.stdout.contains("mode=dry-run"));
    }

    #[test]
    fn dashboard_once_is_attached() {
        let outcome = run(&args(&["dashboard", "--once", "--plain"]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stderr.contains("ABI Diagnostics Dashboard"));
        assert!(outcome.stderr.contains("System"));
        assert!(outcome.stderr.contains("Plugins"));
        assert!(outcome.stderr.contains("WDBX Storage"));
        assert!(outcome.stderr.contains("Scheduler"));
        assert!(outcome.stderr.contains("Memory"));
    }

    #[test]
    fn nn_train_is_attached() {
        let outcome = run(&args(&["nn", "train", "hello world hello world "]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stdout.contains("nn train:"));
        assert!(outcome.stdout.contains("improved=true"));
    }

    #[test]
    fn train_and_complete_are_attached() {
        let train = run(&args(&["train", "example"]));
        assert_eq!(train.exit_code, 0);
        assert!(train.stdout.contains("training accepted"));

        // The helper holds the shared environment lock while `complete` reads
        // ABI_WDBX_PATH, so parallel store tests cannot redirect this call.
        let complete = with_isolated_store(|store_path| {
            let outcome = run(&args(&["complete", "hello world"]));
            assert!(
                store_path.exists(),
                "completion should write only to its isolated test store"
            );
            outcome
        });
        assert_eq!(complete.exit_code, 0);
        assert!(complete.stdout.contains("persisted=true"));
        assert!(complete.stdout.contains("profile="));
        assert!(
            complete.stdout.contains("Abbey:")
                || complete.stdout.contains("Aviva")
                || complete.stdout.contains("ABI")
        );
    }
}
