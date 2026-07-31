//! `abi agent tui` — the line-mode agent REPL.
//!
//! Interactive raw-mode TUI is **not linked**. This is the honest non-TTY
//! fallback: read stdin lines until EOF, dispatch `/slash` commands, and
//! complete free text through the local persona router with budgeted
//! `file_context`. Extracted from `agent.rs` so that module owns dispatch and
//! the one-shot subcommands while REPL state and slash handling live here.

use std::fmt::Write as _;

use abi_ai::{analyze_sentiment, route_to_profile, select_best_profile};

use crate::app::Outcome;

const TUI_HELP: &str = "\
ABI agent line-mode (interactive raw TUI not linked)
commands:
  /help              this text
  /model <id>        switch the completion model (alias-resolved)
  /profile           show profile routing status
  /quit /exit /q     leave the REPL
  /status /stat      session counters and model
  /context           file_context / open-file summary
  /history /hist     recent turn previews
  /reset             clear history and bump session id
  /features /feat    build-time feature migration flags
  /clear /cls        clear screen (ANSI)
  free text          local persona completion with file_context
";

/// Longest model id the line-mode REPL will accept via `/model`.
///
/// Ported from Zig's `MODEL_STORAGE_BYTES`. Zig needed this as a fixed-size
/// backing array for `state.config.model`; here it survives only as the same
/// length cap on an owned `String`, so a `/model` argument this long is
/// rejected the same way in both ports.
const MODEL_STORAGE_BYTES: usize = 128;

/// Whether `id` is acceptable as a `/model` argument.
///
/// Ported from Zig's `validModelId`: every byte must be printable ASCII
/// excluding space (`0x21..=0x7e`), and the id must be non-empty and no longer
/// than [`MODEL_STORAGE_BYTES`]. Zig additionally checked
/// `std.ascii.isWhitespace`, which is redundant here (and there): every ASCII
/// whitespace byte — space, tab, newline, CR, and the rest — is already below
/// `0x21`, so the range check alone rejects internal whitespace such as a
/// two-word id. The range's upper bound also rejects DEL (`0x7f`) and every
/// non-ASCII byte, including each byte of a multi-byte UTF-8 character, so a
/// model id is ASCII by construction rather than by a separate check.
fn valid_model_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MODEL_STORAGE_BYTES
        && id.bytes().all(|byte| (0x21..0x7f).contains(&byte))
}

struct LineModeState {
    session_id: i64,
    turn_count: usize,
    history: Vec<String>,
    /// Owned rather than `&'static str`: `/model` can set this to a freeform,
    /// non-catalog id (Zig accepted any `validModelId` string, not only known
    /// catalog entries), which has no `'static` lifetime to borrow.
    model: String,
}

enum SlashAction {
    Quit,
    Continue,
}

fn emit_line(responses: &mut String, stdout: &mut std::io::Stdout, line: &str) {
    use std::io::Write;
    let _ = writeln!(responses, "{line}");
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

fn slash_status(state: &LineModeState, responses: &mut String, stdout: &mut std::io::Stdout) {
    // `session_id=`, not `session=`: matches Zig's `formatStatusLine`.
    emit_line(
        responses,
        stdout,
        &format!(
            "status: session_id={} turns={} history={} model={} provider={} sea=off live=off store=off mode=line",
            state.session_id,
            state.turn_count,
            state.history.len(),
            state.model,
            abi_ai::models::provider_of(&state.model).label(),
        ),
    );
}

/// `/profile`: report the routing mode and current model. Ported from Zig's
/// `showProfileStatus`.
fn slash_profile(state: &LineModeState, responses: &mut String, stdout: &mut std::io::Stdout) {
    emit_line(
        responses,
        stdout,
        &format!(
            "profile: adaptive router active; model={}; turns={}",
            state.model, state.turn_count
        ),
    );
}

/// `/model <id>`: switch the completion model. Ported from Zig's `applyModel`.
///
/// An empty argument prints usage without touching `state.model`. An argument
/// that fails [`valid_model_id`] (after alias resolution — so `/model
/// fable-5` still normalizes to `claude-fable-5` before validation) reports the
/// same rejection message and, deliberately, leaves `state.model` unchanged:
/// the whole point of validating first is that a rejected `/model` cannot move
/// the session into an inconsistent state.
fn slash_model(
    state: &mut LineModeState,
    arg: &str,
    responses: &mut String,
    stdout: &mut std::io::Stdout,
) {
    if arg.is_empty() {
        emit_line(responses, stdout, "usage: /model <id>");
        return;
    }
    let canonical = abi_ai::models::canonical(arg);
    if !valid_model_id(canonical) {
        emit_line(
            responses,
            stdout,
            &format!(
                "model id must be printable non-whitespace ASCII and at most {MODEL_STORAGE_BYTES} bytes"
            ),
        );
        return;
    }
    state.model = canonical.to_string();
    emit_line(responses, stdout, &format!("model set to {}", state.model));
}

fn slash_context(responses: &mut String, stdout: &mut std::io::Stdout) {
    let sample = abi_ai::build_agent_context(
        "(context probe)",
        std::path::Path::new("."),
        512,
        &abi_ai::AgentContextOptions {
            include_tree: true,
            include_git_diff: false,
            ..abi_ai::AgentContextOptions::default()
        },
    );
    let preview: String = sample.chars().take(240).collect();
    emit_line(
        responses,
        stdout,
        &format!(
            "context: budget={} file_context=on tree=on preview={preview}…",
            abi_ai::DEFAULT_BUDGET_BYTES
        ),
    );
}

fn slash_history(state: &LineModeState, responses: &mut String, stdout: &mut std::io::Stdout) {
    if state.history.is_empty() {
        emit_line(responses, stdout, "history: (empty)");
        return;
    }
    emit_line(
        responses,
        stdout,
        &format!("history: {} turn(s)", state.history.len()),
    );
    for (i, h) in state.history.iter().enumerate().rev().take(8) {
        let short: String = h.chars().take(80).collect();
        emit_line(
            responses,
            stdout,
            &format!("  [{}] {short}", state.history.len() - i),
        );
    }
}

fn handle_slash(
    trimmed: &str,
    state: &mut LineModeState,
    responses: &mut String,
    stdout: &mut std::io::Stdout,
) -> SlashAction {
    use std::io::Write;

    let cmd = trimmed
        .split_whitespace()
        .next()
        .unwrap_or(trimmed)
        .to_ascii_lowercase();
    // Everything after the command token, with the single separating
    // whitespace run removed but internal whitespace preserved — so
    // `/model two words` yields the argument `"two words"`, not `"two"`, which
    // is what makes `valid_model_id` see (and reject) the embedded space.
    let arg = trimmed
        .split_once(char::is_whitespace)
        .map_or("", |(_, rest)| rest.trim());
    match cmd.as_str() {
        "/quit" | "/exit" | "/q" => SlashAction::Quit,
        "/help" | "/h" => {
            emit_line(responses, stdout, TUI_HELP.trim_end());
            SlashAction::Continue
        }
        "/model" => {
            slash_model(state, arg, responses, stdout);
            SlashAction::Continue
        }
        "/profile" => {
            slash_profile(state, responses, stdout);
            SlashAction::Continue
        }
        "/status" | "/stat" => {
            slash_status(state, responses, stdout);
            SlashAction::Continue
        }
        "/context" => {
            slash_context(responses, stdout);
            SlashAction::Continue
        }
        "/history" | "/hist" => {
            slash_history(state, responses, stdout);
            SlashAction::Continue
        }
        "/reset" => {
            state.turn_count = 0;
            state.history.clear();
            emit_line(
                responses,
                stdout,
                &format!("session reset (id={})", state.session_id),
            );
            SlashAction::Continue
        }
        "/features" | "/feat" => {
            emit_line(
                responses,
                stdout,
                "features: ai=on sea=on nn=on wdbx=on gpu=detect-only tui=line-mode connectors=local+live-anthropic",
            );
            SlashAction::Continue
        }
        "/clear" | "/cls" => {
            let _ = write!(stdout, "\x1b[2J\x1b[H");
            let _ = stdout.flush();
            emit_line(responses, stdout, "(cleared)");
            SlashAction::Continue
        }
        other => {
            emit_line(
                responses,
                stdout,
                &format!("unknown command: {other} (try /help)"),
            );
            SlashAction::Continue
        }
    }
}

/// Line-mode agent REPL for non-TTY / scripted use.
///
/// Reads stdin lines until EOF; each non-empty line is completed via the local
/// persona router with budgeted `file_context`. Interactive raw-mode TUI is not
/// linked — this is the honest non-TTY fallback Zig also takes when stdin is
/// not a terminal.
pub(crate) fn line_mode() -> Outcome {
    use std::io::{BufRead, Write};

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut banner = String::new();
    let _ = writeln!(
        banner,
        "ABI agent line-mode (interactive raw TUI not linked). Type a prompt or /help; empty line or EOF to quit."
    );
    let _ = stdout.write_all(banner.as_bytes());
    let _ = stdout.flush();

    let mut responses = String::new();
    let root = std::path::Path::new(".");
    let mut state = LineModeState {
        session_id: abi_foundation::time::unix_ms(),
        turn_count: 0,
        history: Vec::new(),
        model: abi_ai::models::DEFAULT_MODEL.to_string(),
    };

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if trimmed.starts_with('/') {
            if matches!(
                handle_slash(trimmed, &mut state, &mut responses, &mut stdout),
                SlashAction::Quit
            ) {
                break;
            }
            continue;
        }

        let augmented = abi_ai::build_agent_context(
            trimmed,
            root,
            abi_ai::DEFAULT_BUDGET_BYTES,
            &abi_ai::AgentContextOptions {
                include_tree: true,
                include_git_diff: true,
                git_stat_only: true,
                ..abi_ai::AgentContextOptions::default()
            },
        );
        let selected = select_best_profile(analyze_sentiment(trimmed));
        let body = route_to_profile(selected, &augmented);
        state.turn_count += 1;
        state.history.push(format!("{trimmed} → {body}"));
        emit_line(&mut responses, &mut stdout, &body);
        let _ = writeln!(responses);
        let _ = writeln!(stdout);
        let _ = stdout.flush();
    }
    if responses.is_empty() {
        return Outcome {
            stdout: "ABI agent line-mode ready (no input lines).\n".into(),
            stderr: String::new(),
            exit_code: 0,
        };
    }
    Outcome {
        stdout: responses,
        stderr: String::new(),
        exit_code: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_mode_state(model: &str) -> LineModeState {
        LineModeState {
            session_id: 42,
            turn_count: 0,
            history: Vec::new(),
            model: model.to_owned(),
        }
    }

    /// Dispatch one REPL line and return only what it wrote to `responses`.
    ///
    /// `handle_slash` also writes to a live `std::io::Stdout`, so this prints
    /// during a `cargo test` run without `--nocapture` shows nothing — libtest
    /// captures it — but running with `--nocapture` will show duplicate lines.
    /// That is a property of the REPL loop under test, not of the test.
    fn dispatch(line: &str, state: &mut LineModeState) -> String {
        let mut responses = String::new();
        let mut stdout = std::io::stdout();
        handle_slash(line, state, &mut responses, &mut stdout);
        responses
    }

    #[test]
    fn model_command_sets_a_valid_model_and_reports_it() {
        let mut state = line_mode_state(abi_ai::models::DEFAULT_MODEL);
        let responses = dispatch("/model abi-local", &mut state);
        assert_eq!(state.model, "abi-local");
        assert!(responses.contains("model set to abi-local"));
    }

    #[test]
    fn model_command_rejects_a_whitespace_containing_id() {
        // The exact case the smoke test exercises: a two-word "id" must be
        // rejected, and rejection must not perturb the session's model.
        let mut state = line_mode_state("abi-local");
        let responses = dispatch("/model two words", &mut state);
        assert_eq!(
            state.model, "abi-local",
            "a rejected /model must leave state unchanged"
        );
        assert!(responses.contains("model id must be printable non-whitespace ASCII"));
        assert!(!responses.contains("model set to two"));
    }

    #[test]
    fn model_command_resolves_aliases_before_validating() {
        // "/model fable-5" should store the canonical id, matching the
        // alias-resolution `/model <id>` in the help text promises.
        let mut state = line_mode_state(abi_ai::models::DEFAULT_MODEL);
        dispatch("/model fable-5", &mut state);
        assert_eq!(state.model, "claude-fable-5");
    }

    #[test]
    fn model_command_with_no_argument_prints_usage_and_changes_nothing() {
        let mut state = line_mode_state("abi-local");
        let responses = dispatch("/model", &mut state);
        assert_eq!(state.model, "abi-local");
        assert!(responses.contains("usage: /model <id>"));
    }

    #[test]
    fn profile_command_reports_the_current_model_and_turn_count() {
        let mut state = line_mode_state("abi-local");
        state.turn_count = 3;
        let responses = dispatch("/profile", &mut state);
        assert!(responses.contains("profile: adaptive router active; model=abi-local; turns=3"));
    }

    #[test]
    fn status_reports_session_id_not_session() {
        // The Rust port originally printed `session=`; the smoke test (and
        // Zig's formatStatusLine) expect `session_id=`.
        let mut state = line_mode_state("abi-local");
        let responses = dispatch("/status", &mut state);
        assert!(responses.contains("status: session_id=42"));
    }

    #[test]
    fn help_lists_the_model_and_profile_commands() {
        assert!(TUI_HELP.contains("/model"));
        assert!(TUI_HELP.contains("/profile"));
    }

    #[test]
    fn valid_model_id_matches_the_zig_reference_cases() {
        // Verbatim from Zig's "validModelId rejects controls whitespace and
        // overlong ids" test.
        assert!(valid_model_id("abi-local"));
        assert!(valid_model_id("ollama/qwen2"));
        assert!(!valid_model_id(""));
        assert!(!valid_model_id("two words"));
        assert!(!valid_model_id("bad\tid"));
        assert!(!valid_model_id("bad\x1bid"));
        assert!(valid_model_id(&"a".repeat(MODEL_STORAGE_BYTES)));
        assert!(!valid_model_id(&"a".repeat(MODEL_STORAGE_BYTES + 1)));
    }
}
