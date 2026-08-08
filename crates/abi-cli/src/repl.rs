//! `abi agent tui` — agent REPL with a small dependency-free TTY editor.
//!
//! Real TTYs get bounded history, UTF-8-safe cursor editing, and completion.
//! Pipes retain the deterministic `BufRead::lines` fallback: read until EOF,
//! dispatch `/slash` commands, and complete free text through the local
//! persona router with budgeted `file_context`.

use std::fmt::Write as _;

use abi_ai::{analyze_sentiment, route_to_profile, select_best_profile};

use crate::app::Outcome;
use crate::repl_editor::{EditorState, InteractiveRead, read_interactive_line};

const LINE_MODE_HELP: &str = "\
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

const TTY_HELP: &str = "\
ABI agent REPL (TTY editor; deterministic line-mode on pipes)
commands:
  /help              this text
  /model <id>        switch the completion model (alias-resolved)
  /profile           show profile routing status
  /sea [on|off|status|toggle]
                     show or change the session-local SEA toggle
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
    /// Session-local only. This toggle changes REPL routing metadata and never
    /// contacts a live service or mutates the durable SEA store.
    sea_enabled: bool,
    interactive: bool,
}

enum SlashAction {
    Quit,
    Continue,
}

fn emit_line(responses: &mut Option<String>, stdout: &mut std::io::Stdout, line: &str) {
    use std::io::Write;
    if let Some(responses) = responses {
        let _ = writeln!(responses, "{line}");
    }
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

fn slash_status(
    state: &LineModeState,
    responses: &mut Option<String>,
    stdout: &mut std::io::Stdout,
) {
    // `session_id=`, not `session=`: matches Zig's `formatStatusLine`.
    emit_line(
        responses,
        stdout,
        &format!(
            "status: session_id={} turns={} history={} model={} provider={} sea={} live=off store=off mode={}",
            state.session_id,
            state.turn_count,
            state.history.len(),
            state.model,
            abi_ai::models::provider_of(&state.model).label(),
            if state.sea_enabled { "on" } else { "off" },
            if state.interactive { "editor" } else { "line" },
        ),
    );
}

/// `/profile`: report the routing mode and current model. Ported from Zig's
/// `showProfileStatus`.
fn slash_profile(
    state: &LineModeState,
    responses: &mut Option<String>,
    stdout: &mut std::io::Stdout,
) {
    emit_line(
        responses,
        stdout,
        &format!(
            "profile: adaptive router active; model={}; turns={}",
            state.model, state.turn_count
        ),
    );
}

fn slash_sea(
    state: &mut LineModeState,
    arg: &str,
    responses: &mut Option<String>,
    stdout: &mut std::io::Stdout,
) {
    let next = match arg.to_ascii_lowercase().as_str() {
        "" | "status" => None,
        "on" => Some(true),
        "off" => Some(false),
        "toggle" => Some(!state.sea_enabled),
        _ => {
            emit_line(responses, stdout, "usage: /sea [on|off|status|toggle]");
            return;
        }
    };
    if let Some(enabled) = next {
        state.sea_enabled = enabled;
    }
    emit_line(
        responses,
        stdout,
        &format!(
            "sea: {} (session-local; live services=off)",
            if state.sea_enabled { "on" } else { "off" }
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
    responses: &mut Option<String>,
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

fn slash_context(responses: &mut Option<String>, stdout: &mut std::io::Stdout) {
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

fn slash_history(
    state: &LineModeState,
    responses: &mut Option<String>,
    stdout: &mut std::io::Stdout,
) {
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
    responses: &mut Option<String>,
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
            let help = if state.interactive {
                TTY_HELP
            } else {
                LINE_MODE_HELP
            };
            emit_line(responses, stdout, help.trim_end());
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
        "/sea" => {
            slash_sea(state, arg, responses, stdout);
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

fn new_state(interactive: bool) -> LineModeState {
    LineModeState {
        session_id: abi_foundation::time::unix_ms(),
        turn_count: 0,
        history: Vec::new(),
        model: abi_ai::models::DEFAULT_MODEL.to_string(),
        sea_enabled: false,
        interactive,
    }
}

fn handle_input(
    line: &str,
    state: &mut LineModeState,
    responses: &mut Option<String>,
    stdout: &mut std::io::Stdout,
) -> SlashAction {
    use std::io::Write;

    let trimmed = line.trim();
    if trimmed.is_empty() {
        return SlashAction::Quit;
    }
    if trimmed.starts_with('/') {
        return handle_slash(trimmed, state, responses, stdout);
    }

    let augmented = abi_ai::build_agent_context(
        trimmed,
        std::path::Path::new("."),
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
    emit_line(responses, stdout, &body);
    if let Some(responses) = responses {
        let _ = writeln!(responses);
    }
    let _ = writeln!(stdout);
    let _ = stdout.flush();
    SlashAction::Continue
}

fn finish_outcome(responses: String) -> Outcome {
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

/// Deterministic non-TTY path. Its `BufRead::lines` and empty-line behavior is
/// deliberately kept separate from the raw editor.
fn buffered_line_mode() -> Outcome {
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

    let mut responses = Some(String::new());
    let mut state = new_state(false);

    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if matches!(
            handle_input(&line, &mut state, &mut responses, &mut stdout),
            SlashAction::Quit
        ) {
            break;
        }
    }
    finish_outcome(responses.expect("buffered mode captures responses"))
}

fn interactive_mode() -> Outcome {
    use std::io::Write;

    let _raw = match crate::terminal::RawMode::enter() {
        Ok(raw) => raw,
        Err(err) => {
            let _ = writeln!(
                std::io::stderr(),
                "note: raw editor unavailable ({err}); using buffered line-mode"
            );
            return buffered_line_mode();
        }
    };
    let mut stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let _ = writeln!(
        stdout,
        "ABI agent TTY editor. Type a prompt or /help; empty line, Ctrl-D, or Ctrl-C to quit."
    );
    let _ = stdout.flush();

    let mut state = new_state(true);
    let mut editor = EditorState::default();
    let mut decoder = crate::terminal::KeyDecoder::default();
    let mut responses = None;
    while let InteractiveRead::Line(line) =
        read_interactive_line(&mut stdin, &mut stdout, &mut editor, &mut decoder)
    {
        if matches!(
            handle_input(&line, &mut state, &mut responses, &mut stdout),
            SlashAction::Quit
        ) {
            break;
        }
    }
    Outcome {
        stdout: String::new(),
        stderr: String::new(),
        exit_code: 0,
    }
}

/// Agent REPL entrypoint: raw editor on a real TTY, deterministic line reads
/// for redirected/scripted input.
pub(crate) fn line_mode() -> Outcome {
    use std::io::IsTerminal;

    if std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        interactive_mode()
    } else {
        buffered_line_mode()
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
            sea_enabled: false,
            interactive: false,
        }
    }

    /// Dispatch one REPL line and return only what it wrote to `responses`.
    ///
    /// Unit dispatch selects the same capture path used by redirected input;
    /// interactive mode selects `None` and therefore only streams once.
    fn dispatch(line: &str, state: &mut LineModeState) -> String {
        let mut responses = Some(String::new());
        let mut stdout = std::io::stdout();
        handle_slash(line, state, &mut responses, &mut stdout);
        responses.expect("unit dispatch captures responses")
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
    fn sea_toggle_is_session_local_and_reflected_in_status() {
        let mut state = line_mode_state("abi-local");
        assert!(dispatch("/sea", &mut state).contains("sea: off"));
        assert!(dispatch("/sea on", &mut state).contains("sea: on"));
        assert!(state.sea_enabled);
        assert!(dispatch("/status", &mut state).contains("sea=on"));
        assert!(dispatch("/sea toggle", &mut state).contains("sea: off"));
        assert!(!state.sea_enabled);
    }

    #[test]
    fn help_lists_the_model_and_profile_commands() {
        assert!(LINE_MODE_HELP.contains("/model"));
        assert!(TTY_HELP.contains("/profile"));
        assert!(!LINE_MODE_HELP.contains("/sea"));
        assert!(TTY_HELP.contains("/sea"));
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
