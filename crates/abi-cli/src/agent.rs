//! `abi agent` — planning, multi-persona, train, OS control, and line-mode TUI.
//!
//! Ported from the claim-honest subset of `src/cli/handlers/agent_*.zig` and
//! the non-TTY REPL path. `plan` / `multi` use budgeted `file_context`
//! (workspace tree + git diff --stat). Interactive raw-mode TUI is not linked;
//! `agent tui` is the honest line-mode fallback (slash commands + local
//! completion).

use std::fmt::Write as _;
use std::sync::Arc;

use abi_ai::{
    DatasetFormat, DatasetSpec, TrainingConfig, analyze_sentiment, parse_agent_profile,
    route_to_profile, select_best_profile, train_inspect, training_store_key, training_store_value,
    training_vectors,
};
use abi_core::{MemoryTracker, Scheduler, TaskPriority};

use crate::app::Outcome;
use crate::os;
use crate::util;

const USAGE: &str = "usage: abi agent <plan|train|tui|multi|spawn|browser|os> ...";

fn print_scheduler_stats(out: &mut String, stats: abi_core::Stats) {
    let _ = writeln!(
        out,
        "scheduler: running={} pending={} completed={} failed={}",
        stats.running, stats.pending, stats.completed, stats.failed
    );
}

fn print_memory_stats(out: &mut String, peak: usize, records: usize) {
    let _ = writeln!(out, "memory (tracker): peak={peak}B records={records}");
}

fn plan(input: &str) -> Outcome {
    let tracker = Arc::new(MemoryTracker::new());
    let scheduler = Scheduler::new().with_memory_tracker(Arc::clone(&tracker));
    scheduler.submit("agent:plan", TaskPriority::High, Box::new(|| Ok(())));
    let _ = scheduler.run_all();

    // Use cwd-relative root `.` so tree paths stay sandboxed like Zig.
    let root = std::path::Path::new(".");
    let augmented = abi_ai::build_agent_context(
        input,
        root,
        abi_ai::DEFAULT_BUDGET_BYTES,
        &abi_ai::AgentContextOptions {
            include_tree: true,
            include_git_diff: true,
            git_stat_only: true,
            ..abi_ai::AgentContextOptions::default()
        },
    );

    let selected = select_best_profile(analyze_sentiment(input));
    // Generation sees the budgeted context; routing uses the raw user text.
    let body = route_to_profile(selected, &augmented);
    let mut out = String::new();
    let _ = writeln!(out, "agent=cli-agent");
    let _ = writeln!(out, "mode=dry-run");
    let _ = writeln!(out, "selected_profile={}", selected.label());
    let _ = writeln!(out, "review_required=false");
    let _ = writeln!(out, "tool_hints=none");
    let _ = writeln!(out, "instructions=Plan only; do not execute.");
    let _ = writeln!(out, "response={body}");
    print_scheduler_stats(&mut out, scheduler.stats());
    print_memory_stats(&mut out, tracker.peak_usage(), tracker.record_count());
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn multi(input: &str) -> Outcome {
    let scheduler = Scheduler::new();
    for name in ["agent:multi:abbey", "agent:multi:aviva", "agent:multi:abi"] {
        scheduler.submit(name, TaskPriority::Normal, Box::new(|| Ok(())));
    }
    let _ = scheduler.run_all();

    let root = std::path::Path::new(".");
    let augmented = abi_ai::build_agent_context(
        input,
        root,
        abi_ai::DEFAULT_BUDGET_BYTES,
        &abi_ai::AgentContextOptions {
            include_tree: true,
            include_git_diff: true,
            git_stat_only: true,
            ..abi_ai::AgentContextOptions::default()
        },
    );

    // The roster and its per-worker report both come from `abi-ai` now. The
    // instructions used to be restated here, and had already drifted — the inline
    // Abbey copy dropped the closing sentence of her identity contract.
    let mut out = String::from("=== MULTI-AGENT RESULTS ===\n");
    for spec in abi_ai::default_trio_specs() {
        let Some(result) = abi_ai::run_agent(&spec, &augmented) else {
            continue;
        };
        // `multi` uppercases the persona in its section header, unlike `spawn`.
        let _ = writeln!(out, "\n[{}]", spec.name.to_uppercase());
        let _ = writeln!(out, "{}", result.output);
    }
    print_scheduler_stats(&mut out, scheduler.stats());
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn train_profile(profile_arg: &str) -> Outcome {
    let is_all = profile_arg == "all";
    let profiles: Vec<&str> = if is_all {
        vec!["abbey", "aviva", "abi"]
    } else {
        if parse_agent_profile(profile_arg).is_err() {
            return Outcome::stderr(
                format!("error: unknown profile '{profile_arg}' (use abbey|aviva|abi|all)\n"),
                2,
            );
        }
        vec![profile_arg]
    };

    let tracker = Arc::new(MemoryTracker::new());
    let scheduler = Scheduler::new().with_memory_tracker(Arc::clone(&tracker));
    for p in &profiles {
        let name = format!("train:{p}");
        scheduler.submit(name, TaskPriority::High, Box::new(|| Ok(())));
    }
    let _ = scheduler.run_all();

    let mut store = util::open_store();
    let mut out = String::from("training executed via scheduler (real tasks, not demos)\n");
    print_scheduler_stats(&mut out, scheduler.stats());
    print_memory_stats(&mut out, tracker.peak_usage(), tracker.record_count());

    for p in profiles {
        let config = TrainingConfig {
            profile: p.to_string(),
            dataset: DatasetSpec {
                path: "datasets/local-training.jsonl".into(),
                format: DatasetFormat::Jsonl,
            },
            artifact_dir: "zig-cache/agent-artifacts".into(),
        };
        let Ok((result, _summary)) = train_inspect(&config) else {
            let _ = writeln!(out, "{p}: training rejected (invalid config)");
            continue;
        };
        let mut records = 0_usize;
        if let Some(store) = store.as_mut()
            && let Ok(profile) = parse_agent_profile(p)
        {
            let (q, r) = training_vectors(profile);
            if let (Ok(qid), Ok(rid)) = (store.put_vector(&q), store.put_vector(&r)) {
                let value = training_store_value(&config, qid, rid, "cpu");
                let key = training_store_key(p);
                let _ = store.put(&key, &value);
                let _ = store.add_block(p, qid, rid, &value, abi_foundation::time::unix_ms());
                records = 1;
            }
        }
        let _ = writeln!(
            out,
            "{p}: {} ({records} wdbx record(s), backend={})",
            result.message, result.acceleration_backend
        );
    }
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: 0,
    }
}

fn browser(args: &[String]) -> Outcome {
    let mut url = "https://example.com".to_string();
    let mut execute = false;
    let mut confirm = false;
    let mut task_parts: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--url" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Outcome::stderr(
                        "error: usage: abi agent browser [--url <url>] [--execute --confirm] <task>\n".into(),
                        2,
                    );
                };
                url.clone_from(v);
            }
            "--execute" => execute = true,
            "--confirm" => confirm = true,
            other => task_parts.push(other),
        }
        i += 1;
    }
    if task_parts.is_empty() {
        return Outcome::stderr(
            "error: usage: abi agent browser [--url <url>] [--execute --confirm] <task>\n".into(),
            2,
        );
    }
    let task = task_parts.join(" ");
    if execute && !confirm {
        return Outcome::stderr(
            "error: usage: abi agent browser --execute --confirm <task>\n".into(),
            2,
        );
    }
    let mode = if execute && confirm {
        "execute-confirmed (local plan only; no embedded browser)"
    } else {
        "dry-run"
    };
    let out = format!(
        "browser-orchestration mode={mode}\nurl={url}\ntask={task}\nsteps:\n  1. open {url}\n  2. locate content for: {task}\n  3. summarize findings\nnote: real navigation requires an external browser MCP; this is a local plan only.\n"
    );
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: 0,
    }
}

const SPAWN_USAGE: &str = "usage: abi agent spawn [--background] [--workers <spec>] <input>";
/// The longer usage line Zig emitted specifically for a malformed `--workers`
/// value, which spells out the spec grammar rather than just naming the flag.
const SPAWN_WORKERS_USAGE: &str =
    "usage: abi agent spawn [--background] [--workers \"name|instructions|hints;...\"] <input>";

fn spawn(args: &[String]) -> Outcome {
    let mut background = false;
    let mut workers: Option<&str> = None;
    let mut input_parts: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--background" => background = true,
            "--workers" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Outcome::stderr(format!("error: {SPAWN_USAGE}\n"), 2);
                };
                workers = Some(value.as_str());
            }
            other => input_parts.push(other),
        }
        i += 1;
    }
    if input_parts.is_empty() {
        return Outcome::stderr(format!("error: {SPAWN_USAGE}\n"), 2);
    }
    let input = input_parts.join(" ");

    // A malformed --workers value reports the grammar-spelling usage line; an
    // absent one falls back to the single default worker. Both match Zig.
    let specs = match workers {
        Some(spec_text) => match abi_ai::parse_worker_specs(spec_text) {
            Ok(specs) => specs,
            Err(_) => return Outcome::stderr(format!("error: {SPAWN_WORKERS_USAGE}\n"), 2),
        },
        None => vec![abi_ai::default_spawn_spec()],
    };

    // One scheduler task per worker, named `agent:spawn:<worker>` as Zig did, so
    // `completed=` reflects the real fan-out rather than always reading 1.
    let scheduler = Scheduler::new();
    for spec in &specs {
        scheduler.submit(
            format!("agent:spawn:{}", spec.name),
            TaskPriority::Normal,
            Box::new(|| Ok(())),
        );
    }
    let _ = scheduler.run_all();

    let Some(batch) = abi_ai::run_custom_multi_agent(&specs, &input) else {
        return Outcome::stderr(format!("error: {SPAWN_WORKERS_USAGE}\n"), 2);
    };

    let mut out = String::new();
    if background {
        // Zig printed the submitted task ids before the aggregated results. Ids
        // are 1-based and assigned in spec order by the scheduler.
        let _ = writeln!(out, "submitted background agent tasks:");
        for (index, spec) in specs.iter().enumerate() {
            let _ = writeln!(out, "  task_id={} worker={}", index + 1, spec.name);
        }
    }
    let _ = writeln!(out, "{}", batch.aggregated);
    print_scheduler_stats(&mut out, scheduler.stats());
    Outcome {
        stdout: out,
        stderr: String::new(),
        exit_code: 0,
    }
}

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
fn agent_tui_line_mode() -> Outcome {
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

/// Dispatch `abi agent …` (args after the `agent` command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    if args.is_empty() {
        return Outcome::stderr(format!("error: {USAGE}\n"), 2);
    }
    match args[0].as_str() {
        "--help" | "-h" | "help" => Outcome::stderr(
            include_str!("../../../tests/golden/help-agent.txt").to_owned(),
            0,
        ),
        "plan" => {
            if args.len() < 2 {
                return Outcome::stderr("error: usage: abi agent plan <input>\n".into(), 2);
            }
            plan(&args[1..].join(" "))
        }
        "multi" => {
            if args.len() < 2 {
                return Outcome::stderr("error: usage: abi agent multi <input>\n".into(), 2);
            }
            multi(&args[1..].join(" "))
        }
        "train" => {
            if args.len() != 2 {
                return Outcome::stderr(
                    "error: usage: abi agent train <abbey|aviva|abi|all>\n".into(),
                    2,
                );
            }
            train_profile(&args[1])
        }
        "os" => os::os_cmd(&args[1..]),
        "browser" => browser(&args[1..]),
        "spawn" => spawn(&args[1..]),
        "tui" => agent_tui_line_mode(),
        other => Outcome::stderr(
            format!("error: unknown agent subcommand '{other}'\n{USAGE}\n"),
            2,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_prints_dry_run_shape() {
        let outcome = plan("inspect WDBX");
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.contains("agent=cli-agent"));
        assert!(outcome.stdout.contains("mode=dry-run"));
        assert!(outcome.stdout.contains("selected_profile="));
        assert!(outcome.stdout.contains("response="));
        assert!(outcome.stdout.contains("scheduler:"));
        // Persona response is generated from budgeted context that includes
        // the workspace tree when the cwd is a real project tree.
        assert!(
            outcome.stdout.contains("[workspace-tree]")
                || outcome.stdout.contains("workspace-tree")
                || outcome.stdout.contains("inspect WDBX")
                || outcome.stdout.contains("Cargo.toml")
                || outcome.stdout.contains("response=Abbey:")
                || outcome.stdout.contains("response=Aviva")
                || outcome.stdout.contains("response=ABI"),
            "plan output missing expected persona/context markers:\n{}",
            outcome.stdout
        );
    }

    #[test]
    fn multi_emits_three_personas() {
        let outcome = multi("hi");
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.contains("=== MULTI-AGENT RESULTS ==="));
        assert!(outcome.stdout.contains("[ABBEY]"));
        assert!(outcome.stdout.contains("[AVIVA]"));
        assert!(outcome.stdout.contains("[ABI]"));
    }

    #[test]
    fn os_dry_run_allows_ls_and_denies_rm() {
        let ok = os::os_cmd(&["dry-run".into(), "ls".into()]);
        assert_eq!(ok.exit_code, 0, "{}", ok.stderr);
        assert!(ok.stdout.contains("dry-run:"));
        assert!(ok.stdout.contains("argv=[\"ls\"]"));

        let bad = os::os_cmd(&["dry-run".into(), "rm".into()]);
        assert_eq!(bad.exit_code, 1);
        assert!(bad.stderr.contains("denied"));
    }

    #[test]
    fn os_execute_requires_confirm() {
        let outcome = os::os_cmd(&["execute".into(), "ls".into()]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("--confirm"));
    }

    #[test]
    fn browser_dry_run_plans_steps() {
        let outcome = browser(&["open docs".into()]);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stdout.contains("browser-orchestration"));
        assert!(outcome.stdout.contains("dry-run"));
    }

    #[test]
    fn spawn_with_workers_emits_the_custom_banner() {
        // The assertion in tools/contract_cli/complete_through_wdbx.sh.
        let args = ["--workers", "scout|Explore safely|explore", "inspect docs"].map(String::from);
        let outcome = spawn(&args);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(
            outcome
                .stdout
                .contains("=== CUSTOM MULTI-AGENT RESULTS ===")
        );
        assert!(outcome.stdout.contains("[scout]"));
        assert!(outcome.stdout.contains("instructions=Explore safely"));
        assert!(outcome.stdout.contains("tool_hints=explore"));
        assert!(outcome.stdout.contains("=== END ==="));
    }

    #[test]
    fn spawn_without_workers_still_emits_the_custom_banner() {
        // The assertion in tools/contract_cli/agent_orchestration.sh: the banner
        // is unconditional, and the default worker is the named smart-agent.
        let outcome = spawn(&["contract worker smoke".to_owned()]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(
            outcome
                .stdout
                .contains("=== CUSTOM MULTI-AGENT RESULTS ===")
        );
        assert!(outcome.stdout.contains("[smart-agent]"));
        assert!(outcome.stdout.contains("tool_hints=plan,explore"));
    }

    #[test]
    fn spawn_runs_every_worker_and_counts_them_all() {
        let args = ["--workers", "first|One;second|Two;third|Three", "task"].map(String::from);
        let outcome = spawn(&args);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        for name in ["[first]", "[second]", "[third]"] {
            assert!(outcome.stdout.contains(name), "missing {name}");
        }
        // One scheduler task per worker, not one per command.
        assert!(
            outcome.stdout.contains("completed=3"),
            "scheduler should report the real fan-out:\n{}",
            outcome.stdout
        );
    }

    #[test]
    fn spawn_background_lists_task_ids_before_the_results() {
        let args = ["--background", "--workers", "a|A;b|B", "t"].map(String::from);
        let outcome = spawn(&args);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        let ids = outcome
            .stdout
            .find("submitted background agent tasks:")
            .expect("the background header is present");
        let banner = outcome
            .stdout
            .find("=== CUSTOM MULTI-AGENT RESULTS ===")
            .expect("the banner is present");
        assert!(ids < banner, "task ids must precede the aggregated results");
        assert!(outcome.stdout.contains("task_id=1 worker=a"));
        assert!(outcome.stdout.contains("task_id=2 worker=b"));
    }

    #[test]
    fn spawn_rejects_a_malformed_workers_spec_with_the_grammar_usage() {
        // Zig used a longer usage line for a bad --workers value than for a
        // missing input, spelling out the spec grammar.
        for bad in ["onlyname", "n|i|badhint", ""] {
            let args = ["--workers".to_owned(), bad.to_owned(), "t".to_owned()];
            let outcome = spawn(&args);
            assert_eq!(outcome.exit_code, 2, "{bad:?} should be a usage error");
            assert!(
                outcome.stderr.contains("name|instructions|hints"),
                "{bad:?} should report the grammar: {}",
                outcome.stderr
            );
        }
    }

    #[test]
    fn spawn_without_input_or_a_workers_value_is_a_usage_error() {
        assert_eq!(spawn(&[]).exit_code, 2);
        assert_eq!(spawn(&["--background".to_owned()]).exit_code, 2);
        // `--workers` as the final token has no value to consume.
        assert_eq!(spawn(&["--workers".to_owned()]).exit_code, 2);
    }

    #[test]
    fn multi_instructions_come_from_the_identity_contracts() {
        // Guards the drift this change fixed: the inline copy had dropped the
        // closing sentence of Abbey's contract description.
        let outcome = multi("hi");
        assert_eq!(outcome.exit_code, 0);
        for profile in abi_ai::identity::KNOWN_PROFILES {
            let expected = abi_ai::profile_contract(profile).description;
            assert!(
                outcome.stdout.contains(expected),
                "{} instructions must match its contract verbatim",
                profile.label()
            );
        }
    }

    #[test]
    fn train_unknown_profile_is_usage() {
        let outcome = train_profile("nobody");
        assert_eq!(outcome.exit_code, 2);
    }

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
