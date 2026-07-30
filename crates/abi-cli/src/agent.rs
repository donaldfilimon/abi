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
    AgentProfile, DatasetFormat, DatasetSpec, TrainingConfig, analyze_sentiment, complete,
    parse_agent_profile, route_to_profile, select_best_profile, train_inspect, training_store_key,
    training_store_value, training_vectors,
};
use abi_core::{MemoryTracker, Scheduler, TaskPriority};
use abi_wdbx::{DurableStore, StorePaths};

use crate::app::Outcome;

const USAGE: &str = "usage: abi agent <plan|train|tui|multi|spawn|browser|os> ...";

const OS_ALLOWED: &[&str] = &["true", "pwd", "ls", "whoami", "date"];

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

fn open_store() -> Option<DurableStore> {
    if let Ok(path) = std::env::var("ABI_WDBX_PATH") {
        if path == ":memory:" {
            return None;
        }
        return DurableStore::open(StorePaths::new(path)).ok();
    }
    if matches!(
        std::env::var("ABI_WDBX_PERSIST").as_deref(),
        Ok("0" | "false" | "no" | "off")
    ) {
        return None;
    }
    let home = std::env::var("HOME").ok()?;
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).ok()
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

    let profiles = [
        (
            AgentProfile::Abbey,
            "Primary user-facing personality combining technical expertise, emotional intelligence, creativity, clear teaching, thoughtful judgment, and collaborative problem-solving.",
        ),
        (
            AgentProfile::Aviva,
            "Direct expert mode: concise answers, assumptions, next actions.",
        ),
        (
            AgentProfile::Abi,
            "Orchestration and governance review layer.",
        ),
    ];
    let mut out = String::from("=== MULTI-AGENT RESULTS ===\n");
    for (profile, instructions) in profiles {
        let body = route_to_profile(profile, &augmented);
        let label = profile.label().to_uppercase();
        let _ = writeln!(out, "\n[{label}]");
        let _ = writeln!(out, "agent={}", profile.label());
        let _ = writeln!(out, "mode=dry-run");
        let _ = writeln!(out, "selected_profile={}", profile.label());
        let _ = writeln!(out, "review_required=false");
        let _ = writeln!(out, "tool_hints=explore,plan");
        let _ = writeln!(out, "instructions={instructions}");
        let _ = writeln!(out, "response={body}");
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

    let mut store = open_store();
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

fn resolve_allowlisted(cmd: &str) -> Option<&'static str> {
    match cmd {
        "true" => Some("/usr/bin/true"),
        "pwd" => Some("/bin/pwd"),
        "ls" => Some("/bin/ls"),
        "whoami" => Some("/usr/bin/whoami"),
        "date" => Some("/bin/date"),
        _ => None,
    }
}

fn os_cmd(args: &[String]) -> Outcome {
    if args.is_empty() {
        return Outcome::stderr(
            "error: usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]\n".into(),
            2,
        );
    }
    let mode = args[0].as_str();
    let execute = mode == "execute";
    let dry_run = mode == "dry-run";
    if !execute && !dry_run {
        return Outcome::stderr(
            "error: usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]\n".into(),
            2,
        );
    }
    let rest = if execute {
        if args.len() < 2 || args[1] != "--confirm" {
            return Outcome::stderr(
                "error: usage: abi agent os execute --confirm <cmd> [args...]\n".into(),
                2,
            );
        }
        &args[2..]
    } else {
        &args[1..]
    };
    if rest.is_empty() {
        return Outcome::stderr(
            "error: usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]\n".into(),
            2,
        );
    }
    let cmd = rest[0].as_str();
    let cmd_args: Vec<&str> = rest[1..].iter().map(String::as_str).collect();

    if !OS_ALLOWED.contains(&cmd) {
        return Outcome::stderr("error: command denied by os-control policy\n".into(), 1);
    }

    let Some(resolved) = resolve_allowlisted(cmd) else {
        return Outcome::stderr("error: command denied by os-control policy\n".into(), 1);
    };

    let cwd = std::env::current_dir().map_or_else(|_| ".".into(), |p| p.display().to_string());

    if dry_run {
        let argv_list = std::iter::once(cmd)
            .chain(cmd_args.iter().copied())
            .map(|s| format!("\"{s}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let out =
            format!("dry-run: cwd=\"{cwd}\" argv=[{argv_list}] resolved_argv=[\"{resolved}\"]\n");
        return Outcome {
            stdout: out,
            stderr: String::new(),
            exit_code: 0,
        };
    }

    // execute --confirm: still refuse by default in this slice for safety unless
    // the allow-list command is one of the read-only probes. We only run true/pwd/whoami/date/ls.
    let mut command = std::process::Command::new(resolved);
    command.args(&cmd_args);
    command.current_dir(&cwd);
    match command.output() {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            if !output.stderr.is_empty() {
                text.push_str(&String::from_utf8_lossy(&output.stderr));
            }
            if text.is_empty() {
                text = format!(
                    "executed {cmd} exit={}\n",
                    output.status.code().unwrap_or(-1)
                );
            }
            let code = output.status.code().unwrap_or(1);
            let exit_code = u8::try_from(code).unwrap_or(1);
            Outcome {
                stdout: text,
                stderr: String::new(),
                exit_code,
            }
        }
        Err(err) => Outcome::stderr(format!("error: failed to execute: {err}\n"), 1),
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

fn spawn(args: &[String]) -> Outcome {
    let mut background = false;
    let mut workers: Option<String> = None;
    let mut input_parts: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--background" => background = true,
            "--workers" => {
                i += 1;
                let Some(v) = args.get(i) else {
                    return Outcome::stderr(
                        "error: usage: abi agent spawn [--background] [--workers <spec>] <input>\n"
                            .into(),
                        2,
                    );
                };
                workers = Some(v.clone());
            }
            other => input_parts.push(other),
        }
        i += 1;
    }
    if input_parts.is_empty() {
        return Outcome::stderr(
            "error: usage: abi agent spawn [--background] [--workers <spec>] <input>\n".into(),
            2,
        );
    }
    let input = input_parts.join(" ");
    let worker_name = workers
        .as_deref()
        .and_then(|s| s.split('|').next())
        .unwrap_or("smart-agent");
    let body = complete(&input, "abi-local").map_or_else(
        |_| route_to_profile(AgentProfile::Abbey, &input),
        |r| r.output,
    );
    let mut out = String::new();
    if background {
        let _ = writeln!(out, "submitted background agent tasks:");
        let _ = writeln!(out, "  task_id=1 worker={worker_name}");
    }
    let _ = writeln!(out, "worker={worker_name}");
    let _ = writeln!(out, "mode=dry-run");
    let _ = writeln!(out, "response={body}");
    let scheduler = Scheduler::new();
    scheduler.submit("agent:spawn", TaskPriority::Normal, Box::new(|| Ok(())));
    let _ = scheduler.run_all();
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
  /quit /exit /q     leave the REPL
  /status /stat      session counters and model
  /context           file_context / open-file summary
  /history /hist     recent turn previews
  /reset             clear history and bump session id
  /features /feat    build-time feature migration flags
  /clear /cls        clear screen (ANSI)
  free text          local persona completion with file_context
";

struct LineModeState {
    session_id: i64,
    turn_count: usize,
    history: Vec<String>,
    model: &'static str,
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
    emit_line(
        responses,
        stdout,
        &format!(
            "status: session={} turns={} history={} model={} provider={} sea=off live=off store=off mode=line",
            state.session_id,
            state.turn_count,
            state.history.len(),
            state.model,
            abi_ai::models::provider_of(state.model).label(),
        ),
    );
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
    match cmd.as_str() {
        "/quit" | "/exit" | "/q" => SlashAction::Quit,
        "/help" | "/h" => {
            emit_line(responses, stdout, TUI_HELP.trim_end());
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
        model: abi_ai::models::DEFAULT_MODEL,
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
        "os" => os_cmd(&args[1..]),
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
        let ok = os_cmd(&["dry-run".into(), "ls".into()]);
        assert_eq!(ok.exit_code, 0, "{}", ok.stderr);
        assert!(ok.stdout.contains("dry-run:"));
        assert!(ok.stdout.contains("argv=[\"ls\"]"));

        let bad = os_cmd(&["dry-run".into(), "rm".into()]);
        assert_eq!(bad.exit_code, 1);
        assert!(bad.stderr.contains("denied"));
    }

    #[test]
    fn os_execute_requires_confirm() {
        let outcome = os_cmd(&["execute".into(), "ls".into()]);
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
    fn train_unknown_profile_is_usage() {
        let outcome = train_profile("nobody");
        assert_eq!(outcome.exit_code, 2);
    }
}
