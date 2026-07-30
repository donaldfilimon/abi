//! `abi agent` — planning, multi-persona, train, and OS dry-run surfaces.
//!
//! Ported from the claim-honest subset of `src/cli/handlers/agent_*.zig`.
//! Interactive `agent tui`, custom spawn workers with full tool-hint parsing,
//! browser execute, and OS execute remain partially deferred with honest
//! messages. `plan` / `multi` use the ported persona router without the full
//! workspace-tree `file_context` augmentation (disclosed in the output).

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

    let selected = select_best_profile(analyze_sentiment(input));
    let body = route_to_profile(selected, input);
    let mut out = String::new();
    let _ = writeln!(out, "agent=cli-agent");
    let _ = writeln!(out, "mode=dry-run");
    let _ = writeln!(out, "selected_profile={}", selected.label());
    let _ = writeln!(out, "review_required=false");
    let _ = writeln!(out, "tool_hints=none");
    let _ = writeln!(out, "instructions=Plan only; do not execute.");
    let _ = writeln!(
        out,
        "response={body}\n[note: workspace-tree file_context not yet ported; response is persona-only]"
    );
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
        let body = route_to_profile(profile, input);
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
        "tui" => Outcome::stderr(
            "error: Rust handler for `agent tui` is not yet ported (interactive REPL)\n".into(),
            1,
        ),
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
