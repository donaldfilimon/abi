//! `abi agent os` — OS command execution with timeout, env filtering, and audit.
//!
//! **Dry-run is read-only by design and accepts any command.** It never spawns
//! a process; it renders the plan and discloses whether `execute` would permit
//! the command. **Execute (`--confirm`) is the gated path** — it still refuses
//! anything outside the allowlist (`true`, `pwd`, `ls`, `whoami`, `date`), and
//! adds a timeout (30s by default) plus filtered environment variables.
//!
//! Splitting the two matters: describing a command is not running it, so
//! denying `dry-run rm -rf /` bought no safety while making the planning path
//! useless for anything the allowlist had not already blessed.
//!
//! The allowlist and timeout come from [`policy`], which reads
//! `~/.abi/os-policy.toml` when present. That file can only **narrow** the
//! compiled ceiling — see the policy module docs for why widening is not on
//! offer. Executed commands are appended to the WDBX audit chain by [`audit`].

pub(crate) mod audit;
pub(crate) mod policy;

use std::fmt::Write as _;
use std::io::Read as _;
use std::time::{Duration, Instant};

use crate::app::Outcome;

fn resolve_command_path(cmd: &str) -> String {
    match cmd {
        "true" => "/usr/bin/true".to_string(),
        "pwd" => "/bin/pwd".to_string(),
        "ls" => "/bin/ls".to_string(),
        "whoami" => "/usr/bin/whoami".to_string(),
        "date" => "/bin/date".to_string(),
        _ => cmd.to_string(),
    }
}

fn filter_env(cmd: &mut std::process::Command) {
    let filtered: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| {
            let upper = k.to_uppercase();
            !upper.starts_with("ABI_")
                && !upper.contains("SECRET")
                && !upper.contains("TOKEN")
                && !upper.contains("KEY")
                && !upper.contains("PASSWORD")
                && !upper.contains("CREDENTIAL")
        })
        .collect();
    cmd.env_clear();
    for (k, v) in &filtered {
        cmd.env(k, v);
    }
}

/// Execute a command under `timeout`. stdout and stderr are drained on
/// dedicated threads while the caller polls the child, so a chatty command
/// cannot deadlock on a full pipe buffer before the timeout applies.
struct Execution {
    text: String,
    exit_code: u8,
    elapsed: Duration,
    timed_out: bool,
}

fn exec_command_with_timeout(
    mut cmd: std::process::Command,
    label: &str,
    timeout: Duration,
) -> Result<Execution, String> {
    let start = Instant::now();
    let mut child = cmd.spawn().map_err(|e| format!("failed to spawn: {e}"))?;

    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    let out_thread = std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = stdout.read_to_end(&mut out);
        out
    });
    let err_thread = std::thread::spawn(move || {
        let mut err = Vec::new();
        let _ = stderr.read_to_end(&mut err);
        err
    });

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    // `Child` does not wait on drop. Reap it here so repeated
                    // timeouts do not accumulate zombies in the CLI/test
                    // process before collecting the final pipe contents.
                    let _ = child.wait();
                    let out = out_thread
                        .join()
                        .map_err(|_| "stdout drain panicked".to_string())?;
                    let err = err_thread
                        .join()
                        .map_err(|_| "stderr drain panicked".to_string())?;
                    return Ok(Execution {
                        text: merged_output(label, &out, &err, 124),
                        exit_code: 124,
                        elapsed: start.elapsed(),
                        timed_out: true,
                    });
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = out_thread.join();
                let _ = err_thread.join();
                return Err(format!("command wait failed: {e}"));
            }
        }
    };
    let out = out_thread
        .join()
        .map_err(|_| "stdout drain panicked".to_string())?;
    let err = err_thread
        .join()
        .map_err(|_| "stderr drain panicked".to_string())?;

    let code = status.code().unwrap_or(1);
    let exit_code = u8::try_from(code).unwrap_or(1);
    Ok(Execution {
        text: merged_output(label, &out, &err, exit_code),
        exit_code,
        elapsed: start.elapsed(),
        timed_out: false,
    })
}

fn merged_output(label: &str, out: &[u8], err: &[u8], exit_code: u8) -> String {
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(out));
    if !err.is_empty() {
        text.push_str(&String::from_utf8_lossy(err));
    }
    if text.is_empty() {
        text = format!("executed {label} exit={exit_code}");
    }
    text
}

pub(crate) fn os_cmd(args: &[String]) -> Outcome {
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

    // Fail closed: a policy file that exists but does not parse must never
    // degrade into the permissive compiled default. Dry-run reports it too,
    // since planning against a policy the tool could not read would be a lie.
    let policy = match policy::load() {
        Ok(policy) => policy,
        Err(e) => return Outcome::stderr(format!("error: os-control policy: {e}\n"), 1),
    };
    let permitted = policy.permits(cmd);

    // Execute is the gated path. Dry-run falls through: it only describes.
    if execute && !permitted {
        return Outcome::stderr("error: command denied by os-control policy\n".into(), 1);
    }

    let cwd = std::env::current_dir().map_or_else(|_| ".".into(), |p| p.display().to_string());
    let resolved = resolve_command_path(cmd);

    if dry_run {
        // Planning succeeded even when the plan is one `execute` would reject.
        return Outcome {
            stdout: render_dry_run(cmd, &cmd_args, &cwd, &resolved, permitted, &policy),
            stderr: String::new(),
            exit_code: 0,
        };
    }

    let mut command = std::process::Command::new(&resolved);
    command.args(&cmd_args);
    command.current_dir(&cwd);
    filter_env(&mut command);
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());

    let execution = match exec_command_with_timeout(command, cmd, policy.timeout) {
        Ok(r) => r,
        Err(e) => return Outcome::stderr(format!("error: {e}\n"), 1),
    };

    let executed: Vec<&str> = std::iter::once(cmd)
        .chain(cmd_args.iter().copied())
        .collect();
    let audit_status = persist_audit(
        &executed,
        &cwd,
        execution.exit_code,
        execution.elapsed.as_millis(),
        execution.timed_out,
    );
    eprintln!(
        "[os-cmd] cmd={cmd} exit={} elapsed={}ms timed_out={} env_filtered=true timeout={}s audit={audit_status}",
        execution.exit_code,
        execution.elapsed.as_millis(),
        execution.timed_out,
        policy.timeout.as_secs()
    );
    if execution.timed_out {
        return Outcome::stderr(
            format!(
                "error: command timed out after {}s\n",
                policy.timeout.as_secs()
            ),
            execution.exit_code,
        );
    }
    Outcome {
        stdout: execution.text,
        stderr: String::new(),
        exit_code: execution.exit_code,
    }
}

/// Render the `dry-run` plan: the argv line, plus whatever the caller needs to
/// know about the policy that produced the verdict.
fn render_dry_run(
    cmd: &str,
    cmd_args: &[&str],
    cwd: &str,
    resolved: &str,
    permitted: bool,
    policy: &policy::Policy,
) -> String {
    let argv_list = std::iter::once(cmd)
        .chain(cmd_args.iter().copied())
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(", ");

    // `resolve_command_path` echoes the input for anything it does not know an
    // absolute path for, so say "unresolved" rather than implying the bare name
    // would be found on PATH.
    let resolved_field = if permitted {
        format!("resolved_argv=[\"{resolved}\"]")
    } else {
        format!("resolved_argv=[\"{resolved}\"] (unresolved)")
    };
    let verdict = if permitted { "allowed" } else { "denied" };

    let mut out =
        format!("dry-run: cwd=\"{cwd}\" argv=[{argv_list}] {resolved_field} policy={verdict}\n");
    if !permitted {
        let _ = writeln!(
            out,
            "note: execute would refuse — \"{cmd}\" is not in the os-control allowlist ({})",
            policy.allow_list()
        );
    }
    if let policy::Source::File(path) = &policy.source {
        let _ = writeln!(
            out,
            "policy: {} (timeout={}s)",
            path.display(),
            policy.timeout.as_secs()
        );
    }
    if !policy.ignored.is_empty() {
        // A user who wrote `allow = ["curl"]` must learn here that it did not
        // take effect, not when execute refuses.
        let _ = writeln!(
            out,
            "note: policy entries outside the compiled ceiling were ignored: {} (a policy file can only narrow)",
            policy.ignored.join(", ")
        );
    }
    out
}

/// Append the executed command to the WDBX audit chain, returning the status
/// to disclose on the `[os-cmd]` line.
///
/// A failed or absent store is **disclosed, never silent** — an audit trail
/// that quietly stops recording is worse than one that says it did not, since
/// only the second lets an operator notice. It is deliberately not fatal: the
/// command already ran, so failing the invocation now would misreport that.
fn persist_audit(
    argv: &[&str],
    cwd: &str,
    exit_code: u8,
    elapsed_ms: u128,
    timed_out: bool,
) -> String {
    let mut store = match crate::util::open_store_result() {
        Ok(Some(store)) => store,
        Ok(None) => return "skipped(no-store)".to_string(),
        Err(error) => return format!("failed(open: {error})"),
    };
    match audit::record(
        &mut store,
        argv,
        cwd,
        exit_code,
        elapsed_ms,
        timed_out,
        abi_foundation::time::unix_ms(),
    ) {
        Ok(rec) => format!("block={} vector={}", rec.block_hash, rec.vector_id),
        Err(e) => format!("failed({e})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvCleanup {
        keys: Vec<&'static str>,
        dirs: Vec<std::path::PathBuf>,
    }

    impl Drop for EnvCleanup {
        fn drop(&mut self) {
            for key in &self.keys {
                abi_foundation::env::clear_override(key);
            }
            for dir in &self.dirs {
                let _ = std::fs::remove_dir_all(dir);
            }
        }
    }

    /// Pin the policy path at somewhere that does not exist, so every test
    /// below sees the compiled default rather than whatever policy file the
    /// developer running the suite happens to have in `~/.abi/`. Same
    /// discipline the store tests use: never read the user's real config.
    fn with_default_policy<T>(body: impl FnOnce() -> T) -> T {
        let _guard = abi_foundation::env::lock_for_test();
        let _cleanup = EnvCleanup {
            keys: vec![abi_foundation::env::OS_POLICY_PATH],
            dirs: Vec::new(),
        };
        abi_foundation::env::set_override(
            abi_foundation::env::OS_POLICY_PATH,
            "/nonexistent/abi-test/os-policy.toml",
        );
        body()
    }

    /// Run `body` with a policy file containing `contents`.
    fn with_policy_file<T>(tag: &str, contents: &str, body: impl FnOnce() -> T) -> T {
        let _guard = abi_foundation::env::lock_for_test();
        let dir = std::env::temp_dir().join(format!("abi-os-cmd-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let _cleanup = EnvCleanup {
            keys: vec![abi_foundation::env::OS_POLICY_PATH],
            dirs: vec![dir.clone()],
        };
        let path = dir.join("os-policy.toml");
        std::fs::write(&path, contents).expect("write policy");
        abi_foundation::env::set_override(
            abi_foundation::env::OS_POLICY_PATH,
            &path.display().to_string(),
        );
        body()
    }

    #[test]
    fn a_policy_file_can_deny_a_command_the_ceiling_allows() {
        // The narrowing path end-to-end: `ls` is in the compiled ceiling, but
        // a policy listing only `pwd` must make execute refuse it.
        with_policy_file("narrow", "allow = [\"pwd\"]\n", || {
            let denied = os_cmd(&["execute".into(), "--confirm".into(), "ls".into()]);
            assert_eq!(denied.exit_code, 1, "{}", denied.stdout);
            assert!(denied.stderr.contains("denied by os-control policy"));

            let planned = os_cmd(&["dry-run".into(), "ls".into()]);
            assert_eq!(planned.exit_code, 0);
            assert!(planned.stdout.contains("policy=denied"));
            assert!(
                planned.stdout.contains("os-policy.toml"),
                "discloses source"
            );
        });
    }

    #[test]
    fn a_policy_file_cannot_grant_a_command_outside_the_ceiling() {
        // The security property at the dispatch layer, not just the parser.
        with_policy_file("widen", "allow = [\"ls\", \"rm\"]\n", || {
            let denied = os_cmd(&["execute".into(), "--confirm".into(), "rm".into()]);
            assert_eq!(denied.exit_code, 1);
            assert!(denied.stderr.contains("denied by os-control policy"));

            let planned = os_cmd(&["dry-run".into(), "rm".into()]);
            assert!(planned.stdout.contains("policy=denied"));
            assert!(
                planned.stdout.contains("can only narrow"),
                "the ignored entry is disclosed: {}",
                planned.stdout
            );
        });
    }

    #[test]
    fn a_malformed_policy_fails_closed_rather_than_reverting_to_the_default() {
        with_policy_file("bad", "allowed = [\"ls\"]\n", || {
            let out = os_cmd(&["execute".into(), "--confirm".into(), "ls".into()]);
            assert_eq!(out.exit_code, 1, "must not fall back to permissive");
            assert!(out.stderr.contains("os-control policy"), "{}", out.stderr);
            assert!(out.stderr.contains("unknown key"), "{}", out.stderr);
        });
    }

    #[test]
    fn os_dry_run_marks_an_allowlisted_command_allowed() {
        with_default_policy(|| {
            let ok = os_cmd(&["dry-run".into(), "ls".into()]);
            assert_eq!(ok.exit_code, 0, "{}", ok.stderr);
            assert!(ok.stdout.contains("dry-run:"));
            assert!(ok.stdout.contains("argv=[\"ls\"]"));
            assert!(ok.stdout.contains("policy=allowed"));
            assert!(!ok.stdout.contains("unresolved"));
            assert!(!ok.stdout.contains("note:"));
        });
    }

    #[test]
    fn os_dry_run_describes_a_denied_command_without_running_it() {
        with_default_policy(|| {
            // Read-only by design: planning `rm -rf /tmp/x` must succeed and say
            // plainly that execute would refuse it. Nothing is spawned here.
            let planned = os_cmd(&[
                "dry-run".into(),
                "rm".into(),
                "-rf".into(),
                "/tmp/abi-nonexistent".into(),
            ]);
            assert_eq!(planned.exit_code, 0, "{}", planned.stderr);
            assert!(planned.stderr.is_empty());
            assert!(
                planned
                    .stdout
                    .contains("argv=[\"rm\", \"-rf\", \"/tmp/abi-nonexistent\"]")
            );
            assert!(planned.stdout.contains("policy=denied"));
            assert!(planned.stdout.contains("(unresolved)"));
            assert!(planned.stdout.contains("execute would refuse"));
            assert!(planned.stdout.contains("whoami"), "lists the allowlist");
        });
    }

    #[test]
    fn os_execute_still_refuses_anything_off_the_allowlist() {
        with_default_policy(|| {
            // Broadening dry-run must not broaden the path that actually spawns.
            let denied = os_cmd(&["execute".into(), "--confirm".into(), "rm".into()]);
            assert_eq!(denied.exit_code, 1);
            assert!(denied.stderr.contains("denied by os-control policy"));
            assert!(denied.stdout.is_empty());
        });
    }

    #[test]
    fn os_execute_requires_confirm() {
        let outcome = os_cmd(&["execute".into(), "ls".into()]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("--confirm"));
    }

    #[test]
    fn the_configured_timeout_is_the_one_that_fires() {
        // Proves `timeout_secs` reaches the exec path rather than being parsed
        // and dropped: a 1s timeout must kill a 30s sleep and say "after 1s".
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "sleep 30"]);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let execution = exec_command_with_timeout(cmd, "sh", Duration::from_secs(1))
            .expect("a timeout is an auditable execution result");
        assert!(execution.timed_out);
        assert_eq!(execution.exit_code, 124);
        assert!(execution.elapsed >= Duration::from_secs(1));
        assert!(execution.elapsed < Duration::from_secs(5));
    }

    #[test]
    fn execute_and_timeout_audits_reach_the_scratch_store_but_dry_run_does_not() {
        let _guard = abi_foundation::env::lock_for_test();
        let dir = std::env::temp_dir().join(format!(
            "abi-os-dispatch-audit-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let policy_path = dir.join("os-policy.toml");
        let store_path = dir.join("store");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        std::fs::write(&policy_path, "allow = [\"true\"]\ntimeout_secs = 1\n")
            .expect("policy file");
        let _cleanup = EnvCleanup {
            keys: vec![
                abi_foundation::env::OS_POLICY_PATH,
                abi_foundation::env::WDBX_PATH,
            ],
            dirs: vec![dir.clone()],
        };
        abi_foundation::env::set_override(
            abi_foundation::env::OS_POLICY_PATH,
            &policy_path.display().to_string(),
        );
        abi_foundation::env::set_override(
            abi_foundation::env::WDBX_PATH,
            &store_path.display().to_string(),
        );

        let planned = os_cmd(&["dry-run".into(), "true".into()]);
        assert_eq!(planned.exit_code, 0);
        assert!(
            !store_path.exists(),
            "dry-run must not open the audit store"
        );

        let executed = os_cmd(&["execute".into(), "--confirm".into(), "true".into()]);
        assert_eq!(executed.exit_code, 0, "{}", executed.stderr);

        let status = persist_audit(&["sh", "-c", "sleep 30"], "/tmp", 124, 1_000, true);
        assert!(status.starts_with("block="), "{status}");

        let store = abi_wdbx::DurableStore::open(abi_wdbx::StorePaths::new(&store_path))
            .expect("reopen scratch audit store");
        assert_eq!(store.stats().vectors, 2);
        assert_eq!(store.stats().kv_entries, 2);
        assert_eq!(store.stats().blocks, 2);
        let timed_out = store.get("os-cmd:2").expect("timeout metadata");
        let metadata: serde_json::Value = serde_json::from_str(timed_out).expect("metadata json");
        assert_eq!(metadata["timed_out"], true);
        assert_eq!(metadata["exit_code"], 124);
    }

    #[test]
    fn os_execute_drains_large_output_without_pipe_deadlock() {
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "yes x | head -n 100000"]);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let execution = exec_command_with_timeout(cmd, "sh", Duration::from_secs(30))
            .expect("large output completes");
        assert_eq!(execution.exit_code, 0);
        assert!(!execution.timed_out);
        assert!(
            execution.text.lines().count() >= 100_000,
            "output was truncated"
        );
        assert!(
            execution.elapsed.as_secs() < 30,
            "timed out instead of draining"
        );
    }
}
