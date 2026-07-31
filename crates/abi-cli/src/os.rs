//! `abi agent os` — OS command execution with timeout, env filtering, and audit.
//!
//! **Dry-run is read-only by design and accepts any command.** It never spawns
//! a process; it renders the plan and discloses whether `execute` would permit
//! the command. **Execute (`--confirm`) is the gated path** — it still refuses
//! anything outside the allowlist (`true`, `pwd`, `ls`, `whoami`, `date`), and
//! adds a 30-second timeout plus filtered environment variables.
//!
//! Splitting the two matters: describing a command is not running it, so
//! denying `dry-run rm -rf /` bought no safety while making the planning path
//! useless for anything the allowlist had not already blessed.

use std::fmt::Write as _;
use std::io::Read as _;
use std::time::{Duration, Instant};

use crate::app::Outcome;

const OS_ALLOWED: &[&str] = &["true", "pwd", "ls", "whoami", "date"];

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

/// Execute a command with a 30s timeout. stdout and stderr are drained on
/// dedicated threads while the caller polls the child, so a chatty command
/// cannot deadlock on a full pipe buffer before the timeout applies.
fn exec_command_with_timeout(
    mut cmd: std::process::Command,
    label: &str,
) -> Result<(String, u8, Duration), String> {
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

    let timeout = Duration::from_secs(30);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = out_thread.join();
                    let _ = err_thread.join();
                    return Err("command timed out after 30s".into());
                }
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                let _ = child.kill();
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

    let elapsed = start.elapsed();
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&out));
    if !err.is_empty() {
        text.push_str(&String::from_utf8_lossy(&err));
    }
    if text.is_empty() {
        text = format!("executed {label} exit={}", status.code().unwrap_or(-1));
    }
    let code = status.code().unwrap_or(1);
    let exit_code = u8::try_from(code).unwrap_or(1);
    Ok((text, exit_code, elapsed))
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

    let permitted = OS_ALLOWED.contains(&cmd);

    // Execute is the gated path. Dry-run falls through: it only describes.
    if execute && !permitted {
        return Outcome::stderr("error: command denied by os-control policy\n".into(), 1);
    }

    let cwd = std::env::current_dir().map_or_else(|_| ".".into(), |p| p.display().to_string());
    let resolved = resolve_command_path(cmd);

    let argv_list = std::iter::once(cmd)
        .chain(cmd_args.iter().copied())
        .map(|s| format!("\"{s}\""))
        .collect::<Vec<_>>()
        .join(", ");

    if dry_run {
        // `resolve_command_path` echoes the input for anything it does not know
        // an absolute path for, so say "unresolved" rather than implying the
        // bare name would be found on PATH.
        let resolved_field = if permitted {
            format!("resolved_argv=[\"{resolved}\"]")
        } else {
            format!("resolved_argv=[\"{resolved}\"] (unresolved)")
        };
        let policy = if permitted { "allowed" } else { "denied" };
        let mut out =
            format!("dry-run: cwd=\"{cwd}\" argv=[{argv_list}] {resolved_field} policy={policy}\n");
        if !permitted {
            let allowed = OS_ALLOWED.join(", ");
            let _ = writeln!(
                out,
                "note: execute would refuse — \"{cmd}\" is not in the os-control allowlist ({allowed})"
            );
        }
        // Planning succeeded even when the plan is one `execute` would reject.
        return Outcome {
            stdout: out,
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

    let (text, exit_code, elapsed) = match exec_command_with_timeout(command, cmd) {
        Ok(r) => r,
        Err(e) => return Outcome::stderr(format!("error: {e}\n"), 1),
    };
    eprintln!(
        "[os-cmd] cmd={cmd} exit={exit_code} elapsed={}ms env_filtered=true",
        elapsed.as_millis()
    );
    Outcome {
        stdout: text,
        stderr: String::new(),
        exit_code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn os_dry_run_marks_an_allowlisted_command_allowed() {
        let ok = os_cmd(&["dry-run".into(), "ls".into()]);
        assert_eq!(ok.exit_code, 0, "{}", ok.stderr);
        assert!(ok.stdout.contains("dry-run:"));
        assert!(ok.stdout.contains("argv=[\"ls\"]"));
        assert!(ok.stdout.contains("policy=allowed"));
        assert!(!ok.stdout.contains("unresolved"));
        assert!(!ok.stdout.contains("note:"));
    }

    #[test]
    fn os_dry_run_describes_a_denied_command_without_running_it() {
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
    }

    #[test]
    fn os_execute_still_refuses_anything_off_the_allowlist() {
        // Broadening dry-run must not broaden the path that actually spawns.
        let denied = os_cmd(&["execute".into(), "--confirm".into(), "rm".into()]);
        assert_eq!(denied.exit_code, 1);
        assert!(denied.stderr.contains("denied by os-control policy"));
        assert!(denied.stdout.is_empty());
    }

    #[test]
    fn os_execute_requires_confirm() {
        let outcome = os_cmd(&["execute".into(), "ls".into()]);
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("--confirm"));
    }

    #[test]
    fn os_execute_drains_large_output_without_pipe_deadlock() {
        let mut cmd = std::process::Command::new("sh");
        cmd.args(["-c", "yes x | head -n 100000"]);
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        let (text, code, elapsed) =
            exec_command_with_timeout(cmd, "sh").expect("large output completes");
        assert_eq!(code, 0);
        assert!(text.lines().count() >= 100_000, "output was truncated");
        assert!(elapsed.as_secs() < 30, "timed out instead of draining");
    }
}
