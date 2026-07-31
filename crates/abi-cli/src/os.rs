//! `abi agent os` — OS command execution with timeout, env filtering, and audit.
//!
//! Dry-run and execute both gate on the allowlist (`true`, `pwd`, `ls`,
//! `whoami`, `date`). Execute (`--confirm`) adds a 30-second timeout and
//! filtered environment variables.

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

    if !OS_ALLOWED.contains(&cmd) {
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
        let out =
            format!("dry-run: cwd=\"{cwd}\" argv=[{argv_list}] resolved_argv=[\"{resolved}\"]\n");
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
