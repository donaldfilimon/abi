//! Parent-watchdog acceptance using real finite helper processes.
use super::*;
use abi_agent_runtime::CancellationToken;
use std::io::Write as _;
fn command(program: &str, args: &[&str]) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    cmd
}
fn run(
    runner: &mut Runner,
    cmd: Command,
    token: &CancellationToken,
    timeout_ms: u64,
    output: usize,
) -> Result<Vec<u8>, ToolExecutionError> {
    runner.run_command(
        cmd,
        ProcessLimits {
            timeout_ms,
            max_output_bytes: output,
            cleanup_grace_ms: 30,
        },
        ToolExecutionContext {
            cancellation: token,
            deadline: Instant::now() + Duration::from_secs(3),
        },
    )
}
fn assert_reaped(runner: &Runner) {
    assert!(runner.child.is_none());
    let pid = runner.last_pid.unwrap();
    let mut status = 0;
    // SAFETY: nonblocking query of a known direct child; no PID signaling.
    assert_eq!(
        unsafe { libc::waitpid(i32::try_from(pid).unwrap(), &raw mut status, libc::WNOHANG) },
        -1
    );
    assert_eq!(
        io::Error::last_os_error().raw_os_error(),
        Some(libc::ECHILD)
    );
}
fn cases() {
    let token = CancellationToken::new();
    let mut runner = Runner::default();
    assert_eq!(
        run(
            &mut runner,
            command("/usr/bin/env", &[]),
            &token,
            1000,
            4096
        )
        .unwrap(),
        b""
    );
    assert_reaped(&runner);
    assert_eq!(
        run(
            &mut runner,
            command("/bin/sh", &["-c", "printf ok"]),
            &token,
            1000,
            4096
        )
        .unwrap(),
        b"ok"
    );
    assert_reaped(&runner);
    for script in [
        "exit 1",
        "printf error >&2",
        "/bin/sleep 2",
        "printf 12345; printf 67890 >&2",
        "while :; do printf 1234567890; done",
    ] {
        let started = Instant::now();
        assert!(
            run(
                &mut runner,
                command("/bin/sh", &["-c", script]),
                &token,
                30,
                8
            )
            .is_err(),
            "{script}"
        );
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(runner.cleanup(250), "bounded follow-up reap: {script}");
        assert_reaped(&runner);
    }
    // Killing ordinary same-group descendants must close inherited pipes.
    assert_eq!(
        run(
            &mut runner,
            command("/bin/sh", &["-c", "/bin/sleep 2 & printf done; exit 0"]),
            &token,
            1000,
            4096
        )
        .unwrap(),
        b"done"
    );
    assert_reaped(&runner);
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let prior = runner.last_pid;
    assert!(
        run(
            &mut runner,
            command("/bin/sleep", &["2"]),
            &cancelled,
            30,
            8
        )
        .is_err()
    );
    assert_eq!(runner.last_pid, prior);
    let cancelled = CancellationToken::new();
    let sender = cancelled.clone();
    let signal = thread::spawn(move || {
        thread::sleep(Duration::from_millis(20));
        sender.cancel();
    });
    assert!(
        run(
            &mut runner,
            command("/bin/sleep", &["2"]),
            &cancelled,
            1000,
            8
        )
        .is_err()
    );
    signal.join().unwrap();
    assert!(runner.cleanup(250));
    assert_reaped(&runner);
}
fn fault_cases() {
    let token = CancellationToken::new();
    let mut runner = Runner::default();
    for fault in ["kill", "wait", "delayed-reap"] {
        match fault {
            "kill" => runner.fail_kill_once = true,
            "wait" => runner.fail_wait_once = true,
            "delayed-reap" => runner.hold_reap = true,
            _ => unreachable!(),
        }
        assert!(
            run(
                &mut runner,
                command("/bin/sh", &["-c", "printf ok"]),
                &token,
                1000,
                4096
            )
            .is_err()
        );
        assert!(runner.child.is_some());
        let prior = runner.last_pid;
        if fault == "delayed-reap" {
            assert!(
                run(
                    &mut runner,
                    command("/bin/sh", &["-c", "printf forbidden"]),
                    &token,
                    1000,
                    4096
                )
                .is_err()
            );
            assert_eq!(runner.last_pid, prior, "must not spawn while unreaped");
            runner.hold_reap = false;
        }
        assert_eq!(
            run(
                &mut runner,
                command("/bin/sh", &["-c", "printf recovered"]),
                &token,
                1000,
                4096
            )
            .unwrap(),
            b"recovered"
        );
        assert_reaped(&runner);
    }
}
#[test]
fn supervisor_child() {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input).unwrap();
    if input == "run" {
        cases();
        fault_cases();
    }
}
#[test]
fn supervisor_failure_and_cleanup_cases_have_a_parent_watchdog() {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "platform_process::unix::tests::supervisor_child",
            "--nocapture",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"run").unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("supervisor acceptance exceeded parent watchdog");
        }
        thread::sleep(Duration::from_millis(5));
    }
}
