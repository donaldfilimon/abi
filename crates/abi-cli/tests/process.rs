//! Process-level compatibility tests for the Rust `abi` executable.

use std::process::{Command, Output};

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_abi"))
        .args(arguments)
        .output()
        .expect("Rust abi executable should run")
}

fn run_with_env(arguments: &[&str], environment: &[(&str, &str)]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_abi"))
        .args(arguments)
        .envs(environment.iter().copied())
        .output()
        .expect("Rust abi executable should run")
}

#[test]
fn top_level_help_matches_the_zig_capture_on_stderr() {
    let output = run(&["help"]);
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert_eq!(
        output.stderr,
        include_bytes!("../../../tests/golden/help.txt")
    );
}

#[test]
fn json_and_shell_completion_match_the_zig_captures() {
    let json = run(&["help", "--json"]);
    assert!(json.status.success());
    assert!(json.stdout.is_empty());
    assert_eq!(
        json.stderr,
        include_bytes!("../../../tests/golden/help.json")
    );

    for (shell, expected) in [
        (
            "bash",
            include_bytes!("../../../tests/golden/completion.bash").as_slice(),
        ),
        (
            "zsh",
            include_bytes!("../../../tests/golden/completion.zsh").as_slice(),
        ),
        (
            "fish",
            include_bytes!("../../../tests/golden/completion.fish").as_slice(),
        ),
    ] {
        let completion = run(&["help", "--completion", shell]);
        assert!(completion.status.success());
        assert!(completion.stdout.is_empty());
        assert_eq!(completion.stderr, expected);
    }
}

#[test]
fn direct_help_and_exit_codes_cross_the_real_process_boundary() {
    let help = run(&["backends", "--help"]);
    assert!(help.status.success());
    assert!(help.stdout.is_empty());
    assert_eq!(
        help.stderr,
        include_bytes!("../../../tests/golden/help-backends.txt")
    );

    let backends = run(&["backends"]);
    assert!(backends.status.success());
    assert!(backends.stdout.is_empty());
    assert!(
        backends
            .stderr
            .starts_with(b"ABI Framework  0.1.0\nRust nightly")
    );
    assert!(
        String::from_utf8_lossy(&backends.stderr)
            .contains("Native accelerator kernels: not linked")
    );

    let unknown = run(&["complte"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(unknown.stdout.is_empty());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("hint: did you mean `complete`?"));

    let unported = run(&["agent", "plan"]);
    assert_eq!(unported.status.code(), Some(1));
    assert_eq!(
        unported.stderr,
        b"error: Rust handler for `agent` is not yet ported\n"
    );

    let dash = run(&["dashboard", "--once", "--plain"]);
    assert!(dash.status.success());
    let dash_text = String::from_utf8_lossy(&dash.stderr);
    assert!(dash_text.contains("ABI Diagnostics Dashboard"));
    assert!(dash_text.contains("System"));
    assert!(dash_text.contains("Memory"));

    let train = run(&["train", "example"]);
    assert!(train.status.success());
    assert!(String::from_utf8_lossy(&train.stdout).contains("training accepted"));

    let nn = run(&["nn", "train", "hello world hello world "]);
    assert!(nn.status.success());
    assert!(String::from_utf8_lossy(&nn.stdout).contains("improved=true"));
}

#[test]
fn simulate_and_scheduler_cross_the_real_process_boundary() {
    let simulate_help = run(&["wdbx", "simulate", "--help", "ignored-like-zig"]);
    assert!(simulate_help.status.success());
    assert!(simulate_help.stdout.is_empty());
    assert!(
        simulate_help
            .stderr
            .starts_with(b"usage: abi wdbx simulate [options]\n")
    );

    let simulation = run(&[
        "wdbx",
        "simulate",
        "--initial",
        "A",
        "--rule",
        "A->AB",
        "--depth",
        "2",
        "--format",
        "json",
        "--quiet",
    ]);
    assert!(simulation.status.success());
    assert!(simulation.stdout.is_empty());
    let json: serde_json::Value =
        serde_json::from_slice(&simulation.stderr).expect("canonical JSON on stderr");
    assert_eq!(json["format"], "abi-multiway-v1");
    assert_eq!(json["termination"], "max_depth");
    assert_eq!(json["complete"], false);

    let scheduler = run(&["scheduler", "status"]);
    assert!(scheduler.status.success());
    assert!(scheduler.stdout.is_empty());
    assert_eq!(
        scheduler.stderr,
        include_bytes!("../../../tests/golden/scheduler-status.txt")
    );

    let invalid = run(&["wdbx", "simulate", "--depth", "nope"]);
    assert_eq!(invalid.status.code(), Some(2));
    assert!(invalid.stdout.is_empty());
    assert_eq!(
        invalid.stderr,
        b"simulate: --depth: 'nope' is not a valid non-negative integer\n"
    );
}

#[test]
fn auth_status_and_logout_cross_the_real_process_boundary() {
    let path = std::env::temp_dir().join(format!(
        "abi-auth-process-{}-{}.json",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock after epoch")
            .as_nanos()
    ));
    std::fs::write(
        &path,
        r#"{"openai_api_key":"sk-test","twilio_account_sid":"AC-test"}"#,
    )
    .expect("write isolated credential fixture");
    let path_text = path.to_str().expect("UTF-8 temporary path");
    let environment = [
        ("ABI_CREDENTIALS_BACKEND", "file"),
        ("ABI_CREDENTIALS_PATH", path_text),
    ];

    let status = run_with_env(&["auth", "status"], &environment);
    assert!(status.status.success());
    assert!(status.stdout.is_empty());
    assert_eq!(
        status.stderr,
        b"Authentication Status:\n  Backend:   file (~/.abi/credentials.json)\n  OpenAI:    configured\n  Anthropic: not configured\n  Discord:   not configured\n  Grok:      not configured\n  Twilio:    not configured\n"
    );

    let logout = run_with_env(&["auth", "logout"], &environment);
    assert!(logout.status.success());
    assert!(logout.stdout.is_empty());
    assert_eq!(logout.stderr, b"Logged out. Credentials cleared.\n");
    assert!(!path.exists());
}
