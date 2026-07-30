//! Process-level compatibility tests for the Rust `abi` executable.

use std::process::{Command, Output};

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_abi"))
        .args(arguments)
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

    let unknown = run(&["complte"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(unknown.stdout.is_empty());
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("hint: did you mean `complete`?"));

    let unported = run(&["backends"]);
    assert_eq!(unported.status.code(), Some(1));
    assert_eq!(
        unported.stderr,
        b"error: Rust handler for `backends` is not yet ported\n"
    );
}
