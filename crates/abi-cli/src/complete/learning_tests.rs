//! Learning requests must not silently dispatch a non-learning backend.

use super::*;

#[test]
fn learning_rejects_live_transport_before_dispatch() {
    let outcome = run(&[
        "--learn".into(),
        "--live".into(),
        "--model".into(),
        "gpt-5".into(),
        "hello".into(),
    ]);
    assert_eq!(outcome.exit_code, 2);
    assert!(
        outcome
            .stderr
            .contains("--learn cannot combine with --live")
    );
}

#[test]
fn learning_rejects_bridge_models_before_connecting() {
    let request = parse_complete_args(&[
        "--learn".into(),
        "--model".into(),
        "ollama/test".into(),
        "hello".into(),
    ]);
    assert!(is_local_bridge_model("ollama/test"));
    let Err(outcome) = request else {
        panic!("bridge request silently bypasses learning");
    };
    assert_eq!(outcome.exit_code, 2);
    assert!(
        outcome
            .stderr
            .contains("--learn does not support local-bridge models")
    );
}

#[test]
fn local_learning_keeps_explicit_persona_and_separator_input() {
    let Ok(request) = parse_complete_args(&[
        "--learn".into(),
        "--".into(),
        "Aviva, explain --live without using it".into(),
    ]) else {
        panic!("ordinary local learning should remain supported");
    };
    assert!(request.learn);
    assert!(!request.live);
    assert_eq!(request.input, "Aviva, explain --live without using it");
}
