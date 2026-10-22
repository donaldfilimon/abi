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

/// High-value addition: --learn without a persistent store now emits a clear,
/// actionable error instead of silently falling back to a non-learning completion.
/// Covers the recall/adaptation intent and improves error UX.
#[test]
fn learning_without_persistent_store_emits_clear_error() {
    // In the test build there is deliberately no default store; run() for --learn
    // will hit the Ok(None) path and must not dispatch local.
    let outcome = run(&["--learn".into(), "recall test input".into()]);
    assert_eq!(outcome.exit_code, 1);
    assert!(outcome.stderr.contains("requires a persistent WDBX store"));
    // Ensure we did not fall back to ordinary completion output.
    assert!(!outcome.stdout.contains("requested_model="));
}

/// The `self_learn_cycle` field must not report a completed
/// persistence+adaptation+recall cycle when the learning write was skipped or
/// its outcome is unknown. A constant claim of a completed cycle would
/// contradict the `persisted=` and `wdbx_status=` fields on the same output.
#[test]
fn self_learn_cycle_reports_the_outcome_that_actually_happened() {
    let committed = render_local(&MetaReport {
        requested_model: "m",
        profile: "p",
        audit_passed: true,
        escore: 1.0,
        audit_vetoed: false,
        persisted: true,
        learn: Some((3, true, "persistence+adaptation+recall")),
        kv: 0,
        vectors: 0,
        blocks: 0,
        query_id: None,
        response_id: None,
        block_hex: None,
        wdbx_status: None,
        output: "out",
    });
    assert!(committed.contains("self_learn_cycle=persistence+adaptation+recall"));
    assert!(committed.contains("persisted=true"));

    let skipped = render_local(&MetaReport {
        requested_model: "m",
        profile: "p",
        audit_passed: false,
        escore: 0.0,
        audit_vetoed: true,
        persisted: false,
        learn: Some((0, false, "adaptation-only")),
        kv: 0,
        vectors: 0,
        blocks: 0,
        query_id: None,
        response_id: None,
        block_hex: None,
        wdbx_status: Some("audit rejected; learning write skipped"),
        output: "out",
    });
    assert!(skipped.contains("self_learn_cycle=adaptation-only"));
    assert!(!skipped.contains("self_learn_cycle=persistence+adaptation+recall"));
    assert!(skipped.contains("persisted=false"));
    assert!(skipped.contains("audit rejected; learning write skipped"));
}

/// The checked-in SEA learn smoke inspects only the FIRST output line, on
/// purpose: the generated response text may repeat arbitrary markers, so a
/// later line must never be able to satisfy a persistence check. Every marker
/// the smoke requires must therefore share the first line. Splitting the
/// learn fields onto their own line hid `provider=local`/`persisted=` from
/// that check and failed a healthy run with "missing metadata: provider=local".
#[test]
fn learn_markers_all_share_the_first_output_line() {
    let text = render_local(&MetaReport {
        requested_model: "claude-fable-5",
        profile: "abbey",
        audit_passed: true,
        escore: 1.0,
        audit_vetoed: false,
        persisted: true,
        learn: Some((2, true, "persistence+adaptation+recall")),
        kv: 1,
        vectors: 2,
        blocks: 1,
        query_id: None,
        response_id: None,
        block_hex: None,
        wdbx_status: None,
        output: "untrusted response: persisted=true evidence_count=99",
    });
    let first = text.lines().next().unwrap_or_default();
    for marker in [
        "learn=true",
        "provider=local",
        "generation_engine=persona-template",
        "persisted=true",
        "adapted=true",
        "requested_model=claude-fable-5",
        "evidence_count=2",
    ] {
        assert!(
            first.contains(marker),
            "first output line is missing {marker}: {first}"
        );
    }
    // The markers must be space-delimited so the smoke's whitespace-bounded
    // match finds them, and the learn cycle must stay on that same line.
    assert!(first.contains(" self_learn_cycle=persistence+adaptation+recall "));
    // A non-learning completion must not claim to have learned.
    let plain = render_local(&MetaReport {
        requested_model: "claude-fable-5",
        profile: "abbey",
        audit_passed: true,
        escore: 1.0,
        audit_vetoed: false,
        persisted: false,
        learn: None,
        kv: 0,
        vectors: 0,
        blocks: 0,
        query_id: None,
        response_id: None,
        block_hex: None,
        wdbx_status: Some("no persistent WDBX path configured"),
        output: "out",
    });
    assert!(
        !plain
            .lines()
            .next()
            .unwrap_or_default()
            .contains("learn=true")
    );
}
