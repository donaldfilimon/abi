//! Focused multiway simulator tests.

use std::sync::atomic::AtomicBool;

use super::{
    ConfigError, MultiwayConfig, MultiwayExportError, ParseRuleError, Rule, Termination,
    build_causal_edges, compute_multiway_metrics, export_multiway_dot, export_multiway_json,
    load_multiway_export, multiway_config_hash, parse_rule, persist_multiway, resume_multiway,
    run_multiway, validate_config,
};

fn config(initial: &[&str], rules: &[(&str, &str)]) -> MultiwayConfig {
    let mut config = MultiwayConfig::new(
        initial
            .iter()
            .map(|payload| payload.as_bytes().to_vec())
            .collect(),
        rules
            .iter()
            .map(|(lhs, rhs)| Rule::new(lhs.as_bytes(), rhs.as_bytes()))
            .collect(),
    );
    config.max_states = 100;
    config.max_events = 1_000;
    config.max_payload = 64;
    config
}

#[test]
fn rule_parsing_hashing_and_validation_match_the_oracle() {
    assert_eq!(
        parse_rule(" A -> AB ").expect("rule"),
        Rule::new(b"A", b"AB")
    );
    assert_eq!(parse_rule("AB"), Err(ParseRuleError::MissingArrow));
    assert_eq!(parse_rule(" ->B"), Err(ParseRuleError::EmptyLhs));
    let rule = Rule::new(b"A", b"AB");
    assert_eq!(rule.canonical_bytes(), b"A->AB");
    assert_eq!(rule.content_hash(), Rule::new(b"A", b"AB").content_hash());

    let mut invalid = config(&["A"], &[("A", "B")]);
    invalid.max_depth = 0;
    assert_eq!(validate_config(&invalid), Err(ConfigError::ZeroBound));
}

#[test]
fn overlapping_matches_preserve_rule_then_offset_order() {
    let mut config = config(&["AAA"], &[("AA", "B")]);
    config.max_depth = 1;
    let result = run_multiway(&config, None);
    assert_eq!(result.events.len(), 2);
    assert_eq!(result.events[0].position, 0);
    assert_eq!(result.events[1].position, 1);
    assert_eq!(
        result
            .states
            .iter()
            .map(|state| state.payload.as_slice())
            .collect::<Vec<_>>(),
        [b"AAA".as_slice(), b"BA".as_slice(), b"AB".as_slice()]
    );
}

#[test]
fn duplicate_events_survive_state_deduplication() {
    let mut config = config(&["A"], &[("A", "B"), ("A", "B")]);
    config.max_depth = 1;
    let result = run_multiway(&config, None);
    let metrics = compute_multiway_metrics(&result);
    assert_eq!(result.states.len(), 2);
    assert_eq!(metrics.event_count, 2);
    assert_eq!(metrics.unique_transitions, 1);
}

#[test]
fn cycles_convergence_and_empty_payloads_are_measured() {
    let cycle = run_multiway(&config(&["A"], &[("A", "B"), ("B", "A")]), None);
    let cycle_metrics = compute_multiway_metrics(&cycle);
    assert!(cycle.complete);
    assert!(cycle_metrics.has_cycle);
    assert_eq!(cycle_metrics.weakly_connected_components, 1);

    let convergence = run_multiway(&config(&["AB"], &[("A", "C"), ("B", "C")]), None);
    assert_eq!(compute_multiway_metrics(&convergence).convergent_states, 1);

    let shrinking = run_multiway(&config(&["AA"], &[("A", "")]), None);
    assert!(shrinking.complete);
    assert_eq!(shrinking.states.last().expect("empty").payload, b"");
}

#[test]
fn every_hard_cap_is_atomic_and_never_exceeded() {
    let mut states = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
    states.max_depth = 10;
    states.max_states = 7;
    let state_result = run_multiway(&states, None);
    assert_eq!(state_result.termination, Termination::MaxStates);
    assert!(state_result.states.len() <= 7);

    let mut events = states.clone();
    events.max_states = 10_000;
    events.max_events = 9;
    let event_result = run_multiway(&events, None);
    assert_eq!(event_result.termination, Termination::MaxEvents);
    assert!(event_result.events.len() <= 9);

    let mut payload = config(&["A"], &[("A", "AAAA")]);
    payload.max_depth = 10;
    payload.max_payload = 8;
    assert_eq!(
        run_multiway(&payload, None).termination,
        Termination::PayloadLimit
    );

    let mut memory = config(&["A"], &[("A", "AA")]);
    memory.max_depth = 20;
    memory.max_memory_bytes = 512;
    assert_eq!(
        run_multiway(&memory, None).termination,
        Termination::AllocationFailure
    );
}

#[test]
fn max_depth_and_cancellation_return_resumable_partial_state() {
    let mut depth = config(&["A"], &[("A", "AA")]);
    depth.max_depth = 2;
    let depth_result = run_multiway(&depth, None);
    assert_eq!(depth_result.termination, Termination::MaxDepth);
    assert!(!depth_result.complete);
    assert_ne!(depth_result.frontier.len(), 0);
    assert!(
        depth_result
            .states
            .iter()
            .all(|state| state.depth <= depth.max_depth)
    );

    let cancelled = AtomicBool::new(true);
    let cancelled_result = run_multiway(&depth, Some(&cancelled));
    assert_eq!(cancelled_result.termination, Termination::Cancelled);
    assert!(!cancelled_result.complete);
}

#[test]
fn repeated_runs_are_structurally_deterministic() {
    let config = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
    let first = run_multiway(&config, None);
    let second = run_multiway(&config, None);
    assert_eq!(first.states, second.states);
    assert_eq!(first.events, second.events);
    assert_eq!(first.states_per_depth, second.states_per_depth);
    assert_eq!(first.events_per_depth, second.events_per_depth);
    assert_eq!(first.termination, second.termination);
}

#[test]
fn canonical_exports_are_byte_deterministic_and_dot_keeps_multiplicity() {
    let mut config = config(&["A"], &[("A", "B"), ("A", "B")]);
    config.max_depth = 1;
    let first = run_multiway(&config, None);
    let second = run_multiway(&config, None);
    let first_json = export_multiway_json(&config, &first, &compute_multiway_metrics(&first))
        .expect("first export");
    let second_json = export_multiway_json(&config, &second, &compute_multiway_metrics(&second))
        .expect("second export");
    assert_eq!(first_json, second_json);
    assert!(first_json.starts_with("{\"format\":\"abi-multiway-v1\""));
    assert!(first_json.contains("\"resume\":{\"depth\":1"));

    let dot = export_multiway_dot(&config, &first).expect("dot");
    assert!(dot.contains("digraph multiway"));
    assert_eq!(dot.matches("s0 -> s1").count(), 2);
}

#[test]
fn interrupted_resume_matches_an_uninterrupted_canonical_export() {
    let mut full = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
    full.max_depth = 5;
    full.max_states = 500;
    full.max_events = 5_000;
    let direct = run_multiway(&full, None);
    let direct_export = export_multiway_json(&full, &direct, &compute_multiway_metrics(&direct))
        .expect("direct export");

    let mut shallow = full.clone();
    shallow.max_depth = 3;
    let partial = run_multiway(&shallow, None);
    assert_eq!(partial.termination, Termination::MaxDepth);
    let partial_export =
        export_multiway_json(&shallow, &partial, &compute_multiway_metrics(&partial))
            .expect("partial export");
    let resumed = resume_multiway(&partial_export, &full, None).expect("resume");
    let resumed_export = export_multiway_json(&full, &resumed, &compute_multiway_metrics(&resumed))
        .expect("resumed export");

    assert_eq!(resumed_export, direct_export);
}

#[test]
fn resume_rejects_mismatch_malformed_and_completed_documents() {
    let mut partial_config = config(&["A"], &[("A", "AB")]);
    partial_config.max_depth = 1;
    let partial = run_multiway(&partial_config, None);
    let export = export_multiway_json(
        &partial_config,
        &partial,
        &compute_multiway_metrics(&partial),
    )
    .expect("export");
    let mismatch = config(&["A"], &[("A", "BB")]);
    assert!(matches!(
        resume_multiway(&export, &mismatch, None),
        Err(MultiwayExportError::ConfigMismatch)
    ));
    assert!(matches!(
        resume_multiway("{\"format\":\"abi-multiway-v1\"}", &partial_config, None),
        Err(MultiwayExportError::MalformedExport)
    ));
    assert!(matches!(
        resume_multiway("{\"format\":\"nope\"}", &partial_config, None),
        Err(MultiwayExportError::UnsupportedFormat)
    ));

    let complete_config = config(&["A"], &[("A", "B"), ("B", "A")]);
    let complete = run_multiway(&complete_config, None);
    let complete_export = export_multiway_json(
        &complete_config,
        &complete,
        &compute_multiway_metrics(&complete),
    )
    .expect("complete export");
    assert!(matches!(
        resume_multiway(&complete_export, &complete_config, None),
        Err(MultiwayExportError::AlreadyComplete)
    ));
}

#[test]
fn token_lineage_links_producer_events_to_consumers() {
    let mut config = config(&["A"], &[("A", "AB"), ("B", "C")]);
    config.max_depth = 2;
    let result = run_multiway(&config, None);
    let edges = build_causal_edges(&config, &result);
    assert!(
        edges
            .iter()
            .any(|edge| edge.parent == 0 && edge.child > edge.parent)
    );
}

#[test]
fn wdbx_persistence_round_trips_latest_and_config_hash() {
    let directory = abi_foundation::temp_path::temp_file_path("abi_multiway_persist", "store");
    std::fs::create_dir_all(&directory).expect("fixture directory");
    let paths = crate::StorePaths {
        dir: directory.clone(),
        base: "multiway".to_owned(),
    };

    let mut config = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
    config.max_depth = 3;
    let result = run_multiway(&config, None);
    let export =
        export_multiway_json(&config, &result, &compute_multiway_metrics(&result)).expect("export");
    persist_multiway(paths.clone(), &config, &result, &export).expect("persist");
    assert_eq!(load_multiway_export(&paths, None).expect("latest"), export);
    let config_hash = super::hex_hash(multiway_config_hash(&config).expect("config hash"));
    assert_eq!(
        load_multiway_export(&paths, Some(&config_hash)).expect("by hash"),
        export
    );
    let reopened = crate::VersionedStore::open(paths).expect("reopen");
    assert!(reopened.stats().blocks >= 1);
    assert_eq!(
        reopened.get(&format!(
            "multiway:state:{}",
            super::hex_hash(result.states[0].hash)
        )),
        Some("A".to_owned())
    );

    std::fs::remove_dir_all(directory).expect("cleanup");
}
