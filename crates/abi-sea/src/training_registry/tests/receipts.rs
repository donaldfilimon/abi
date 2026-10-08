use super::*;
use crate::training_artifacts::publish_artifact;
use abi_ai::training_contracts::*;
use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::Path;
fn prepared(registry: &mut TrainingRegistry, root: &Path) -> RunReceipt {
    let record = record();
    evidence(registry, &record);
    registry.admit(record.clone()).unwrap();
    let manifest = snapshot(&record);
    registry.register_snapshot(manifest.clone()).unwrap();
    let base = publish_artifact(
        registry,
        root,
        &sha256_hex(b"base weights"),
        b"base weights",
    )
    .unwrap();
    let tokenizer =
        publish_artifact(registry, root, &sha256_hex(b"tokenizer"), b"tokenizer").unwrap();
    RunReceipt {
        version: 1,
        run_id: "run".into(),
        inputs: RunInputs {
            profile: TrainingProfile::Abbey,
            base,
            tokenizer,
            snapshot_id: manifest.root_digest,
            seed: 42,
            configuration_digest: "c".repeat(64),
            initial_tensor_digest: "a".repeat(64),
            trainer_principal: "teacher".into(),
            classification: ModelClassification::ProductionLlm,
        },
        state: RunState::Prepared,
        step_count: 0,
        adapter: None,
        tensor_change: None,
        evaluation_id: None,
        provenance: "trusted caller test attestation; not actual training".into(),
    }
}
fn submit(
    registry: &mut TrainingRegistry,
    root: &Path,
    run: &RunReceipt,
) -> Result<(), RegistryError> {
    let eligibility = registry.eligibility(&run.inputs.snapshot_id)?;
    registry.register_run("operator", root, &eligibility, run.clone())
}
fn root(temp: &Temp) -> PathBuf {
    let root = temp.0.join("artifacts");
    std::fs::DirBuilder::new()
        .mode(0o700)
        .create(&root)
        .unwrap();
    root
}
fn complete(
    registry: &mut TrainingRegistry,
    root: &Path,
    classification: ModelClassification,
) -> ModelRegistration {
    let mut run = prepared(registry, root);
    run.inputs.classification = classification;
    submit(registry, root, &run).unwrap();
    run.state = RunState::Running;
    submit(registry, root, &run).unwrap();
    let adapter = publish_artifact(
        registry,
        root,
        &sha256_hex(b"changed adapter weights"),
        b"changed adapter weights",
    )
    .unwrap();
    registry
        .register_evaluation(
            "operator",
            root,
            EvaluationReceipt {
                version: 1,
                id: "evaluation".into(),
                run_id: run.run_id.clone(),
                inputs: run.inputs.clone(),
                adapter: adapter.clone(),
                evaluator_principal: "reviewer".into(),
                passed: true,
                provenance: "independent trusted test attestation".into(),
            },
        )
        .unwrap();
    run.adapter = Some(adapter.clone());
    run.state = RunState::Completed;
    run.step_count = 10;
    run.tensor_change = Some(TensorChange {
        before_digest: run.inputs.initial_tensor_digest.clone(),
        after_digest: "b".repeat(64),
        changed_tensors: 1,
    });
    run.evaluation_id = Some("evaluation".into());
    submit(registry, root, &run).unwrap();
    let teacher = TeacherIdentity {
        base_digest: run.inputs.base.sha256.clone(),
        adapter_digest: adapter.sha256.clone(),
        tokenizer_digest: run.inputs.tokenizer.sha256.clone(),
        generation_configuration_digest: "f".repeat(64),
        qualification_digest: "e".repeat(64),
    };
    registry
        .register_qualification(
            "operator",
            QualificationEvidence {
                digest: teacher.qualification_digest.clone(),
                artifact: ArtifactIdentity {
                    base_digest: teacher.base_digest.clone(),
                    adapter_digest: teacher.adapter_digest.clone(),
                    tokenizer_digest: teacher.tokenizer_digest.clone(),
                    generation_configuration_digest: teacher
                        .generation_configuration_digest
                        .clone(),
                },
                artifact_principal: "teacher".into(),
                reviewer_principal: "reviewer".into(),
                purpose: QualificationPurpose::Teacher,
                passed: true,
                verified_weight_artifact: true,
            },
        )
        .unwrap();
    ModelRegistration {
        version: 1,
        id: "model".into(),
        profile: run.inputs.profile,
        completed_run_id: run.run_id,
        base: run.inputs.base,
        adapter,
        tokenizer: run.inputs.tokenizer,
        snapshot_id: run.inputs.snapshot_id,
        evaluation_id: "evaluation".into(),
        classification,
        teacher,
    }
}
#[test]
fn run_resume_completion_and_stale_receipts_refuse() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    let mut run = prepared(&mut registry, &root);
    let old = registry.eligibility(&run.inputs.snapshot_id).unwrap();
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Running;
    assert!(
        registry
            .register_run("operator", &root, &old, run.clone())
            .is_err()
    );
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Completed;
    assert!(submit(&mut registry, &root, &run).is_err());
    run.state = RunState::Interrupted;
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Running;
    run.inputs.seed += 1;
    assert!(submit(&mut registry, &root, &run).is_err());
    run.inputs.seed -= 1;
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Failed;
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Running;
    assert!(submit(&mut registry, &root, &run).is_err());
}
#[test]
fn model_selection_checks_bytes_exact_identity_reopen_and_revocation() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    let model = complete(&mut registry, &root, ModelClassification::ProductionLlm);
    assert!(select_model(&model, &registry, &root).is_err());
    registry
        .register_model("operator", &root, model.clone())
        .unwrap();
    assert!(
        select_model(&model, &registry, &root)
            .unwrap()
            .operator_cutover_required
    );
    let mut wrong = model.clone();
    wrong.teacher.qualification_digest = "f".repeat(64);
    assert!(registry.register_model("operator", &root, wrong).is_err());
    let mut wrong = model.clone();
    wrong.evaluation_id = "other".into();
    assert!(registry.register_model("operator", &root, wrong).is_err());
    drop(registry);
    let mut registry = temp.open();
    select_model(&model, &registry, &root).unwrap();
    let path = root.join(&model.adapter.relative_path);
    std::fs::write(&path, b"corrupt").unwrap();
    assert!(select_model(&model, &registry, &root).is_err());
    std::fs::write(&path, b"changed adapter weights").unwrap();
    std::fs::remove_file(&path).unwrap();
    assert!(select_model(&model, &registry, &root).is_err());
    std::fs::write(&path, b"changed adapter weights").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    registry
        .revoke(
            "record",
            RevocationDecision {
                operator_identity: "operator".into(),
                decision_id: "revoke".into(),
                reason: "withdrawn".into(),
            },
        )
        .unwrap();
    assert!(select_model(&model, &registry, &root).is_err());
    drop(registry);
    let registry = temp.open();
    assert!(select_model(&model, &registry, &root).is_err());
}
#[test]
fn fixture_model_cannot_qualify_even_when_registration_lies() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    let mut model = complete(&mut registry, &root, ModelClassification::Fixture);
    assert!(
        registry
            .register_model("operator", &root, model.clone())
            .is_err()
    );
    model.classification = ModelClassification::ProductionLlm;
    assert!(
        registry
            .register_model("operator", &root, model.clone())
            .is_err()
    );
    assert!(select_model(&model, &registry, &root).is_err());
}
#[test]
fn evaluation_rejects_self_alias_failures_and_wrong_run_subjects() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    let mut run = prepared(&mut registry, &root);
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Running;
    submit(&mut registry, &root, &run).unwrap();
    let adapter =
        publish_artifact(&mut registry, &root, &sha256_hex(b"adapter"), b"adapter").unwrap();
    let mut evaluation = EvaluationReceipt {
        version: 1,
        id: "evaluation".into(),
        run_id: run.run_id.clone(),
        inputs: run.inputs.clone(),
        adapter,
        evaluator_principal: "teacher-alias".into(),
        passed: true,
        provenance: "test".into(),
    };
    assert!(
        registry
            .register_evaluation("operator", &root, evaluation.clone())
            .is_err()
    );
    evaluation.evaluator_principal = "reviewer".into();
    evaluation.passed = false;
    assert!(
        registry
            .register_evaluation("operator", &root, evaluation.clone())
            .is_err()
    );
    evaluation.passed = true;
    evaluation.inputs.seed += 1;
    assert!(
        registry
            .register_evaluation("operator", &root, evaluation.clone())
            .is_err()
    );
    evaluation.inputs.seed -= 1;
    registry
        .register_evaluation("operator", &root, evaluation.clone())
        .unwrap();
    run.state = RunState::Completed;
    run.step_count = 1;
    run.adapter = Some(evaluation.adapter);
    run.tensor_change = Some(TensorChange {
        before_digest: run.inputs.initial_tensor_digest.clone(),
        after_digest: "b".repeat(64),
        changed_tensors: 1,
    });
    run.evaluation_id = Some("missing".into());
    assert!(submit(&mut registry, &root, &run).is_err());
}
#[test]
fn reopen_revalidates_corrupt_persisted_run_history() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    let run = prepared(&mut registry, &root);
    submit(&mut registry, &root, &run).unwrap();
    drop(registry);
    let mut store = DurableStore::open_directory(&temp.0).unwrap();
    let mut state: serde_json::Value = serde_json::from_str(store.get(STATE_KEY).unwrap()).unwrap();
    state["runs"]["history"]["run"][0]["state"] = serde_json::json!("Completed");
    store.put(STATE_KEY, &state.to_string()).unwrap();
    drop(store);
    assert!(
        TrainingRegistry::new(DurableStore::open_directory(&temp.0).unwrap(), policy()).is_err()
    );
}

#[test]
fn revoked_snapshot_refuses_actual_new_run_resume_and_completion() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    let mut run = prepared(&mut registry, &root);
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Running;
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Interrupted;
    submit(&mut registry, &root, &run).unwrap();
    let old = registry.eligibility(&run.inputs.snapshot_id).unwrap();
    registry
        .revoke(
            "record",
            RevocationDecision {
                operator_identity: "operator".into(),
                decision_id: "withdraw".into(),
                reason: "no longer approved".into(),
            },
        )
        .unwrap();
    run.state = RunState::Running;
    assert!(matches!(
        registry.register_run("operator", &root, &old, run.clone()),
        Err(RegistryError::Revoked(_))
    ));
    run.state = RunState::Completed;
    assert!(matches!(
        registry.register_run("operator", &root, &old, run.clone()),
        Err(RegistryError::Revoked(_))
    ));
    run.run_id = "new-run".into();
    run.state = RunState::Prepared;
    assert!(matches!(
        registry.register_run("operator", &root, &old, run),
        Err(RegistryError::Revoked(_))
    ));
}

#[test]
fn uncertain_artifact_registration_poison_prevents_receipt_publication() {
    let temp = Temp::new();
    let root = root(&temp);
    let mut registry = temp.open();
    registry.fail_after_write = true;
    assert!(publish_artifact(&mut registry, &root, &sha256_hex(b"artifact"), b"artifact").is_err());
    assert!(matches!(
        registry.artifact_session_healthy(),
        Err(RegistryError::PoisonedSession)
    ));
    drop(registry);
    let store = DurableStore::open_directory(&temp.0).unwrap();
    let state: serde_json::Value = serde_json::from_str(store.get(STATE_KEY).unwrap()).unwrap();
    assert_eq!(state["runs"]["models"], serde_json::json!({}));
    assert_eq!(state["runs"]["history"], serde_json::json!({}));
}

fn held_out_snapshot(registry: &mut TrainingRegistry, mixed: bool) -> SnapshotManifest {
    let mut held = record();
    held.id = "held-out".into();
    held.family_id = "held-out-family".into();
    held.permission.as_mut().unwrap().approved_use_decision_id = "held-out-permission".into();
    held.approval.as_mut().unwrap().decision_id = "held-out-approval".into();
    evidence(registry, &held);
    registry.admit(held.clone()).unwrap();
    let mut records = vec![held.clone()];
    let mut splits = [(held.family_id.clone(), Split::Evaluation)]
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    if mixed {
        records.push(record());
        splits.insert("family".into(), Split::Train);
    }
    let manifest = build_snapshot(
        &records,
        &SnapshotPolicy {
            family_splits: splits,
            permitted_specification_digests: ["a".repeat(64)].into(),
        },
    )
    .unwrap();
    registry.register_snapshot(manifest.clone()).unwrap();
    manifest
}
#[test]
fn held_out_and_mixed_snapshots_refuse_training_run() {
    for mixed in [false, true] {
        let temp = Temp::new();
        let mut registry = temp.open();
        let root = root(&temp);
        let mut run = prepared(&mut registry, &root);
        let manifest = held_out_snapshot(&mut registry, mixed);
        run.inputs.snapshot_id = manifest.root_digest;
        let receipt = registry.eligibility(&run.inputs.snapshot_id).unwrap();
        assert_eq!(
            registry.approved_records(&receipt).unwrap().len(),
            if mixed { 2 } else { 1 }
        );
        assert!(
            matches!(registry.training_records(&receipt), Err(RegistryError::Invalid(message))
            if message == "training snapshot contains held-out records")
        );
        let before = serde_json::to_value(&registry.state).unwrap();
        assert!(matches!(submit(&mut registry, &root, &run),
            Err(RegistryError::Invalid(message)) if message == "training snapshot contains held-out records"));
        assert_eq!(serde_json::to_value(&registry.state).unwrap(), before);
        run.state = RunState::Running;
        assert!(submit(&mut registry, &root, &run).is_err());
    }
}
#[test]
fn persisted_held_out_run_refuses_reopen() {
    let temp = Temp::new();
    let mut registry = temp.open();
    let root = root(&temp);
    let run = prepared(&mut registry, &root);
    submit(&mut registry, &root, &run).unwrap();
    let manifest = held_out_snapshot(&mut registry, false);
    let mut state = serde_json::to_value(&registry.state).unwrap();
    state["runs"]["history"]["run"][0]["inputs"]["snapshot_id"] = manifest.root_digest.into();
    registry.store.put(STATE_KEY, &state.to_string()).unwrap();
    drop(registry);
    assert!(
        matches!(TrainingRegistry::new(DurableStore::open_directory(&temp.0).unwrap(), policy()),
        Err(RegistryError::Invalid(message)) if message == "training snapshot contains held-out records")
    );
}
