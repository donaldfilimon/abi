//! Synthetic fixtures only: these tests never train or qualify actual models.
#![cfg(unix)]
use abi_ai::training_contracts::*;
use abi_sea::training_artifacts::publish_artifact;
use abi_sea::training_registry::*;
use abi_wdbx::durable::DurableStore;
use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn all_profiles_export_reopen_and_fixture_receipts_remain_unqualified() {
    let temp = Temp::new();
    let root = temp.root();
    let mut registry = temp.open();
    let records = seed_records();
    let json = serde_json::to_string(&records).unwrap();
    for name in ["Abbey", "Aviva", "Abi", "Quesar"] {
        assert!(json.contains(&format!("\"{name}\"")));
    }
    assert_eq!(
        serde_json::from_str::<Vec<ApprovedRecord>>(&json).unwrap(),
        records
    );
    for record in &records {
        assert!(matches!(
            registry.admit(record.clone()),
            Err(RegistryError::Missing(_))
        ));
        admit_seed(&mut registry, record);
    }
    let manifest = build_snapshot(
        &records,
        &SnapshotPolicy {
            family_splits: records
                .iter()
                .map(|r| (r.family_id.clone(), Split::Train))
                .collect(),
            permitted_specification_digests: [sha256_hex(b"seed specification fixture")].into(),
        },
    )
    .unwrap();
    registry.register_snapshot(manifest.clone()).unwrap();
    let eligibility = registry.eligibility(&manifest.root_digest).unwrap();
    let exported = registry.approved_records(&eligibility).unwrap();
    assert_eq!(exported.len(), 4);
    assert_eq!(
        build_snapshot(
            &exported,
            &SnapshotPolicy {
                family_splits: records
                    .iter()
                    .map(|r| (r.family_id.clone(), Split::Train))
                    .collect(),
                permitted_specification_digests: manifest.specification_digests.clone(),
            }
        )
        .unwrap(),
        manifest
    );
    registry.checkpoint().unwrap();
    drop(registry);
    let mut registry = temp.open();
    registry.validate_eligibility(&eligibility).unwrap();
    assert_eq!(registry.snapshot(&manifest.root_digest).unwrap(), manifest);
    assert_eq!(registry.approved_records(&eligibility).unwrap(), exported);
    let fixtures = fixture_artifacts(&mut registry, &root);
    for profile in [
        TrainingProfile::Abbey,
        TrainingProfile::Aviva,
        TrainingProfile::Abi,
        TrainingProfile::Quesar,
    ] {
        exercise_profile(
            &mut registry,
            &root,
            &records,
            &manifest,
            profile,
            &fixtures,
        );
    }
    drop(registry);
    let registry = temp.open();
    assert_eq!(
        registry
            .approved_records(&registry.eligibility(&manifest.root_digest).unwrap())
            .unwrap(),
        exported
    );
}

#[test]
fn revocation_invalidates_export_and_actual_resume_after_reopen() {
    let temp = Temp::new();
    let root = temp.root();
    let mut registry = temp.open();
    let record = seed_records().remove(0);
    admit_seed(&mut registry, &record);
    let manifest = build_snapshot(
        std::slice::from_ref(&record),
        &SnapshotPolicy {
            family_splits: [(record.family_id.clone(), Split::Train)].into(),
            permitted_specification_digests: [record.sources[0].document_digest.clone()].into(),
        },
    )
    .unwrap();
    registry.register_snapshot(manifest.clone()).unwrap();
    let mut run = RunReceipt {
        version: 1,
        run_id: "resume-fixture".into(),
        inputs: RunInputs {
            profile: record.profile,
            base: artifact(&mut registry, &root, b"base fixture"),
            tokenizer: artifact(&mut registry, &root, b"tokenizer fixture"),
            snapshot_id: manifest.root_digest.clone(),
            seed: 1,
            configuration_digest: sha256_hex(b"configuration"),
            initial_tensor_digest: sha256_hex(b"initial"),
            trainer_principal: "trainer".into(),
            classification: ModelClassification::Fixture,
        },
        state: RunState::Prepared,
        step_count: 0,
        adapter: None,
        tensor_change: None,
        evaluation_id: None,
        provenance: "unqualified fixture".into(),
    };
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Running;
    submit(&mut registry, &root, &run).unwrap();
    run.state = RunState::Interrupted;
    submit(&mut registry, &root, &run).unwrap();
    let eligibility = registry.eligibility(&manifest.root_digest).unwrap();
    registry
        .revoke(
            &record.id,
            RevocationDecision {
                operator_identity: "operator".into(),
                decision_id: "withdraw-seed".into(),
                reason: "fixture permission withdrawn".into(),
            },
        )
        .unwrap();
    drop(registry);
    let mut registry = temp.open();
    assert!(matches!(
        registry.eligibility(&manifest.root_digest),
        Err(RegistryError::Revoked(_))
    ));
    assert!(matches!(
        registry.approved_records(&eligibility),
        Err(RegistryError::Revoked(_))
    ));
    run.state = RunState::Running;
    assert!(matches!(
        registry.register_run("operator", &root, &eligibility, run.clone()),
        Err(RegistryError::Revoked(_))
    ));
    run.run_id = "new-fixture".into();
    run.state = RunState::Prepared;
    assert!(matches!(
        registry.register_run("operator", &root, &eligibility, run),
        Err(RegistryError::Revoked(_))
    ));
}

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "abi-training-foundation-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .unwrap();
        Self(path)
    }
    fn root(&self) -> PathBuf {
        let root = self.0.join("artifacts");
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&root)
            .unwrap();
        root
    }
    fn open(&self) -> TrainingRegistry {
        TrainingRegistry::new(
            DurableStore::open_directory(&self.0).unwrap(),
            AuthorityPolicy {
                principals: ["operator", "reviewer", "trainer"]
                    .into_iter()
                    .map(|p| (p.into(), p.into()))
                    .chain([("trainer-alias".into(), "trainer".into())])
                    .collect(),
                evidence_registrars: ["operator".into()].into(),
                human_approvers: ["reviewer".into()].into(),
                permission_operators: ["operator".into()].into(),
                revocation_operators: ["operator".into()].into(),
                qualification_reviewers: ["reviewer".into()].into(),
            },
        )
        .unwrap()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("fixture cleanup failed: {error}");
        }
    }
}
fn seed_records() -> Vec<ApprovedRecord> {
    [
        TrainingProfile::Abbey,
        TrainingProfile::Aviva,
        TrainingProfile::Abi,
        TrainingProfile::Quesar,
    ]
    .into_iter()
    .map(|profile| ApprovedRecord {
        id: format!("seed-{profile:?}"),
        profile,
        family_id: format!("family-{profile:?}"),
        input: "synthetic specification question".into(),
        target: "synthetic approved answer".into(),
        sources: vec![SourceRef {
            repository_revision: "synthetic-revision".into(),
            document_digest: sha256_hex(b"seed specification fixture"),
            requirement_id: "fixture-requirement".into(),
        }],
        permission: Some(Permission {
            approved_use_decision_id: format!("permission-{profile:?}"),
        }),
        approval: Some(Approval {
            approver_identity: "reviewer".into(),
            decision_id: format!("approval-{profile:?}"),
            evidence: ApprovalKind::Human {},
        }),
        teacher: None,
    })
    .collect()
}
fn admit_seed(registry: &mut TrainingRegistry, record: &ApprovedRecord) {
    let digest = sha256_hex(&canonical_record_bytes(record).unwrap());
    registry
        .register_permission(
            "operator",
            PermissionDecision {
                id: record
                    .permission
                    .as_ref()
                    .unwrap()
                    .approved_use_decision_id
                    .clone(),
                operator_identity: "operator".into(),
                record_digest: digest.clone(),
                approved_use: "training".into(),
            },
        )
        .unwrap();
    registry
        .register_approval(
            "operator",
            ApprovalDecision {
                id: record.approval.as_ref().unwrap().decision_id.clone(),
                approver_identity: "reviewer".into(),
                record_digest: digest,
                approved_use: "training".into(),
                evidence: ApprovalKind::Human {},
            },
        )
        .unwrap();
    let mut altered = record.clone();
    altered.target.push_str(" adversarial alteration");
    assert!(matches!(
        registry.admit(altered),
        Err(RegistryError::Invalid(_))
    ));
    registry.admit(record.clone()).unwrap();
}
fn artifact(registry: &mut TrainingRegistry, root: &Path, bytes: &[u8]) -> ArtifactRef {
    publish_artifact(registry, root, &sha256_hex(bytes), bytes).unwrap()
}
fn submit(
    registry: &mut TrainingRegistry,
    root: &Path,
    run: &RunReceipt,
) -> Result<(), RegistryError> {
    let eligibility = registry.eligibility(&run.inputs.snapshot_id)?;
    registry.register_run("operator", root, &eligibility, run.clone())
}

struct FixtureArtifacts {
    base: ArtifactRef,
    tokenizer: ArtifactRef,
    adapter: ArtifactRef,
    teacher: TeacherIdentity,
}
fn fixture_artifacts(registry: &mut TrainingRegistry, root: &Path) -> FixtureArtifacts {
    let base = artifact(registry, root, b"fixture base, no model tensors");
    let tokenizer = artifact(registry, root, b"fixture tokenizer");
    let adapter = artifact(registry, root, b"fake candidate, unqualified");
    let teacher = TeacherIdentity {
        base_digest: base.sha256.clone(),
        adapter_digest: adapter.sha256.clone(),
        tokenizer_digest: tokenizer.sha256.clone(),
        generation_configuration_digest: sha256_hex(b"generation fixture"),
        qualification_digest: sha256_hex(b"unqualified candidate"),
    };
    let qualification = QualificationEvidence {
        digest: teacher.qualification_digest.clone(),
        artifact: ArtifactIdentity {
            base_digest: teacher.base_digest.clone(),
            adapter_digest: teacher.adapter_digest.clone(),
            tokenizer_digest: teacher.tokenizer_digest.clone(),
            generation_configuration_digest: teacher.generation_configuration_digest.clone(),
        },
        artifact_principal: "trainer".into(),
        reviewer_principal: "reviewer".into(),
        purpose: QualificationPurpose::Teacher,
        passed: true,
        verified_weight_artifact: false,
    };
    assert!(matches!(
        registry.register_qualification("operator", qualification),
        Err(RegistryError::Invalid(_))
    ));
    FixtureArtifacts {
        base,
        tokenizer,
        adapter,
        teacher,
    }
}
fn exercise_profile(
    registry: &mut TrainingRegistry,
    root: &Path,
    records: &[ApprovedRecord],
    manifest: &SnapshotManifest,
    profile: TrainingProfile,
    fixtures: &FixtureArtifacts,
) {
    let FixtureArtifacts {
        base,
        tokenizer,
        adapter,
        teacher,
    } = fixtures;
    let mut run = RunReceipt {
        version: 1,
        run_id: format!("fixture-{profile:?}"),
        inputs: RunInputs {
            profile,
            base: base.clone(),
            tokenizer: tokenizer.clone(),
            snapshot_id: manifest.root_digest.clone(),
            seed: 7,
            configuration_digest: sha256_hex(b"optimizer fixture"),
            initial_tensor_digest: sha256_hex(b"initial fixture"),
            trainer_principal: "trainer".into(),
            classification: ModelClassification::Fixture,
        },
        state: RunState::Prepared,
        step_count: 0,
        adapter: None,
        tensor_change: None,
        evaluation_id: None,
        provenance: "synthetic contract fixture; no training performed".into(),
    };
    assert!(matches!(
        submit(registry, root, &run),
        Err(RegistryError::Invalid(message)) if message == "run profile mismatch"
    ));
    let profile_records: Vec<_> = records
        .iter()
        .filter(|record| record.profile == profile)
        .cloned()
        .collect();
    assert_eq!(profile_records.len(), 1);
    let profile_manifest = build_snapshot(
        &profile_records,
        &SnapshotPolicy {
            family_splits: profile_records
                .iter()
                .map(|record| (record.family_id.clone(), Split::Train))
                .collect(),
            permitted_specification_digests: manifest.specification_digests.clone(),
        },
    )
    .unwrap();
    registry
        .register_snapshot(profile_manifest.clone())
        .unwrap();
    run.inputs
        .snapshot_id
        .clone_from(&profile_manifest.root_digest);
    submit(registry, root, &run).unwrap();
    run.state = RunState::Running;
    submit(registry, root, &run).unwrap();
    let (run, evaluation) = complete_fixture_run(registry, root, run, adapter);
    let mut model = ModelRegistration {
        version: 1,
        id: format!("fake-model-{profile:?}"),
        profile,
        completed_run_id: run.run_id,
        base: base.clone(),
        adapter: adapter.clone(),
        tokenizer: tokenizer.clone(),
        snapshot_id: profile_manifest.root_digest.clone(),
        evaluation_id: evaluation.id,
        classification: ModelClassification::Fixture,
        teacher: teacher.clone(),
    };
    assert!(
        registry
            .register_model("operator", root, model.clone())
            .is_err()
    );
    model.classification = ModelClassification::ProductionLlm;
    assert!(
        registry
            .register_model("operator", root, model.clone())
            .is_err()
    );
    assert!(select_model(&model, registry, root).is_err());
}
fn complete_fixture_run(
    registry: &mut TrainingRegistry,
    root: &Path,
    mut run: RunReceipt,
    adapter: &ArtifactRef,
) -> (RunReceipt, EvaluationReceipt) {
    let profile = run.inputs.profile;
    let mut evaluation = EvaluationReceipt {
        version: 1,
        id: format!("eval-{profile:?}"),
        run_id: run.run_id.clone(),
        inputs: run.inputs.clone(),
        adapter: adapter.clone(),
        evaluator_principal: "trainer-alias".into(),
        passed: true,
        provenance: "fake evaluation of fixture bytes".into(),
    };
    assert!(matches!(
        registry.register_evaluation("operator", root, evaluation.clone()),
        Err(RegistryError::Unauthorized)
    ));
    evaluation.evaluator_principal = "reviewer".into();
    registry
        .register_evaluation("operator", root, evaluation.clone())
        .unwrap();
    run.state = RunState::Completed;
    run.step_count = 1;
    run.adapter = Some(adapter.clone());
    run.evaluation_id = Some(evaluation.id.clone());
    assert!(matches!(
        submit(registry, root, &run),
        Err(RegistryError::Invalid(_))
    ));
    run.tensor_change = Some(TensorChange {
        before_digest: run.inputs.initial_tensor_digest.clone(),
        after_digest: sha256_hex(b"changed fixture"),
        changed_tensors: 1,
    });
    submit(registry, root, &run).unwrap();
    (run, evaluation)
}
