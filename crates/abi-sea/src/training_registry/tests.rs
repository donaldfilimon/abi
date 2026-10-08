use super::*;
use abi_ai::training_contracts::{
    Approval, Permission, SourceRef, Split, TeacherIdentity, TrainingProfile,
};
use abi_wdbx::DurableError;
use std::os::unix::fs::DirBuilderExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
static COUNTER: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().canonicalize().unwrap().join(format!(
            "abi-training-registry-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new().mode(0o700).create(&p).unwrap();
        Self(p)
    }
    fn open(&self) -> TrainingRegistry {
        TrainingRegistry::new(DurableStore::open_directory(&self.0).unwrap(), policy()).unwrap()
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.0) {
            eprintln!("temporary registry cleanup failed: {error}");
        }
    }
}
fn policy() -> AuthorityPolicy {
    AuthorityPolicy {
        principals: ["operator", "reviewer", "teacher", "evaluator"]
            .into_iter()
            .map(|p| (p.into(), p.into()))
            .chain([("teacher-alias".into(), "teacher".into())])
            .collect(),
        evidence_registrars: ["operator".into()].into(),
        human_approvers: ["reviewer".into(), "teacher".into()].into(),
        permission_operators: ["operator".into()].into(),
        revocation_operators: ["operator".into()].into(),
        qualification_reviewers: ["reviewer".into(), "teacher".into()].into(),
    }
}
fn record() -> ApprovedRecord {
    ApprovedRecord {
        id: "record".into(),
        profile: TrainingProfile::Abbey,
        family_id: "family".into(),
        input: "question".into(),
        target: "answer".into(),
        sources: vec![SourceRef {
            repository_revision: "revision".into(),
            document_digest: "a".repeat(64),
            requirement_id: "req".into(),
        }],
        permission: Some(Permission {
            approved_use_decision_id: "permission".into(),
        }),
        approval: Some(Approval {
            approver_identity: "reviewer".into(),
            decision_id: "approval".into(),
            evidence: ApprovalKind::Human {},
        }),
        teacher: None,
    }
}
fn evidence(reg: &mut TrainingRegistry, r: &ApprovedRecord) {
    let subject = sha256_hex(&canonical_record_bytes(r).unwrap());
    reg.register_permission(
        "operator",
        PermissionDecision {
            id: r
                .permission
                .as_ref()
                .unwrap()
                .approved_use_decision_id
                .clone(),
            operator_identity: "operator".into(),
            record_digest: subject.clone(),
            approved_use: "training".into(),
        },
    )
    .unwrap();
    let a = r.approval.as_ref().unwrap();
    reg.register_approval(
        "operator",
        ApprovalDecision {
            id: a.decision_id.clone(),
            approver_identity: a.approver_identity.clone(),
            record_digest: subject,
            approved_use: "training".into(),
            evidence: a.evidence.clone(),
        },
    )
    .unwrap();
}
fn snapshot(r: &ApprovedRecord) -> SnapshotManifest {
    build_snapshot(
        std::slice::from_ref(r),
        &SnapshotPolicy {
            family_splits: [(r.family_id.clone(), Split::Train)].into(),
            permitted_specification_digests: ["a".repeat(64)].into(),
        },
    )
    .unwrap()
}
fn revocation() -> RevocationDecision {
    RevocationDecision {
        operator_identity: "operator".into(),
        decision_id: "revoke".into(),
        reason: "permission withdrawn".into(),
    }
}
fn qualification(
    purpose: QualificationPurpose,
    principal: &str,
    adapter: char,
) -> QualificationEvidence {
    QualificationEvidence {
        digest: adapter.to_string().repeat(64),
        artifact: ArtifactIdentity {
            base_digest: "b".repeat(64),
            adapter_digest: adapter.to_string().repeat(64),
            tokenizer_digest: "c".repeat(64),
            generation_configuration_digest: "d".repeat(64),
        },
        artifact_principal: principal.into(),
        reviewer_principal: "reviewer".into(),
        purpose,
        passed: true,
        verified_weight_artifact: true,
    }
}
fn teacher_record() -> (ApprovedRecord, QualificationEvidence) {
    let q = qualification(QualificationPurpose::Teacher, "teacher", 'e');
    let mut r = record();
    r.teacher = Some(TeacherIdentity {
        base_digest: q.artifact.base_digest.clone(),
        adapter_digest: q.artifact.adapter_digest.clone(),
        tokenizer_digest: q.artifact.tokenizer_digest.clone(),
        generation_configuration_digest: q.artifact.generation_configuration_digest.clone(),
        qualification_digest: q.digest.clone(),
    });
    (r, q)
}
#[test]
fn admission_requires_registered_decisions_and_exact_subject() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    assert!(matches!(
        reg.admit(r.clone()),
        Err(RegistryError::Missing(_))
    ));
    evidence(&mut reg, &r);
    let mut changed = r.clone();
    changed.target = "changed".into();
    assert!(matches!(reg.admit(changed), Err(RegistryError::Invalid(_))));
    reg.admit(r).unwrap();
}
#[test]
fn checkpoint_and_wal_reopen_preserve_admission() {
    for checkpoint in [false, true] {
        let temp = Temp::new();
        let mut reg = temp.open();
        let r = record();
        evidence(&mut reg, &r);
        reg.admit(r.clone()).unwrap();
        let m = snapshot(&r);
        reg.register_snapshot(m.clone()).unwrap();
        let receipt = reg.eligibility(&m.root_digest).unwrap();
        if checkpoint {
            reg.checkpoint().unwrap();
        }
        drop(reg);
        let reopened = temp.open();
        reopened.validate_eligibility(&receipt).unwrap();
    }
}
#[test]
fn duplicate_and_snapshot_mismatches_refuse() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    let m = snapshot(&r);
    assert!(reg.register_snapshot(m.clone()).is_err());
    evidence(&mut reg, &r);
    reg.admit(r.clone()).unwrap();
    reg.admit(r.clone()).unwrap();
    let mut changed = r.clone();
    changed.target = "changed".into();
    changed
        .permission
        .as_mut()
        .unwrap()
        .approved_use_decision_id = "permission2".into();
    changed.approval.as_mut().unwrap().decision_id = "approval2".into();
    evidence(&mut reg, &changed);
    assert!(matches!(
        reg.admit(changed),
        Err(RegistryError::Conflict(_))
    ));
    let mut wrong = m.clone();
    wrong.specification_digests = ["b".repeat(64)].into();
    assert!(reg.register_snapshot(wrong).is_err());
    let mut wrong = m.clone();
    wrong.records[0].digest = "b".repeat(64);
    assert!(reg.register_snapshot(wrong).is_err());
    let mut wrong = m.clone();
    wrong.version = 2;
    assert!(reg.register_snapshot(wrong).is_err());
    reg.register_snapshot(m).unwrap();
}
#[test]
fn revocation_blocks_export_new_run_and_resume_irreversibly() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    evidence(&mut reg, &r);
    reg.admit(r.clone()).unwrap();
    let m = snapshot(&r);
    reg.register_snapshot(m.clone()).unwrap();
    let receipt = reg.eligibility(&m.root_digest).unwrap();
    reg.revoke(&r.id, revocation()).unwrap();
    assert!(matches!(
        reg.eligibility(&m.root_digest),
        Err(RegistryError::Revoked(_))
    ));
    assert!(reg.validate_eligibility(&receipt).is_err());
    assert!(reg.register_snapshot(m.clone()).is_err());
    reg.admit(r).unwrap();
    assert!(reg.eligibility(&m.root_digest).is_err());
    let mut changed = revocation();
    changed.reason = "undo".into();
    assert!(matches!(
        reg.revoke("record", changed),
        Err(RegistryError::Conflict(_))
    ));
    drop(reg);
    assert!(temp.open().eligibility(&m.root_digest).is_err());
}
#[test]
fn any_mutation_invalidates_old_receipt() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    evidence(&mut reg, &r);
    reg.admit(r.clone()).unwrap();
    let m = snapshot(&r);
    reg.register_snapshot(m.clone()).unwrap();
    let receipt = reg.eligibility(&m.root_digest).unwrap();
    reg.register_qualification(
        "operator",
        qualification(QualificationPurpose::Teacher, "teacher", 'e'),
    )
    .unwrap();
    assert!(matches!(
        reg.validate_eligibility(&receipt),
        Err(RegistryError::StaleEligibility)
    ));
    assert!(reg.eligibility("unknown").is_err());
}
#[test]
fn writer_session_excludes_competition() {
    let temp = Temp::new();
    let reg = temp.open();
    assert!(matches!(
        DurableStore::open_directory(&temp.0),
        Err(DurableError::WriterBusy { .. })
    ));
    drop(reg);
    let _reopened = temp.open();
}
#[test]
fn privilege_use_and_qualification_outcomes_are_enforced() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    let subject = sha256_hex(&canonical_record_bytes(&r).unwrap());
    let d = PermissionDecision {
        id: "permission".into(),
        operator_identity: "operator".into(),
        record_digest: subject,
        approved_use: "training".into(),
    };
    assert!(matches!(
        reg.register_permission("reviewer", d.clone()),
        Err(RegistryError::Unauthorized)
    ));
    let mut wrong = d;
    wrong.approved_use = "inference".into();
    assert!(reg.register_permission("operator", wrong).is_err());
    for weights in [false, true] {
        let mut q = qualification(QualificationPurpose::Teacher, "teacher", 'e');
        q.verified_weight_artifact = weights;
        q.passed = false;
        assert!(reg.register_qualification("operator", q).is_err());
    }
    let mut q = qualification(QualificationPurpose::Teacher, "teacher", 'e');
    q.verified_weight_artifact = false;
    assert!(reg.register_qualification("operator", q).is_err());
    let mut q = qualification(QualificationPurpose::Teacher, "teacher", 'e');
    q.reviewer_principal = "teacher-alias".into();
    assert!(reg.register_qualification("operator", q).is_err());
}
#[test]
fn teacher_identity_and_independence_resolve_real_evidence() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let (r, q) = teacher_record();
    evidence(&mut reg, &r);
    assert!(reg.admit(r.clone()).is_err());
    reg.register_qualification("operator", q).unwrap();
    reg.admit(r.clone()).unwrap();
    let mut wrong = r.clone();
    wrong.id = "wrong".into();
    wrong.teacher.as_mut().unwrap().base_digest = "f".repeat(64);
    wrong.permission.as_mut().unwrap().approved_use_decision_id = "p2".into();
    wrong.approval.as_mut().unwrap().decision_id = "a2".into();
    evidence(&mut reg, &wrong);
    assert!(reg.admit(wrong).is_err());
    let mut alias = r;
    alias.id = "alias".into();
    alias.permission.as_mut().unwrap().approved_use_decision_id = "p3".into();
    alias.approval.as_mut().unwrap().decision_id = "a3".into();
    alias.approval.as_mut().unwrap().approver_identity = "teacher-alias".into();
    evidence(&mut reg, &alias);
    assert!(matches!(reg.admit(alias), Err(RegistryError::Unauthorized)));
}
#[test]
fn qualified_evaluator_binds_artifact_principal_and_purpose() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let (mut r, teacher) = teacher_record();
    reg.register_qualification("operator", teacher).unwrap();
    let evaluator = qualification(QualificationPurpose::Evaluator, "evaluator", 'f');
    reg.register_qualification("operator", evaluator.clone())
        .unwrap();
    r.approval.as_mut().unwrap().approver_identity = "evaluator".into();
    r.approval.as_mut().unwrap().evidence = ApprovalKind::QualifiedEvaluator {
        artifact_digest: evaluator.artifact.adapter_digest,
        qualification_digest: evaluator.digest,
    };
    evidence(&mut reg, &r);
    reg.admit(r).unwrap();
    let mut bad = qualification(QualificationPurpose::Teacher, "evaluator", 'a');
    bad.digest = "b".repeat(64);
    reg.register_qualification("operator", bad.clone()).unwrap();
    let decision = ApprovalDecision {
        id: "bad".into(),
        approver_identity: "evaluator".into(),
        record_digest: "a".repeat(64),
        approved_use: "training".into(),
        evidence: ApprovalKind::QualifiedEvaluator {
            artifact_digest: bad.artifact.adapter_digest,
            qualification_digest: bad.digest,
        },
    };
    assert!(reg.register_approval("operator", decision).is_err());
}
#[test]
fn unsupported_record_schema_refuses_on_reopen() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    evidence(&mut reg, &r);
    reg.admit(r).unwrap();
    let key = format!("training:v1:record:{}", reg.state.records["record"]);
    let original = reg.store.get(&key).unwrap();
    let mut blob: serde_json::Value = serde_json::from_str(original).unwrap();
    blob["version"] = 2.into();
    reg.store.put(&key, &blob.to_string()).unwrap();
    drop(reg);
    assert!(matches!(
        TrainingRegistry::new(DurableStore::open_directory(&temp.0).unwrap(), policy()),
        Err(RegistryError::Invalid(_))
    ));
}
#[test]
fn uncertain_publication_poison_requires_recovery() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    evidence(&mut reg, &r);
    reg.admit(r.clone()).unwrap();
    let m = snapshot(&r);
    reg.register_snapshot(m.clone()).unwrap();
    let receipt = reg.eligibility(&m.root_digest).unwrap();
    assert_eq!(reg.snapshot(&m.root_digest).unwrap(), m);
    assert_eq!(reg.approved_records(&receipt).unwrap(), vec![r.clone()]);
    reg.fail_after_write = true;
    assert!(matches!(
        reg.revoke(&r.id, revocation()),
        Err(RegistryError::Storage(_))
    ));
    assert!(!reg.state.revocations.contains_key(&r.id)); // Simulates persisted/in-memory divergence.
    assert!(matches!(
        reg.eligibility(&m.root_digest),
        Err(RegistryError::PoisonedSession)
    ));
    assert!(matches!(
        reg.validate_eligibility(&receipt),
        Err(RegistryError::PoisonedSession)
    ));
    assert!(matches!(reg.admit(r), Err(RegistryError::PoisonedSession)));
    assert!(matches!(
        reg.checkpoint(),
        Err(RegistryError::PoisonedSession)
    ));
    assert!(matches!(
        reg.snapshot(&m.root_digest),
        Err(RegistryError::PoisonedSession)
    ));
    assert!(matches!(
        reg.approved_records(&receipt),
        Err(RegistryError::PoisonedSession)
    ));
    drop(reg);
    assert!(matches!(
        temp.open().eligibility(&m.root_digest),
        Err(RegistryError::Revoked(_))
    ));
}
#[test]
fn persisted_manifest_is_recomputed_before_eligibility() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let r = record();
    evidence(&mut reg, &r);
    reg.admit(r.clone()).unwrap();
    let manifest = snapshot(&r);
    reg.register_snapshot(manifest.clone()).unwrap();
    let mut state = reg.state.clone();
    state
        .snapshots
        .get_mut(&manifest.root_digest)
        .unwrap()
        .version = 2;
    reg.store
        .put(STATE_KEY, &serde_json::to_string(&state).unwrap())
        .unwrap();
    drop(reg);
    assert!(matches!(
        TrainingRegistry::new(DurableStore::open_directory(&temp.0).unwrap(), policy()),
        Err(RegistryError::Invalid(_))
    ));
}
#[test]
fn wal_budget_checkpoints_before_cumulative_growth_and_reopens() {
    let temp = Temp::new();
    let mut reg = temp.open();
    reg.wal_budget = 1200;
    // Existing frames consume the WAL budget, including frames before registry ownership.
    reg.store.put("prefill", &"x".repeat(1000)).unwrap();
    let wal = abi_wdbx::wal::wal_path(reg.store.paths());
    let before = std::fs::metadata(&wal).unwrap().len();
    reg.write("training:v1:test", "small").unwrap();
    let after = std::fs::metadata(&wal).unwrap().len();
    assert!(after < before);
    assert_eq!(
        reg.store.recovery_source(),
        abi_wdbx::wal::RecoverySource::Segment
    );
    assert!(after <= reg.wal_budget);
    drop(reg);
    let reopened = temp.open();
    assert_eq!(reopened.store.get("training:v1:test"), Some("small"));
    assert_eq!(
        reopened.store.get("prefill"),
        Some("x".repeat(1000).as_str())
    );
}
#[test]
fn wal_budget_rejects_oversized_frame_before_any_write() {
    let temp = Temp::new();
    let mut reg = temp.open();
    reg.wal_budget = 1200;
    let wal = abi_wdbx::wal::wal_path(reg.store.paths());
    let before = std::fs::read(&wal).unwrap();
    // Escaped characters require substantially more encoded space than source bytes.
    assert!(matches!(
        reg.write("training:v1:test", &"\u{0001}".repeat(200)),
        Err(RegistryError::Invalid(_))
    ));
    assert_eq!(std::fs::read(&wal).unwrap(), before);
    assert!(!reg.poisoned);
    assert!(reg.store.get("training:v1:test").is_none());
    reg.write("training:v1:test", "small").unwrap();
}
#[test]
fn wal_frame_bound_covers_encoding_and_checked_arithmetic() {
    for (key, value) in [
        ("key", "ascii"),
        ("\"\\\u{0001}", "\n\r\t\u{0000}"),
        ("ü", "😀"),
    ] {
        let encoded = serde_json::json!({"type":"kv", "key":key, "value":value}).to_string();
        assert!(
            u64::try_from(encoded.len() + 10).unwrap()
                <= frame_size_bound(key.len(), value.len()).unwrap()
        );
    }
    assert!(frame_size_bound(usize::MAX, 1).is_err());
    assert!(frame_size_bound(usize::MAX / 6, 0).is_err());
}
#[test]
fn wal_budget_accepts_exact_bound_without_checkpoint() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let wal = abi_wdbx::wal::wal_path(reg.store.paths());
    let frame = frame_size_bound("training:v1:test".len(), "small".len()).unwrap();
    reg.wal_budget = 64 + frame;
    let before = reg.store.checkpoint_epoch();
    reg.write("training:v1:test", "small").unwrap();
    assert_eq!(reg.store.checkpoint_epoch(), before);
    assert!(std::fs::metadata(wal).unwrap().len() <= reg.wal_budget);
}

#[cfg(unix)]
mod receipts;
mod split_isolation;

#[cfg(unix)]
#[test]
fn public_store_root_refuses_before_registry_writes() {
    use std::os::unix::fs::PermissionsExt;
    let temp = Temp::new();
    let store = DurableStore::open_directory(&temp.0).unwrap();
    std::fs::set_permissions(&temp.0, std::fs::Permissions::from_mode(0o755)).unwrap();
    let wal = abi_wdbx::wal::wal_path(store.paths());
    let before = std::fs::read(&wal).unwrap();
    assert!(matches!(
        TrainingRegistry::new(store, policy()),
        Err(RegistryError::Invalid(_))
    ));
    assert_eq!(std::fs::read(&wal).unwrap(), before);
}
#[cfg(unix)]
#[test]
fn symlink_store_root_refuses() {
    use std::os::unix::fs::symlink;
    let temp = Temp::new();
    let link = temp.0.join("alias");
    symlink(&temp.0, &link).unwrap();
    let store = DurableStore::open_directory(&link).unwrap();
    assert!(matches!(
        TrainingRegistry::new(store, policy()),
        Err(RegistryError::Invalid(_))
    ));
}

#[test]
fn store_symlink_and_hardlink_files_refuse() {
    use std::os::unix::fs::symlink;
    for hard in [false, true] {
        let temp = Temp::new();
        let store = DurableStore::open_directory(&temp.0).unwrap();
        let wal = abi_wdbx::wal::wal_path(store.paths());
        if hard {
            std::fs::hard_link(&wal, temp.0.join("alias")).unwrap();
        } else {
            let real = temp.0.join("real-wal");
            std::fs::rename(&wal, &real).unwrap();
            symlink(&real, &wal).unwrap();
        }
        assert!(matches!(
            TrainingRegistry::new(store, policy()),
            Err(RegistryError::Invalid(_))
        ));
    }
}
#[test]
fn private_root_protects_backend_files_after_checkpoint_reopen() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let temp = Temp::new();
    let mut registry = temp.open();
    let r = record();
    evidence(&mut registry, &r);
    registry.admit(r.clone()).unwrap();
    let manifest = snapshot(&r);
    registry.register_snapshot(manifest.clone()).unwrap();
    registry.checkpoint().unwrap();
    assert_eq!(
        std::fs::symlink_metadata(&temp.0).unwrap().mode() & 0o777,
        0o700
    );
    for entry in std::fs::read_dir(&temp.0).unwrap() {
        let path = entry.unwrap().path();
        assert_eq!(path.parent(), Some(temp.0.as_path()));
        let meta = std::fs::symlink_metadata(&path).unwrap();
        assert!(!meta.file_type().is_symlink());
        assert_eq!(meta.nlink(), 1);
        // Pin the accepted ambient backend mode under the protected root.
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o644)).unwrap();
    }
    drop(registry);
    let reopened = temp.open();
    let receipt = reopened.eligibility(&manifest.root_digest).unwrap();
    assert_eq!(reopened.training_records(&receipt).unwrap(), vec![r]);
    assert_eq!(reopened.snapshot(&manifest.root_digest).unwrap(), manifest);
    assert_eq!(
        std::fs::symlink_metadata(&temp.0).unwrap().mode() & 0o777,
        0o700
    );
}

#[test]
fn training_export_preserves_identity_and_refuses_revocation() {
    let temp = Temp::new();
    let mut registry = temp.open();
    let r = record();
    evidence(&mut registry, &r);
    registry.admit(r.clone()).unwrap();
    let manifest = snapshot(&r);
    registry.register_snapshot(manifest.clone()).unwrap();
    let receipt = registry.eligibility(&manifest.root_digest).unwrap();
    let exported = registry.training_records(&receipt).unwrap();
    assert_eq!(exported, vec![r.clone()]);
    assert_eq!(snapshot(&exported[0]), manifest);
    registry.revoke(&r.id, revocation()).unwrap();
    assert!(matches!(
        registry.training_records(&receipt),
        Err(RegistryError::Revoked(_))
    ));
    drop(registry);
    let reopened = temp.open();
    assert!(matches!(
        reopened.training_records(&receipt),
        Err(RegistryError::Revoked(_))
    ));
}
