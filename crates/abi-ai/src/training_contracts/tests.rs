use super::*;
fn d(c: char) -> String {
    c.to_string().repeat(64)
}
fn record() -> ApprovedRecord {
    ApprovedRecord {
        id: "one".into(),
        profile: TrainingProfile::Abbey,
        family_id: "family".into(),
        input: "question".into(),
        target: "answer".into(),
        sources: vec![SourceRef {
            repository_revision: "revision".into(),
            document_digest: d('a'),
            requirement_id: "req".into(),
        }],
        permission: Some(Permission {
            approved_use_decision_id: "use-1".into(),
        }),
        approval: Some(Approval {
            approver_identity: "reviewer".into(),
            decision_id: "review-1".into(),
            evidence: ApprovalKind::Human {},
        }),
        teacher: None,
    }
}
fn teacher() -> TeacherIdentity {
    TeacherIdentity {
        base_digest: d('b'),
        adapter_digest: d('c'),
        tokenizer_digest: d('d'),
        generation_configuration_digest: d('e'),
        qualification_digest: d('f'),
    }
}
fn policy() -> SnapshotPolicy {
    SnapshotPolicy {
        family_splits: BTreeMap::from([("family".into(), Split::Train)]),
        permitted_specification_digests: BTreeSet::from([d('a')]),
    }
}
#[test]
fn unknown_profile_refuses() {
    assert!(serde_json::from_str::<TrainingProfile>("\"Other\"").is_err());
}
#[test]
fn absent_permission_refuses() {
    let mut r = record();
    r.permission = None;
    assert_eq!(validate_record(&r), Err(ContractError::MissingPermission));
}
#[test]
fn absent_approval_refuses() {
    let mut r = record();
    r.approval = None;
    assert_eq!(validate_record(&r), Err(ContractError::MissingApproval));
}
#[test]
fn unqualified_teacher_refuses() {
    let mut r = record();
    let mut t = teacher();
    t.qualification_digest.clear();
    r.teacher = Some(t);
    assert_eq!(validate_record(&r), Err(ContractError::UnqualifiedTeacher));
}
#[test]
fn teacher_required_artifacts_refuse() {
    for field in 0..4 {
        let mut r = record();
        let mut t = teacher();
        let fields = [
            &mut t.base_digest,
            &mut t.adapter_digest,
            &mut t.tokenizer_digest,
            &mut t.generation_configuration_digest,
        ];
        fields.into_iter().nth(field).unwrap().clear();
        r.teacher = Some(t);
        assert!(matches!(
            validate_record(&r),
            Err(ContractError::InvalidDigest(_))
        ));
    }
}
#[test]
fn teacher_self_approval_refuses() {
    let mut r = record();
    r.teacher = Some(teacher());
    r.approval.as_mut().unwrap().evidence = ApprovalKind::QualifiedEvaluator {
        artifact_digest: d('c'),
        qualification_digest: d('f'),
    };
    assert_eq!(validate_record(&r), Err(ContractError::TeacherSelfApproval));
}
#[test]
fn independent_approvals_accept() {
    let mut r = record();
    r.teacher = Some(teacher());
    assert_eq!(validate_record(&r), Ok(()));
    r.approval.as_mut().unwrap().evidence = ApprovalKind::QualifiedEvaluator {
        artifact_digest: d('a'),
        qualification_digest: d('b'),
    };
    assert_eq!(validate_record(&r), Ok(()));
}
#[test]
fn malformed_evaluator_refuses() {
    let mut r = record();
    r.approval.as_mut().unwrap().evidence = ApprovalKind::QualifiedEvaluator {
        artifact_digest: d('a'),
        qualification_digest: String::new(),
    };
    assert_eq!(
        validate_record(&r),
        Err(ContractError::InvalidDigest(
            "evaluator_qualification_digest"
        ))
    );
}
#[test]
fn permutation_is_deterministic() {
    let a = record();
    let mut b = a.clone();
    b.id = "two".into();
    let x = build_snapshot(&[a.clone(), b.clone()], &policy()).unwrap();
    let y = build_snapshot(&[b, a], &policy()).unwrap();
    assert_eq!(
        serde_json::to_vec(&x).unwrap(),
        serde_json::to_vec(&y).unwrap()
    );
    assert_eq!(x.root_digest.len(), 64);
}
#[test]
fn duplicate_changed_content_refuses() {
    let a = record();
    let mut b = a.clone();
    b.target = "changed".into();
    assert_eq!(
        build_snapshot(&[a, b], &policy()),
        Err(ContractError::DuplicateId("one".into()))
    );
}
#[test]
fn equal_duplicates_collapse() {
    let r = record();
    assert_eq!(
        build_snapshot(&[r.clone(), r], &policy())
            .unwrap()
            .records
            .len(),
        1
    );
}
#[test]
fn families_share_split() {
    let a = record();
    let mut b = a.clone();
    b.id = "two".into();
    let mut p = policy();
    p.family_splits.insert("family".into(), Split::Evaluation);
    let m = build_snapshot(&[a, b], &p).unwrap();
    assert!(m.records.iter().all(|r| r.split == Split::Evaluation));
}
#[test]
fn teacher_evaluation_refuses() {
    let mut r = record();
    r.teacher = Some(teacher());
    let mut p = policy();
    p.family_splits.insert("family".into(), Split::Evaluation);
    assert_eq!(
        build_snapshot(&[r], &p),
        Err(ContractError::TeacherInEvaluation("one".into()))
    );
}
#[test]
fn absent_family_refuses() {
    let mut p = policy();
    p.family_splits.clear();
    assert_eq!(
        build_snapshot(&[record()], &p),
        Err(ContractError::MissingFamilySplit("family".into()))
    );
}
#[test]
fn changed_specification_refuses() {
    let mut r = record();
    r.sources[0].document_digest = d('b');
    assert_eq!(
        build_snapshot(&[r], &policy()),
        Err(ContractError::UnpermittedSpecification(d('b')))
    );
}
#[test]
fn source_order_and_duplicates_are_sets() {
    let mut r = record();
    let mut other = r.sources[0].clone();
    other.requirement_id = "other".into();
    r.sources.push(other);
    let a = canonical_record_bytes(&r).unwrap();
    r.sources.reverse();
    r.sources.push(r.sources[0].clone());
    assert_eq!(a, canonical_record_bytes(&r).unwrap());
}
#[test]
fn canonical_encoding_preserves_delimiters_and_unicode() {
    let mut a = record();
    a.input = "a\u{1f}b".into();
    a.target = "c\n\"é".into();
    let mut b = a.clone();
    b.input = "a".into();
    b.target = "b\u{1f}c\n\"é".into();
    assert_ne!(
        canonical_record_bytes(&a).unwrap(),
        canonical_record_bytes(&b).unwrap()
    );
    let bytes = canonical_record_bytes(&a).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["record"]["input"], a.input);
    assert_eq!(value["version"], 1);
}
#[test]
fn strict_unknown_fields_and_numeric_config_refuse() {
    let mut v = serde_json::to_value(record()).unwrap();
    v["surprise"] = true.into();
    assert!(serde_json::from_value::<ApprovedRecord>(v).is_err());
    let mut v = serde_json::to_value(teacher()).unwrap();
    v["generation_configuration_digest"] = 0.5.into();
    assert!(serde_json::from_value::<TeacherIdentity>(v).is_err());
    assert!(
        serde_json::from_value::<ApprovalKind>(serde_json::json!({"kind":"Human","extra":true}))
            .is_err()
    );
}
#[test]
fn uppercase_digest_and_blank_identity_refuse() {
    let mut r = record();
    r.sources[0].document_digest = d('A');
    assert!(matches!(
        validate_record(&r),
        Err(ContractError::InvalidDigest(_))
    ));
    r = record();
    r.approval.as_mut().unwrap().approver_identity = " ".into();
    assert_eq!(
        validate_record(&r),
        Err(ContractError::EmptyIdentity("approver_identity"))
    );
}
#[test]
fn sha256_known_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn run_receipts_require_real_transition_validation() {
    let run = super::receipts::tests::prepared();
    let mut completed = run.clone();
    completed.state = RunState::Completed;
    assert!(validate_run_transition(&run, &completed).is_err());
}

#[test]
fn canonical_v1_record_and_snapshot_fixed_vectors() {
    let mut r = record();
    r.input = "a\u{1f}b".into();
    r.target = "c\n\"é".into();
    let mut other = r.sources[0].clone();
    other.requirement_id = "other".into();
    r.sources.push(other);
    // Independent fixed UTF-8/preimage vectors pin nulls, sorted sources and escaping.
    let expected_record = r#"{"record":{"approval":{"approver_identity":"reviewer","decision_id":"review-1","evidence":{"kind":"Human"}},"family_id":"family","id":"one","input":"a\u001fb","permission":{"approved_use_decision_id":"use-1"},"profile":"Abbey","sources":[{"document_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repository_revision":"revision","requirement_id":"other"},{"document_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repository_revision":"revision","requirement_id":"req"}],"target":"c\n\"é","teacher":null},"version":1}"#;
    assert_eq!(
        canonical_record_bytes(&r).unwrap(),
        expected_record.as_bytes()
    );
    assert_eq!(
        sha256_hex(expected_record.as_bytes()),
        "45d2e9aea25bd74a2983ed17a37dc2e295d9e42754dd6a6deee3fa7d46ee5d71"
    );
    let manifest = build_snapshot(&[r], &policy()).unwrap();
    let expected_preimage = r#"{"records":[{"digest":"45d2e9aea25bd74a2983ed17a37dc2e295d9e42754dd6a6deee3fa7d46ee5d71","family_id":"family","id":"one","profile":"Abbey","split":"Train"}],"specification_digests":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],"version":1}"#;
    let actual_preimage = json_bytes(&serde_json::json!({
        "version": manifest.version,
        "records": manifest.records,
        "specification_digests": manifest.specification_digests,
    }))
    .unwrap();
    assert_eq!(actual_preimage, expected_preimage.as_bytes());
    assert_eq!(
        manifest.root_digest,
        "9c0676bd239bbca7849582e37501e971f4b4c788580573577511bb5ecff3014c"
    );
    assert_eq!(
        sha256_hex(expected_preimage.as_bytes()),
        manifest.root_digest
    );
}
