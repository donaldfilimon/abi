//! Aggregate retained snapshot split isolation and legacy recovery regressions.
use super::*;

fn admit_record(reg: &mut TrainingRegistry, id: &str, family: &str) -> ApprovedRecord {
    let mut r = record();
    r.id = id.into();
    r.family_id = family.into();
    r.permission.as_mut().unwrap().approved_use_decision_id = format!("permission-{id}");
    r.approval.as_mut().unwrap().decision_id = format!("approval-{id}");
    evidence(reg, &r);
    reg.admit(r.clone()).unwrap();
    r
}
fn manifest(records: &[ApprovedRecord], split: Split) -> SnapshotManifest {
    build_snapshot(
        records,
        &SnapshotPolicy {
            family_splits: records
                .iter()
                .map(|r| (r.family_id.clone(), split))
                .collect(),
            permitted_specification_digests: records
                .iter()
                .flat_map(|r| r.sources.iter().map(|s| s.document_digest.clone()))
                .collect(),
        },
    )
    .unwrap()
}
fn refused_without_publication(reg: &mut TrainingRegistry, next: &SnapshotManifest) {
    let revision = reg.state.revision;
    let state = reg.store.get(STATE_KEY).unwrap().to_owned();
    assert!(
        matches!(
            reg.register_snapshot(next.clone()),
            Err(RegistryError::Invalid(_))
        ),
        "a valid individual manifest must still refuse a conflicting retained split"
    );
    assert_eq!(reg.state.revision, revision);
    assert_eq!(reg.store.get(STATE_KEY), Some(state.as_str()));
    assert!(!reg.state.snapshots.contains_key(&next.root_digest));
    assert!(matches!(
        reg.eligibility(&next.root_digest),
        Err(RegistryError::Missing(_))
    ));
}

#[test]
fn aggregate_split_isolation_same_record_cannot_move_between_separate_snapshots() {
    for (first, second) in [
        (Split::Evaluation, Split::Train),
        (Split::Train, Split::Evaluation),
    ] {
        let temp = Temp::new();
        let mut reg = temp.open();
        let r = admit_record(&mut reg, "same-record", "scenario-family");
        let original = manifest(std::slice::from_ref(&r), first);
        let collision = manifest(std::slice::from_ref(&r), second);
        assert_ne!(original.root_digest, collision.root_digest);
        reg.register_snapshot(original.clone()).unwrap();
        let receipt = reg.eligibility(&original.root_digest).unwrap();
        refused_without_publication(&mut reg, &collision);
        reg.validate_eligibility(&receipt).unwrap();
        assert_eq!(reg.snapshot(&original.root_digest).unwrap(), original);
    }
}

#[test]
fn aggregate_split_isolation_related_record_ids_survive_wal_and_checkpoint_reopen() {
    for checkpoint in [false, true] {
        let temp = Temp::new();
        let mut reg = temp.open();
        let held_out = admit_record(&mut reg, "held-out", "same-source-family");
        let related = admit_record(&mut reg, "related", "same-source-family");
        let evaluation = manifest(&[held_out], Split::Evaluation);
        let training = manifest(&[related], Split::Train);
        reg.register_snapshot(evaluation.clone()).unwrap();
        if checkpoint {
            reg.checkpoint().unwrap();
        }
        drop(reg);
        let mut reopened = temp.open();
        assert_eq!(
            reopened.snapshot(&evaluation.root_digest).unwrap(),
            evaluation
        );
        refused_without_publication(&mut reopened, &training);
    }
}

#[test]
fn aggregate_split_isolation_train_family_cannot_later_claim_related_held_out_record() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let trained = admit_record(&mut reg, "trained", "same-source-family");
    let related = admit_record(&mut reg, "related-held-out", "same-source-family");
    let training = manifest(&[trained], Split::Train);
    let evaluation = manifest(&[related], Split::Evaluation);
    reg.register_snapshot(training.clone()).unwrap();
    let receipt = reg.eligibility(&training.root_digest).unwrap();
    refused_without_publication(&mut reg, &evaluation);
    assert_eq!(reg.training_records(&receipt).unwrap().len(), 1);
}

#[test]
fn aggregate_split_isolation_revocation_does_not_release_held_out_family_after_reopen() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let held_out = admit_record(&mut reg, "revoked-held-out", "same-source-family");
    let related = admit_record(&mut reg, "new-related-id", "same-source-family");
    let evaluation = manifest(std::slice::from_ref(&held_out), Split::Evaluation);
    let training = manifest(&[related], Split::Train);
    reg.register_snapshot(evaluation.clone()).unwrap();
    reg.revoke(&held_out.id, revocation()).unwrap();
    reg.checkpoint().unwrap();
    drop(reg);
    let mut reopened = temp.open();
    assert!(matches!(
        reopened.eligibility(&evaluation.root_digest),
        Err(RegistryError::Revoked(_))
    ));
    refused_without_publication(&mut reopened, &training);
    assert_eq!(
        reopened.state.snapshots[&evaluation.root_digest],
        evaluation
    );
    assert!(reopened.state.revocations.contains_key(&held_out.id));
}

#[test]
fn aggregate_split_isolation_same_split_subsets_and_idempotence_remain_compatible() {
    for split in [Split::Train, Split::Evaluation] {
        let temp = Temp::new();
        let mut reg = temp.open();
        let one = admit_record(&mut reg, "one", "same-source-family");
        let two = admit_record(&mut reg, "two", "same-source-family");
        let first = manifest(std::slice::from_ref(&one), split);
        let second = manifest(std::slice::from_ref(&two), split);
        let combined = manifest(&[two.clone(), one], split);
        for m in [&first, &second, &combined] {
            reg.register_snapshot(m.clone()).unwrap();
            let current = reg.eligibility(&m.root_digest).unwrap();
            reg.register_snapshot(m.clone()).unwrap();
            reg.validate_eligibility(&current).unwrap();
        }
        reg.checkpoint().unwrap();
        drop(reg);
        let reopened = temp.open();
        let receipt = reopened.eligibility(&second.root_digest).unwrap();
        assert_eq!(reopened.snapshot(&combined.root_digest).unwrap(), combined);
        if split == Split::Train {
            assert_eq!(reopened.training_records(&receipt).unwrap(), vec![two]);
        } else {
            assert!(matches!(
                reopened.training_records(&receipt),
                Err(RegistryError::Invalid(_))
            ));
            assert_eq!(reopened.approved_records(&receipt).unwrap(), vec![two]);
        }
    }
}

/// The old `register_snapshot` accepted this exact version1 state. Publishing the
/// fixture directly preserves the actual WAL path while bypassing the repaired
/// admission guard. It is not a new production mutation API.
fn publish_legacy_collision(reg: &mut TrainingRegistry) -> SnapshotManifest {
    let held_out = admit_record(reg, "legacy-evaluation", "legacy-family");
    let related = admit_record(reg, "legacy-training", "legacy-family");
    let evaluation = manifest(&[held_out], Split::Evaluation);
    let training = manifest(&[related], Split::Train);
    reg.register_snapshot(evaluation).unwrap();
    let mut legacy = reg.state.clone();
    legacy
        .snapshots
        .insert(training.root_digest.clone(), training.clone());
    reg.publish(legacy).unwrap();
    training
}

#[test]
fn aggregate_split_isolation_legacy_conflicts_refuse_wal_and_checkpoint_reopen() {
    for checkpoint in [false, true] {
        let temp = Temp::new();
        let mut reg = temp.open();
        publish_legacy_collision(&mut reg);
        if checkpoint {
            reg.checkpoint().unwrap();
        }
        drop(reg);
        let reopened =
            TrainingRegistry::new(DurableStore::open_directory(&temp.0).unwrap(), policy());
        assert!(
            matches!(reopened, Err(RegistryError::Invalid(_))),
            "replay must validate the aggregate, not each manifest in isolation"
        );
    }
}

#[test]
fn aggregate_split_isolation_training_export_refuses_a_legacy_contaminated_registry() {
    let temp = Temp::new();
    let mut reg = temp.open();
    let training = publish_legacy_collision(&mut reg);
    // Eligibility may remain structural; train export must independently refuse
    // a family held out elsewhere in the retained registry.
    let receipt = Eligibility {
        snapshot_id: training.root_digest.clone(),
        registry_revision: reg.state.revision,
    };
    assert!(matches!(
        reg.training_records(&receipt),
        Err(RegistryError::Invalid(_))
    ));
}
