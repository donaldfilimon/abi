//! Immutable proposal, compensation, identity-separation, and replay tests.

use abi_capability::{
    ApprovalLevel, ChangeApproval, ChangeSet, CompensationClass, Digest, Principal, PrincipalKind,
    Reversibility, RiskClass, Scope,
};

fn digest(byte: u8) -> Digest {
    Digest::from_bytes([byte; 32])
}

fn principal(id: &str, kind: PrincipalKind) -> Principal {
    Principal::new(id, kind).expect("synthetic identity is bounded")
}

fn scope() -> Scope {
    Scope::new(
        "org_ref",
        "deploy_ref",
        "tenant_ref",
        "discord_recording",
        Some("guild_ref"),
        digest(8),
        digest(9),
    )
    .expect("synthetic scope is bounded")
}

fn change_set(generator_digest: Digest) -> ChangeSet {
    ChangeSet::new(
        "operation_ref",
        principal("requester_ref", PrincipalKind::HumanSubject),
        principal("abbey_service", PrincipalKind::Service),
        scope(),
        "discord.channel.organize",
        "2.0.0",
        digest(1),
        CompensationClass::ExactRestore,
        RiskClass::Medium,
        ApprovalLevel::A3Admin,
        digest(2),
        digest(3),
        digest(4),
        digest(5),
        generator_digest,
        1_000,
        301_000,
        120_000,
    )
    .expect("synthetic change set is valid")
}

#[test]
fn compensation_classes_map_exactly_to_the_existing_risk_vocabulary() {
    assert_eq!(
        CompensationClass::ExactRestore.reversibility(),
        Reversibility::Reversible
    );
    assert_eq!(
        CompensationClass::BestEffort.reversibility(),
        Reversibility::ReversibleWithLoss
    );
    assert_eq!(
        CompensationClass::None.reversibility(),
        Reversibility::Irreversible
    );
}

#[test]
fn every_immutable_field_is_bound_and_replay_is_deterministic() {
    let first = change_set(digest(6));
    let replay = change_set(digest(6));
    assert_eq!(first, replay);
    assert_eq!(first.change_set_digest, first.computed_digest());

    let human_edited_or_regenerated = change_set(digest(7));
    assert_ne!(
        first.change_set_digest, human_edited_or_regenerated.change_set_digest,
        "a changed generator or edited proposal must receive a new digest"
    );

    let mut tampered = first.clone();
    tampered.snapshot_digest = digest(42);
    assert_ne!(tampered.change_set_digest, tampered.computed_digest());
}

#[test]
fn proposal_author_and_approver_are_distinct_and_expiring() {
    let change_set = change_set(digest(6));
    let approval = ChangeApproval::approve(
        "decision_ref",
        &change_set,
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        None,
        ApprovalLevel::A3Admin,
        120_000,
        2_000,
    )
    .expect("distinct authorized human can approve");
    assert_eq!(approval.change_set_digest, change_set.change_set_digest);

    for forbidden in [
        principal("requester_ref", PrincipalKind::HumanSubject),
        principal("abbey_service", PrincipalKind::Service),
        principal("member_ref", PrincipalKind::HumanSubject),
    ] {
        assert!(
            ChangeApproval::approve(
                "decision_ref",
                &change_set,
                forbidden,
                None,
                ApprovalLevel::A3Admin,
                120_000,
                2_000,
            )
            .is_err()
        );
    }

    assert!(
        ChangeApproval::approve(
            "decision_ref",
            &change_set,
            principal("admin_ref", PrincipalKind::GuildAdministrator),
            None,
            ApprovalLevel::A3Admin,
            301_000,
            301_000,
        )
        .is_err(),
        "an expired proposal cannot be approved"
    );
}

#[test]
fn exact_restore_requires_rollback_and_bounded_execution_windows() {
    let result = ChangeSet::new(
        "operation_ref",
        principal("requester_ref", PrincipalKind::HumanSubject),
        principal("abbey_service", PrincipalKind::Service),
        scope(),
        "discord.channel.organize",
        "2.0.0",
        digest(1),
        CompensationClass::ExactRestore,
        RiskClass::Medium,
        ApprovalLevel::A3Admin,
        digest(2),
        digest(3),
        Digest::default(),
        digest(5),
        digest(6),
        1_000,
        301_001,
        120_001,
    );
    assert!(result.is_err());
}
