//! C2 for the local PR #814 change/approval slice, without an actuator or store.

use abi_capability::{
    ApprovalLevel, ApprovalState, ChangeApproval, ChangeSet, Digest, Principal, PrincipalKind,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::process::{Command, Stdio};

const RECORDING: &str = include_str!("fixtures/change_set_replay.json");

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Recording {
    contract_major: u64,
    contract_revision: u64,
    corpus_digest: String,
    kernel_version: String,
    proposal: ChangeSet,
    events: Vec<Event>,
    expected_accepted: Vec<bool>,
    transcript_digest: Digest,
}

#[derive(Deserialize)]
#[serde(tag = "event", rename_all = "snake_case", deny_unknown_fields)]
enum Event {
    Issue {
        decision_id: String,
        approved_by: Principal,
        coapproved_by: Option<Principal>,
        level: ApprovalLevel,
        expires_at_ms: u64,
        now_ms: u64,
    },
    Check {
        now_ms: u64,
    },
    Invalidate {
        state: ApprovalState,
        now_ms: u64,
    },
    Snapshot {
        digest: Digest,
        reseal: bool,
        now_ms: u64,
    },
}

#[derive(Serialize)]
struct ReplayResult {
    proposal_digest: Digest,
    approval: Option<ChangeApproval>,
}

fn recording() -> Recording {
    serde_json::from_str(RECORDING).expect("frozen synthetic recording decodes")
}

fn principal(id: &str, kind: PrincipalKind) -> Principal {
    Principal::new(id, kind).expect("bounded synthetic principal")
}

fn approve(proposal: &ChangeSet) -> ChangeApproval {
    ChangeApproval::approve(
        "decision_ref",
        proposal,
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        None,
        ApprovalLevel::A3Admin,
        120_000,
        1_000,
    )
    .expect("original proposal can be approved")
}

fn can_approve(proposal: &ChangeSet) -> bool {
    ChangeApproval::approve(
        "decision_ref",
        proposal,
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        None,
        ApprovalLevel::A3Admin,
        120_000,
        2_000,
    )
    .is_ok()
}

fn edited<T: Serialize + for<'de> Deserialize<'de>>(
    original: &T,
    pointer: &str,
    replacement: Value,
) -> T {
    let mut value = serde_json::to_value(original).expect("closed fields serialize");
    let field = value
        .pointer_mut(pointer)
        .expect("mutation targets an existing field");
    assert_ne!(*field, replacement, "{pointer} must actually change");
    *field = replacement;
    serde_json::from_value(value).expect("mutation remains structurally decodable")
}

// This also runs normally under the workspace gate. The parent test invokes it
// twice in separate OS processes, with no inherited stdin or shared replay state.
#[test]
fn replay_worker_emits_checked_transcript() {
    let recording = recording();
    let manifest: Value =
        serde_json::from_str(include_str!("../../../contracts/abbey/manifest.json"))
            .expect("manifest decodes");
    assert_eq!(manifest["contract_major"], recording.contract_major);
    assert_eq!(manifest["contract_revision"], recording.contract_revision);
    assert_eq!(manifest["aggregate_digest"], recording.corpus_digest);
    assert_eq!(recording.kernel_version, env!("CARGO_PKG_VERSION"));
    assert_eq!(
        recording.proposal.change_set_digest,
        recording.proposal.computed_digest()
    );

    let mut proposal = recording.proposal;
    let mut approval: Option<ChangeApproval> = None;
    let mut transcript = Vec::new();
    for event in recording.events {
        let now_ms = match event {
            Event::Issue {
                decision_id,
                approved_by,
                coapproved_by,
                level,
                expires_at_ms,
                now_ms,
            } => {
                approval = ChangeApproval::approve(
                    decision_id,
                    &proposal,
                    approved_by,
                    coapproved_by,
                    level,
                    expires_at_ms,
                    now_ms,
                )
                .ok();
                now_ms
            }
            Event::Check { now_ms } => now_ms,
            Event::Invalidate { state, now_ms } => {
                approval
                    .as_mut()
                    .expect("decision precedes invalidation")
                    .state = state;
                now_ms
            }
            Event::Snapshot {
                digest,
                reseal,
                now_ms,
            } => {
                proposal.snapshot_digest = digest;
                if reseal {
                    proposal.change_set_digest = proposal.computed_digest();
                }
                now_ms
            }
        };
        transcript.push(ReplayResult {
            proposal_digest: proposal.change_set_digest,
            approval: approval
                .as_ref()
                .filter(|decision| decision.validate_for(&proposal, now_ms).is_ok())
                .cloned(),
        });
    }
    assert_eq!(
        transcript
            .iter()
            .map(|result| result.approval.is_some())
            .collect::<Vec<_>>(),
        recording.expected_accepted,
        "frozen decisions must match, including every refusal"
    );
    let bytes = serde_json::to_vec(&transcript).expect("content-free transcript serializes");
    assert_eq!(
        Sha256::digest(&bytes).as_slice(),
        recording.transcript_digest.as_bytes(),
        "the independent golden commitment pins every emitted decision field"
    );
    println!(
        "\nC2_TRANSCRIPT={}",
        String::from_utf8(bytes).expect("JSON is UTF-8")
    );
}

#[test]
fn replay_after_process_restart_is_byte_identical() {
    let run = || {
        let output = Command::new(std::env::current_exe().expect("test executable exists"))
            .args([
                "--exact",
                "replay_worker_emits_checked_transcript",
                "--nocapture",
            ])
            .stdin(Stdio::null())
            .output()
            .expect("fresh replay process runs");
        assert!(
            output.status.success(),
            "replay failed: {}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .expect("test output is UTF-8")
            .lines()
            .find_map(|line| line.strip_prefix("C2_TRANSCRIPT="))
            .expect("child emits its checked transcript")
            .to_owned()
    };
    assert_eq!(run(), run());
}

#[test]
fn every_immutable_field_rejects_approval_when_tampered() {
    let proposal = recording().proposal;
    let approval = approve(&proposal);
    let changed_digest = json!(Digest::from_bytes([42; 32]));
    let mutations = [
        ("/operation_id", json!("operation_other")),
        ("/change_set_digest", changed_digest.clone()),
        ("/requested_by/id", json!("requester_other")),
        ("/requested_by/kind", json!("GuildManager")),
        ("/proposed_by/id", json!("service_other")),
        ("/proposed_by/kind", json!("HumanSubject")),
        ("/scope/organization_id", json!("org_other")),
        ("/scope/deployment_id", json!("deploy_other")),
        ("/scope/tenant_id", json!("tenant_other")),
        ("/scope/platform", json!("local")),
        ("/scope/guild_ref", json!("guild_other")),
        ("/scope/resource_digest", changed_digest.clone()),
        ("/scope/subject_digest", changed_digest.clone()),
        ("/capability_id", json!("discord.channel.inspect")),
        ("/capability_version", json!("2.0.1")),
        ("/package_digest", changed_digest.clone()),
        ("/compensation_class", json!("BestEffort")),
        ("/risk", json!("High")),
        ("/required_approval", json!("A4Owner")),
        ("/precondition_digest", changed_digest.clone()),
        ("/expected_postcondition_digest", changed_digest.clone()),
        ("/rollback_digest", changed_digest.clone()),
        ("/snapshot_digest", changed_digest.clone()),
        ("/generator_digest", changed_digest),
        ("/created_at_ms", json!(999)),
        ("/expires_at_ms", json!(300_999)),
        ("/prepared_ttl_ms", json!(119_999)),
    ];
    let mut covered = BTreeSet::new();
    for (pointer, replacement) in mutations {
        let tampered = edited(&proposal, pointer, replacement);
        assert!(
            !can_approve(&tampered),
            "new approval accepted {pointer} tampering"
        );
        assert!(
            approval.validate_for(&tampered, 2_000).is_err(),
            "reused {pointer}"
        );
        covered.insert(pointer.split('/').nth(1).expect("field path").to_owned());
    }
    let fields = serde_json::to_value(&proposal).expect("proposal serializes");
    assert_eq!(
        covered,
        fields.as_object().unwrap().keys().cloned().collect()
    );
}

#[test]
fn regenerated_proposal_invalidates_old_approval_and_requires_a_new_decision() {
    let original = recording().proposal;
    let old_approval = approve(&original);
    let mut regenerated = original.clone();
    regenerated.generator_digest = Digest::from_bytes([7; 32]);
    regenerated.change_set_digest = regenerated.computed_digest();
    assert_ne!(original.change_set_digest, regenerated.change_set_digest);
    assert!(old_approval.validate_for(&regenerated, 2_000).is_err());
    let new_approval = ChangeApproval::approve(
        "decision_regenerated",
        &regenerated,
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        None,
        ApprovalLevel::A3Admin,
        120_000,
        2_000,
    )
    .expect("explicit new approval can bind the regenerated proposal");
    assert!(new_approval.validate_for(&regenerated, 2_000).is_ok());
    assert!(new_approval.validate_for(&original, 2_000).is_err());
}

#[test]
fn resealing_does_not_bypass_proposal_construction_rules() {
    let original = recording().proposal;
    for (pointer, replacement) in [
        ("/operation_id", json!("")),
        ("/proposed_by/kind", json!("HumanSubject")),
        ("/proposed_by/id", json!("requester_ref")),
        ("/capability_id", json!("invalid")),
        ("/capability_version", json!("2.0")),
        ("/expires_at_ms", json!(1_000)),
        ("/expires_at_ms", json!(301_001)),
        ("/prepared_ttl_ms", json!(0)),
        ("/prepared_ttl_ms", json!(120_001)),
        ("/rollback_digest", json!(Digest::default())),
    ] {
        let mut tampered = edited(&original, pointer, replacement);
        tampered.change_set_digest = tampered.computed_digest();
        assert!(!can_approve(&tampered), "resealing bypassed {pointer}");
    }
}

#[test]
fn proposal_and_approval_expiry_boundaries_are_exclusive() {
    let proposal = recording().proposal;
    for (now_ms, expires_at_ms, accepted) in [
        (999, 120_000, false),
        (1_000, 120_000, true),
        (2_000, 2_000, false),
        (2_000, 2_001, true),
        (2_000, 301_000, true),
        (2_000, 301_001, false),
        (300_999, 301_000, true),
        (301_000, 301_000, false),
        (u64::MAX, u64::MAX, false),
    ] {
        assert_eq!(
            ChangeApproval::approve(
                "decision_ref",
                &proposal,
                principal("admin_ref", PrincipalKind::GuildAdministrator),
                None,
                ApprovalLevel::A3Admin,
                expires_at_ms,
                now_ms,
            )
            .is_ok(),
            accepted,
            "now={now_ms}, expires={expires_at_ms}"
        );
    }
    let approval = approve(&proposal);
    for (now_ms, accepted) in [
        (999, false),
        (1_000, true),
        (119_999, true),
        (120_000, false),
        (301_000, false),
        (u64::MAX, false),
    ] {
        assert_eq!(
            approval.validate_for(&proposal, now_ms).is_ok(),
            accepted,
            "{now_ms}"
        );
    }
}

#[test]
fn invalidated_decisions_stay_invalid_after_deserialization() {
    let proposal = recording().proposal;
    let original = approve(&proposal);
    for state in [
        ApprovalState::Denied,
        ApprovalState::Cancelled,
        ApprovalState::Expired,
        ApprovalState::Consumed,
    ] {
        let invalidated = edited(&original, "/state", json!(state));
        let before = serde_json::to_vec(&invalidated).unwrap();
        let reloaded: ChangeApproval = serde_json::from_slice(&before).unwrap();
        assert!(
            reloaded.validate_for(&proposal, 2_000).is_err(),
            "{state:?}"
        );
        assert_eq!(
            serde_json::to_vec(&reloaded).unwrap(),
            before,
            "validation cannot revive a decision"
        );
    }
}

#[test]
fn reloaded_approval_rechecks_authority_binding_and_expiry() {
    let proposal = recording().proposal;
    let original = approve(&proposal);
    for (pointer, replacement) in [
        ("/decision_id", json!("")),
        ("/change_set_digest", json!(Digest::from_bytes([42; 32]))),
        ("/approved_by/id", json!("requester_ref")),
        ("/approved_by/id", json!("abbey_service")),
        ("/approved_by/kind", json!("Service")),
        ("/approved_by/kind", json!("GuildManager")),
        ("/level", json!("A2Manager")),
        ("/level", json!("A5DualControl")),
        (
            "/coapproved_by",
            json!(principal("admin_two", PrincipalKind::GuildAdministrator)),
        ),
        ("/expires_at_ms", json!(2_000)),
        ("/expires_at_ms", json!(301_001)),
    ] {
        let tampered = edited(&original, pointer, replacement);
        assert!(
            tampered.validate_for(&proposal, 2_000).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn dual_control_replay_requires_two_distinct_authorized_humans() {
    let mut proposal = recording().proposal;
    proposal.required_approval = ApprovalLevel::A5DualControl;
    proposal.change_set_digest = proposal.computed_digest();
    let decision = ChangeApproval::approve(
        "decision_dual",
        &proposal,
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        Some(principal("owner_ref", PrincipalKind::GuildOwner)),
        ApprovalLevel::A5DualControl,
        120_000,
        2_000,
    )
    .expect("two distinct authorized humans satisfy dual control");
    let bytes = serde_json::to_vec(&decision).unwrap();
    let reloaded: ChangeApproval = serde_json::from_slice(&bytes).unwrap();
    assert!(reloaded.validate_for(&proposal, 2_000).is_ok());
    for forbidden in [
        principal("requester_ref", PrincipalKind::GuildOwner),
        principal("abbey_service", PrincipalKind::GuildOwner),
        principal("admin_ref", PrincipalKind::GuildAdministrator),
        principal("service_ref", PrincipalKind::Service),
        principal("manager_ref", PrincipalKind::GuildManager),
    ] {
        let invalidated = edited(&reloaded, "/coapproved_by", json!(forbidden));
        assert!(invalidated.validate_for(&proposal, 2_000).is_err());
    }
}
