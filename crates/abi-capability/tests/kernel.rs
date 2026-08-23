//! Core deny-by-default and approval-required behavior.

use abi_agent_runtime::CancellationToken;
use abi_capability::{
    ApprovalLevel, AuthorizationContext, BoundedMemoryAuditSink, CapabilityPackage,
    CapabilityRegistry, Decision, Digest, FixedClock, Grant, Kernel, PlatformFacts, Principal,
    PrincipalKind, RecordingActuator, Request, Reversibility, RiskClass, Scope, SideEffect,
};
use std::collections::BTreeSet;

fn digest(byte: u8) -> Digest {
    Digest::from_bytes([byte; 32])
}

fn principal(id: &str, kind: PrincipalKind) -> Principal {
    Principal::new(id, kind).expect("synthetic principal is bounded")
}

fn scope(tenant: &str) -> Scope {
    Scope::new(
        "org_ref",
        "deploy_ref",
        tenant,
        "discord_recording",
        Some("guild_ref"),
        digest(8),
        digest(9),
    )
    .expect("synthetic scope is bounded")
}

fn package() -> CapabilityPackage {
    CapabilityPackage::new(
        "discord.channel.organize",
        "2.0.0",
        digest(3),
        SideEffect::PlatformWrite,
        Reversibility::Reversible,
        RiskClass::Medium,
        ApprovalLevel::A2Manager,
        ["manage_channels"],
        ["channel_state_matches"],
    )
    .expect("synthetic package compiles")
}

fn request() -> Request {
    Request::new(
        "attempt_v2",
        "call_v2",
        principal("member_ref", PrincipalKind::HumanSubject),
        scope("tenant_ref"),
        "discord.channel.organize",
        "2.0.0",
        digest(3),
        digest(5),
        "tenant_v2",
        "guild_v2",
        "safety_v2",
        4,
    )
    .expect("synthetic request is bounded")
}

#[test]
fn no_grant_denies_and_audits_without_recording_an_effect() {
    let registry = CapabilityRegistry::compile([package()]).expect("package registry compiles");
    let mut audit = BoundedMemoryAuditSink::new(8, 4096);
    let mut actuator = RecordingActuator::new(BTreeSet::from(["channel_state_matches"]));
    let context = AuthorizationContext {
        now_ms: &FixedClock::new(1_000),
        cancellation: &CancellationToken::new(),
        grants: &[],
        approval: None,
        used_decision_ids: &BTreeSet::new(),
        platform: &PlatformFacts::new(
            BTreeSet::from(["manage_channels"]),
            4,
            "tenant_v2",
            "guild_v2",
            "safety_v2",
        ),
    };

    let outcome = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
        .expect("a refusal is a successful closed decision");

    assert_eq!(outcome.receipt.decision, Decision::Deny);
    assert_eq!(outcome.receipt.reason.as_str(), "no_matching_grant");
    assert!(outcome.receipt.redacted);
    assert_eq!(audit.records().len(), 1);
    assert_eq!(actuator.records().len(), 0);
}

#[test]
fn exact_grant_without_required_manager_approval_is_not_treated_as_denial() {
    let package = package();
    let registry = CapabilityRegistry::compile([package.clone()]).expect("registry compiles");
    let grant = Grant::active(
        "grant_v2",
        principal("owner_ref", PrincipalKind::GuildOwner),
        request().principal.clone(),
        request().scope.clone(),
        &package,
        0,
        2_000,
        ApprovalLevel::A2Manager,
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    )
    .expect("grant is bounded");
    let mut audit = BoundedMemoryAuditSink::new(8, 4096);
    let mut actuator = RecordingActuator::new(BTreeSet::from(["channel_state_matches"]));
    let grants = [grant];
    let used = BTreeSet::new();
    let clock = FixedClock::new(1_000);
    let cancellation = CancellationToken::new();
    let platform = PlatformFacts::new(
        BTreeSet::from(["manage_channels"]),
        4,
        "tenant_v2",
        "guild_v2",
        "safety_v2",
    );
    let context = AuthorizationContext {
        now_ms: &clock,
        cancellation: &cancellation,
        grants: &grants,
        approval: None,
        used_decision_ids: &used,
        platform: &platform,
    };

    let outcome = Kernel::new(&registry)
        .authorize_and_record(&request(), &context, &mut audit, &mut actuator)
        .expect("approval request is a closed result");

    assert_eq!(outcome.receipt.decision, Decision::ApprovalRequired);
    assert_eq!(outcome.receipt.reason.as_str(), "approval_missing");
    assert_eq!(actuator.records().len(), 0);
}
