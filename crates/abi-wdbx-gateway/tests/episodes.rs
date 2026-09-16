//! Scratch-only tests for the canonical episode gate (WDBX v3 over gRPC).
#![cfg(unix)]

use abi_wdbx_gateway::proto::wdbx_gateway_server::WdbxGateway;
use abi_wdbx_gateway::proto::{
    GetKvRequest, KvEntry, ProposeEpisodeWriteRequest, PutKvRequest, StatsRequest,
    VerifyEpisodeRequest, WatchMutationsRequest,
};
use abi_wdbx_gateway::{
    BearerToken, EventHub, GatewayConfig, GatewayService, Limits, StoreExecutor,
};
use futures_util::StreamExt as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::PathBuf;
use std::sync::Arc;
use tonic::{Code, Request};
use uuid::Uuid;

const TOKEN: &str = "gateway-test-token";

struct Scratch {
    root: PathBuf,
    token: PathBuf,
}

impl Scratch {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!("abi-gateway-{label}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let token = root.join("token");
        std::fs::write(&token, format!("{TOKEN}\n")).unwrap();
        std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
        Self { root, token }
    }

    fn store(&self) -> PathBuf {
        self.root.join("store")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn authenticated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    request
}

fn service(scratch: &Scratch, limits: Limits) -> (GatewayService, Arc<EventHub>) {
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    let executor = StoreExecutor::open(&scratch.store(), limits.blocking_jobs).unwrap();
    let events = Arc::new(EventHub::new(&limits));
    (
        GatewayService::new(token, limits, executor, Arc::clone(&events)),
        events,
    )
}

fn episode_policy(learning_enabled: bool) -> abi_wdbx::v3::episode::StorePolicy {
    use abi_wdbx::v3::episode::{GuildEpisodePolicy, StorePolicy};

    StorePolicy {
        contract_revision: 2,
        contract_digest: [9; 32],
        guilds: std::collections::BTreeMap::from([(
            "guild_ref".to_owned(),
            GuildEpisodePolicy {
                learning_enabled,
                policy_version: "policy_v1".into(),
                token_budget: 1_000,
                storage_budget_bytes: 1 << 20,
                current_consent_epoch: Some(7),
            },
        )]),
    }
}

fn episode_proposal_json(request_id: &str, operation_id: &str) -> Vec<u8> {
    use abi_wdbx::v3::episode::{
        ActorKind, ActorRef, EpisodeEvent, EpisodeSource, EpisodeWrite, EvidenceLevel,
    };

    let write = EpisodeWrite {
        request_id: request_id.into(),
        operation_id: operation_id.into(),
        contract_revision: 2,
        contract_digest: [9; 32],
        guild_ref: "guild_ref".into(),
        consent_epoch: None,
        source_type: EpisodeSource::Proposal,
        policy_version: "policy_v1".into(),
        evidence_level: EvidenceLevel::C1,
        event: EpisodeEvent::Proposal {
            requested_by: ActorRef {
                principal_id: "requester_ref".into(),
                kind: ActorKind::HumanSubject,
            },
            proposed_by: ActorRef {
                principal_id: "abbey_service".into(),
                kind: ActorKind::Service,
            },
        },
        token_cost: 3,
        expected_commitment: None,
        quiet: false,
    };
    serde_json::to_vec(&write).unwrap()
}

fn episode_memory_candidate_json(request_id: &str, operation_id: &str) -> Vec<u8> {
    use abi_wdbx::v3::episode::{
        ActorKind, ActorRef, EpisodeEvent, EpisodeSource, EpisodeWrite, EvidenceLevel,
        MemoryCandidate, MemoryClass, RetentionClass,
    };

    let write = EpisodeWrite {
        request_id: request_id.into(),
        operation_id: operation_id.into(),
        contract_revision: 2,
        contract_digest: [9; 32],
        guild_ref: "guild_ref".into(),
        consent_epoch: None,
        source_type: EpisodeSource::DiscordGuild,
        policy_version: "policy_v1".into(),
        evidence_level: EvidenceLevel::C1,
        event: EpisodeEvent::MemoryCandidate {
            recorded_by: ActorRef {
                principal_id: "abbey_service".into(),
                kind: ActorKind::Service,
            },
            candidate: MemoryCandidate {
                class: MemoryClass::Embedding,
                retention: RetentionClass::Durable,
                payload_commitment: [4; 32],
                payload_bytes: 1_536,
                dimension: Some(384),
                embedding_version: Some("abbey-embedding-v1".into()),
                member_scoped: true,
                supersedes: None,
                forgets: None,
            },
        },
        token_cost: 1,
        expected_commitment: None,
        quiet: false,
    };
    serde_json::to_vec(&write).unwrap()
}

fn episode_approval_json(request_id: &str, operation_id: &str) -> Vec<u8> {
    use abi_wdbx::v3::episode::{
        ActorKind, ActorRef, EpisodeEvent, EpisodeSource, EpisodeWrite, EvidenceLevel,
    };

    let write = EpisodeWrite {
        request_id: request_id.into(),
        operation_id: operation_id.into(),
        contract_revision: 2,
        contract_digest: [9; 32],
        guild_ref: "guild_ref".into(),
        consent_epoch: None,
        source_type: EpisodeSource::DiscordGuild,
        policy_version: "policy_v1".into(),
        evidence_level: EvidenceLevel::C1,
        event: EpisodeEvent::Approval {
            approved_by: ActorRef {
                principal_id: "admin_ref".into(),
                kind: ActorKind::GuildAdministrator,
            },
        },
        token_cost: 1,
        expected_commitment: None,
        quiet: false,
    };
    serde_json::to_vec(&write).unwrap()
}

#[tokio::test]
async fn memory_candidate_receipts_pass_through_as_completed_single_event_operations() {
    let scratch = Scratch::new("memory");
    let limits = Limits::default();
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    let executor = StoreExecutor::open_with_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
    )
    .unwrap();
    let events = Arc::new(EventHub::new(&limits));
    let service = GatewayService::new(token, limits, executor, Arc::clone(&events));

    let appended = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_memory_candidate_json("req_m1", "mem_1"),
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(appended.decision, "appended");
    let receipt = appended.receipt.clone().unwrap();
    assert_eq!(receipt.event_kind, "memory_candidate");
    assert_eq!(receipt.terminal_status, "completed");
    assert_eq!(receipt.previous_digest, Vec::<u8>::new());
    assert!(receipt.redacted);
    let serialized = format!("{receipt:?}");
    assert!(!serialized.contains("payload"));
    assert!(!serialized.contains("abbey-embedding"));

    // The operation is closed in the same append: a second candidate on it
    // is a replay (as is any operation-opening event on a taken identifier),
    // and any lifecycle event on it is an invalid transition.
    let replay = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_memory_candidate_json("req_m2", "mem_1"),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(replay.code(), Code::AlreadyExists);
    let follow = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_approval_json("req_m3", "mem_1"),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(follow.code(), Code::FailedPrecondition);
    assert!(follow.message().contains("episode_transition_invalid"));

    let verified = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: appended.episode_digest.clone(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(verified.found);
    assert_eq!(verified.receipt.unwrap().event_kind, "memory_candidate");
}

#[tokio::test]
async fn episode_gate_previews_appends_rejects_replays_and_verifies() {
    let scratch = Scratch::new("episodes");
    let limits = Limits::default();
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    let executor = StoreExecutor::open_with_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
    )
    .unwrap();
    let events = Arc::new(EventHub::new(&limits));
    let service = GatewayService::new(token, limits, executor, Arc::clone(&events));
    let mut watch = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
        .unwrap()
        .into_inner();

    let preview = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: true,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(preview.decision, "preview");
    assert_eq!(preview.episode_digest.len(), 32);
    assert!(preview.receipt.is_none());

    let appended = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(appended.decision, "appended");
    assert_eq!(appended.episode_digest, preview.episode_digest);
    let receipt = appended.receipt.clone().unwrap();
    assert_eq!(receipt.guild_ref, "guild_ref");
    assert_eq!(receipt.event_kind, "proposal");
    assert_eq!(receipt.evidence_level, "C1");
    assert_eq!(receipt.previous_digest, Vec::<u8>::new());
    assert_eq!(receipt.terminal_status, "");
    assert!(receipt.redacted);
    let notice = watch.next().await.unwrap().unwrap();
    assert_eq!(notice.kind, "propose_episode_write");
    assert_eq!(notice.transaction_id.len(), 64);

    let replay = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(replay.code(), Code::AlreadyExists);
    assert_eq!(replay.message(), "episode_replay");

    let verified = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: appended.episode_digest.clone(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(verified.found);
    assert_eq!(verified.receipt.unwrap().sequence, receipt.sequence);

    let unknown = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: vec![0; 32],
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(!unknown.found);
    assert!(unknown.receipt.is_none());

    let short_digest = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: vec![0; 31],
        }))
        .await
        .unwrap_err();
    assert_eq!(short_digest.code(), Code::InvalidArgument);
}

#[tokio::test]
async fn verify_episode_finds_receipt_beyond_former_2048_window() {
    let scratch = Scratch::new("episodes-whole-ledger-verify");
    let limits = Limits {
        requests_per_second: 10_000,
        ..Default::default()
    };
    let mut policy = episode_policy(true);
    let guild_policy = policy.guilds.get_mut("guild_ref").unwrap();
    guild_policy.token_budget = 1_000_000;
    guild_policy.storage_budget_bytes = 32 * 1024 * 1024;
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    let executor =
        StoreExecutor::open_with_episodes(&scratch.store(), limits.blocking_jobs, Some(policy))
            .unwrap();
    let events = Arc::new(EventHub::new(&limits));
    let service = GatewayService::new(token, limits, executor, events);

    let count = 2_049_usize;
    let mut target = None;
    for index in 0..count {
        target = Some(
            service
                .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
                    episode_write_json: episode_proposal_json(
                        &format!("window_req_{index}"),
                        &format!("window_op_{index}"),
                    ),
                    preview_only: false,
                }))
                .await
                .unwrap()
                .into_inner(),
        );
    }
    let target = target.unwrap();
    let target_receipt = target.receipt.unwrap();
    assert_eq!(target_receipt.sequence, u64::try_from(count).unwrap());

    let verified = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: target.episode_digest,
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(verified.found);
    let verified_receipt = verified.receipt.unwrap();
    assert_eq!(verified_receipt.sequence, u64::try_from(count).unwrap());
    assert_eq!(verified_receipt.request_id, "window_req_2048");
}

#[tokio::test]
async fn episode_gate_rejects_unconfigured_disabled_unauthenticated_and_malformed_writes() {
    let scratch = Scratch::new("episodes-unconfigured");
    let (service, _) = service(&scratch, Limits::default());
    let propose = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(propose.code(), Code::FailedPrecondition);
    let verify = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: vec![0; 32],
        }))
        .await
        .unwrap_err();
    assert_eq!(verify.code(), Code::FailedPrecondition);

    let disabled = Scratch::new("episodes-disabled");
    let limits = Limits::default();
    let token = Arc::new(BearerToken::load(&disabled.token).unwrap());
    let executor = StoreExecutor::open_with_episodes(
        &disabled.store(),
        limits.blocking_jobs,
        Some(episode_policy(false)),
    )
    .unwrap();
    let events = Arc::new(EventHub::new(&limits));
    let service = GatewayService::new(token, limits, executor, events);
    let denied = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(denied.code(), Code::FailedPrecondition);
    assert_eq!(denied.message(), "episode_learning_disabled");

    let unauthenticated = service
        .propose_episode_write(Request::new(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: true,
        }))
        .await
        .unwrap_err();
    assert_eq!(unauthenticated.code(), Code::Unauthenticated);
    let malformed = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: br#"{"request_id":"x","unknown":1}"#.to_vec(),
            preview_only: true,
        }))
        .await
        .unwrap_err();
    assert_eq!(malformed.code(), Code::InvalidArgument);

    let mut config = GatewayConfig::loopback(disabled.store(), disabled.token.clone());
    config.episode_policy = Some(disabled.root.join("missing-policy.json"));
    assert!(config.validate_pre_bind().is_err());
    let policy_file = disabled.root.join("policy.json");
    std::fs::write(
        &policy_file,
        serde_json::to_vec(&episode_policy(true)).unwrap(),
    )
    .unwrap();
    config.episode_policy = Some(policy_file);
    config.validate_pre_bind().unwrap();
}

#[tokio::test]
async fn episode_gate_survives_reopen_beside_the_v2_store() {
    let scratch = Scratch::new("episodes-reopen");
    let limits = Limits::default();
    let open = || {
        let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
        let executor = StoreExecutor::open_with_episodes(
            &scratch.store(),
            limits.blocking_jobs,
            Some(episode_policy(true)),
        )
        .unwrap();
        let events = Arc::new(EventHub::new(&limits));
        GatewayService::new(token, limits.clone(), executor, events)
    };

    let service = open();
    let appended = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    service
        .put_kv(authenticated(PutKvRequest {
            entries: vec![KvEntry {
                key: "beside-episodes".into(),
                value: "v2-still-works".into(),
            }],
        }))
        .await
        .unwrap();
    let stats = service
        .stats(authenticated(StatsRequest {}))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(stats.kv_entries, 1);
    drop(service);

    let reopened = open();
    let verified = reopened
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: appended.episode_digest.clone(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(verified.found);
    assert_eq!(verified.receipt.unwrap().request_id, "req_1");
    let replay_after_reopen = reopened
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_1", "op_1"),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(replay_after_reopen.code(), Code::AlreadyExists);
    let kv = reopened
        .get_kv(authenticated(GetKvRequest {
            key: "beside-episodes".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(kv.value, "v2-still-works");
    let unknown_guild = reopened
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "no_such_guild".into(),
            episode_digest: appended.episode_digest,
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(!unknown_guild.found);
    assert!(unknown_guild.receipt.is_none());
}

/// Write a 32-byte Ed25519 secret to `name` under the scratch root, owner-only.
fn write_signing_key(scratch: &Scratch, name: &str, fill: u8) -> PathBuf {
    let path = scratch.root.join(name);
    std::fs::write(&path, [fill; 32]).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path
}

/// Append one proposal through a gateway service and return its digest.
async fn append_one(executor: StoreExecutor, limits: &Limits, scratch: &Scratch) -> Vec<u8> {
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    let service = GatewayService::new(
        token,
        limits.clone(),
        executor,
        Arc::new(EventHub::new(limits)),
    );
    let appended = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json("req_s1", "op_s1"),
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(appended.decision, "appended");
    appended.episode_digest
    // `service` drops here, releasing the episode ledger's writer lock.
}

/// Reopen the ledger read-side and report one record's signature state.
fn stored_signature(
    scratch: &Scratch,
    digest: &[u8],
    signer: &abi_wdbx::v3::episode::EpisodeSigner,
) -> abi_wdbx::v3::episode::SignatureStatus {
    let digest: [u8; 32] = digest.try_into().unwrap();
    let store = abi_wdbx::v3::episode::EpisodeStore::open(
        scratch.store().join("episodes"),
        episode_policy(true),
    )
    .unwrap();
    let known = signer.key_id().clone();
    let verifying = signer.verifying_key();
    store
        .signature_status("guild_ref", &digest, |id| {
            (*id == known).then_some(verifying)
        })
        .unwrap()
        .expect("the appended digest is in the ledger")
}

#[tokio::test]
async fn configured_signing_key_signs_every_appended_episode() {
    use abi_wdbx::v3::episode::{EpisodeSigner, SignatureStatus};

    let scratch = Scratch::new("episodes-signed");
    let limits = Limits::default();
    let key = write_signing_key(&scratch, "episode-signing.key", 7);
    let signer = EpisodeSigner::from_key_file(&key).unwrap();

    let executor = StoreExecutor::open_with_signed_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
        Some(EpisodeSigner::from_key_file(&key).unwrap()),
    )
    .unwrap();
    let digest = append_one(executor, &limits, &scratch).await;

    assert_eq!(
        stored_signature(&scratch, &digest, &signer),
        SignatureStatus::Valid(signer.key_id().clone())
    );
}

#[tokio::test]
async fn unconfigured_signing_key_still_appends_unsigned_episodes() {
    use abi_wdbx::v3::episode::{EpisodeSigner, SignatureStatus};

    let scratch = Scratch::new("episodes-unsigned");
    let limits = Limits::default();
    let key = write_signing_key(&scratch, "unused.key", 7);
    let signer = EpisodeSigner::from_key_file(&key).unwrap();

    let executor = StoreExecutor::open_with_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
    )
    .unwrap();
    let digest = append_one(executor, &limits, &scratch).await;

    assert_eq!(
        stored_signature(&scratch, &digest, &signer),
        SignatureStatus::Unsigned
    );
}

#[test]
fn episode_signing_key_refuses_a_missing_policy_and_the_membership_key() {
    use abi_wdbx::v3::episode::EpisodeSigner;
    use abi_wdbx_gateway::GatewayError;

    let scratch = Scratch::new("episodes-signing-refusals");
    let limits = Limits::default();
    let key = write_signing_key(&scratch, "episode-signing.key", 7);

    // Nothing to sign without a policy.
    let without_policy = StoreExecutor::open_with_signed_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        None,
        Some(EpisodeSigner::from_key_file(&key).unwrap()),
    );
    assert!(matches!(
        without_policy,
        Err(GatewayError::Configuration(_))
    ));

    // The first open creates the membership keypair; a copy of its secret
    // must be refused as an episode key even though the path differs.
    drop(StoreExecutor::open(&scratch.store(), limits.blocking_jobs).unwrap());
    let membership_secret = scratch
        .store()
        .join("gateway-membership")
        .join("signing.key");
    let copied = scratch.root.join("copied-membership.key");
    std::fs::copy(&membership_secret, &copied).unwrap();
    std::fs::set_permissions(&copied, std::fs::Permissions::from_mode(0o600)).unwrap();
    let reused = StoreExecutor::open_with_signed_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
        Some(EpisodeSigner::from_key_file(&copied).unwrap()),
    );
    assert!(matches!(reused, Err(GatewayError::Configuration(_))));

    // A distinct key on the same store is accepted.
    StoreExecutor::open_with_signed_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
        Some(EpisodeSigner::from_key_file(&key).unwrap()),
    )
    .unwrap();
}

#[test]
fn episode_signing_key_is_validated_before_bind() {
    let scratch = Scratch::new("episodes-signing-config");
    let policy_file = scratch.root.join("policy.json");
    std::fs::write(
        &policy_file,
        serde_json::to_vec(&episode_policy(true)).unwrap(),
    )
    .unwrap();
    let key = write_signing_key(&scratch, "episode-signing.key", 7);

    let mut config = GatewayConfig::loopback(scratch.store(), scratch.token.clone());
    config.episode_signing_key = Some(key.clone());
    assert!(
        config.validate_pre_bind().is_err(),
        "a signing key without a policy must be refused"
    );

    config.episode_policy = Some(policy_file);
    config.validate_pre_bind().unwrap();

    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        config.validate_pre_bind().is_err(),
        "a group- or world-readable signing key must be refused"
    );

    config.episode_signing_key = Some(scratch.root.join("missing.key"));
    assert!(config.validate_pre_bind().is_err());
}

/// Build a service over an executor, with the standard test token.
fn service_over(executor: StoreExecutor, limits: &Limits, scratch: &Scratch) -> GatewayService {
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    GatewayService::new(
        token,
        limits.clone(),
        executor,
        Arc::new(EventHub::new(limits)),
    )
}

async fn verify(
    service: &GatewayService,
    digest: &[u8],
) -> abi_wdbx_gateway::proto::VerifyEpisodeResponse {
    service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "guild_ref".into(),
            episode_digest: digest.to_vec(),
        }))
        .await
        .unwrap()
        .into_inner()
}

#[tokio::test]
async fn verify_episode_reports_signature_state_against_the_gateway_key() {
    use abi_wdbx::v3::episode::EpisodeSigner;

    let scratch = Scratch::new("episodes-verify-signature");
    let limits = Limits::default();
    let key_a = write_signing_key(&scratch, "episode-a.key", 7);
    let key_b = write_signing_key(&scratch, "episode-b.key", 8);
    let id_a = EpisodeSigner::from_key_file(&key_a)
        .unwrap()
        .key_id()
        .as_str()
        .to_owned();
    let open = |key: Option<&PathBuf>| {
        StoreExecutor::open_with_signed_episodes(
            &scratch.store(),
            limits.blocking_jobs,
            Some(episode_policy(true)),
            key.map(|path| EpisodeSigner::from_key_file(path).unwrap()),
        )
        .unwrap()
    };

    // Appended under key A, verified by the gateway holding key A.
    let service = service_over(open(Some(&key_a)), &limits, &scratch);
    let digest = append_via(&service, "req_v1", "op_v1").await;
    let signed = verify(&service, &digest).await;
    assert!(signed.found);
    assert_eq!(signed.signature_status, "valid");
    assert_eq!(signed.signer_key_id, id_a);

    // A digest that was never appended carries no signature fields.
    let missing = verify(&service, &[0x55; 32]).await;
    assert!(!missing.found);
    assert_eq!(missing.signature_status, "");
    assert_eq!(missing.signer_key_id, "");
    drop(service);

    // The same record, seen by a gateway that now holds key B, is
    // unverifiable under B: reported as unknown, never as invalid.
    let service = service_over(open(Some(&key_b)), &limits, &scratch);
    let rotated = verify(&service, &digest).await;
    assert_eq!(rotated.signature_status, "unknown_key");
    assert_eq!(rotated.signer_key_id, id_a);
    drop(service);

    // A gateway with no key cannot resolve any key id either.
    let service = service_over(open(None), &limits, &scratch);
    let keyless = verify(&service, &digest).await;
    assert_eq!(keyless.signature_status, "unknown_key");
    assert_eq!(keyless.signer_key_id, id_a);

    // And a record it appends itself is unsigned, with no key id.
    let unsigned_digest = append_via(&service, "req_v2", "op_v2").await;
    let unsigned = verify(&service, &unsigned_digest).await;
    assert_eq!(unsigned.signature_status, "unsigned");
    assert_eq!(unsigned.signer_key_id, "");
}

async fn append_via(service: &GatewayService, request_id: &str, operation_id: &str) -> Vec<u8> {
    let appended = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: episode_proposal_json(request_id, operation_id),
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(appended.decision, "appended");
    appended.episode_digest
}

fn memory_edge_json(
    request_id: &str,
    governance: bool,
    edge: abi_wdbx::v3::episode::MemoryEdge,
) -> Vec<u8> {
    use abi_wdbx::v3::episode::{
        ActorKind, ActorRef, EpisodeEvent, EpisodeSource, EpisodeWrite, EvidenceLevel,
    };

    let recorded_by = if governance {
        ActorRef {
            principal_id: "admin_ref".into(),
            kind: ActorKind::GuildAdministrator,
        }
    } else {
        ActorRef {
            principal_id: "abbey_service".into(),
            kind: ActorKind::Service,
        }
    };
    let write = EpisodeWrite {
        request_id: request_id.into(),
        operation_id: format!("op_{request_id}"),
        contract_revision: 2,
        contract_digest: [9; 32],
        guild_ref: "guild_ref".into(),
        consent_epoch: None,
        source_type: EpisodeSource::DiscordGuild,
        policy_version: "policy_v1".into(),
        evidence_level: EvidenceLevel::C1,
        event: EpisodeEvent::MemoryEdge { recorded_by, edge },
        token_cost: 1,
        expected_commitment: None,
        quiet: false,
    };
    serde_json::to_vec(&write).unwrap()
}

async fn append_json(service: &GatewayService, episode_write_json: Vec<u8>) -> [u8; 32] {
    let appended = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json,
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(appended.decision, "appended");
    appended.episode_digest.as_slice().try_into().unwrap()
}

fn quarantine_edge(target: [u8; 32]) -> abi_wdbx::v3::episode::MemoryEdge {
    abi_wdbx::v3::episode::MemoryEdge {
        kind: abi_wdbx::v3::episode::MemoryEdgeKind::Quarantines,
        target,
        counterpart: None,
        reason: abi_wdbx::v3::episode::EdgeReason::OperatorReport,
    }
}

fn resolution_edge(target: [u8; 32]) -> abi_wdbx::v3::episode::MemoryEdge {
    abi_wdbx::v3::episode::MemoryEdge {
        kind: abi_wdbx::v3::episode::MemoryEdgeKind::Resolves,
        target,
        counterpart: None,
        reason: abi_wdbx::v3::episode::EdgeReason::ReviewedValid,
    }
}

#[tokio::test]
async fn verify_episode_reports_memory_edge_state() {
    use abi_wdbx::v3::episode::{EdgeReason, MemoryEdge, MemoryEdgeKind};

    let scratch = Scratch::new("episodes-verify-edges");
    let limits = Limits::default();
    let executor = StoreExecutor::open_with_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
    )
    .unwrap();
    let service = service_over(executor, &limits, &scratch);

    let a = append_json(&service, episode_memory_candidate_json("req_a", "mem_a")).await;
    let b = append_json(&service, episode_memory_candidate_json("req_b", "mem_b")).await;
    let flag = append_json(
        &service,
        memory_edge_json("req_q", false, quarantine_edge(a)),
    )
    .await;
    let pair = append_json(
        &service,
        memory_edge_json(
            "req_c",
            false,
            MemoryEdge {
                kind: MemoryEdgeKind::Contradicts,
                target: a.min(b),
                counterpart: Some(a.max(b)),
                reason: EdgeReason::ConflictingObservation,
            },
        ),
    )
    .await;

    // A flagged candidate is still found, and says why it is suspect.
    let flagged = verify(&service, &a).await;
    assert!(flagged.found);
    assert!(!flagged.memory_forgotten);
    assert_eq!(flagged.open_quarantine_edge, flag.to_vec());
    assert_eq!(flagged.open_contradictions.len(), 1);
    assert_eq!(flagged.open_contradictions[0].counterpart, b.to_vec());
    assert_eq!(flagged.open_contradictions[0].edge, pair.to_vec());
    assert_eq!(flagged.memory_edge_status, "");

    let other = verify(&service, &b).await;
    assert_eq!(other.open_quarantine_edge, Vec::<u8>::new());
    assert_eq!(other.open_contradictions[0].counterpart, a.to_vec());

    let open_edge = verify(&service, &flag).await;
    assert_eq!(
        open_edge.receipt.as_ref().unwrap().event_kind,
        "memory_edge"
    );
    assert_eq!(open_edge.memory_edge_status, "open");
    assert_eq!(open_edge.open_contradictions, vec![]);

    // A service may not lift its own flag: the store's reason comes back.
    let refused = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: memory_edge_json("req_r0", false, resolution_edge(flag)),
            preview_only: false,
        }))
        .await
        .unwrap_err();
    assert_eq!(refused.code(), Code::FailedPrecondition);
    assert_eq!(refused.message(), "episode_transition_invalid");

    let resolution = append_json(
        &service,
        memory_edge_json("req_r1", true, resolution_edge(flag)),
    )
    .await;
    assert_eq!(verify(&service, &flag).await.memory_edge_status, "closed");
    assert_eq!(
        verify(&service, &a).await.open_quarantine_edge,
        Vec::<u8>::new()
    );
    // A resolution is an edge episode, but not a closable one.
    assert_eq!(verify(&service, &resolution).await.memory_edge_status, "");
    // Records that are not memory carry no edge state.
    let proposal = append_via(&service, "req_p", "op_p").await;
    let plain = verify(&service, &proposal).await;
    assert!(plain.found);
    assert!(!plain.memory_forgotten);
    assert_eq!(plain.open_quarantine_edge, Vec::<u8>::new());
    assert_eq!(plain.open_contradictions, vec![]);
    assert_eq!(plain.memory_edge_status, "");
}
