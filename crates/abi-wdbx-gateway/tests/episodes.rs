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
