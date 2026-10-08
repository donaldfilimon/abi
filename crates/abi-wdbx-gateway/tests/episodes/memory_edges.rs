//! Memory-edge projection and resolution provenance.

use super::*;

#[test]
fn resolver_field_is_additive_for_older_protobuf_peers() {
    use abi_wdbx_gateway::proto::VerifyEpisodeResponse;
    use prost::Message as _;

    // An older peer knows fields 1 and 8 but has no field 9.
    #[derive(prost::Message)]
    struct LegacyResponse {
        #[prost(bool, tag = "1")]
        found: bool,
        #[prost(string, tag = "8")]
        memory_edge_status: String,
    }
    let current = VerifyEpisodeResponse {
        found: true,
        memory_edge_status: "closed".into(),
        memory_edge_resolution: vec![7; 32],
        ..Default::default()
    };
    let legacy = LegacyResponse::decode(current.encode_to_vec().as_slice()).unwrap();
    assert!(legacy.found);
    assert_eq!(legacy.memory_edge_status, "closed");
    let without_resolver =
        VerifyEpisodeResponse::decode(legacy.encode_to_vec().as_slice()).unwrap();
    assert!(without_resolver.found);
    assert_eq!(without_resolver.memory_edge_status, "closed");
    assert_eq!(without_resolver.memory_edge_resolution, Vec::<u8>::new());
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
    assert_eq!(flagged.memory_edge_resolution, Vec::<u8>::new());

    let other = verify(&service, &b).await;
    assert_eq!(other.open_quarantine_edge, Vec::<u8>::new());
    assert_eq!(other.open_contradictions[0].counterpart, a.to_vec());

    let open_edge = verify(&service, &flag).await;
    assert_eq!(
        open_edge.receipt.as_ref().unwrap().event_kind,
        "memory_edge"
    );
    assert_eq!(open_edge.memory_edge_status, "open");
    assert_eq!(open_edge.memory_edge_resolution, Vec::<u8>::new());
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
        verify(&service, &flag).await.memory_edge_resolution,
        resolution
    );
    assert_eq!(
        verify(&service, &a).await.open_quarantine_edge,
        Vec::<u8>::new()
    );
    // A resolution is an edge episode, but not a closable one.
    assert_eq!(verify(&service, &resolution).await.memory_edge_status, "");
    assert_eq!(
        verify(&service, &resolution).await.memory_edge_resolution,
        Vec::<u8>::new()
    );
    // Records that are not memory carry no edge state.
    let proposal = append_via(&service, "req_p", "op_p").await;
    let plain = verify(&service, &proposal).await;
    assert!(plain.found);
    assert!(!plain.memory_forgotten);
    assert_eq!(plain.open_quarantine_edge, Vec::<u8>::new());
    assert_eq!(plain.open_contradictions, vec![]);
    assert_eq!(plain.memory_edge_status, "");
    assert_eq!(plain.memory_edge_resolution, Vec::<u8>::new());

    assert_cycle_reopen_projection(service, &scratch, &limits, a, flag, resolution, pair).await;
}

async fn assert_cycle_reopen_projection(
    service: GatewayService,
    scratch: &Scratch,
    limits: &Limits,
    a: [u8; 32],
    flag: [u8; 32],
    resolution: [u8; 32],
    pair: [u8; 32],
) {
    // Closing a different kind of edge must return its own canonical resolver.
    let pair_resolution = append_json(
        &service,
        memory_edge_json("req_pair_resolution", true, resolution_edge(pair)),
    )
    .await;
    assert_eq!(
        verify(&service, &pair).await.memory_edge_resolution,
        pair_resolution
    );
    let next_flag = append_json(
        &service,
        memory_edge_json("req_next_flag", false, quarantine_edge(a)),
    )
    .await;
    assert_eq!(
        verify(&service, &next_flag).await.memory_edge_resolution,
        Vec::<u8>::new()
    );
    assert_eq!(
        verify(&service, &flag).await.memory_edge_resolution,
        resolution
    );
    let next_resolution = append_json(
        &service,
        memory_edge_json("req_next_resolution", true, resolution_edge(next_flag)),
    )
    .await;
    drop(service);

    // The gateway rebuilds the same projection after releasing and reopening
    // the scratch store; its additive field also survives protobuf encoding.
    let executor = StoreExecutor::open_with_episodes(
        &scratch.store(),
        limits.blocking_jobs,
        Some(episode_policy(true)),
    )
    .unwrap();
    let service = service_over(executor, limits, scratch);
    for (edge, resolver) in [
        (flag, resolution),
        (pair, pair_resolution),
        (next_flag, next_resolution),
    ] {
        use prost::Message as _;
        let found = verify(&service, &edge).await;
        assert_eq!(found.memory_edge_status, "closed");
        assert_eq!(found.memory_edge_resolution, resolver);
        let decoded = abi_wdbx_gateway::proto::VerifyEpisodeResponse::decode(
            found.encode_to_vec().as_slice(),
        )
        .unwrap();
        assert_eq!(decoded.memory_edge_resolution, resolver);
    }
    let missing = verify(&service, &[0x55; 32]).await;
    assert!(!missing.found);
    assert_eq!(missing.memory_edge_resolution, Vec::<u8>::new());
    let wrong_guild = service
        .verify_episode(authenticated(VerifyEpisodeRequest {
            guild_ref: "other_guild".into(),
            episode_digest: flag.to_vec(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(!wrong_guild.found);
    assert_eq!(wrong_guild.memory_edge_resolution, Vec::<u8>::new());
}
