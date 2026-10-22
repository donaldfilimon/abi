//! Consumer regressions over the real scratch-only WDBX episode ledger.

use super::*;
use abi_wdbx::v3::episode::{
    ActorKind, ActorRef, EpisodeEvent, EpisodeSource, EpisodeWrite, StorePolicy, TerminalReason,
    TerminalStatus,
};
use abi_wdbx_gateway::proto::ProposeEpisodeWriteResponse;

fn open_gateway(scratch: &Scratch, policy: StorePolicy) -> GatewayService {
    let limits = Limits::default();
    let executor =
        StoreExecutor::open_with_episodes(&scratch.store(), limits.blocking_jobs, Some(policy))
            .unwrap();
    service_over(executor, &limits, scratch)
}

fn ledger_path(scratch: &Scratch) -> PathBuf {
    scratch.store().join("episodes/episodes.v1.jsonl")
}

fn lifecycle_write(request_id: &str, event: Option<EpisodeEvent>) -> EpisodeWrite {
    let mut write: EpisodeWrite =
        serde_json::from_slice(&episode_proposal_json(request_id, "hardening_op")).unwrap();
    // Every event in this operation carries the same source binding.
    write.source_type = EpisodeSource::DiscordGuild;
    if let Some(event) = event {
        write.event = event;
    }
    write
}

fn actor(principal_id: &str, kind: ActorKind) -> ActorRef {
    ActorRef {
        principal_id: principal_id.into(),
        kind,
    }
}

async fn append_write(
    service: &GatewayService,
    write: &EpisodeWrite,
) -> ProposeEpisodeWriteResponse {
    let response = service
        .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
            episode_write_json: serde_json::to_vec(write).unwrap(),
            preview_only: false,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(response.decision, "appended");
    response
}

async fn seed_inexact_compensation(service: &GatewayService) {
    append_write(service, &lifecycle_write("hardening_proposal", None)).await;
    let events = [
        EpisodeEvent::Approval {
            approved_by: actor("admin_ref", ActorKind::GuildAdministrator),
        },
        EpisodeEvent::Execution {
            executed_by: actor("abbey_service", ActorKind::Service),
            voice: None,
        },
        EpisodeEvent::Compensation {
            compensated_by: actor("abbey_service", ActorKind::Service),
            exact_restore_observed: false,
        },
    ];
    for (index, event) in events.into_iter().enumerate() {
        append_write(
            service,
            &lifecycle_write(&format!("hardening_step_{index}"), Some(event)),
        )
        .await;
    }
}

async fn assert_rejected_without_mutation(
    service: &GatewayService,
    scratch: &Scratch,
    write: &EpisodeWrite,
    modes: &[bool],
    label: &str,
) {
    let before = std::fs::read(ledger_path(scratch)).unwrap();
    let mut watch = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
        .unwrap()
        .into_inner();
    for &preview_only in modes {
        let error = service
            .propose_episode_write(authenticated(ProposeEpisodeWriteRequest {
                episode_write_json: serde_json::to_vec(write).unwrap(),
                preview_only,
            }))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::FailedPrecondition);
        assert_eq!(error.message(), label);
        assert_eq!(std::fs::read(ledger_path(scratch)).unwrap(), before);
    }
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(25), watch.next())
            .await
            .is_err(),
        "a rejected episode must not publish a mutation notice"
    );
}

async fn assert_receipt_verifiable(
    service: &GatewayService,
    appended: &ProposeEpisodeWriteResponse,
) {
    let verified = verify(service, &appended.episode_digest).await;
    assert!(verified.found);
    assert_eq!(verified.receipt, appended.receipt);
}

#[tokio::test]
async fn inexact_compensation_rejects_success_before_and_after_gateway_reopen() {
    let scratch = Scratch::new("ledger-inexact-compensation");
    let service = open_gateway(&scratch, episode_policy(true));
    seed_inexact_compensation(&service).await;
    let compensated = lifecycle_write(
        "hardening_terminal",
        Some(EpisodeEvent::Terminal {
            status: TerminalStatus::Compensated,
            reason: TerminalReason::Completed,
        }),
    );
    assert_rejected_without_mutation(
        &service,
        &scratch,
        &compensated,
        &[true, false],
        "episode_transition_invalid",
    )
    .await;
    let before_reopen = std::fs::read(ledger_path(&scratch)).unwrap();
    drop(service);

    let service = open_gateway(&scratch, episode_policy(true));
    assert_eq!(std::fs::read(ledger_path(&scratch)).unwrap(), before_reopen);
    assert_rejected_without_mutation(
        &service,
        &scratch,
        &compensated,
        &[true, false],
        "episode_transition_invalid",
    )
    .await;

    // Reusing the rejected request ID proves rejection did not consume it.
    let failed = lifecycle_write(
        "hardening_terminal",
        Some(EpisodeEvent::Terminal {
            status: TerminalStatus::Failed,
            reason: TerminalReason::MediaFailure,
        }),
    );
    let appended = append_write(&service, &failed).await;
    let receipt = appended.receipt.as_ref().unwrap();
    assert_eq!(receipt.sequence, 5);
    assert_eq!(receipt.event_kind, "terminal");
    assert_eq!(receipt.terminal_status, "failed");
    assert_receipt_verifiable(&service, &appended).await;
    drop(service);
    let service = open_gateway(&scratch, episode_policy(true));
    assert_receipt_verifiable(&service, &appended).await;
}

#[tokio::test]
async fn padded_ledger_reopen_charges_actual_bytes_before_next_gateway_append() {
    let scratch = Scratch::new("ledger-physical-byte-budget");
    let service = open_gateway(&scratch, episode_policy(true));
    let seed: EpisodeWrite =
        serde_json::from_slice(&episode_proposal_json("budget_seed", "budget_seed_op")).unwrap();
    let appended = append_write(&service, &seed).await;
    drop(service); // Release both the gateway executor and the episode writer lock.

    let path = ledger_path(&scratch);
    let original = std::fs::read(&path).unwrap();
    // JSON permits leading whitespace. Keep one valid JSONL record, but make
    // its physical size materially larger than its canonical serialization.
    let mut padded = vec![b' '; 8_192];
    padded.extend_from_slice(&original);
    std::fs::write(&path, &padded).unwrap();
    let mut policy = episode_policy(true);
    policy
        .guilds
        .get_mut("guild_ref")
        .unwrap()
        .storage_budget_bytes = u64::try_from(padded.len()).unwrap() + 1;

    let service = open_gateway(&scratch, policy);
    assert_eq!(std::fs::read(&path).unwrap(), padded);
    assert_receipt_verifiable(&service, &appended).await;
    let next: EpisodeWrite =
        serde_json::from_slice(&episode_proposal_json("budget_next", "budget_next_op")).unwrap();
    assert_rejected_without_mutation(
        &service,
        &scratch,
        &next,
        &[false],
        "episode_storage_budget_exhausted",
    )
    .await;
    assert_eq!(std::fs::read(&path).unwrap(), padded);
    assert_receipt_verifiable(&service, &appended).await;
}
