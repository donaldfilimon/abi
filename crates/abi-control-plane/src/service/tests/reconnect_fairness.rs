//! Real HTTP regressions for bounded reconnect progress and native proof lifetime.
use super::relay::{enroll, runtime};
use super::*;
use crate::*;
use std::sync::atomic::{AtomicUsize, Ordering};

fn observed(app: Router, calls: Arc<AtomicUsize>, proofs: Arc<Mutex<Vec<NodeMessage>>>) -> Router {
    app.layer(axum::middleware::from_fn(
        move |request: axum::extract::Request, next: axum::middleware::Next| {
            let calls = calls.clone();
            let proofs = proofs.clone();
            async move {
                if request.uri().path() != "/v1/relay/node" {
                    return next.run(request).await;
                }
                let (parts, body) = request.into_parts();
                let bytes = axum::body::to_bytes(body, 384 * 1024).await.unwrap();
                let message: NodeMessage = serde_json::from_slice(&bytes).unwrap();
                calls.fetch_add(1, Ordering::SeqCst);
                proofs.lock().unwrap().push(message);
                next.run(axum::extract::Request::from_parts(parts, Body::from(bytes)))
                    .await
            }
        },
    ))
}
fn seed_retained(
    f: &Fixture,
    cert: &NodeCertificate,
    runtime: &mut DurableRelayRuntime<abi_worker::FileControlStore>,
) {
    let mut authority = f.authority.lock().unwrap();
    let mut relay = f.relay.lock().unwrap();
    let time = now().unwrap();
    let account = Account {
        account_id: cert.claim.account_id.clone(),
        tenant_id: cert.claim.tenant_id.clone(),
        active_job_limit: 2,
        queued_job_limit: 10,
    };
    for _ in 0..20 {
        let input = InferenceJob {
            prompt: "private-reconnect-fairness".into(),
            tokens: 10,
            model: "local-echo-v1".into(),
        };
        let submission = prepare_job(
            cert,
            &authority.public_key(),
            &RelayKey::generate().unwrap(),
            &input,
            time,
        )
        .unwrap();
        let policy = DeliveryPolicy::for_submission(&submission, time + 3600).unwrap();
        relay
            .submit_durable(
                &mut authority,
                &account,
                DurableSubmission {
                    submission: submission.clone(),
                    delivery: policy.clone(),
                },
                time,
            )
            .unwrap();
        let mut dispatch = Dispatch {
            submission,
            signature: vec![],
        };
        dispatch.signature = authority
            .key
            .sign(&dispatch.bytes().unwrap())
            .to_bytes()
            .to_vec();
        runtime
            .outbox
            .prepare(
                &dispatch,
                &SignedDeliveryPolicy::sign(policy, &authority.key).unwrap(),
            )
            .unwrap();
        let manifest = runtime
            .reconcile_local(time)
            .unwrap()
            .into_iter()
            .find(|m| m.binding.job_id == dispatch.submission.binding.job_id)
            .unwrap();
        assert_eq!(manifest.outcome, RecordedOutcome::Abandoned);
        let request = RecoveryRequest {
            protocol: "abi.relay-recovery.v1".into(),
            action: RecoveryAction::Inspect {
                job_id: manifest.binding.job_id.clone(),
                manifest,
            },
        };
        let reply = relay.recover(&mut authority, cert, &request, time).unwrap();
        assert!(reply.permit.is_none());
    }
}
#[tokio::test]
async fn twenty_retained_manifests_do_not_starve_fresh_poll_and_epoch_reset_is_bounded() {
    Box::pin(tokio::time::timeout(
        std::time::Duration::from_secs(15),
        async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let f = fixture_with_origin(origin).await;
            let browser = browser(&f, "fair-reconnect").await;
            let (cert, key, encryption) = enroll(&f, &browser, 61).await;
            let mut durable = DurableRelayRuntime::new(
                runtime(&f, &cert, &key, encryption, "fair-worker.log"),
                FileRelayOutbox::create(
                    &f.dir.path().join("fair-outbox.sqlite"),
                    OutboxLimits::default(),
                )
                .unwrap(),
            )
            .unwrap();
            seed_retained(&f, &cert, &mut durable);
            let client = RelayClient::new(cert, key, true).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let proofs = Arc::new(Mutex::new(Vec::new()));
            let app = observed(f.app.clone(), calls.clone(), proofs);
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let first = client
                .recover_pending(&mut durable, now().unwrap())
                .await
                .unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 16);
            assert_eq!(first.restored, 16);
            assert_eq!(first.pending, 4);
            let before = calls.load(Ordering::SeqCst);
            let second = client
                .recover_pending(&mut durable, now().unwrap())
                .await
                .unwrap();
            assert_eq!(second.restored, 20);
            assert_eq!(second.pending, 0);
            assert!(calls.load(Ordering::SeqCst) - before <= 16);
            assert_eq!(client.recovered.lock().unwrap().len(), 20);
            let (mut durable, fresh_poll) = client
                .run_durable_once(durable, Arc::new(abi_agent_runtime::EchoProvider::new()))
                .await;
            assert!(
                fresh_poll.unwrap().is_none(),
                "fresh poll must become reachable after bounded recovery"
            );
            let old_epoch = client
                .recovered
                .lock()
                .unwrap()
                .values()
                .next()
                .unwrap()
                .0
                .clone();
            // Rotate the real router's replay epoch and transient cache under its exclusive guard.
            // Process restart itself is independently covered by reconnect_process.
            {
                let mut relay = f.relay.lock().unwrap();
                relay.epoch = uuid::Uuid::new_v4().to_string();
                relay.frames.clear();
                relay.results.clear();
                relay.terminals.clear();
            }
            let mut completed = false;
            for _ in 0..3 {
                let before = calls.load(Ordering::SeqCst);
                let round = client
                    .recover_pending(&mut durable, now().unwrap())
                    .await
                    .unwrap();
                assert!(
                    calls.load(Ordering::SeqCst) - before <= 16,
                    "epoch repair must preserve per-round total exchange budget"
                );
                if round.pending == 0 {
                    assert_eq!(round.restored, 20);
                    completed = true;
                    break;
                }
            }
            assert!(
                completed,
                "epoch reset must recover every retained manifest within bounded rounds"
            );
            let memos = client.recovered.lock().unwrap();
            assert_eq!(memos.len(), 20);
            assert!(memos.values().all(|(epoch, _)| *epoch != old_epoch));
            drop(memos);
            server.abort();
        },
    ))
    .await
    .unwrap();
}
#[tokio::test]
async fn native_five_second_proofs_expire_before_the_existing_live_proof_budget_deadlocks() {
    Box::pin(tokio::time::timeout(
        std::time::Duration::from_secs(15),
        async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let origin = format!("http://{}", listener.local_addr().unwrap());
            let f = fixture_with_origin(origin).await;
            let browser = browser(&f, "native-proof-budget").await;
            let (cert, key, _) = enroll(&f, &browser, 63).await;
            let client = RelayClient::new(cert, key, true).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let proofs = Arc::new(Mutex::new(Vec::new()));
            let app = observed(f.app.clone(), calls, proofs.clone());
            let server = tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap();
            });
            let issued = now().unwrap();
            for _ in 0..64 {
                client.exchange(NodeAction::Poll, issued).await.unwrap();
            }
            assert!(
                client.exchange(NodeAction::Poll, issued).await.is_err(),
                "existing 64-live-proof limit must still hold"
            );
            assert!(
                proofs
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|m| m.proof.expires_at == issued + 5)
            );
            // Wait for this exact credential expiry boundary, rather than timing asynchronous work with sleeps.
            let expires = std::time::UNIX_EPOCH
                + std::time::Duration::from_secs(u64::try_from(issued + 5).unwrap());
            if let Ok(remaining) = expires.duration_since(std::time::SystemTime::now()) {
                tokio::time::sleep(remaining).await;
            }
            for _ in 0..16 {
                client
                    .exchange(NodeAction::Poll, now().unwrap())
                    .await
                    .unwrap();
            }
            let live: i64 = f
                .authority
                .lock()
                .unwrap()
                .db
                .query_row(
                    "SELECT count(*) FROM challenges WHERE node=?1 AND expires>?2",
                    rusqlite::params![client.certificate.node_id, now().unwrap()],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(live, 16);
            server.abort();
        },
    ))
    .await
    .unwrap();
}
