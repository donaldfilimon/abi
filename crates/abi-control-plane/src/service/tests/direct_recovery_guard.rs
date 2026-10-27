//! Direct runtime callers retain the same fail-closed uncertainty boundary as reconnect.
use super::relay::{enroll, runtime};
use super::*;
use crate::delivery_outbox::faults::{Fault, Guard};
use crate::{
    Dispatch, DispatchPermit, DurableRelayRuntime, FileRelayOutbox, InferenceJob, NodeCertificate,
    OutboxLimits, RelayKey, SignedDeliveryPolicy, prepare_job,
};
use crate::{OutboxPacket, PacketId, RecordedOutcome, RelayRuntime};
use abi_worker::{FileControlStore, FileControlStoreLimits};
use std::sync::atomic::{AtomicUsize, Ordering};
struct Count(AtomicUsize);
impl abi_agent_runtime::ModelProvider for Count {
    fn id(&self) -> &'static str {
        "direct-recovery-guard"
    }
    fn run(
        &self,
        request: &abi_agent_runtime::ModelRequest,
        run: &mut abi_agent_runtime::RunContext<'_>,
    ) -> std::result::Result<(), abi_agent_runtime::RuntimeError> {
        self.0.fetch_add(1, Ordering::SeqCst);
        abi_agent_runtime::EchoProvider::new().run(request, run)
    }
}
fn dispatch(
    f: &Fixture,
    cert: &NodeCertificate,
    time: i64,
    issued: i64,
) -> (Dispatch, SignedDeliveryPolicy, DispatchPermit) {
    let authority = f.authority.lock().unwrap();
    let submission = prepare_job(
        cert,
        &authority.public_key(),
        &RelayKey::generate().unwrap(),
        &InferenceJob {
            prompt: "private-direct-guard".into(),
            tokens: 10,
            model: "local-echo-v1".into(),
        },
        time,
    )
    .unwrap();
    let policy = SignedDeliveryPolicy::sign(
        crate::DeliveryPolicy::for_submission(&submission, time + 86400).unwrap(),
        &authority.key,
    )
    .unwrap();
    let mut permit = DispatchPermit {
        binding: submission.binding.clone(),
        submission_digest: crate::authority::hash(&crate::enrollment::json(&submission).unwrap()),
        issued_at: issued,
        expires_at: issued + 5,
        signature: vec![],
    };
    permit.signature = authority
        .key
        .sign(&permit.bytes().unwrap())
        .to_bytes()
        .to_vec();
    let mut dispatch = Dispatch {
        submission,
        signature: vec![],
    };
    dispatch.signature = authority
        .key
        .sign(&dispatch.bytes().unwrap())
        .to_bytes()
        .to_vec();
    (dispatch, policy, permit)
}
#[tokio::test]
async fn direct_execution_after_uncertainty_refuses_before_prepare_or_provider() {
    let f = fixture().await;
    let browser = browser(&f, "direct-uncertainty").await;
    let (cert, key, encryption) = enroll(&f, &browser, 115).await;
    let time = now().unwrap();
    let mut durable = DurableRelayRuntime::new(
        runtime(&f, &cert, &key, encryption, "direct-worker.log"),
        FileRelayOutbox::create(
            &f.dir.path().join("direct-outbox.sqlite"),
            OutboxLimits::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let provider = Count(AtomicUsize::new(0));
    let cancel = abi_worker::CancellationToken::new();
    let (old, policy, expired) = dispatch(&f, &cert, time, time - 6);
    assert!(
        durable
            .execute_durable(
                &old,
                &policy,
                &expired,
                &provider,
                || time,
                &cancel,
                |_| Ok(())
            )
            .is_err()
    );
    assert!(durable.unresolved_live);
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    let (fresh, policy, permit) = dispatch(&f, &cert, time, time);
    fresh
        .verify(&f.authority.lock().unwrap().public_key())
        .unwrap();
    permit
        .verify(&f.authority.lock().unwrap().public_key(), time)
        .unwrap();
    assert!(
        durable
            .execute_durable(
                &fresh,
                &policy,
                &permit,
                &provider,
                || time,
                &cancel,
                |_| Ok(())
            )
            .is_err()
    );
    assert!(durable.unresolved_live);
    assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    assert!(
        durable
            .outbox
            .entry(&fresh.submission.binding.job_id)
            .unwrap()
            .is_none()
    );
}
fn reopen(
    f: &Fixture,
    cert: &NodeCertificate,
    key: &SigningKey,
) -> DurableRelayRuntime<FileControlStore> {
    let runtime = RelayRuntime::new(
        cert.clone(),
        key.clone(),
        RelayKey::from_seed(&[117; 32]).unwrap(),
        f.authority.lock().unwrap().public_key(),
        "local-echo-v1",
        FileControlStore::open(
            &f.dir.path().join("terminal.log"),
            FileControlStoreLimits::default(),
        )
        .unwrap(),
        now().unwrap(),
    )
    .unwrap();
    let outbox = FileRelayOutbox::open(
        &f.dir.path().join("terminal.sqlite"),
        OutboxLimits::default(),
    )
    .unwrap();
    DurableRelayRuntime::new(runtime, outbox).unwrap()
}
fn terminal_ids(durable: &DurableRelayRuntime<FileControlStore>, d: &Dispatch) -> Vec<PacketId> {
    let b = &d.submission.binding;
    let entry = durable.outbox.entry(&b.job_id).unwrap().unwrap();
    let terminals = entry
        .packets
        .iter()
        .filter(|p| matches!(p, OutboxPacket::Terminal { .. }))
        .count();
    assert_eq!(terminals, 1);
    assert_eq!(
        entry.manifest.as_ref().unwrap().outcome,
        RecordedOutcome::Completed
    );
    entry.packets.iter().map(|p| p.id(b).unwrap()).collect()
}
#[tokio::test]
async fn terminal_outbox_failure_reopens_with_one_terminal_and_refuses_fresh_dispatch() {
    for fault in [Fault::Full, Fault::PartialWrite, Fault::Sync] {
        let f = fixture().await;
        let browser = browser(&f, "terminal-outbox").await;
        let (cert, key, encryption) = enroll(&f, &browser, 116).await;
        let time = now().unwrap();
        let outbox_path = f.dir.path().join("terminal.sqlite");
        let guard = Guard::new(&outbox_path);
        let mut durable = DurableRelayRuntime::new(
            runtime(&f, &cert, &key, encryption, "terminal.log"),
            FileRelayOutbox::create(&outbox_path, OutboxLimits::default()).unwrap(),
        )
        .unwrap();
        let provider = Count(AtomicUsize::new(0));
        let cancel = abi_worker::CancellationToken::new();
        let (d, policy, permit) = dispatch(&f, &cert, time, time);
        let mut armed = false;
        let clock = || {
            if !armed && provider.0.load(Ordering::SeqCst) == 1 {
                // The next outbox write is the terminal append in finish_local.
                guard.arm(fault);
                armed = true;
            }
            time
        };
        assert!(
            durable
                .execute_durable(&d, &policy, &permit, &provider, clock, &cancel, |_| Ok(()))
                .is_err()
        );
        assert!(armed);
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
        assert!(durable.outbox.poisoned());
        let (fresh, policy, permit) = dispatch(&f, &cert, time, time);
        assert!(
            durable
                .execute_durable(
                    &fresh,
                    &policy,
                    &permit,
                    &provider,
                    || time,
                    &cancel,
                    |_| Ok(())
                )
                .is_err()
        );
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
        drop(durable);
        drop(guard);
        let mut durable = reopen(&f, &cert, &key);
        durable.reconcile_local(time).unwrap();
        let ids1 = terminal_ids(&durable, &d);
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
        drop(durable);
        let mut durable = reopen(&f, &cert, &key);
        durable.reconcile_local(time).unwrap();
        assert_eq!(terminal_ids(&durable, &d), ids1);
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
    }
}
