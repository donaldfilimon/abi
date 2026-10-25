//! Direct runtime callers retain the same fail-closed uncertainty boundary as reconnect.
use super::relay::{enroll, runtime};
use super::*;
use crate::{
    Dispatch, DispatchPermit, DurableRelayRuntime, FileRelayOutbox, InferenceJob, NodeCertificate,
    OutboxLimits, RelayKey, SignedDeliveryPolicy, prepare_job,
};
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
