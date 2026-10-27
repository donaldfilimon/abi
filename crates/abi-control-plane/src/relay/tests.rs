use super::*;
use crate::{
    JobBinding, NodeAction, NodeCertificate, PairRequest, RelayKey, seal, types::VerifiedIdentity,
};
use ed25519_dalek::SigningKey;
struct Fixture {
    dir: tempfile::TempDir,
    authority: Authority,
    relay: Relay,
    alice: Account,
    bob: Account,
    node_a: NodeCertificate,
    node_b: NodeCertificate,
    key_a: SigningKey,
    key_b: SigningKey,
    enc_a: RelayKey,
    enc_b: RelayKey,
    reply: RelayKey,
    reply_b: RelayKey,
}
fn enroll(
    authority: &mut Authority,
    account: &Account,
    key: &SigningKey,
    enc: &RelayKey,
) -> NodeCertificate {
    let ticket = authority
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: Some(enc.public_key()),
                capabilities: vec!["model.infer".into()],
            },
            100,
        )
        .unwrap();
    let claim = authority
        .claim_pairing(account, &ticket.user_code, 100)
        .unwrap();
    authority
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            100,
        )
        .unwrap()
}
fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let mut authority = Authority::open(
        &dir.path().join("authority.sqlite"),
        true,
        "https://abi.example",
    )
    .unwrap();
    let mut account = |subject: &str| {
        authority
            .account(
                &VerifiedIdentity {
                    issuer: "https://issuer.example".into(),
                    subject: subject.into(),
                },
                100,
            )
            .unwrap()
    };
    let alice = account("alice");
    let bob = account("bob");
    let key_a = SigningKey::from_bytes(&[41; 32]);
    let key_b = SigningKey::from_bytes(&[42; 32]);
    let enc_a = RelayKey::from_seed(&[101; 32]).unwrap();
    let enc_b = RelayKey::from_seed(&[102; 32]).unwrap();
    let reply = RelayKey::from_seed(&[103; 32]).unwrap();
    let reply_b = RelayKey::from_seed(&[104; 32]).unwrap();
    let node_a = enroll(&mut authority, &alice, &key_a, &enc_a);
    let node_b = enroll(&mut authority, &bob, &key_b, &enc_b);
    let relay = Relay::open(&mut authority, RelayLimits::default(), 100).unwrap();
    Fixture {
        dir,
        authority,
        relay,
        alice,
        bob,
        node_a,
        node_b,
        key_a,
        key_b,
        enc_a,
        enc_b,
        reply,
        reply_b,
    }
}
fn submission(f: &Fixture, account: &Account, node: &NodeCertificate) -> Submission {
    submission_at(f, account, node, 100, 200)
}
fn submission_at(
    f: &Fixture,
    account: &Account,
    node: &NodeCertificate,
    submitted_at: i64,
    expires_at: i64,
) -> Submission {
    let binding = JobBinding {
        protocol: "abi.relay-job.v1".into(),
        audience: "https://abi.example".into(),
        job_id: uuid::Uuid::new_v4().to_string(),
        account_id: account.account_id.clone(),
        tenant_id: account.tenant_id.clone(),
        node_id: node.node_id.clone(),
        capability: "model.infer".into(),
        expires_at,
        submitted_at,
        max_result_bytes: 16384,
        reply_key: if account.account_id == f.alice.account_id {
            f.reply.public_key()
        } else {
            f.reply_b.public_key()
        },
    };
    let payload = seal(
        &node.claim.encryption_public_key.unwrap(),
        &binding.aad("request").unwrap(),
        br#"{"prompt":"private-tenant-content-canary","tokens":10,"model":"local-echo-v1"}"#,
    )
    .unwrap();
    Submission { binding, payload }
}
#[test]
fn accounts_can_only_dispatch_to_their_own_enrolled_nodes_once() {
    let mut f = fixture();
    let job = submission(&f, &f.alice, &f.node_a);
    let id = job.binding.job_id.clone();
    let receipt = f
        .relay
        .submit(&mut f.authority, &f.alice, job.clone(), 101)
        .unwrap();
    assert_eq!(receipt.state, "queued");
    assert!(
        f.relay
            .submit(&mut f.authority, &f.bob, job.clone(), 101)
            .is_err()
    );
    assert!(f.relay.view(&mut f.authority, &f.bob, &id, 101).is_err());
    let poll = NodeMessage::sign(&f.node_b, &f.key_b, NodeAction::Poll, 102).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &poll, 102)
            .unwrap()
            .dispatch
            .is_none()
    );
    let poll = NodeMessage::sign(&f.node_a, &f.key_a, NodeAction::Poll, 102).unwrap();
    let dispatch = f
        .relay
        .node(&mut f.authority, &poll, 102)
        .unwrap()
        .dispatch
        .unwrap();
    dispatch.verify(&f.authority.public_key()).unwrap();
    assert!(f.relay.node(&mut f.authority, &poll, 102).is_err());
    let retry = NodeMessage::sign(&f.node_a, &f.key_a, NodeAction::Poll, 103).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &retry, 103)
            .unwrap()
            .dispatch
            .is_none()
    );
    assert!(
        f.enc_a
            .open(
                &job.binding.aad("request").unwrap(),
                &dispatch.submission.payload
            )
            .is_ok()
    );
    assert!(
        f.enc_b
            .open(
                &job.binding.aad("request").unwrap(),
                &dispatch.submission.payload
            )
            .is_err()
    );
    assert_eq!(
        f.relay
            .submit(&mut f.authority, &f.alice, job, 103)
            .unwrap()
            .state,
        "offered"
    );
}
struct CountProvider(std::sync::atomic::AtomicUsize);
impl abi_agent_runtime::ModelProvider for CountProvider {
    fn id(&self) -> &'static str {
        "counted-local"
    }
    fn run(
        &self,
        request: &abi_agent_runtime::ModelRequest,
        run: &mut abi_agent_runtime::RunContext<'_>,
    ) -> std::result::Result<(), abi_agent_runtime::RuntimeError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        abi_agent_runtime::EchoProvider::new().run(request, run)
    }
}
fn poll(f: &mut Fixture, node_a: bool, now: i64) -> Dispatch {
    let (cert, key) = if node_a {
        (&f.node_a, &f.key_a)
    } else {
        (&f.node_b, &f.key_b)
    };
    f.relay
        .node(
            &mut f.authority,
            &NodeMessage::sign(cert, key, NodeAction::Poll, now).unwrap(),
            now,
        )
        .unwrap()
        .dispatch
        .unwrap()
}
fn permit(f: &mut Fixture, node_a: bool, id: &str, now: i64) -> DispatchPermit {
    let (cert, key) = if node_a {
        (&f.node_a, &f.key_a)
    } else {
        (&f.node_b, &f.key_b)
    };
    f.relay
        .node(
            &mut f.authority,
            &NodeMessage::sign(cert, key, NodeAction::Inspect { job_id: id.into() }, now).unwrap(),
            now,
        )
        .unwrap()
        .permit
        .unwrap()
}
#[test]
fn two_tenants_execute_encrypted_inference_with_separate_durable_workers_once() {
    use crate::RelayRuntime;
    use abi_worker::{FileControlStore, FileControlStoreLimits, JobResult};
    let mut f = fixture();
    let a = submission(&f, &f.alice, &f.node_a);
    let b = submission(&f, &f.bob, &f.node_b);
    f.relay
        .submit(&mut f.authority, &f.alice, a.clone(), 101)
        .unwrap();
    f.relay
        .submit(&mut f.authority, &f.bob, b.clone(), 101)
        .unwrap();
    let da = poll(&mut f, true, 102);
    let db = poll(&mut f, false, 102);
    let pa = permit(&mut f, true, &a.binding.job_id, 102);
    let pb = permit(&mut f, false, &b.binding.job_id, 102);
    let path = f.dir.path().join("node-a.log");
    let store = FileControlStore::create(&path, FileControlStoreLimits::default()).unwrap();
    let store_b = FileControlStore::create(
        &f.dir.path().join("node-b.log"),
        FileControlStoreLimits::default(),
    )
    .unwrap();
    let mut ra = RelayRuntime::new(
        f.node_a.clone(),
        f.key_a.clone(),
        RelayKey::from_seed(&[101; 32]).unwrap(),
        f.authority.public_key(),
        "local-echo-v1",
        store,
        102,
    )
    .unwrap();
    let mut rb = RelayRuntime::new(
        f.node_b.clone(),
        f.key_b.clone(),
        RelayKey::from_seed(&[102; 32]).unwrap(),
        f.authority.public_key(),
        "local-echo-v1",
        store_b,
        102,
    )
    .unwrap();
    let ca = CountProvider(std::sync::atomic::AtomicUsize::new(0));
    let cb = CountProvider(std::sync::atomic::AtomicUsize::new(0));
    assert!(rb.execute(&da, &pa, &cb, || 103).is_err());
    let output_a = ra.execute(&da, &pa, &ca, || 103).unwrap().unwrap();
    let output_b = rb.execute(&db, &pb, &cb, || 103).unwrap().unwrap();
    assert!(ra.execute(&da, &pa, &ca, || 104).unwrap().is_some());
    drop(ra);
    let store = FileControlStore::open(&path, FileControlStoreLimits::default()).unwrap();
    let mut ra = RelayRuntime::new(
        f.node_a.clone(),
        f.key_a.clone(),
        RelayKey::from_seed(&[101; 32]).unwrap(),
        f.authority.public_key(),
        "local-echo-v1",
        store,
        104,
    )
    .unwrap();
    assert!(ra.execute(&da, &pa, &ca, || 104).unwrap().is_some());
    assert_eq!(ca.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    assert_eq!(cb.0.load(std::sync::atomic::Ordering::SeqCst), 1);
    for (cert, key, id, payload) in [
        (&f.node_a, &f.key_a, &a.binding.job_id, &output_a),
        (&f.node_b, &f.key_b, &b.binding.job_id, &output_b),
    ] {
        let message = NodeMessage::sign(
            cert,
            key,
            NodeAction::Complete {
                job_id: id.clone(),
                payload: payload.clone(),
            },
            104,
        )
        .unwrap();
        f.relay.node(&mut f.authority, &message, 104).unwrap();
    }
    let view = f
        .relay
        .view(&mut f.authority, &f.alice, &a.binding.job_id, 104)
        .unwrap();
    assert_eq!(view.receipt.state, "completed");
    let bytes = f
        .reply
        .open(&a.binding.aad("result").unwrap(), &view.result.unwrap())
        .unwrap();
    let result: JobResult = serde_json::from_slice(&bytes).unwrap();
    assert!(result.verify_digest());
    assert_eq!(result.status, abi_worker::JobStatus::Succeeded);
    assert!(
        String::from_utf8_lossy(&result.chunks[0].bytes).contains("private-tenant-content-canary")
    );
    assert!(
        f.reply
            .open(&b.binding.aad("result").unwrap(), &output_b)
            .is_err()
    );
    no_control_plane_content(&f, &a);
}
fn no_control_plane_content(f: &Fixture, a: &Submission) {
    let cp = std::fs::read(f.dir.path().join("authority.sqlite")).unwrap();
    assert!(
        !cp.windows(b"private-tenant-content-canary".len())
            .any(|b| b == b"private-tenant-content-canary")
    );
    assert!(!cp.windows(16).any(|b| b == &a.payload.ciphertext[..16]));
}
#[test]
fn quotas_enforce_two_active_and_ten_queued_per_account() {
    let mut f = fixture();
    for _ in 0..10 {
        let s = submission(&f, &f.alice, &f.node_a);
        f.relay.submit(&mut f.authority, &f.alice, s, 101).unwrap();
    }
    let s = submission(&f, &f.alice, &f.node_a);
    assert!(matches!(
        f.relay.submit(&mut f.authority, &f.alice, s, 101),
        Err(Error::Full)
    ));
    poll(&mut f, true, 102);
    poll(&mut f, true, 102);
    for _ in 0..2 {
        let s = submission(&f, &f.alice, &f.node_a);
        f.relay.submit(&mut f.authority, &f.alice, s, 103).unwrap();
    }
    let message = NodeMessage::sign(&f.node_a, &f.key_a, NodeAction::Poll, 103).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &message, 103)
            .unwrap()
            .dispatch
            .is_none()
    );
    let bob = submission(&f, &f.bob, &f.node_b);
    f.relay.submit(&mut f.authority, &f.bob, bob, 103).unwrap();
    poll(&mut f, false, 103);
}
#[test]
fn revocation_cancellation_and_restart_never_redispatch_lost_handoffs() {
    let mut f = fixture();
    let job = submission(&f, &f.alice, &f.node_a);
    let id = job.binding.job_id.clone();
    f.relay
        .submit(&mut f.authority, &f.alice, job, 101)
        .unwrap();
    poll(&mut f, true, 102);
    assert!(f.relay.cancel(&mut f.authority, &f.bob, &id, 102).is_err());
    f.authority
        .revoke(&f.alice, &f.node_a.node_id, 103)
        .unwrap();
    let message = NodeMessage::sign(
        &f.node_a,
        &f.key_a,
        NodeAction::Inspect { job_id: id.clone() },
        103,
    )
    .unwrap();
    assert!(f.relay.node(&mut f.authority, &message, 103).is_err());
    let b = submission(&f, &f.bob, &f.node_b);
    let bid = b.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.bob, b, 103).unwrap();
    poll(&mut f, false, 103);
    drop(f.relay);
    f.relay = Relay::open(&mut f.authority, RelayLimits::default(), 104).unwrap();
    assert_eq!(
        f.relay
            .view(&mut f.authority, &f.bob, &bid, 104)
            .unwrap()
            .receipt
            .state,
        "unresolved"
    );
    let poll = NodeMessage::sign(&f.node_b, &f.key_b, NodeAction::Poll, 104).unwrap();
    assert!(
        f.relay
            .node(&mut f.authority, &poll, 104)
            .unwrap()
            .dispatch
            .is_none()
    );
}
#[test]
fn node_operation_mutation_and_foreign_results_are_refused() {
    let mut f = fixture();
    let a = submission(&f, &f.alice, &f.node_a);
    let id = a.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, a, 101).unwrap();
    poll(&mut f, true, 102);
    permit(&mut f, true, &id, 102);
    let mut message = NodeMessage::sign(&f.node_a, &f.key_a, NodeAction::Poll, 102).unwrap();
    message.action = NodeAction::Fail { job_id: id.clone() };
    assert!(f.relay.node(&mut f.authority, &message, 102).is_err());
    let foreign = NodeMessage::sign(
        &f.node_b,
        &f.key_b,
        NodeAction::Fail { job_id: id.clone() },
        102,
    )
    .unwrap();
    assert!(f.relay.node(&mut f.authority, &foreign, 102).is_err());
    assert_eq!(
        f.relay
            .view(&mut f.authority, &f.alice, &id, 102)
            .unwrap()
            .receipt
            .state,
        "dispatched"
    );
    let cancelled = f
        .relay
        .cancel(&mut f.authority, &f.alice, &id, 103)
        .unwrap();
    assert_eq!(cancelled.state, "cancel_requested");
}
#[test]
fn authority_permit_must_still_be_fresh_after_durable_admission() {
    let mut f = fixture();
    let s = submission(&f, &f.alice, &f.node_a);
    let id = s.binding.job_id.clone();
    f.relay.submit(&mut f.authority, &f.alice, s, 101).unwrap();
    let dispatch = poll(&mut f, true, 102);
    let permit = permit(&mut f, true, &id, 102);
    let mut runtime = crate::RelayRuntime::new(
        f.node_a.clone(),
        f.key_a.clone(),
        RelayKey::from_seed(&[101; 32]).unwrap(),
        f.authority.public_key(),
        "local-echo-v1",
        abi_worker::MemoryControlStore::new(),
        103,
    )
    .unwrap();
    let provider = CountProvider(std::sync::atomic::AtomicUsize::new(0));
    let mut ticks = [103, 103, 103, 108, 108].into_iter();
    assert!(
        runtime
            .execute(&dispatch, &permit, &provider, || ticks
                .next()
                .unwrap_or(108))
            .is_err()
    );
    assert_eq!(provider.0.load(std::sync::atomic::Ordering::SeqCst), 0);
}
#[test]
fn completed_ciphertext_retry_does_not_charge_the_memory_quota_twice() {
    let mut f = fixture();
    f.relay.limits.memory_bytes = 65_584;
    let mut s = submission(&f, &f.alice, &f.node_a);
    s.binding.max_result_bytes = 65_536;
    let id = s.binding.job_id.clone();
    f.relay
        .submit(&mut f.authority, &f.alice, s.clone(), 101)
        .unwrap();
    poll(&mut f, true, 102);
    permit(&mut f, true, &id, 102);
    let output = seal(
        &f.reply.public_key(),
        &s.binding.aad("result").unwrap(),
        &vec![0; 65_536],
    )
    .unwrap();
    for _ in 0..2 {
        let message = NodeMessage::sign(
            &f.node_a,
            &f.key_a,
            NodeAction::Complete {
                job_id: id.clone(),
                payload: output.clone(),
            },
            103,
        )
        .unwrap();
        assert_eq!(
            f.relay
                .node(&mut f.authority, &message, 103)
                .unwrap()
                .receipt
                .unwrap()
                .state,
            "completed"
        );
    }
    assert!(
        f.relay
            .submit(&mut f.authority, &f.alice, s, 104)
            .unwrap()
            .result_available
    );
}

mod streaming;

mod dispatch_recheck;
mod recovery;
mod tools;
