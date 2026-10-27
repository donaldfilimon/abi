use super::*;
use crate::{PairRequest, Relay, RelayLimits};
use ed25519_dalek::Signer as _;

fn identity(subject: &str) -> VerifiedIdentity {
    VerifiedIdentity {
        issuer: "https://issuer.example".into(),
        subject: subject.into(),
    }
}
fn open() -> (tempfile::TempDir, Authority) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let store = Authority::open(
        &dir.path().join("authority.sqlite"),
        true,
        "https://abi.example",
    )
    .unwrap();
    (dir, store)
}
fn pair_node(store: &mut Authority, account: &Account, seed: u8, now: i64) -> String {
    let key = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
    let ticket = store
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: None,
                capabilities: vec!["model.infer".into()],
            },
            now,
        )
        .unwrap();
    let claim = store
        .claim_pairing(account, &ticket.user_code, now)
        .unwrap();
    store
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            now,
        )
        .unwrap()
        .node_id
}
fn insert_job(store: &Authority, id: &str, account: &Account, node: &str, state: &str) {
    store
        .db
        .execute(
            "INSERT INTO relay_jobs(id,account,node,binding,digest,state,deadline) VALUES(?1,?2,?3,'b','d',?4,0)",
            params![id, account.account_id, node, state],
        )
        .unwrap();
}

#[test]
fn metrics_count_accounts_sessions_nodes_and_jobs() {
    let (_dir, mut store) = open();
    let alice = store.account(&identity("alice"), 100).unwrap();
    let bob = store.account(&identity("bob"), 100).unwrap();
    store.new_session(&alice, 100).unwrap();
    store.new_session(&bob, 100).unwrap();
    let kept = pair_node(&mut store, &alice, 7, 101);
    let revoked = pair_node(&mut store, &bob, 8, 101);
    store.revoke(&bob, &revoked, 102).unwrap();
    Relay::open(&mut store, RelayLimits::default(), 103).unwrap();
    insert_job(&store, "job-1", &alice, &kept, "queued");
    insert_job(&store, "job-2", &alice, &kept, "completed");
    insert_job(&store, "job-3", &alice, &kept, "mystery");
    let m = store.operator_metrics(104, 1_000).unwrap();
    assert_eq!(m.accounts, 2);
    assert_eq!(m.sessions_active, 2);
    assert_eq!((m.nodes_active, m.nodes_revoked), (1, 1));
    assert_eq!(m.jobs.get("queued"), Some(&1));
    assert_eq!(m.jobs.get("completed"), Some(&1));
    assert_eq!(m.jobs.get("other"), Some(&1));
    assert_eq!(m.audit_events.get("node.revoked"), Some(&1));
    assert_eq!(m.audit_events.get("session.created"), Some(&2));
}

#[test]
fn metrics_contain_no_identifiers() {
    let (_dir, mut store) = open();
    let alice = store.account(&identity("subject-alice-7f3e"), 100).unwrap();
    let session = store.new_session(&alice, 100).unwrap();
    let node = pair_node(&mut store, &alice, 9, 101);
    Relay::open(&mut store, RelayLimits::default(), 102).unwrap();
    insert_job(&store, "job-id-5c1d", &alice, &node, "dispatched");
    let json = serde_json::to_string(&store.operator_metrics(103, 1_000).unwrap()).unwrap();
    for secret in [
        alice.account_id.as_str(),
        alice.tenant_id.as_str(),
        "subject-alice-7f3e",
        "issuer.example",
        "abi.example",
        session.token.as_str(),
        session.csrf.as_str(),
        node.as_str(),
        "job-id-5c1d",
    ] {
        assert!(!json.contains(secret), "metrics leaked {secret}: {json}");
    }
}

#[test]
fn expired_sessions_and_old_audit_are_excluded() {
    let (_dir, mut store) = open();
    let alice = store.account(&identity("alice"), 100).unwrap();
    store.new_session(&alice, 100).unwrap();
    let far = 100 + 10_i64.pow(12);
    let m = store.operator_metrics(far, 1_000).unwrap();
    assert_eq!(m.accounts, 1);
    assert_eq!(m.sessions_active, 0);
    assert!(m.audit_events.is_empty());
    assert!(m.jobs.is_empty());
    assert_eq!(m.audit_window_ms, 1_000);
}
