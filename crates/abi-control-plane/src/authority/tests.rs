use super::*;
use crate::{NodeCertificate, PairClaim, PairRequest, PairTicket};
fn identity(subject: &str) -> VerifiedIdentity {
    VerifiedIdentity {
        issuer: "https://issuer.example".into(),
        subject: subject.into(),
    }
}
#[test]
fn verified_accounts_survive_restart_and_do_not_cross_tenants() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let path = dir.path().join("authority.sqlite");
    let mut store = Authority::open(&path, true, "https://abi.example").unwrap();
    let alice = store.account(&identity("alice"), 100).unwrap();
    let bob = store.account(&identity("bob"), 100).unwrap();
    assert_ne!(alice.tenant_id, bob.tenant_id);
    assert_eq!((alice.active_job_limit, alice.queued_job_limit), (2, 10));
    drop(store);
    let mut reopened = Authority::open(&path, false, "https://abi.example").unwrap();
    assert_eq!(reopened.account(&identity("alice"), 101).unwrap(), alice);
}
#[test]
fn missing_authority_requires_explicit_initialization() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        Authority::open(
            &dir.path().join("missing.sqlite"),
            false,
            "https://abi.example"
        )
        .is_err()
    );
}

fn setup() -> (tempfile::TempDir, Authority, Account, Account) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let mut store = Authority::open(
        &dir.path().join("authority.sqlite"),
        true,
        "https://abi.example",
    )
    .unwrap();
    let alice = store.account(&identity("alice"), 100).unwrap();
    let bob = store.account(&identity("bob"), 100).unwrap();
    (dir, store, alice, bob)
}
fn pair(
    store: &mut Authority,
    account: &Account,
    key: &ed25519_dalek::SigningKey,
    now: i64,
) -> (PairTicket, PairClaim, NodeCertificate) {
    use ed25519_dalek::Signer as _;
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
    let certificate = store
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            now,
        )
        .unwrap();
    (ticket, claim, certificate)
}
#[test]
fn pairing_needs_both_browser_ownership_and_exact_local_key_confirmation() {
    use ed25519_dalek::{Signer as _, SigningKey};
    let (_dir, mut store, alice, bob) = setup();
    let key = SigningKey::from_bytes(&[7; 32]);
    let ticket = store
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: None,
                capabilities: vec!["model.infer".into()],
            },
            101,
        )
        .unwrap();
    assert!(
        store
            .poll_pairing(&ticket.enrollment_id, &ticket.user_code, 101)
            .is_err()
    );
    assert!(
        store
            .poll_pairing(&ticket.enrollment_id, &ticket.poll_secret, 101)
            .unwrap()
            .is_none()
    );
    assert!(
        store
            .confirm_pairing(&ticket.enrollment_id, &ticket.poll_secret, &[0; 64], 101)
            .is_err()
    );
    let claim = store.claim_pairing(&alice, &ticket.user_code, 102).unwrap();
    assert!(store.claim_pairing(&bob, &ticket.user_code, 102).is_err());
    assert_eq!(
        store
            .poll_pairing(&ticket.enrollment_id, &ticket.poll_secret, 102)
            .unwrap(),
        Some(claim.clone())
    );
    assert!(store.nodes(&alice).unwrap().is_empty());
    let mut changed = claim.clone();
    changed.tenant_id = bob.tenant_id.clone();
    assert!(
        store
            .confirm_pairing(
                &ticket.enrollment_id,
                &ticket.poll_secret,
                &key.sign(&changed.signing_bytes().unwrap()).to_bytes(),
                103
            )
            .is_err()
    );
    let signature = key.sign(&claim.signing_bytes().unwrap()).to_bytes();
    let cert = store
        .confirm_pairing(&ticket.enrollment_id, &ticket.poll_secret, &signature, 103)
        .unwrap();
    cert.verify(&store.public_key(), "https://abi.example", 103)
        .unwrap();
    assert!(
        cert.verify(&store.public_key(), "https://other.example", 103)
            .is_err()
    );
    assert_eq!(
        store
            .confirm_pairing(&ticket.enrollment_id, &ticket.poll_secret, &signature, 104)
            .unwrap(),
        cert
    );
    assert_eq!(store.nodes(&alice).unwrap().len(), 1);
    assert!(store.nodes(&bob).unwrap().is_empty());
    assert!(store.revoke(&bob, &cert.node_id, 104).is_err());
}
#[test]
fn challenge_replay_mutation_expiry_revocation_and_deleted_accounts_are_refused() {
    use ed25519_dalek::{Signer as _, SigningKey};
    let (dir, mut store, alice, _bob) = setup();
    let key = SigningKey::from_bytes(&[8; 32]);
    let (_, _, cert) = pair(&mut store, &alice, &key, 101);
    let challenge = store.challenge(&cert.node_id, 102).unwrap();
    let signature = key.sign(&challenge.signing_bytes().unwrap()).to_bytes();
    let mut changed = challenge.clone();
    changed.audience = "https://other.example".into();
    assert!(
        store
            .authenticate_node(
                &changed,
                &key.sign(&changed.signing_bytes().unwrap()).to_bytes(),
                102
            )
            .is_err()
    );
    assert!(store.authenticate_node(&challenge, &[0; 64], 102).is_err());
    assert_eq!(
        store
            .authenticate_node(&challenge, &signature, 102)
            .unwrap(),
        cert
    );
    assert!(
        store
            .authenticate_node(&challenge, &signature, 102)
            .is_err()
    );
    let stale = store.challenge(&cert.node_id, 103).unwrap();
    assert!(
        store
            .authenticate_node(
                &stale,
                &key.sign(&stale.signing_bytes().unwrap()).to_bytes(),
                163
            )
            .is_err()
    );
    let fresh = store.challenge(&cert.node_id, 104).unwrap();
    store.revoke(&alice, &cert.node_id, 105).unwrap();
    drop(store);
    let mut store = Authority::open(
        &dir.path().join("authority.sqlite"),
        false,
        "https://abi.example",
    )
    .unwrap();
    assert!(
        store
            .authenticate_node(
                &fresh,
                &key.sign(&fresh.signing_bytes().unwrap()).to_bytes(),
                106
            )
            .is_err()
    );
    let (_, _, other) = pair(&mut store, &alice, &key, 107);
    let challenge = store.challenge(&other.node_id, 108).unwrap();
    store.delete_account(&alice, 109).unwrap();
    assert!(
        store
            .authenticate_node(
                &challenge,
                &key.sign(&challenge.signing_bytes().unwrap()).to_bytes(),
                109
            )
            .is_err()
    );
    let replacement = store.account(&identity("alice"), 110).unwrap();
    assert_ne!(replacement.tenant_id, alice.tenant_id);
    assert!(store.nodes(&replacement).unwrap().is_empty());
}
#[test]
fn sessions_are_hashed_csrf_bound_expiring_and_removed_on_deletion() {
    let (_dir, mut store, alice, _bob) = setup();
    let session = store.new_session(&alice, 101).unwrap();
    assert!(store.session(&session.token, Some("wrong"), 102).is_err());
    assert_eq!(
        store
            .session(&session.token, Some(&session.csrf), 102)
            .unwrap(),
        alice
    );
    let stored: (String, String) = store
        .db
        .query_row("SELECT token,csrf FROM sessions", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_ne!(stored.0, session.token);
    assert_ne!(stored.1, session.csrf);
    assert!(store.session(&session.token, None, 28_901).is_err());
    store.delete_account(&alice, 103).unwrap();
    assert!(store.session(&session.token, None, 104).is_err());
}
#[test]
fn expiry_and_capability_escalation_are_refused() {
    let (_dir, mut store, alice, _bob) = setup();
    let key = ed25519_dalek::SigningKey::from_bytes(&[9; 32]);
    for capabilities in [
        vec![],
        vec!["operator.train".into()],
        vec!["model.infer".into(), "model.infer".into()],
    ] {
        assert!(
            store
                .begin_pairing(
                    &PairRequest {
                        public_key: key.verifying_key().to_bytes(),
                        encryption_public_key: None,
                        capabilities
                    },
                    101
                )
                .is_err()
        );
    }
    let ticket = store
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: None,
                capabilities: vec!["model.infer".into()],
            },
            101,
        )
        .unwrap();
    assert!(store.claim_pairing(&alice, &ticket.user_code, 401).is_err());
    assert!(
        store
            .poll_pairing(&ticket.enrollment_id, &ticket.poll_secret, 401)
            .is_err()
    );
}
#[test]
fn audit_failure_rolls_back_authority_mutation_and_history_is_append_only() {
    let (_dir, mut store, _alice, _bob) = setup();
    assert!(store.db.execute("DELETE FROM audit", []).is_err());
    assert!(
        store
            .db
            .execute("UPDATE audit SET kind='hidden'", [])
            .is_err()
    );
    store.db.execute_batch("CREATE TRIGGER audit_fail BEFORE INSERT ON audit BEGIN SELECT RAISE(ABORT,'unavailable'); END;").unwrap();
    assert!(store.account(&identity("charlie"), 101).is_err());
    let count: i64 = store
        .db
        .query_row(
            "SELECT count(*) FROM accounts WHERE subject='charlie'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
}
#[test]
fn simultaneous_browser_claims_have_exactly_one_winner() {
    let (dir, mut store, alice, bob) = setup();
    let key = ed25519_dalek::SigningKey::from_bytes(&[10; 32]);
    let ticket = store
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: None,
                capabilities: vec!["model.infer".into()],
            },
            101,
        )
        .unwrap();
    let path = dir.path().join("authority.sqlite");
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let handles = [alice, bob]
        .into_iter()
        .map(|account| {
            let path = path.clone();
            let barrier = barrier.clone();
            let code = ticket.user_code.clone();
            std::thread::spawn(move || {
                let mut store = Authority::open(&path, false, "https://abi.example").unwrap();
                barrier.wait();
                store.claim_pairing(&account, &code, 102).is_ok()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        handles
            .into_iter()
            .map(|h| usize::from(h.join().unwrap()))
            .sum::<usize>(),
        1
    );
}
#[test]
fn wrong_audience_symlinks_and_public_permissions_refuse_startup() {
    use std::os::unix::fs::{PermissionsExt as _, symlink};
    let (dir, store, _alice, _bob) = setup();
    drop(store);
    let path = dir.path().join("authority.sqlite");
    assert!(Authority::open(&path, false, "https://other.example").is_err());
    let link = dir.path().join("link.sqlite");
    symlink(&path, &link).unwrap();
    assert!(Authority::open(&link, false, "https://abi.example").is_err());
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
    assert!(Authority::open(&path, false, "https://abi.example").is_err());
}
#[test]
fn anonymous_pairing_does_not_spend_the_administrative_audit_budget() {
    let (_dir, mut store, _alice, _bob) = setup();
    let before: i64 = store
        .db
        .query_row("SELECT count(*) FROM audit", [], |r| r.get(0))
        .unwrap();
    let key = ed25519_dalek::SigningKey::from_bytes(&[17; 32]);
    store
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: None,
                capabilities: vec!["model.infer".into()],
            },
            101,
        )
        .unwrap();
    let after: i64 = store
        .db
        .query_row("SELECT count(*) FROM audit", [], |r| r.get(0))
        .unwrap();
    assert_eq!(before, after);
}
#[test]
fn obtaining_local_challenges_cannot_exhaust_node_authentication() {
    use ed25519_dalek::{Signer as _, SigningKey};
    let (_dir, mut store, alice, _bob) = setup();
    let key = SigningKey::from_bytes(&[18; 32]);
    let (_, _, certificate) = pair(&mut store, &alice, &key, 101);
    for _ in 0..100 {
        store.challenge(&certificate.node_id, 102).unwrap();
    }
    let challenge = store.challenge(&certificate.node_id, 102).unwrap();
    store
        .authenticate_node(
            &challenge,
            &key.sign(&challenge.signing_bytes().unwrap()).to_bytes(),
            102,
        )
        .unwrap();
}
#[test]
fn sqlite_reopen_cannot_traverse_a_replaceable_public_ancestor() {
    use std::os::unix::fs::PermissionsExt as _;
    let outer = tempfile::tempdir().unwrap();
    std::fs::set_permissions(outer.path(), std::fs::Permissions::from_mode(0o777)).unwrap();
    let inner = outer.path().join("private");
    std::fs::create_dir(&inner).unwrap();
    std::fs::set_permissions(&inner, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(Authority::open(&inner.join("authority.sqlite"), true, "https://abi.example").is_err());
}
#[test]
fn confirmed_credential_and_used_nonce_persist_across_restarts() {
    use ed25519_dalek::{Signer as _, SigningKey};
    let (dir, mut store, alice, _bob) = setup();
    let key = SigningKey::from_bytes(&[20; 32]);
    let (ticket, claim, cert) = pair(&mut store, &alice, &key, 101);
    let proof = store.challenge(&cert.node_id, 102).unwrap();
    let signature = key.sign(&proof.signing_bytes().unwrap()).to_bytes();
    store.authenticate_node(&proof, &signature, 102).unwrap();
    drop(store);
    let mut store = Authority::open(
        &dir.path().join("authority.sqlite"),
        false,
        "https://abi.example",
    )
    .unwrap();
    assert!(store.authenticate_node(&proof, &signature, 103).is_err());
    assert_eq!(
        store
            .confirm_pairing(
                &ticket.enrollment_id,
                &ticket.poll_secret,
                &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
                103
            )
            .unwrap(),
        cert
    );
}
#[test]
fn account_deletion_clears_subject_bytes_from_the_active_database() {
    let (dir, mut store, _alice, _bob) = setup();
    let subject = "delete-subject-sensitive-canary-40ff0eef";
    let account = store.account(&identity(subject), 101).unwrap();
    store.delete_account(&account, 102).unwrap();
    drop(store);
    let bytes = std::fs::read(dir.path().join("authority.sqlite")).unwrap();
    assert!(
        !bytes
            .windows(subject.len())
            .any(|window| window == subject.as_bytes())
    );
}
