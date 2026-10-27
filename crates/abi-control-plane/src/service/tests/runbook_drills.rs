//! Executable drills cited by `RUNBOOKS.md`. Each drill exercises one operator
//! procedure against local fixtures only; none is staging or separate-host evidence.
use super::*;

const ORIGIN: &str = "http://127.0.0.1:3000";

/// Enroll one node for `account` over HTTP and return its key and credential.
async fn enroll(f: &Fixture, account: &Browser, seed: u8) -> (SigningKey, crate::NodeCertificate) {
    let key = SigningKey::from_bytes(&[seed; 32]);
    let response = request(
        &f.app,
        "POST",
        "/v1/enrollments",
        "",
        None,
        None,
        Some(serde_json::json!({"public_key":key.verifying_key().to_bytes(),"capabilities":["model.infer"]})),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let ticket: crate::PairTicket = serde_json::from_value(json(response).await).unwrap();
    let claim: crate::PairClaim = serde_json::from_value(
        json(
            request(
                &f.app,
                "POST",
                "/v1/enrollments/claim",
                &account.cookie,
                Some(ORIGIN),
                Some(&account.csrf),
                Some(serde_json::json!({"user_code":ticket.user_code})),
            )
            .await,
        )
        .await,
    )
    .unwrap();
    let certificate = f
        .authority
        .lock()
        .unwrap()
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            now().unwrap(),
        )
        .unwrap();
    (key, certificate)
}

/// A node-built proof. Built locally (as a node may) so that it is sent even when
/// the authority would refuse to preview a challenge for a revoked node.
fn proof(key: &SigningKey, node_id: &str) -> (crate::NodeChallenge, Vec<u8>) {
    let challenge = crate::NodeChallenge {
        protocol: "abi.node-proof.v1".into(),
        audience: ORIGIN.into(),
        node_id: node_id.into(),
        nonce: crate::authority::secret(),
        expires_at: now().unwrap() + 30,
    };
    let signature = key
        .sign(&challenge.signing_bytes().unwrap())
        .to_bytes()
        .to_vec();
    (challenge, signature)
}

async fn authenticate_status(f: &Fixture, key: &SigningKey, node_id: &str) -> StatusCode {
    let (challenge, signature) = proof(key, node_id);
    request(
        &f.app,
        "POST",
        "/v1/nodes/authenticate",
        "",
        None,
        None,
        Some(serde_json::json!({"challenge":challenge,"signature":signature})),
    )
    .await
    .status()
}

async fn revoke(f: &Fixture, account: &Browser, node_id: &str) {
    let path = format!("/v1/nodes/{node_id}/revoke");
    let response = request(
        &f.app,
        "POST",
        &path,
        &account.cookie,
        Some(ORIGIN),
        Some(&account.csrf),
        None,
    )
    .await;
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
}

/// Runbook 1: node compromise / tenant revocation.
#[tokio::test]
async fn drill_node_revocation_refuses_authentication() {
    let f = fixture().await;
    let alice = browser(&f, "alice").await;
    let (key, cert) = enroll(&f, &alice, 51).await;
    assert_eq!(
        authenticate_status(&f, &key, &cert.node_id).await,
        StatusCode::OK
    );
    revoke(&f, &alice, &cert.node_id).await;
    assert_eq!(
        authenticate_status(&f, &key, &cert.node_id).await,
        StatusCode::UNAUTHORIZED
    );
    let view = json(
        request(
            &f.app,
            "GET",
            "/v1/account",
            &alice.cookie,
            None,
            None,
            None,
        )
        .await,
    )
    .await;
    assert_eq!(view["nodes"][0]["revoked"], true);
}

/// Runbook 3: backup and restore. A restored copy of the closed database keeps
/// revocations and account deletions; it never resurrects them.
#[tokio::test]
async fn drill_restore_preserves_revocation_and_deletion() {
    let mut f = fixture().await;
    let alice = browser(&f, "alice").await;
    let bob = browser(&f, "bob").await;
    let (revoked_key, revoked) = enroll(&f, &alice, 52).await;
    let (kept_key, kept) = enroll(&f, &alice, 53).await;
    let (bob_key, bob_node) = enroll(&f, &bob, 54).await;
    let bob_account = json(
        request(&f.app, "GET", "/v1/account", &bob.cookie, None, None, None).await,
    )
    .await["account"]["account_id"]
        .as_str()
        .unwrap()
        .to_owned();
    revoke(&f, &alice, &revoked.node_id).await;
    assert_eq!(
        request(
            &f.app,
            "DELETE",
            "/v1/account",
            &bob.cookie,
            Some(ORIGIN),
            Some(&bob.csrf),
            None
        )
        .await
        .status(),
        StatusCode::NO_CONTENT
    );
    // Stop the service: dropping the fixture closes every database handle.
    let source_dir = std::mem::replace(&mut f.dir, tempfile::tempdir().unwrap());
    drop(f);
    let restore_dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        restore_dir.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let restored_path = restore_dir.path().join("control.sqlite");
    std::fs::copy(source_dir.path().join("control.sqlite"), &restored_path).unwrap();
    std::fs::set_permissions(
        &restored_path,
        std::os::unix::fs::PermissionsExt::from_mode(0o600),
    )
    .unwrap();
    drop(source_dir);
    let mut restored = Authority::open(&restored_path, false, ORIGIN).unwrap();
    let clock = now().unwrap();
    assert!(restored.challenge(&revoked.node_id, clock).is_err());
    let (challenge, signature) = proof(&revoked_key, &revoked.node_id);
    assert!(
        restored
            .authenticate_node(&challenge, &signature, clock)
            .is_err()
    );
    let (challenge, signature) = proof(&bob_key, &bob_node.node_id);
    assert!(
        restored
            .authenticate_node(&challenge, &signature, clock)
            .is_err()
    );
    let count =
        |sql: &str, id: &str| -> i64 { restored.db.query_row(sql, [id], |r| r.get(0)).unwrap() };
    assert_eq!(
        count("SELECT count(*) FROM accounts WHERE id=?1", &bob_account),
        0
    );
    assert_eq!(
        count("SELECT count(*) FROM nodes WHERE account=?1", &bob_account),
        0
    );
    // Recovery: an unaffected node still authenticates against the restored copy.
    let (challenge, signature) = proof(&kept_key, &kept.node_id);
    restored
        .authenticate_node(&challenge, &signature, clock)
        .unwrap();
}

/// Runbook 2: authority signing-key compromise. There is no rotation command; a
/// freshly initialized authority refuses every credential signed by the old key,
/// even if the old node rows are carried over, and nodes must re-pair.
#[tokio::test]
async fn drill_new_authority_refuses_old_node_certificates() {
    let old = fixture().await;
    let alice = browser(&old, "alice").await;
    let (key, cert) = enroll(&old, &alice, 55).await;
    assert_eq!(
        authenticate_status(&old, &key, &cert.node_id).await,
        StatusCode::OK
    );
    let new = fixture().await;
    let new_key = new.authority.lock().unwrap().public_key();
    assert_ne!(new_key, old.authority.lock().unwrap().public_key());
    assert!(cert.verify(&new_key, ORIGIN, now().unwrap()).is_err());
    assert_eq!(
        authenticate_status(&new, &key, &cert.node_id).await,
        StatusCode::UNAUTHORIZED
    );
    // Even a mistaken carry-over of the old account and node rows is refused,
    // because the credential signature no longer verifies.
    {
        let authority = new.authority.lock().unwrap();
        authority
            .db
            .execute(
                "INSERT INTO accounts VALUES(?1,?2,'carried-issuer','carried-subject')",
                [&cert.claim.account_id, &cert.claim.tenant_id],
            )
            .unwrap();
        authority
            .db
            .execute(
                "INSERT INTO nodes VALUES(?1,?2,?3,0)",
                [
                    &cert.node_id,
                    &cert.claim.account_id,
                    &serde_json::to_string(&cert).unwrap(),
                ],
            )
            .unwrap();
    }
    assert_eq!(
        authenticate_status(&new, &key, &cert.node_id).await,
        StatusCode::UNAUTHORIZED
    );
    // Recovery: re-pairing the same node key with the new authority succeeds.
    let alice_new = browser(&new, "alice").await;
    let (key, cert) = enroll(&new, &alice_new, 55).await;
    assert_eq!(
        authenticate_status(&new, &key, &cert.node_id).await,
        StatusCode::OK
    );
}

/// Runbook 5: emergency shutdown. The served router answers `/healthz`, then stops
/// within 5 s of the shutdown signal even while a keep-alive connection stays open.
#[tokio::test]
async fn drill_graceful_shutdown_completes_inflight_health() {
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    let f = fixture().await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(
        axum::serve(listener, f.app.clone())
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .into_future(),
    );
    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    stream
        .write_all(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        .await
        .unwrap();
    let mut received = Vec::new();
    let mut buffer = [0_u8; 1024];
    let read = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        while !String::from_utf8_lossy(&received).contains(r#"{"status":"ok"}"#) {
            let n = stream.read(&mut buffer).await.unwrap();
            assert_ne!(n, 0, "connection closed before the health response");
            received.extend_from_slice(&buffer[..n]);
        }
    })
    .await;
    assert!(read.is_ok(), "no health response");
    assert!(String::from_utf8_lossy(&received).starts_with("HTTP/1.1 200"));
    stop.send(()).unwrap();
    let finished = tokio::time::timeout(std::time::Duration::from_secs(5), server).await;
    assert!(
        finished.is_ok(),
        "server did not stop within 5 s of the signal"
    );
    finished.unwrap().unwrap().unwrap();
    assert!(tokio::net::TcpStream::connect(address).await.is_err());
    drop(stream);
}
