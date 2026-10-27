//! Process acceptance uses the real HTTP router and private scratch histories.
#![allow(clippy::too_many_lines)] // Explicit sequential crash/network fixture stages.
use super::*;
use crate::*;
use abi_worker::{FileControlStore, FileControlStoreLimits};
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead as _, Read as _, Write as _},
    process::{Child, Command, Stdio},
    time::Duration,
};

#[derive(Serialize, Deserialize)]
struct ChildConfig {
    role: String,
    directory: std::path::PathBuf,
    audience: String,
    certificate: NodeCertificate,
    authority_key: [u8; 32],
    dispatch: Dispatch,
    policy: SignedDeliveryPolicy,
    permit: DispatchPermit,
    time: i64,
    initialize: bool,
    fault: String,
    key_seed: u8,
    encryption_seed: u8,
    reply_seed: u8,
}
struct Process {
    child: Child,
    ready: std::sync::mpsc::Receiver<String>,
}
impl Process {
    fn start(config: &ChildConfig) -> Self {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "service::tests::reconnect_process::recovery_child",
                "--nocapture",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        writeln!(
            child.stdin.as_mut().unwrap(),
            "{}",
            serde_json::to_string(config).unwrap()
        )
        .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, ready) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Some(s) = line.strip_prefix("RECOVERY_READY ") {
                    let _ = tx.send(s.to_owned());
                }
            }
        });
        Self { child, ready }
    }
    fn refused(&mut self) {
        assert!(
            matches!(
                self.ready.recv_timeout(Duration::from_secs(30)),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
            ),
            "failed startup must not announce readiness"
        );
        assert!(!self.child.wait().unwrap().success());
    }
    fn ready(&self) -> String {
        self.ready
            .recv_timeout(Duration::from_secs(30))
            .expect("bounded child boundary")
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
fn wait_boundary(value: &str) {
    boundary(value);
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}
fn boundary(value: &str) {
    println!("RECOVERY_READY {value}");
    std::io::stdout().flush().unwrap();
}
struct Counted(std::path::PathBuf);
impl abi_agent_runtime::ModelProvider for Counted {
    fn id(&self) -> &'static str {
        "durable-counted-fixture"
    }
    fn run(
        &self,
        r: &abi_agent_runtime::ModelRequest,
        c: &mut abi_agent_runtime::RunContext<'_>,
    ) -> std::result::Result<(), abi_agent_runtime::RuntimeError> {
        // A second invocation or marker failure aborts this child, never concealing it.
        let mut marker = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.0)
            .unwrap();
        marker.write_all(b"1").unwrap();
        marker.sync_all().unwrap();
        std::fs::File::open(self.0.parent().unwrap())
            .unwrap()
            .sync_all()
            .unwrap();
        abi_agent_runtime::EchoProvider::new().run(r, c)
    }
}
fn digest_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut value = String::new();
    for b in Sha256::digest(bytes) {
        write!(value, "{b:02x}").unwrap();
    }
    value
}
async fn authority_app(c: &ChildConfig, config: OidcConfig) -> Router {
    let oidc = Oidc::discover(config).await.unwrap();
    let mut authority =
        Authority::open(&c.directory.join("authority.sqlite"), false, &c.audience).unwrap();
    let b = &c.dispatch.submission.binding;
    let account = Account {
        account_id: b.account_id.clone(),
        tenant_id: b.tenant_id.clone(),
        active_job_limit: 2,
        queued_job_limit: 10,
    };
    match c.fault.as_str() {
        "revoked" => authority
            .revoke(&account, &c.certificate.node_id, now().unwrap())
            .unwrap(),
        "deleted" => authority.delete_account(&account, now().unwrap()).unwrap(),
        "expired" => {
            authority
                .db
                .execute(
                    "UPDATE nodes SET certificate=?1 WHERE id=?2",
                    rusqlite::params![
                        serde_json::to_string(&c.certificate).unwrap(),
                        c.certificate.node_id
                    ],
                )
                .unwrap();
        }
        _ => {}
    }
    let limits = RelayLimits {
        memory_bytes: if c.fault == "capacity" {
            65_584
        } else {
            RelayLimits::default().memory_bytes
        },
        ..RelayLimits::default()
    };
    let mut relay = Relay::open(&mut authority, limits, now().unwrap()).unwrap();
    if c.fault == "cancelled" {
        relay
            .cancel(&mut authority, &account, &b.job_id, now().unwrap())
            .unwrap();
    }
    if c.fault == "capacity" {
        let mut binding = b.clone();
        binding.job_id = uuid::Uuid::new_v4().to_string();
        binding.submitted_at = now().unwrap();
        binding.expires_at = binding.submitted_at + 120;
        let payload = seal(
            &c.certificate.claim.encryption_public_key.unwrap(),
            &binding.aad("request").unwrap(),
            &vec![0; 65_536],
        )
        .unwrap();
        relay
            .submit(
                &mut authority,
                &account,
                Submission { binding, payload },
                now().unwrap(),
            )
            .unwrap();
    }
    routes(App {
        authority: Arc::new(Mutex::new(authority)),
        oidc: Arc::new(oidc),
        relay: Arc::new(Mutex::new(relay)),
        operator: Arc::default(),
    })
}

#[test]
fn recovery_child() {
    // Ordinary gate invocation sees EOF and performs no fixture action.
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).unwrap() == 0 {
        return;
    }
    let c: ChildConfig = serde_json::from_str(&line).unwrap();
    if c.role == "authority" {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                let issuer = Issuer {
                    redirect_origin: c.audience.clone(),
                    base: format!("http://{}", listener.local_addr().unwrap()),
                    key: SigningKey::from_bytes(&[42; 32]),
                };
                let app = Router::new()
                    .route("/.well-known/openid-configuration", get(discovery))
                    .route("/jwks", get(jwks))
                    .route("/token", post(token))
                    .with_state(issuer.clone());
                tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
                let config = OidcConfig {
                    issuer: issuer.base,
                    client_id: "abi-client".into(),
                    client_secret: None,
                    origin: c.audience.clone(),
                    loopback_development: true,
                };
                let app = authority_app(&c, config).await;
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
                boundary(&listener.local_addr().unwrap().to_string());
                axum::serve(listener, app).await.unwrap();
            });
    } else {
        let history = c.directory.join("worker.log");
        let outbox = c.directory.join("outbox.sqlite");
        let store = if c.initialize {
            FileControlStore::create(&history, FileControlStoreLimits::default())
        } else {
            FileControlStore::open(&history, FileControlStoreLimits::default())
        }
        .unwrap();
        let mut outbox = if c.initialize {
            FileRelayOutbox::create(
                &outbox,
                if c.fault == "outbox_capacity" {
                    OutboxLimits {
                        job_bytes: 1,
                        ..OutboxLimits::default()
                    }
                } else {
                    OutboxLimits::default()
                },
            )
        } else {
            FileRelayOutbox::open(&outbox, OutboxLimits::default())
        }
        .unwrap();
        let mut runtime = RelayRuntime::new(
            c.certificate,
            SigningKey::from_bytes(&[c.key_seed; 32]),
            RelayKey::from_seed(&[c.encryption_seed; 32]).unwrap(),
            c.authority_key,
            if c.initialize {
                "local-echo-v1"
            } else {
                "changed-model-v2"
            },
            store,
            c.time,
        )
        .unwrap();
        if c.initialize
            && matches!(
                c.role.as_str(),
                "prepared" | "completed" | "completed_legacy" | "admitted"
            )
        {
            outbox.prepare(&c.dispatch, &c.policy).unwrap();
            if c.role == "admitted" {
                runtime.admit_recovery_fixture(&c.dispatch, c.time).unwrap();
            }
            if c.role == "completed_legacy" {
                runtime
                    .execute(
                        &c.dispatch,
                        &c.permit,
                        &Counted(c.directory.join("invocations")),
                        || c.time,
                    )
                    .unwrap();
            }
            if c.role == "completed" {
                runtime
                    .execute_streaming(
                        &c.dispatch,
                        &c.permit,
                        &Counted(c.directory.join("invocations")),
                        || c.time,
                        &abi_worker::CancellationToken::new(),
                        |frame| {
                            let bytes = frame.open(
                                &c.dispatch.submission.binding,
                                &SigningKey::from_bytes(&[c.key_seed; 32])
                                    .verifying_key()
                                    .to_bytes(),
                                &RelayKey::from_seed(&[c.reply_seed; 32]).unwrap(),
                            )?;
                            let digest = digest_hex(&bytes);
                            outbox.append_frame(
                                &c.dispatch.submission.binding.job_id,
                                &frame,
                                &digest,
                            )
                        },
                    )
                    .unwrap();
            }
            wait_boundary(&c.role);
        }
        let mut runtime = DurableRelayRuntime::new(runtime, outbox).unwrap();
        if c.initialize {
            runtime
                .execute_durable(
                    &c.dispatch,
                    &c.policy,
                    &c.permit,
                    &Counted(c.directory.join("invocations")),
                    || c.time,
                    &abi_worker::CancellationToken::new(),
                    |_| {
                        if c.role == "delta" {
                            wait_boundary("delta");
                        }
                        Ok(())
                    },
                )
                .unwrap();
        } else {
            runtime.reconcile_local(c.time).unwrap();
        }
        boundary("synced");
        // Keep handles open until the orchestrator kills us at this synced boundary.
        let mut next = String::new();
        let _ = std::io::stdin().read_line(&mut next);
    }
}
fn post_node(address: &str, message: &NodeMessage) -> NodeReply {
    let (status, body) = post_status(address, message);
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap()
}
fn post_status(address: &str, message: &NodeMessage) -> (u16, Vec<u8>) {
    let body = serde_json::to_vec(message).unwrap();
    let mut stream = std::net::TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    write!(stream,"POST /v1/relay/node HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).unwrap();
    stream.write_all(&body).unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).unwrap();
    let cut = response.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let status = std::str::from_utf8(&response[9..12])
        .unwrap()
        .parse()
        .unwrap();
    (status, response[cut..].to_vec())
}
fn publish_without_ack(address: &str, message: &NodeMessage) {
    let body = serde_json::to_vec(message).unwrap();
    let mut stream = std::net::TcpStream::connect(address).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(15)))
        .unwrap();
    write!(stream, "POST /v1/relay/node HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
    stream.write_all(&body).unwrap();
    // The real router produces response headers only after the metadata commit.
    // Discard the connection before consuming the response body/acknowledgement.
    let mut headers = Vec::new();
    while !headers.ends_with(b"\r\n\r\n") {
        let mut byte = [0];
        stream.read_exact(&mut byte).unwrap();
        headers.push(byte[0]);
        assert!(headers.len() <= 8192);
    }
    assert!(headers.starts_with(b"HTTP/1.1 200"));
}

fn seeded() -> (tempfile::TempDir, ChildConfig, Submission, RelayKey) {
    seeded_at(now().unwrap())
}
fn seeded_at(time: i64) -> (tempfile::TempDir, ChildConfig, Submission, RelayKey) {
    let directory = tempfile::tempdir().unwrap();
    std::fs::set_permissions(
        directory.path(),
        std::os::unix::fs::PermissionsExt::from_mode(0o700),
    )
    .unwrap();
    let audience = "http://127.0.0.1:3000";
    let mut authority =
        Authority::open(&directory.path().join("authority.sqlite"), true, audience).unwrap();
    let account = authority
        .account(
            &crate::types::VerifiedIdentity {
                issuer: "https://fixture.example".into(),
                subject: "alice".into(),
            },
            time,
        )
        .unwrap();
    let key = SigningKey::from_bytes(&[41; 32]);
    let encryption = RelayKey::from_seed(&[101; 32]).unwrap();
    let reply = RelayKey::from_seed(&[103; 32]).unwrap();
    let ticket = authority
        .begin_pairing(
            &PairRequest {
                public_key: key.verifying_key().to_bytes(),
                encryption_public_key: Some(encryption.public_key()),
                capabilities: vec!["model.infer".into()],
            },
            time,
        )
        .unwrap();
    let claim = authority
        .claim_pairing(&account, &ticket.user_code, time)
        .unwrap();
    let certificate = authority
        .confirm_pairing(
            &ticket.enrollment_id,
            &ticket.poll_secret,
            &key.sign(&claim.signing_bytes().unwrap()).to_bytes(),
            time,
        )
        .unwrap();
    let binding = JobBinding {
        protocol: "abi.relay-job.v1".into(),
        audience: audience.into(),
        job_id: uuid::Uuid::new_v4().to_string(),
        account_id: account.account_id.clone(),
        tenant_id: account.tenant_id.clone(),
        node_id: certificate.node_id.clone(),
        capability: "model.infer".into(),
        submitted_at: time,
        expires_at: time + 120,
        max_result_bytes: 16384,
        reply_key: reply.public_key(),
    };
    let submission = Submission {
        payload: seal(
            &encryption.public_key(),
            &binding.aad("request").unwrap(),
            br#"{"prompt":"PROCESS_PRIVATE_CANARY","tokens":10,"model":"local-echo-v1"}"#,
        )
        .unwrap(),
        binding,
    };
    let mut relay = Relay::open(&mut authority, RelayLimits::default(), time).unwrap();
    relay
        .submit_durable(
            &mut authority,
            &account,
            DurableSubmission {
                delivery: DeliveryPolicy::for_submission(&submission, time + 3600).unwrap(),
                submission: submission.clone(),
            },
            time,
        )
        .unwrap();
    let offered = relay
        .node(
            &mut authority,
            &NodeMessage::sign(&certificate, &key, NodeAction::Poll, time).unwrap(),
            time,
        )
        .unwrap();
    let dispatch = offered.dispatch.unwrap();
    let policy = offered.delivery.unwrap();
    let permit = relay
        .node(
            &mut authority,
            &NodeMessage::sign(
                &certificate,
                &key,
                NodeAction::Inspect {
                    job_id: submission.binding.job_id.clone(),
                },
                time,
            )
            .unwrap(),
            time,
        )
        .unwrap()
        .permit
        .unwrap();
    let config = ChildConfig {
        role: "worker".into(),
        directory: directory.path().into(),
        audience: audience.into(),
        certificate: certificate.clone(),
        authority_key: authority.public_key(),
        dispatch,
        policy,
        permit,
        time,
        initialize: true,
        fault: String::new(),
        key_seed: 41,
        encryption_seed: 101,
        reply_seed: 103,
    };
    drop(relay);
    drop(authority);
    (directory, config, submission, reply)
}
#[test]
fn reconnect_process_terminal_survives_worker_and_authority_kills() {
    let (directory, mut config, submission, reply) = seeded();
    let time = config.time;
    let certificate = config.certificate.clone();
    let key = SigningKey::from_bytes(&[41; 32]);
    let worker = Process::start(&config);
    assert_eq!(worker.ready(), "synced");
    drop(worker);
    let before = FileRelayOutbox::open(
        &directory.path().join("outbox.sqlite"),
        OutboxLimits::default(),
    )
    .unwrap();
    let saved_packets = serde_json::to_vec(
        &before
            .entry(&submission.binding.job_id)
            .unwrap()
            .unwrap()
            .packets,
    )
    .unwrap();
    assert!(
        FileRelayOutbox::open(
            &directory.path().join("outbox.sqlite"),
            OutboxLimits::default()
        )
        .is_err(),
        "exclusive process ownership"
    );
    drop(before);
    config.initialize = false;
    config.time = time + 121;
    let worker = Process::start(&config);
    assert_eq!(worker.ready(), "synced");
    drop(worker);
    assert_eq!(
        std::fs::read(directory.path().join("invocations")).unwrap(),
        b"1"
    );
    let outbox = FileRelayOutbox::open(
        &directory.path().join("outbox.sqlite"),
        OutboxLimits::default(),
    )
    .unwrap();
    let entry = outbox.entry(&submission.binding.job_id).unwrap().unwrap();
    let manifest = entry.manifest.unwrap();
    let packets = entry.packets;
    assert_eq!(
        serde_json::to_vec(&packets).unwrap(),
        saved_packets,
        "restart must retain exact ciphertext"
    );
    assert!(!packets.is_empty());
    for p in &packets {
        if let OutboxPacket::Terminal { packet } = p {
            let clear = packet
                .open(&submission.binding, &key.verifying_key().to_bytes(), &reply)
                .unwrap();
            assert_ne!(clear.len(), 0);
            assert!(
                packet
                    .open(
                        &submission.binding,
                        &key.verifying_key().to_bytes(),
                        &RelayKey::from_seed(&[104; 32]).unwrap()
                    )
                    .is_err()
            );
        }
    }
    config.role = "authority".into();
    let mut previous_epoch = None;
    for _ in 0..2 {
        let server = Process::start(&config);
        let address = server.ready();
        let request = RecoveryRequest {
            protocol: "abi.relay-recovery.v1".into(),
            action: RecoveryAction::Inspect {
                job_id: submission.binding.job_id.clone(),
                manifest: manifest.clone(),
            },
        };
        let response = post_node(
            &address,
            &NodeMessage::sign(
                &certificate,
                &key,
                NodeAction::Recovery { request },
                now().unwrap(),
            )
            .unwrap(),
        );
        let grant = response.recovery.unwrap().grant.unwrap();
        if let Some(epoch) = previous_epoch.replace(grant.epoch.clone()) {
            assert_ne!(epoch, grant.epoch, "restart rotates replay authority");
        }
        let fresh = post_node(
            &address,
            &NodeMessage::sign(&certificate, &key, NodeAction::Poll, now().unwrap()).unwrap(),
        );
        assert!(
            fresh.dispatch.is_none(),
            "recovery must never redispatch an offered job"
        );
        for packet in &packets {
            let encoded = serde_json::to_vec(packet).unwrap();
            let lost_ack = RecoveryRequest {
                protocol: "abi.relay-recovery.v1".into(),
                action: RecoveryAction::Publish {
                    grant: grant.clone(),
                    packet: packet.clone(),
                },
            };
            publish_without_ack(
                &address,
                &NodeMessage::sign(
                    &certificate,
                    &key,
                    NodeAction::Recovery { request: lost_ack },
                    now().unwrap(),
                )
                .unwrap(),
            );
            for _ in 0..2 {
                let request = RecoveryRequest {
                    protocol: "abi.relay-recovery.v1".into(),
                    action: RecoveryAction::Publish {
                        grant: grant.clone(),
                        packet: packet.clone(),
                    },
                };
                let response = post_node(
                    &address,
                    &NodeMessage::sign(
                        &certificate,
                        &key,
                        NodeAction::Recovery { request },
                        now().unwrap(),
                    )
                    .unwrap(),
                );
                assert_ne!(response.recovery.unwrap().cached.len(), 0);
            }
            assert_eq!(serde_json::to_vec(packet).unwrap(), encoded);
        }
        drop(server);
    }
    let central = std::fs::read(directory.path().join("authority.sqlite")).unwrap();
    assert!(
        !central
            .windows(b"PROCESS_PRIVATE_CANARY".len())
            .any(|w| w == b"PROCESS_PRIVATE_CANARY")
    );
    let backup = directory.path().join("central-backup.sqlite");
    std::fs::copy(directory.path().join("authority.sqlite"), &backup).unwrap();
    assert_eq!(std::fs::read(backup).unwrap(), central);
    assert!(!directory.path().join("authority.sqlite-journal").exists());
    for packet in packets {
        let cipher = match &packet {
            OutboxPacket::Delta { frame } => &frame.payload.ciphertext,
            OutboxPacket::Terminal { packet } => &packet.payload.ciphertext,
        };
        assert!(!central.windows(cipher.len()).any(|w| w == cipher));
        let encoded = serde_json::to_vec(&packet).unwrap();
        assert!(!central.windows(encoded.len()).any(|w| w == encoded));
    }
}

#[test]
fn reconnect_process_kills_do_not_resume_prepared_admitted_or_partial_generation() {
    for role in [
        "prepared",
        "admitted",
        "delta",
        "completed",
        "completed_legacy",
    ] {
        eprintln!("crash boundary {role}");
        let (directory, mut config, submission, _) = seeded();
        config.role = role.into();
        let worker = Process::start(&config);
        assert_eq!(worker.ready(), role);
        drop(worker);
        let expected = if matches!(role, "prepared" | "admitted") {
            None
        } else {
            Some(b"1".to_vec())
        };
        assert_eq!(
            std::fs::read(directory.path().join("invocations")).ok(),
            expected
        );
        config.initialize = false;
        config.role = "worker".into();
        config.time += 121;
        let worker = Process::start(&config);
        assert_eq!(worker.ready(), "synced");
        drop(worker);
        assert_eq!(
            std::fs::read(directory.path().join("invocations")).ok(),
            expected,
            "reconciliation must never invoke inference"
        );
        let outbox = FileRelayOutbox::open(
            &directory.path().join("outbox.sqlite"),
            OutboxLimits::default(),
        )
        .unwrap();
        let entry = outbox.entry(&submission.binding.job_id).unwrap().unwrap();
        if matches!(role, "completed" | "completed_legacy") {
            assert_eq!(entry.manifest.unwrap().outcome, RecordedOutcome::Completed);
            assert!(
                entry
                    .packets
                    .iter()
                    .any(|p| matches!(p, OutboxPacket::Terminal { .. }))
            );
        } else {
            assert!(
                !entry
                    .packets
                    .iter()
                    .any(|p| matches!(p, OutboxPacket::Terminal { .. }))
            );
            if role == "delta" {
                assert!(!entry.packets.is_empty());
            }
            assert!(
                entry.abandoned
                    || entry
                        .manifest
                        .as_ref()
                        .is_some_and(|m| m.outcome == RecordedOutcome::Unresolved
                            || m.outcome == RecordedOutcome::Abandoned)
            );
        }
    }
}

mod authority_cases;
