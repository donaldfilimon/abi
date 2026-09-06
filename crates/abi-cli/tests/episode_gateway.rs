//! End-to-end: `abi wdbx episode` against a real gateway on loopback.
//!
//! The gateway runs in this test process with a scratch store, a bearer token
//! file, and a JSON `StorePolicy`; the CLI is spawned as a separate process
//! and reaches it only over gRPC.
#![cfg(unix)]

use abi_wdbx_gateway::proto::wdbx_gateway_client::WdbxGatewayClient;
use abi_wdbx_gateway::{GatewayConfig, PreparedGateway, TlsFiles};
use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair, KeyUsagePurpose,
};
use std::collections::BTreeMap;
use std::net::{SocketAddr, TcpListener};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Duration;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};

const TOKEN: &str = "episode-cli-test-token";

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "abi-cli-episode-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn free_loopback() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

fn write_private(path: &Path, bytes: &[u8]) {
    std::fs::write(path, bytes).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
}

fn policy_json() -> Vec<u8> {
    use abi_wdbx::v3::episode::{GuildEpisodePolicy, StorePolicy};

    serde_json::to_vec(&StorePolicy {
        contract_revision: 2,
        contract_digest: [9; 32],
        guilds: BTreeMap::from([(
            "guild_ref".to_owned(),
            GuildEpisodePolicy {
                learning_enabled: true,
                policy_version: "policy_v1".into(),
                token_budget: 1_000,
                storage_budget_bytes: 1 << 20,
                current_consent_epoch: None,
            },
        )]),
    })
    .unwrap()
}

fn write_json(request_id: &str) -> Vec<u8> {
    use abi_wdbx::v3::episode::{
        ActorKind, ActorRef, EpisodeEvent, EpisodeSource, EpisodeWrite, EvidenceLevel,
    };

    serde_json::to_vec(&EpisodeWrite {
        request_id: request_id.into(),
        operation_id: "op_cli_1".into(),
        contract_revision: 2,
        contract_digest: [9; 32],
        guild_ref: "guild_ref".into(),
        consent_epoch: None,
        source_type: EpisodeSource::Proposal,
        policy_version: "policy_v1".into(),
        evidence_level: EvidenceLevel::C1,
        event: EpisodeEvent::Proposal {
            requested_by: ActorRef {
                principal_id: "requester_ref".into(),
                kind: ActorKind::HumanSubject,
            },
            proposed_by: ActorRef {
                principal_id: "abbey_service".into(),
                kind: ActorKind::Service,
            },
        },
        token_cost: 3,
        expected_commitment: None,
        quiet: false,
    })
    .unwrap()
}

fn abi(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_abi"))
        .env("ABI_WDBX_PATH", ":memory:")
        .env_remove("ABI_WDBX_GATEWAY_ENDPOINT")
        .env_remove("ABI_WDBX_GATEWAY_TOKEN_FILE")
        .args(arguments)
        .output()
        .expect("abi executable runs")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn field<'a>(line: &'a str, key: &str) -> &'a str {
    line.split_whitespace()
        .find_map(|pair| {
            pair.strip_prefix(key)
                .and_then(|rest| rest.strip_prefix('='))
        })
        .unwrap_or_else(|| panic!("{key} missing in {line:?}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_proposes_verifies_and_reports_rejections_through_a_live_gateway() {
    let scratch = Scratch::new();
    let token_file = scratch.path("token");
    write_private(&token_file, format!("{TOKEN}\n").as_bytes());
    let policy_file = scratch.path("policy.json");
    std::fs::write(&policy_file, policy_json()).unwrap();
    let write_file = scratch.path("write.json");
    std::fs::write(&write_file, write_json("req_cli_1")).unwrap();

    let mut config = GatewayConfig::loopback(scratch.path("store"), &token_file);
    config.grpc_addr = free_loopback();
    config.events_addr = free_loopback();
    config.episode_policy = Some(policy_file);
    let endpoint = format!("http://{}", config.grpc_addr);
    let gateway = PreparedGateway::prepare(config).await.unwrap();
    let (shutdown, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(gateway.serve(async move {
        let _ = shutdown_receiver.await;
    }));
    tokio::time::timeout(Duration::from_secs(5), async {
        while WdbxGatewayClient::connect(endpoint.clone()).await.is_err() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("gateway listener readiness");

    let token_arg = token_file.to_str().unwrap().to_owned();
    let write_arg = write_file.to_str().unwrap().to_owned();
    let endpoint_clone = endpoint.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let common = [
            "--endpoint",
            endpoint_clone.as_str(),
            "--token-file",
            token_arg.as_str(),
        ];
        let run = |head: &[&str]| {
            let mut arguments: Vec<&str> = head.to_vec();
            arguments.extend_from_slice(&common);
            abi(&arguments)
        };

        let preview = run(&["wdbx", "episode", "propose", &write_arg, "--preview"]);
        assert!(preview.status.success(), "{}", text(&preview.stderr));
        let preview_line = text(&preview.stdout);
        assert_eq!(field(&preview_line, "decision"), "preview");
        let digest = field(&preview_line, "episode_digest").to_owned();
        assert_eq!(digest.len(), 64);

        let appended = run(&["wdbx", "episode", "propose", &write_arg]);
        assert!(appended.status.success(), "{}", text(&appended.stderr));
        let appended_line = text(&appended.stdout);
        assert_eq!(field(&appended_line, "decision"), "appended");
        assert_eq!(field(&appended_line, "episode_digest"), digest);
        assert_eq!(field(&appended_line, "event_kind"), "proposal");
        assert_eq!(field(&appended_line, "previous_digest"), "none");

        let replay = run(&["wdbx", "episode", "propose", &write_arg, "--json"]);
        assert_eq!(replay.status.code(), Some(1));
        assert!(text(&replay.stderr).contains("AlreadyExists: episode_replay"));

        let verified = run(&["wdbx", "episode", "verify", "guild_ref", &digest, "--json"]);
        assert!(verified.status.success(), "{}", text(&verified.stderr));
        let json: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
        assert_eq!(json["found"], "true");
        assert_eq!(json["request_id"], "req_cli_1");

        let missing = run(&["wdbx", "episode", "verify", "guild_ref", &"0".repeat(64)]);
        assert_eq!(missing.status.code(), Some(1));
        assert!(text(&missing.stdout).starts_with("found=false"));
        assert!(text(&missing.stderr).contains("not found"));

        let wrong_token = scratch.path("wrong-token");
        write_private(&wrong_token, b"not-the-token\n");
        let denied = abi(&[
            "wdbx",
            "episode",
            "verify",
            "guild_ref",
            &digest,
            "--endpoint",
            &endpoint_clone,
            "--token-file",
            wrong_token.to_str().unwrap(),
        ]);
        assert_eq!(denied.status.code(), Some(1));
        assert!(text(&denied.stderr).contains("Unauthenticated"));
        scratch
    })
    .await
    .unwrap();

    let _ = shutdown.send(());
    server.await.unwrap().unwrap();
    drop(outcome);
}

struct Pki {
    ca: PathBuf,
    server_cert: PathBuf,
    server_key: PathBuf,
    client_cert: PathBuf,
    client_key: PathBuf,
    ca_pem: String,
    client_cert_pem: String,
    client_key_pem: String,
}

/// One CA signing a `localhost` server certificate and a client certificate,
/// the same shape the gateway's own mTLS test uses.
fn generate_pki(scratch: &Scratch) -> Pki {
    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params.key_usages = vec![
        KeyUsagePurpose::DigitalSignature,
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
    ];
    let ca = ca_params.self_signed(&ca_key).unwrap();
    let server_key = KeyPair::generate().unwrap();
    let mut server_params = CertificateParams::new(vec!["localhost".into()]).unwrap();
    server_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let server = server_params.signed_by(&server_key, &ca, &ca_key).unwrap();
    let client_key = KeyPair::generate().unwrap();
    let mut client_params = CertificateParams::new(vec!["episode-cli".into()]).unwrap();
    client_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
    let client = client_params.signed_by(&client_key, &ca, &ca_key).unwrap();

    let pki = Pki {
        ca: scratch.path("ca.pem"),
        server_cert: scratch.path("server-cert.pem"),
        server_key: scratch.path("server-key.pem"),
        client_cert: scratch.path("client-cert.pem"),
        client_key: scratch.path("client-key.pem"),
        ca_pem: ca.pem(),
        client_cert_pem: client.pem(),
        client_key_pem: client_key.serialize_pem(),
    };
    std::fs::write(&pki.ca, &pki.ca_pem).unwrap();
    std::fs::write(&pki.server_cert, server.pem()).unwrap();
    write_private(&pki.server_key, server_key.serialize_pem().as_bytes());
    std::fs::write(&pki.client_cert, &pki.client_cert_pem).unwrap();
    write_private(&pki.client_key, pki.client_key_pem.as_bytes());
    pki
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_presents_a_client_identity_to_an_mtls_gateway() {
    let scratch = Scratch::new();
    let pki = generate_pki(&scratch);
    let token_file = scratch.path("token");
    write_private(&token_file, format!("{TOKEN}\n").as_bytes());
    let policy_file = scratch.path("policy.json");
    std::fs::write(&policy_file, policy_json()).unwrap();
    let write_file = scratch.path("write.json");
    std::fs::write(&write_file, write_json("req_cli_tls_1")).unwrap();

    let mut config = GatewayConfig::loopback(scratch.path("store"), &token_file);
    config.grpc_addr = free_loopback();
    config.events_addr = free_loopback();
    config.episode_policy = Some(policy_file);
    config.tls = TlsFiles {
        certificate: Some(pki.server_cert.clone()),
        private_key: Some(pki.server_key.clone()),
        client_ca: Some(pki.ca.clone()),
    };
    let endpoint = format!("https://localhost:{}", config.grpc_addr.port());
    let gateway = PreparedGateway::prepare(config).await.unwrap();
    let (shutdown, shutdown_receiver) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(gateway.serve(async move {
        let _ = shutdown_receiver.await;
    }));
    let probe = Endpoint::from_shared(endpoint.clone())
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .domain_name("localhost")
                .ca_certificate(Certificate::from_pem(pki.ca_pem.clone()))
                .identity(Identity::from_pem(
                    pki.client_cert_pem.clone(),
                    pki.client_key_pem.clone(),
                )),
        )
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while probe.clone().connect().await.is_err() {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("mTLS gateway listener readiness");

    let outcome = tokio::task::spawn_blocking(move || {
        let path = |p: &Path| p.to_str().unwrap().to_owned();
        let (token_arg, write_arg) = (path(&token_file), path(&write_file));
        let (ca_arg, cert_arg, key_arg) =
            (path(&pki.ca), path(&pki.client_cert), path(&pki.client_key));
        let common = [
            "--endpoint",
            endpoint.as_str(),
            "--token-file",
            token_arg.as_str(),
            "--ca-cert",
            ca_arg.as_str(),
        ];
        let run = |head: &[&str], tail: &[&str]| {
            let mut arguments: Vec<&str> = head.to_vec();
            arguments.extend_from_slice(&common);
            arguments.extend_from_slice(tail);
            abi(&arguments)
        };

        // With the identity: appended over mTLS.
        let appended = run(
            &["wdbx", "episode", "propose", &write_arg],
            &["--client-cert", &cert_arg, "--client-key", &key_arg],
        );
        assert!(appended.status.success(), "{}", text(&appended.stderr));
        let appended_line = text(&appended.stdout);
        assert_eq!(field(&appended_line, "decision"), "appended");
        let digest = field(&appended_line, "episode_digest").to_owned();

        let verified = run(
            &["wdbx", "episode", "verify", "guild_ref", &digest],
            &["--client-cert", &cert_arg, "--client-key", &key_arg],
        );
        assert!(verified.status.success(), "{}", text(&verified.stderr));
        assert!(text(&verified.stdout).starts_with("found=true"));

        // Without the identity the gateway refuses the handshake or the call;
        // either way the CLI exits 1 and never reports a receipt.
        let anonymous = run(&["wdbx", "episode", "verify", "guild_ref", &digest], &[]);
        assert_eq!(
            anonymous.status.code(),
            Some(1),
            "{}",
            text(&anonymous.stdout)
        );
        assert!(!text(&anonymous.stdout).contains("found=true"));

        // Half an identity is a usage error, before any network activity.
        let half = run(
            &["wdbx", "episode", "verify", "guild_ref", &digest],
            &["--client-cert", &cert_arg],
        );
        assert_eq!(half.status.code(), Some(2));
        assert!(text(&half.stderr).contains("must be given together"));
        scratch
    })
    .await
    .unwrap();

    let _ = shutdown.send(());
    server.await.unwrap().unwrap();
    drop(outcome);
}
