//! Scratch-only gateway integration tests.
#![cfg(unix)]

use abi_wdbx_gateway::proto::wdbx_gateway_client::WdbxGatewayClient;
use abi_wdbx_gateway::proto::wdbx_gateway_server::WdbxGateway;
use abi_wdbx_gateway::proto::{
    GetKvRequest, KvEntry, MembershipAction, MembershipChangeRequest, PutKvRequest,
    PutVectorRequest, ResolveConflictRequest, SearchRequest, StatsRequest, VectorInput,
    WatchMutationsRequest,
};
use abi_wdbx_gateway::{
    BearerToken, EventHub, GatewayConfig, GatewayService, Limits, PreparedGateway, StoreExecutor,
    TlsFiles, validate_websocket_headers,
};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use futures_util::StreamExt as _;
use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair, KeyUsagePurpose,
};
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};
use tonic::{Code, Request};
use uuid::Uuid;

const TOKEN: &str = "gateway-test-token";
const ORIGIN: &str = "https://gateway.test";

struct Scratch {
    root: PathBuf,
    token: PathBuf,
}

impl Scratch {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!("abi-gateway-{label}-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let token = root.join("token");
        std::fs::write(&token, format!("{TOKEN}\n")).unwrap();
        std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
        Self { root, token }
    }

    fn store(&self) -> PathBuf {
        self.root.join("store")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

struct TestPki {
    ca_pem: String,
    client_cert_pem: String,
    client_key_pem: String,
    ca_der: Vec<u8>,
    client_cert_der: Vec<u8>,
    client_key_der: Vec<u8>,
    server_cert: PathBuf,
    server_key: PathBuf,
    client_ca: PathBuf,
}

impl TestPki {
    fn generate(scratch: &Scratch) -> Self {
        let ca_key = KeyPair::generate().unwrap();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
        ];
        let ca = ca_params.self_signed(&ca_key).unwrap();

        let server_key_pair = KeyPair::generate().unwrap();
        let mut server_params = CertificateParams::new(vec!["localhost".into()]).unwrap();
        server_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        let server = server_params
            .signed_by(&server_key_pair, &ca, &ca_key)
            .unwrap();

        let client_key_pair = KeyPair::generate().unwrap();
        let mut client_params = CertificateParams::new(vec!["gateway-client".into()]).unwrap();
        client_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
        let client = client_params
            .signed_by(&client_key_pair, &ca, &ca_key)
            .unwrap();

        let server_cert = scratch.root.join("server-cert.pem");
        let server_key = scratch.root.join("server-key.pem");
        let client_ca = scratch.root.join("client-ca.pem");
        std::fs::write(&server_cert, server.pem()).unwrap();
        std::fs::write(&server_key, server_key_pair.serialize_pem()).unwrap();
        std::fs::write(&client_ca, ca.pem()).unwrap();
        std::fs::set_permissions(&server_key, std::fs::Permissions::from_mode(0o600)).unwrap();
        Self {
            ca_pem: ca.pem(),
            client_cert_pem: client.pem(),
            client_key_pem: client_key_pair.serialize_pem(),
            ca_der: ca.der().to_vec(),
            client_cert_der: client.der().to_vec(),
            client_key_der: client_key_pair.serialize_der(),
            server_cert,
            server_key,
            client_ca,
        }
    }
}

fn websocket_client_tls(pki: &TestPki) -> rustls::ClientConfig {
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

    let mut roots = rustls::RootCertStore::empty();
    roots.add(CertificateDer::from(pki.ca_der.clone())).unwrap();
    rustls::ClientConfig::builder_with_provider(rustls::crypto::ring::default_provider().into())
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_client_auth_cert(
            vec![CertificateDer::from(pki.client_cert_der.clone())],
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pki.client_key_der.clone())),
        )
        .unwrap()
}

fn authenticated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    request
}

fn service(scratch: &Scratch, limits: Limits) -> (GatewayService, Arc<EventHub>) {
    let token = Arc::new(BearerToken::load(&scratch.token).unwrap());
    let executor = StoreExecutor::open(&scratch.store(), limits.blocking_jobs).unwrap();
    let events = Arc::new(EventHub::new(&limits));
    (
        GatewayService::new(token, limits, executor, Arc::clone(&events)),
        events,
    )
}

fn seed_conflict(scratch: &Scratch) {
    use abi_wdbx::{StorePaths, V2Mutation, VersionedStore};

    let mutation = |value: &str| V2Mutation::PutKv {
        key: "alpha".into(),
        value: value.into(),
    };
    let mut left = VersionedStore::open(StorePaths::new(scratch.root.join("left"))).unwrap();
    let left = left.commit_export(vec![mutation("left")]).unwrap();
    let mut right = VersionedStore::open(StorePaths::new(scratch.root.join("right"))).unwrap();
    let right = right.commit_export(vec![mutation("right")]).unwrap();
    let mut target = VersionedStore::open(StorePaths::new(scratch.store())).unwrap();
    target.import_committed(&left).unwrap();
    target.import_committed(&right).unwrap();
}

#[tokio::test]
async fn bearer_auth_and_all_eight_methods_use_a_scratch_store() {
    let scratch = Scratch::new("methods");
    seed_conflict(&scratch);
    let (service, _) = service(&scratch, Limits::default());
    let unauthorized = service
        .stats(Request::new(StatsRequest {}))
        .await
        .unwrap_err();
    assert_eq!(unauthorized.code(), Code::Unauthenticated);
    let mut duplicate_auth = authenticated(StatsRequest {});
    duplicate_auth
        .metadata_mut()
        .append("authorization", format!("Bearer {TOKEN}").parse().unwrap());
    let duplicate_auth = service.stats(duplicate_auth).await.unwrap_err();
    assert_eq!(duplicate_auth.code(), Code::Unauthenticated);

    let mut watch = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
        .unwrap()
        .into_inner();
    let vector = service
        .put_vector(authenticated(PutVectorRequest {
            vectors: vec![VectorInput {
                values: vec![1.0, 0.0],
            }],
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(vector.ids.len(), 1);
    assert_eq!(watch.next().await.unwrap().unwrap().kind, "put_vector");

    let search = service
        .search(authenticated(SearchRequest {
            query: vec![1.0, 0.0],
            limit: 5,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(search.hits[0].id, vector.ids[0]);
    let search_notice = watch.next().await.unwrap().unwrap();
    assert_eq!(search_notice.kind, "search");
    assert_eq!(search_notice.transaction_id, "");

    service
        .put_kv(authenticated(PutKvRequest {
            entries: vec![KvEntry {
                key: "beta".into(),
                value: "private-value".into(),
            }],
        }))
        .await
        .unwrap();
    let current = service
        .get_kv(authenticated(GetKvRequest {
            key: "alpha".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(current.found);
    assert_eq!(current.version_ids.len(), 2);

    service
        .resolve_conflict(authenticated(ResolveConflictRequest {
            key: "alpha".into(),
            version_ids: current.version_ids,
            value: "resolved-value".into(),
        }))
        .await
        .unwrap();
    let resolved = service
        .get_kv(authenticated(GetKvRequest {
            key: "alpha".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resolved.value, "resolved-value");

    let membership = service
        .membership_change(authenticated(MembershipChangeRequest {
            action: MembershipAction::Add as i32,
            node_id: "node-a".into(),
        }))
        .await
        .unwrap()
        .into_inner();
    assert!(membership.changed);

    let stats = service
        .stats(authenticated(StatsRequest {}))
        .await
        .unwrap()
        .into_inner();
    assert_eq!((stats.vectors, stats.kv_entries, stats.members), (1, 2, 1));
}

#[tokio::test]
async fn request_vector_key_value_rate_and_stream_limits_fail_before_dispatch() {
    let scratch = Scratch::new("limits");
    let limits = Limits {
        message_bytes: 20_000,
        batch_items: 1,
        key_bytes: 4,
        value_bytes: 4,
        requests_per_second: 10,
        concurrent_streams: 1,
        ..Limits::default()
    };
    let (service, _) = service(&scratch, limits);

    let dimensions = service
        .put_vector(authenticated(PutVectorRequest {
            vectors: vec![VectorInput {
                values: vec![0.0; 4097],
            }],
        }))
        .await
        .unwrap_err();
    assert_eq!(dimensions.code(), Code::InvalidArgument);
    let key = service
        .get_kv(authenticated(GetKvRequest {
            key: "12345".into(),
        }))
        .await
        .unwrap_err();
    assert_eq!(key.code(), Code::InvalidArgument);
    let value = service
        .put_kv(authenticated(PutKvRequest {
            entries: vec![KvEntry {
                key: "key".into(),
                value: "12345".into(),
            }],
        }))
        .await
        .unwrap_err();
    assert_eq!(value.code(), Code::InvalidArgument);
    let batch = service
        .put_kv(authenticated(PutKvRequest {
            entries: vec![
                KvEntry {
                    key: "one".into(),
                    value: "1".into(),
                },
                KvEntry {
                    key: "two".into(),
                    value: "2".into(),
                },
            ],
        }))
        .await
        .unwrap_err();
    assert_eq!(batch.code(), Code::InvalidArgument);
    let search_limit = service
        .search(authenticated(SearchRequest {
            query: vec![1.0],
            limit: 1_001,
        }))
        .await
        .unwrap_err();
    assert_eq!(search_limit.code(), Code::InvalidArgument);

    let first = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
        .unwrap();
    let Err(second) = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
    else {
        panic!("second stream unexpectedly admitted");
    };
    assert_eq!(second.code(), Code::ResourceExhausted);
    drop(first);
}

#[tokio::test]
async fn encoded_message_and_idle_stream_limits_are_enforced() {
    let scratch = Scratch::new("message-idle");
    let limits = Limits {
        message_bytes: 8,
        idle_seconds: 1,
        ..Limits::default()
    };
    let (service, _) = service(&scratch, limits);
    let oversized = service
        .put_kv(authenticated(PutKvRequest {
            entries: vec![KvEntry {
                key: "key".into(),
                value: "long-value".into(),
            }],
        }))
        .await
        .unwrap_err();
    assert_eq!(oversized.code(), Code::ResourceExhausted);
    let mut watch = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
        .unwrap()
        .into_inner();
    let idle = watch.next().await.unwrap().unwrap_err();
    assert_eq!(idle.code(), Code::DeadlineExceeded);
}

#[tokio::test]
async fn failed_authentication_attempts_consume_the_rate_budget() {
    let scratch = Scratch::new("auth-rate");
    let limits = Limits {
        requests_per_second: 2,
        ..Limits::default()
    };
    let (service, _) = service(&scratch, limits);
    for _ in 0..2 {
        let error = service
            .stats(Request::new(StatsRequest {}))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::Unauthenticated);
    }
    let limited = service
        .stats(authenticated(StatsRequest {}))
        .await
        .unwrap_err();
    assert_eq!(limited.code(), Code::ResourceExhausted);
}

#[tokio::test]
async fn lagged_mutation_watch_is_terminated() {
    let scratch = Scratch::new("lag");
    let limits = Limits {
        event_queue: 1,
        ..Limits::default()
    };
    let (service, events) = service(&scratch, limits);
    let mut watch = service
        .watch_mutations(authenticated(WatchMutationsRequest {}))
        .await
        .unwrap()
        .into_inner();
    events.publish("put_kv", "tx-1", 1);
    events.publish("put_vector", "tx-2", 1);
    let error = watch.next().await.unwrap().unwrap_err();
    assert_eq!(error.code(), Code::ResourceExhausted);
}

#[tokio::test]
async fn websocket_rejects_missing_or_hostile_origin_and_bad_auth() {
    let scratch = Scratch::new("origin");
    let token = BearerToken::load(&scratch.token).unwrap();
    let origins = vec![ORIGIN.into()];
    assert_eq!(
        validate_websocket_headers(&token, &origins, &ws_headers(None, Some(TOKEN))),
        Err(StatusCode::FORBIDDEN)
    );
    assert_eq!(
        validate_websocket_headers(
            &token,
            &origins,
            &ws_headers(Some("https://evil.test"), Some(TOKEN)),
        ),
        Err(StatusCode::FORBIDDEN)
    );
    assert_eq!(
        validate_websocket_headers(&token, &origins, &ws_headers(Some(ORIGIN), Some("wrong"))),
        Err(StatusCode::UNAUTHORIZED)
    );
    assert!(
        validate_websocket_headers(&token, &origins, &ws_headers(Some(ORIGIN), Some(TOKEN)))
            .is_ok()
    );
    let mut duplicate = ws_headers(Some(ORIGIN), Some(TOKEN));
    duplicate.append("origin", HeaderValue::from_static("https://evil.test"));
    assert_eq!(
        validate_websocket_headers(&token, &origins, &duplicate),
        Err(StatusCode::FORBIDDEN)
    );
    let mut duplicate_auth = ws_headers(Some(ORIGIN), Some(TOKEN));
    duplicate_auth.append(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {TOKEN}")).unwrap(),
    );
    assert_eq!(
        validate_websocket_headers(&token, &origins, &duplicate_auth),
        Err(StatusCode::UNAUTHORIZED)
    );
    assert_eq!(
        validate_websocket_headers(&token, &origins, &ws_headers(Some("null"), Some(TOKEN)),),
        Err(StatusCode::FORBIDDEN)
    );
}

fn ws_headers(origin: Option<&str>, token: Option<&str>) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Some(origin) = origin {
        headers.insert("origin", HeaderValue::from_str(origin).unwrap());
    }
    if let Some(token) = token {
        headers.insert(
            "authorization",
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        );
    }
    headers
}

fn free_loopback() -> std::net::SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap()
}

#[tokio::test]
async fn real_grpc_and_websocket_networks_use_two_explicit_listeners() {
    let scratch = Scratch::new("network");
    let mut config = GatewayConfig::loopback(scratch.store(), &scratch.token);
    config.grpc_addr = free_loopback();
    config.events_addr = free_loopback();
    config.allowed_origins = vec![ORIGIN.into()];
    assert_ne!(config.grpc_addr, config.events_addr);
    let expected = (config.grpc_addr, config.events_addr);
    let gateway = PreparedGateway::prepare(config).await.unwrap();
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(gateway.serve(async move {
        let _ = shutdown_receiver.await;
    }));

    let endpoint = format!("http://{}", expected.0);
    let mut client = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            match WdbxGatewayClient::connect(endpoint.clone()).await {
                Ok(client) => break client,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        }
    })
    .await
    .expect("gRPC listener readiness deadline");
    let mut request = format!("ws://{}/v1/events", expected.1)
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", HeaderValue::from_static(ORIGIN));
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {TOKEN}")).unwrap(),
    );
    let (mut events, _) = tokio_tungstenite::connect_async(request).await.unwrap();
    client
        .put_kv(authenticated(PutKvRequest {
            entries: vec![KvEntry {
                key: "net".into(),
                value: "secret-not-in-event".into(),
            }],
        }))
        .await
        .unwrap();
    let message = tokio::time::timeout(std::time::Duration::from_secs(2), events.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let text = message.into_text().unwrap();
    assert!(text.contains("put_kv"));
    assert!(!text.contains("secret-not-in-event"));

    shutdown_sender.send(()).unwrap();
    let addresses = server.await.unwrap().unwrap();
    assert_eq!((addresses.grpc, addresses.events), expected);
}

#[tokio::test]
async fn failed_websocket_upgrades_consume_the_listener_rate_budget() {
    use tokio_tungstenite::tungstenite::Error as WebSocketError;

    let scratch = Scratch::new("ws-rate");
    let mut config = GatewayConfig::loopback(scratch.store(), &scratch.token);
    config.grpc_addr = free_loopback();
    config.events_addr = free_loopback();
    config.allowed_origins = vec![ORIGIN.into()];
    config.limits.requests_per_second = 2;
    let addresses = (config.grpc_addr, config.events_addr);
    let gateway = PreparedGateway::prepare(config).await.unwrap();
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(gateway.serve(async move {
        let _ = shutdown_receiver.await;
    }));
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if WdbxGatewayClient::connect(format!("http://{}", addresses.0))
                .await
                .is_ok()
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("gateway readiness deadline");

    for _ in 0..2 {
        let request = websocket_network_request(addresses.1, "wrong");
        let error = tokio_tungstenite::connect_async(request).await.unwrap_err();
        assert!(matches!(
            error,
            WebSocketError::Http(ref response)
                if response.status() == StatusCode::UNAUTHORIZED
        ));
    }
    let limited = tokio_tungstenite::connect_async(websocket_network_request(addresses.1, TOKEN))
        .await
        .unwrap_err();
    assert!(matches!(
        limited,
        WebSocketError::Http(ref response)
            if response.status() == StatusCode::TOO_MANY_REQUESTS
    ));
    shutdown_sender.send(()).unwrap();
    server.await.unwrap().unwrap();
}

fn websocket_network_request(
    address: std::net::SocketAddr,
    token: &str,
) -> tokio_tungstenite::tungstenite::http::Request<()> {
    let mut request = format!("ws://{address}/v1/events")
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", HeaderValue::from_static(ORIGIN));
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
    );
    request
}

#[tokio::test]
async fn real_tls_and_mtls_handshakes_cover_grpc_and_websocket_listeners() {
    use tokio_tungstenite::Connector;

    let scratch = Scratch::new("mtls");
    let pki = TestPki::generate(&scratch);
    let mut config = GatewayConfig::loopback(scratch.store(), &scratch.token);
    config.grpc_addr = free_loopback();
    config.events_addr = free_loopback();
    config.allowed_origins = vec![ORIGIN.into()];
    config.tls = TlsFiles {
        certificate: Some(pki.server_cert.clone()),
        private_key: Some(pki.server_key.clone()),
        client_ca: Some(pki.client_ca.clone()),
    };
    let addresses = (config.grpc_addr, config.events_addr);
    let gateway = PreparedGateway::prepare(config).await.unwrap();
    let (shutdown_sender, shutdown_receiver) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(gateway.serve(async move {
        let _ = shutdown_receiver.await;
    }));

    let endpoint = Endpoint::from_shared(format!("https://localhost:{}", addresses.0.port()))
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
    let channel = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            match endpoint.clone().connect().await {
                Ok(channel) => break channel,
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(20)).await,
            }
        }
    })
    .await
    .expect("mTLS gRPC readiness deadline");
    let mut client = WdbxGatewayClient::new(channel);
    client.stats(authenticated(StatsRequest {})).await.unwrap();

    let no_identity = Endpoint::from_shared(format!("https://localhost:{}", addresses.0.port()))
        .unwrap()
        .tls_config(
            ClientTlsConfig::new()
                .domain_name("localhost")
                .ca_certificate(Certificate::from_pem(pki.ca_pem.clone())),
        )
        .unwrap()
        .connect()
        .await;
    if let Ok(channel) = no_identity {
        let rejected = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            WdbxGatewayClient::new(channel).stats(authenticated(StatsRequest {})),
        )
        .await
        .expect("no-identity mTLS refusal deadline");
        assert!(rejected.is_err());
    }

    let websocket_tls = websocket_client_tls(&pki);
    let mut request = format!("wss://localhost:{}/v1/events", addresses.1.port())
        .into_client_request()
        .unwrap();
    request
        .headers_mut()
        .insert("origin", HeaderValue::from_static(ORIGIN));
    request.headers_mut().insert(
        "authorization",
        HeaderValue::from_str(&format!("Bearer {TOKEN}")).unwrap(),
    );
    let (mut websocket, _) = tokio_tungstenite::connect_async_tls_with_config(
        request,
        None,
        false,
        Some(Connector::Rustls(Arc::new(websocket_tls))),
    )
    .await
    .unwrap();
    client
        .put_kv(authenticated(PutKvRequest {
            entries: vec![KvEntry {
                key: "tls".into(),
                value: "not-in-event".into(),
            }],
        }))
        .await
        .unwrap();
    let event = tokio::time::timeout(std::time::Duration::from_secs(2), websocket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap()
        .into_text()
        .unwrap();
    assert!(event.contains("put_kv"));
    assert!(!event.contains("not-in-event"));

    shutdown_sender.send(()).unwrap();
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn non_loopback_partial_tls_symlink_and_insecure_secrets_fail_closed() {
    let scratch = Scratch::new("prebind");
    let mut config = GatewayConfig::loopback(scratch.store(), &scratch.token);
    config.grpc_addr = "0.0.0.0:0".parse().unwrap();
    let error = PreparedGateway::prepare(config.clone()).await.unwrap_err();
    assert!(error.to_string().contains("non-loopback"));

    config.tls = TlsFiles {
        certificate: Some(scratch.root.join("cert.pem")),
        private_key: None,
        client_ca: None,
    };
    let error = PreparedGateway::prepare(config).await.unwrap_err();
    assert!(error.to_string().contains("configured together"));

    std::fs::set_permissions(&scratch.token, std::fs::Permissions::from_mode(0o644)).unwrap();
    let error = BearerToken::load(&scratch.token).unwrap_err();
    assert!(error.to_string().contains("group/other"));

    std::fs::set_permissions(&scratch.token, std::fs::Permissions::from_mode(0o600)).unwrap();
    let link = scratch.root.join("token-link");
    std::os::unix::fs::symlink(&scratch.token, &link).unwrap();
    let error = BearerToken::load(Path::new(&link)).unwrap_err();
    assert!(error.to_string().contains("bearer token"));
    assert!(!format!("{:?}", BearerToken::load(&scratch.token).unwrap()).contains(TOKEN));

    let mut invalid_origins =
        GatewayConfig::loopback(scratch.store().join("origins"), &scratch.token);
    invalid_origins.allowed_origins = vec![ORIGIN.into(), ORIGIN.into()];
    assert!(invalid_origins.validate_pre_bind().is_err());
    invalid_origins.allowed_origins = vec!["null".into()];
    assert!(invalid_origins.validate_pre_bind().is_err());

    let mut same_listener = GatewayConfig::loopback(scratch.store().join("same"), &scratch.token);
    same_listener.events_addr = same_listener.grpc_addr;
    assert!(same_listener.validate_pre_bind().is_err());
}
