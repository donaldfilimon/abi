use super::*;
use crate::StorePaths;
use std::io::{Read as _, Write as _};
use std::net::Shutdown;
use std::path::PathBuf;
use std::thread;

struct Fixture {
    dir: PathBuf,
    paths: StorePaths,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let dir = abi_foundation::temp_path::temp_file_path(name, "store");
        std::fs::create_dir_all(&dir).expect("fixture directory");
        Self {
            paths: StorePaths {
                dir: dir.clone(),
                base: "rest".to_string(),
            },
            dir,
        }
    }

    fn open(&self) -> VersionedStore {
        VersionedStore::open(self.paths.clone()).expect("open store")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn pure_routes_cover_health_stats_and_unknown_paths() {
    let fixture = Fixture::new("abi_rest_pure_basics");
    let mut store = fixture.open();

    let health = route(&mut store, "GET", "/health", b"", 1_000);
    assert_eq!(health.status, 200);
    assert_eq!(health.body, r#"{"status":"ok"}"#);

    let stats = route(&mut store, "GET", "/stats", b"", 1_000);
    let stats_json: Value = serde_json::from_str(&stats.body).expect("stats json");
    assert_eq!(stats_json["vectors"], 0);
    assert_eq!(stats_json["backend"], "cpu");
    assert_eq!(stats_json["mode"], "fallback");

    let missing = route(&mut store, "DELETE", "/unknown", b"", 1_000);
    assert_eq!(missing.status, 404);
    assert!(missing.body.contains("no route for DELETE /unknown"));
}

#[test]
fn insert_and_query_kv_survive_reopen_and_escape_json() {
    let fixture = Fixture::new("abi_rest_kv");
    {
        let mut store = fixture.open();
        let inserted = route(
            &mut store,
            "POST",
            "/insert",
            br#"{"key":"agent:abbey","value":"trained \"locally\""}"#,
            1_000,
        );
        assert_eq!(inserted.status, 200);
    }

    let mut store = fixture.open();
    let queried = route(
        &mut store,
        "POST",
        "/query",
        br#"{"key":"agent:abbey"}"#,
        1_000,
    );
    assert_eq!(queried.status, 200);
    let value: Value = serde_json::from_str(&queried.body).expect("query json");
    assert_eq!(value["value"], r#"trained "locally""#);

    let missing = route(&mut store, "POST", "/query", br#"{"key":"missing"}"#, 1_000);
    assert_eq!(missing.status, 404);
}

#[test]
fn vector_query_applies_temporal_causal_and_persona_components() {
    let fixture = Fixture::new("abi_rest_hybrid");
    let mut store = fixture.open();
    let mut ids = Vec::new();
    for _ in 0..2 {
        let response = route(
            &mut store,
            "POST",
            "/insert",
            br#"{"vector":[1.0,0.0,0.0,0.0]}"#,
            1_000,
        );
        assert_eq!(response.status, 200);
        let value: Value = serde_json::from_str(&response.body).expect("insert json");
        ids.push(
            serde_json::from_value::<RecordId>(value["id"].clone()).expect("versioned vector id"),
        );
    }
    store
        .add_temporal_node(ids[0], 1_000 - 24 * 60 * 60 * 1_000)
        .expect("one-day-old node");
    store.add_temporal_node(ids[1], 1_000).expect("new node");
    store
        .add_temporal_edge(ids[0], ids[1])
        .expect("causal edge");

    let response = route(
        &mut store,
        "POST",
        "/query",
        br#"{"vector":[1.0,0.0,0.0,0.0],"limit":2}"#,
        1_000,
    );
    assert_eq!(response.status, 200);
    let value: Value = serde_json::from_str(&response.body).expect("hybrid json");
    assert_eq!(value["ranking"], "hybrid");
    assert_eq!(value["vectors"], 2);
    assert_eq!(
        value["results"][0]["id"],
        serde_json::to_value(ids[1]).unwrap()
    );
    assert_eq!(value["results"][0]["persona"], 0.5);
    assert_eq!(value["results"][0]["causal"], 1.0);
    assert_eq!(value["results"][1]["causal"], 0.6);
    assert_eq!(value["results"][1]["temporal"], 0.5);
}

#[test]
fn insert_block_and_verify_recompute_the_chain() {
    let fixture = Fixture::new("abi_rest_verify");
    let mut store = fixture.open();
    let inserted = route(
        &mut store,
        "POST",
        "/insert",
        br#"{"profile":"abbey","metadata":"local"}"#,
        1_000,
    );
    assert_eq!(inserted.status, 200);
    let verified = route(&mut store, "POST", "/verify", b"{}", 1_000);
    let value: Value = serde_json::from_str(&verified.body).expect("verify json");
    assert_eq!(value["chain_valid"], true);
    assert_eq!(value["blocks"], 1);
}

#[test]
fn malformed_route_inputs_preserve_oracle_errors() {
    let fixture = Fixture::new("abi_rest_errors");
    let mut store = fixture.open();
    let cases: &[(&[u8], &str)] = &[
        (b"{", "invalid json"),
        (b"[]", "expected object"),
        (br#"{"vector":"no"}"#, "vector must be an array"),
        (br#"{"vector":[]}"#, "vector must be non-empty"),
        (br#"{"vector":[1,true]}"#, "vector elements must be numbers"),
    ];
    for (body, message) in cases {
        let response = route(&mut store, "POST", "/insert", body, 1_000);
        assert_eq!(response.status, 400);
        assert!(
            response.body.contains(message),
            "{body:?}: {}",
            response.body
        );
    }

    let empty_query = route(&mut store, "POST", "/query", br#"{"vector":[1,0]}"#, 1_000);
    assert_eq!(empty_query.body, r#"{"results":[],"vectors":0}"#);

    for (limit, message) in [
        ("0", "limit must be between 1 and 100"),
        ("101", "limit must be between 1 and 100"),
        ("1.5", "limit must be an integer"),
        (r#""two""#, "limit must be an integer"),
    ] {
        let body = format!(r#"{{"vector":[1,0],"limit":{limit}}}"#);
        let response = route(&mut store, "POST", "/query", body.as_bytes(), 1_000);
        assert_eq!(response.status, 400);
        assert!(response.body.contains(message));
    }
}

#[test]
fn real_tcp_auth_accepts_only_the_configured_bearer() {
    let fixture = Fixture::new("abi_rest_tcp_auth");
    let config = RestConfig {
        bearer_token: Some("secret".to_string()),
        rate_limiter: RateLimiter::new(3, 0, 0),
    };
    let mut server = RestServer::bind(0, fixture.open(), config).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        for _ in 0..3 {
            server.serve_one().expect("serve request");
        }
        server
    });

    let missing = exchange(port, &[b"GET /health HTTP/1.1\r\n\r\n"]);
    assert!(missing.starts_with("HTTP/1.1 401 Unauthorized"));
    assert!(missing.contains("WWW-Authenticate: Bearer"));
    let wrong = exchange(
        port,
        &[b"GET /health HTTP/1.1\r\nAuthorization: Bearer wrong\r\n\r\n"],
    );
    assert!(wrong.starts_with("HTTP/1.1 401 Unauthorized"));
    let valid = exchange(
        port,
        &[b"GET /health HTTP/1.1\r\nAuthorization: Bearer secret\r\n\r\n"],
    );
    assert!(valid.starts_with("HTTP/1.1 200 OK"));
    assert!(valid.ends_with(r#"{"status":"ok"}"#));

    let server = handle.join().expect("server thread");
    assert_eq!(server.config.rate_limiter.stats().allowed, 3);
}

#[test]
fn failed_auth_consumes_the_bucket_before_a_valid_request() {
    let fixture = Fixture::new("abi_rest_tcp_rate_auth");
    let config = RestConfig {
        bearer_token: Some("secret".to_string()),
        rate_limiter: RateLimiter::new(1, 0, 0),
    };
    let mut server = RestServer::bind(0, fixture.open(), config).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        for _ in 0..2 {
            server.serve_one().expect("serve request");
        }
        server
    });

    let wrong = exchange(
        port,
        &[b"GET /health HTTP/1.1\r\nAuthorization: Bearer wrong\r\n\r\n"],
    );
    assert!(wrong.starts_with("HTTP/1.1 401 Unauthorized"));
    let limited = exchange(
        port,
        &[b"GET /health HTTP/1.1\r\nAuthorization: Bearer secret\r\n\r\n"],
    );
    assert!(limited.starts_with("HTTP/1.1 429 Too Many Requests"));
    assert!(limited.contains("Retry-After: 1"));

    let server = handle.join().expect("server thread");
    let stats = server.config.rate_limiter.stats();
    assert_eq!(stats.allowed, 1);
    assert_eq!(stats.denied, 1);
}

#[test]
fn real_tcp_reassembles_body_and_embeds_rate_stats() {
    let fixture = Fixture::new("abi_rest_tcp_body");
    let config = RestConfig {
        bearer_token: None,
        rate_limiter: RateLimiter::new(5, 0, 0),
    };
    let mut server = RestServer::bind(0, fixture.open(), config).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        for _ in 0..2 {
            server.serve_one().expect("serve request");
        }
        server
    });

    let body = br#"{"key":"agent:abbey","value":"trained"}"#;
    let header = format!(
        "POST /insert HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let inserted = exchange(port, &[header.as_bytes(), &body[..10], &body[10..]]);
    assert!(inserted.starts_with("HTTP/1.1 200 OK"));

    let stats_response = exchange(port, &[b"GET /stats HTTP/1.1\r\n\r\n"]);
    let stats_body = find_body(stats_response.as_bytes()).expect("stats body");
    let stats: Value = serde_json::from_slice(stats_body).expect("stats json");
    assert_eq!(stats["kv_entries"], 1);
    assert_eq!(stats["rate_limit"]["capacity"], 5);
    assert_eq!(stats["rate_limit"]["tokens"], 3);
    assert_eq!(stats["rate_limit"]["allowed"], 2);

    let server = handle.join().expect("server thread");
    assert_eq!(
        server.store().get("agent:abbey").as_deref(),
        Some("trained")
    );
}

#[test]
fn repeated_query_joined_teardown_and_reopen_preserve_searchability() {
    const ITERATIONS: usize = 50;

    let fixture = Fixture::new("abi_rest_tcp_teardown");
    let seed_id = {
        let mut store = fixture.open();
        store.put_vector(&[1.0, 0.0]).expect("seed vector")
    };

    for iteration in 0..ITERATIONS {
        let config = RestConfig {
            bearer_token: None,
            rate_limiter: RateLimiter::new(1, 0, 0),
        };
        let mut server = RestServer::bind(0, fixture.open(), config).expect("bind");
        let port = server.local_port().expect("port");
        let handle = thread::spawn(move || {
            server.serve_one().expect("serve query");
            server
        });

        let body = br#"{"vector":[1.0,0.0],"limit":1}"#;
        let header = format!(
            "POST /query HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let response = exchange(port, &[header.as_bytes(), body]);
        assert!(
            response.starts_with("HTTP/1.1 200 OK"),
            "iteration {iteration}: {response}"
        );
        let response_body = find_body(response.as_bytes()).expect("query response body");
        let value: Value = serde_json::from_slice(response_body).expect("query response json");
        assert_eq!(
            value["results"][0]["id"],
            serde_json::to_value(seed_id).unwrap(),
            "iteration {iteration}"
        );

        // Joining transfers the server (and its VersionedStore) back to this
        // thread. Dropping it before reopening is the real Rust teardown
        // boundary: no borrow can outlive the owner, while WAL/file-handle
        // cleanup still has to make the next open and search succeed.
        drop(handle.join().expect("server thread"));
        let reopened = fixture.open();
        let results = reopened
            .search(&[1.0, 0.0], 1)
            .expect("search after teardown");
        assert_eq!(results[0].id, seed_id, "iteration {iteration}");
    }
}

#[test]
fn real_tcp_rejects_incomplete_and_oversize_requests() {
    let fixture = Fixture::new("abi_rest_tcp_bounds");
    let config = RestConfig {
        bearer_token: None,
        rate_limiter: RateLimiter::new(5, 0, 0),
    };
    let mut server = RestServer::bind(0, fixture.open(), config).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        for _ in 0..2 {
            server.serve_one().expect("serve request");
        }
    });

    let incomplete = exchange(
        port,
        &[b"POST /insert HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}"],
    );
    assert!(incomplete.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(incomplete.ends_with(r#"{"error":"incomplete request"}"#));

    let oversized = format!("POST /insert HTTP/1.1\r\nContent-Length: {MAX_REQUEST_SIZE}\r\n\r\n");
    let too_large = exchange(port, &[oversized.as_bytes()]);
    assert!(too_large.starts_with("HTTP/1.1 413 Payload Too Large"));
    assert!(too_large.ends_with(r#"{"error":"request too large"}"#));

    handle.join().expect("server thread");
}

fn exchange(port: u16, chunks: &[&[u8]]) -> String {
    let mut stream =
        TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect to REST server");
    for chunk in chunks {
        stream.write_all(chunk).expect("write request chunk");
    }
    stream.shutdown(Shutdown::Write).expect("finish request");
    let mut response = String::new();
    stream.read_to_string(&mut response).expect("read response");
    response
}
