//! Loopback-only WDBX REST routing and bounded HTTP/1.1 transport.
//!
//! This is intentionally a small local-control surface, not a general web
//! framework. It accepts one request per connection, caps requests at 64 KiB,
//! optionally checks a bearer token, and always closes the connection.

use crate::rate_limit::{RateLimitStats, RateLimiter};
use crate::{HybridScorer, RecordId, TemporalCausalGraph, VersionedStore};
use abi_foundation::env::WDBX_REST_TOKEN;
use abi_foundation::http::{
    MAX_REQUEST_SIZE, ReadResult, find_body, has_bearer_token, read_request, reason_phrase,
    write_all, write_unauthorized,
};
use serde_json::{Value, json};
use std::fmt::Write as _;
use std::io;
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

/// JSON response returned by the pure router.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestResponse {
    /// HTTP status code.
    pub status: u16,
    /// UTF-8 JSON body.
    pub body: String,
}

impl RestResponse {
    fn json(status: u16, value: &Value) -> Self {
        Self {
            status,
            body: value.to_string(),
        }
    }

    fn error(status: u16, message: impl Into<String>) -> Self {
        Self::json(status, &json!({"error": message.into()}))
    }
}

/// Route one already-framed REST request.
pub fn route(
    store: &mut VersionedStore,
    method: &str,
    path: &str,
    body: &[u8],
    now_ms: i64,
) -> RestResponse {
    match (method, path) {
        ("GET", "/health") => RestResponse::json(200, &json!({"status": "ok"})),
        ("GET", "/stats") => RestResponse::json(200, &store_stats(store)),
        ("POST", "/verify") => RestResponse::json(
            200,
            &json!({
                "chain_valid": store.snapshot().verify_audit_dag().is_ok(),
                "blocks": store.stats().blocks,
            }),
        ),
        ("POST", "/insert") => route_insert(store, body, now_ms),
        ("POST", "/query") => route_query(store, body, now_ms),
        _ => RestResponse::error(404, format!("no route for {method} {path}")),
    }
}

fn route_insert(store: &mut VersionedStore, body: &[u8], now_ms: i64) -> RestResponse {
    let object = match parsed_object(body) {
        Ok(object) => object,
        Err(response) => return response,
    };

    if let Some(profile_value) = object.get("profile") {
        let Some(profile) = profile_value.as_str() else {
            return RestResponse::error(400, "profile must be a string");
        };
        let metadata = object.get("metadata").and_then(Value::as_str).unwrap_or("");
        return match store.add_block(
            profile,
            RecordId::new_v2(),
            RecordId::new_v2(),
            metadata,
            now_ms,
        ) {
            Ok(_) => RestResponse::json(
                200,
                &json!({"inserted": "block", "blocks": store.stats().blocks}),
            ),
            Err(error) => RestResponse::error(500, error.to_string()),
        };
    }

    if let Some(vector_value) = object.get("vector") {
        let vector = match parse_vector(vector_value) {
            Ok(vector) => vector,
            Err(message) => return RestResponse::error(400, message),
        };
        return match store.put_vector(&vector) {
            Ok(id) => RestResponse::json(200, &json!({"inserted": "vector", "id": id})),
            Err(error) => RestResponse::error(500, error.to_string()),
        };
    }

    let key = object.get("key").and_then(Value::as_str);
    let value = object.get("value").and_then(Value::as_str);
    let (Some(key), Some(value)) = (key, value) else {
        return RestResponse::error(400, "need key+value or profile");
    };
    match store.put(key, value) {
        Ok(_) => RestResponse::json(200, &json!({"inserted": "kv"})),
        Err(error) => RestResponse::error(500, error.to_string()),
    }
}

fn route_query(store: &VersionedStore, body: &[u8], now_ms: i64) -> RestResponse {
    let object = match parsed_object(body) {
        Ok(object) => object,
        Err(response) => return response,
    };

    if let Some(key_value) = object.get("key") {
        let Some(key) = key_value.as_str() else {
            return RestResponse::error(400, "key must be a string");
        };
        return match store.get(key) {
            Some(value) => RestResponse::json(200, &json!({"value": value})),
            None => RestResponse::error(404, "not found"),
        };
    }

    let Some(vector_value) = object.get("vector") else {
        return RestResponse::error(400, "need key or vector");
    };
    let vector = match parse_vector(vector_value) {
        Ok(vector) => vector,
        Err(message) => return RestResponse::error(400, message),
    };
    let limit = match parse_limit(object.get("limit")) {
        Ok(limit) => limit,
        Err(message) => return RestResponse::error(400, message),
    };
    if store.stats().vectors == 0 {
        return RestResponse::json(200, &json!({"results": [], "vectors": 0}));
    }

    let snapshot = store.snapshot();
    let graph = TemporalCausalGraph::from_v2_records(&snapshot.preferred_temporal_records());
    let focus_id = snapshot
        .causal_focus_vector_id()
        .unwrap_or(RecordId::Legacy(1));
    let scorer = HybridScorer::new(now_ms);
    let mut ranked = match store.search(&vector, limit) {
        Ok(ranked) => ranked
            .into_iter()
            .map(|result| {
                let components = scorer.score(&graph, focus_id, result.id, result.score, 0.5);
                (result, components)
            })
            .collect::<Vec<_>>(),
        Err(error) => return RestResponse::error(400, error.to_string()),
    };
    ranked.sort_by(|left, right| {
        right
            .1
            .combined()
            .total_cmp(&left.1.combined())
            .then_with(|| left.0.id.cmp(&right.0.id))
    });
    let mut results = String::from("[");
    for (index, (node, components)) in ranked.into_iter().enumerate() {
        if index > 0 {
            results.push(',');
        }
        write!(
            results,
            "{{\"id\":{},\"score\":{:.6},\"semantic\":{:.6},\"temporal\":{:.6},\"causal\":{:.6},\"persona\":{:.6}}}",
            serde_json::to_string(&node.id).expect("RecordId serializes"),
            components.combined(),
            components.semantic,
            components.temporal,
            components.causal,
            components.persona,
        )
        .expect("writing JSON into String cannot fail");
    }
    results.push(']');
    RestResponse {
        status: 200,
        body: format!(
            "{{\"results\":{results},\"vectors\":{},\"ranking\":\"hybrid\"}}",
            store.stats().vectors
        ),
    }
}

fn parsed_object(body: &[u8]) -> Result<serde_json::Map<String, Value>, RestResponse> {
    let value: Value =
        serde_json::from_slice(body).map_err(|_| RestResponse::error(400, "invalid json"))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| RestResponse::error(400, "expected object"))
}

fn parse_vector(value: &Value) -> Result<Vec<f32>, &'static str> {
    let Some(array) = value.as_array() else {
        return Err("vector must be an array");
    };
    if array.is_empty() {
        return Err("vector must be non-empty");
    }
    array
        .iter()
        .map(|component| {
            component
                .as_f64()
                .map(json_number_to_f32)
                .filter(|number| number.is_finite())
                .ok_or("vector elements must be numbers")
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn json_number_to_f32(number: f64) -> f32 {
    // Zig's REST parser uses @floatCast for the same wire-format boundary.
    // Non-finite results are rejected by the caller.
    number as f32
}

fn parse_limit(value: Option<&Value>) -> Result<usize, &'static str> {
    let Some(value) = value else {
        return Ok(10);
    };
    if !value.is_number() || value.as_f64().is_some_and(f64::is_nan) {
        return Err("limit must be an integer");
    }
    let Some(limit) = value.as_u64() else {
        return if value.as_i64().is_some() {
            Err("limit must be between 1 and 100")
        } else {
            Err("limit must be an integer")
        };
    };
    if !(1..=100).contains(&limit) {
        return Err("limit must be between 1 and 100");
    }
    usize::try_from(limit).map_err(|_| "limit must be between 1 and 100")
}

fn store_stats(store: &VersionedStore) -> Value {
    let stats = store.stats();
    json!({
        "kv_entries": stats.kv_entries,
        "vectors": stats.vectors,
        "blocks": stats.blocks,
        "spatial_records": stats.spatial_records,
        "temporal_nodes": stats.temporal_nodes,
        "temporal_edges": stats.temporal_edges,
        "vector_dimensions": store.snapshot().vector_dimensions(),
        "next_vector_id": Value::Null,
        "format_version": 2,
        "backend": "cpu",
        "mode": "fallback",
    })
}

/// Configuration for one local REST service.
#[derive(Debug)]
pub struct RestConfig {
    /// Optional bearer token. `None` preserves the oracle's local auth-off mode.
    pub bearer_token: Option<String>,
    /// Token bucket shared by all connections.
    pub rate_limiter: RateLimiter,
}

impl RestConfig {
    /// Load auth and rate-limit configuration from the process environment.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            bearer_token: std::env::var(WDBX_REST_TOKEN)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            rate_limiter: RateLimiter::from_env(),
        }
    }
}

/// Sequential one-request-per-connection REST server.
#[derive(Debug)]
pub struct RestServer {
    listener: TcpListener,
    store: VersionedStore,
    config: RestConfig,
}

impl RestServer {
    /// Bind exactly `127.0.0.1:port`.
    pub fn bind(port: u16, store: VersionedStore, config: RestConfig) -> io::Result<Self> {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port))?;
        Ok(Self {
            listener,
            store,
            config,
        })
    }

    /// Kernel-selected or configured local port.
    pub fn local_port(&self) -> io::Result<u16> {
        Ok(self.listener.local_addr()?.port())
    }

    /// Accept and handle one connection.
    pub fn serve_one(&mut self) -> io::Result<()> {
        let (stream, _) = self.listener.accept()?;
        handle_connection(&mut self.store, &self.config, stream)
    }

    /// Serve until the process is stopped.
    pub fn run(&mut self) -> io::Result<()> {
        loop {
            self.serve_one()?;
        }
    }

    /// Borrow the underlying durable store.
    #[must_use]
    pub fn store(&self) -> &VersionedStore {
        &self.store
    }
}

fn handle_connection(
    store: &mut VersionedStore,
    config: &RestConfig,
    mut stream: TcpStream,
) -> io::Result<()> {
    let raw = match read_request(&mut stream, MAX_REQUEST_SIZE) {
        ReadResult::Empty => return Ok(()),
        ReadResult::Incomplete => {
            return write_response(
                &mut stream,
                &RestResponse::error(400, "incomplete request"),
                &[],
            );
        }
        ReadResult::TooLarge => {
            return write_response(
                &mut stream,
                &RestResponse::error(413, "request too large"),
                &[],
            );
        }
        ReadResult::Request(raw) => raw,
    };
    let Ok(raw_text) = std::str::from_utf8(&raw) else {
        return write_response(
            &mut stream,
            &RestResponse::error(400, "incomplete request"),
            &[],
        );
    };
    let Some((method, path)) = request_target(raw_text) else {
        return Ok(());
    };

    // Ordering is part of the security contract: failed auth consumes a token.
    if !config.rate_limiter.acquire() {
        return write_response(
            &mut stream,
            &RestResponse::error(429, "too many requests"),
            &[("Retry-After", "1")],
        );
    }
    if let Some(token) = &config.bearer_token
        && !has_bearer_token(raw_text, token)
    {
        return write_unauthorized(&mut stream, "unauthorized");
    }

    let mut response = route(
        store,
        method,
        path,
        find_body(&raw).unwrap_or_default(),
        unix_ms(),
    );
    if path == "/stats" && response.status == 200 {
        response.body = add_rate_stats(&response.body, config.rate_limiter.stats());
    }
    write_response(&mut stream, &response, &[])
}

fn request_target(raw: &str) -> Option<(&str, &str)> {
    let request_line = raw.lines().next()?.trim_end_matches('\r');
    let mut fields = request_line.split(' ');
    Some((fields.next()?, fields.next()?))
}

fn add_rate_stats(body: &str, stats: RateLimitStats) -> String {
    let Ok(mut value) = serde_json::from_str::<Value>(body) else {
        return body.to_string();
    };
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "rate_limit".to_string(),
            serde_json::to_value(stats).expect("rate-limit stats serialize"),
        );
    }
    value.to_string()
}

fn write_response(
    stream: &mut TcpStream,
    response: &RestResponse,
    extra_headers: &[(&str, &str)],
) -> io::Result<()> {
    let phrase = if response.status == 413 {
        "Payload Too Large"
    } else {
        reason_phrase(response.status)
    };
    let mut header = format!(
        "HTTP/1.1 {} {phrase}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        response.status,
        response.body.len()
    );
    for (name, value) in extra_headers {
        header.push_str(name);
        header.push_str(": ");
        header.push_str(value);
        header.push_str("\r\n");
    }
    header.push_str("Connection: close\r\n\r\n");
    write_all(stream, header.as_bytes())?;
    write_all(stream, response.body.as_bytes())
}

fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| i64::try_from(duration.as_millis()).ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
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
                serde_json::from_value::<RecordId>(value["id"].clone())
                    .expect("versioned vector id"),
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

        let oversized =
            format!("POST /insert HTTP/1.1\r\nContent-Length: {MAX_REQUEST_SIZE}\r\n\r\n");
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
}
