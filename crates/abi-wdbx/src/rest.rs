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
mod tests;
