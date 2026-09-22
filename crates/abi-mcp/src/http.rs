//! Custom loopback HTTP compatibility transport for MCP JSON-RPC.
//!
//! Ported from `src/mcp/http_transport.zig`. Listens on `127.0.0.1` only:
//! - `GET /sse` — establishes a persistent SSE stream, returns a stream id
//!   in the `endpoint` event; keeps the connection open for `message` events
//! - `POST /message?id={stream_id}` — publishes the JSON-RPC response as an
//!   SSE `message` event on the named stream
//!
//! Optional bearer auth via `ABI_MCP_HTTP_TOKEN`. Default port 8080
//! (`ABI_MCP_HTTP_PORT`). Not a general web server. This implements the
//! MCP 2024-11-05 persistent HTTP+SSE channel: response-bearing POSTs
//! publish as SSE `message` events rather than replying directly over HTTP.

use std::collections::HashMap;
use std::io::Read;
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::thread;
use std::time::Duration;

use abi_foundation::env::{self, MCP_HTTP_PORT, MCP_HTTP_TOKEN};
use abi_foundation::http::{
    MAX_REQUEST_SIZE, ReadResult, has_bearer_token, read_request, write_all, write_unauthorized,
};

use crate::rpc;
use crate::state::McpState;

/// Default MCP HTTP port when `ABI_MCP_HTTP_PORT` is unset.
pub const DEFAULT_HTTP_PORT: u16 = 8080;

/// Avoid an unbounded stderr stream if a broken peer repeatedly causes
/// connection-level failures. The next error emits one suppression notice.
const MAX_REPORTED_SERVER_ERRORS: usize = 8;

/// Configuration for the custom loopback HTTP listener.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// Loopback port (0 = kernel-assign).
    pub port: u16,
    /// Optional bearer token.
    pub bearer_token: Option<String>,
}

impl HttpConfig {
    /// Load port/token from process env (empty token = auth off).
    #[must_use]
    pub fn from_env() -> Self {
        let port = env::get(MCP_HTTP_PORT)
            .and_then(|raw| raw.parse().ok())
            .filter(|port| *port != 0)
            .unwrap_or(DEFAULT_HTTP_PORT);
        let bearer_token = env::get(MCP_HTTP_TOKEN).filter(|t| !t.is_empty());
        Self { port, bearer_token }
    }
}

/// Shared SSE stream registry: maps stream id to a sender, tracks the next id.
#[derive(Debug)]
pub struct SseRegistry {
    streams: Arc<std::sync::Mutex<HashMap<String, Sender<String>>>>,
    next_id: Arc<AtomicU64>,
}

impl SseRegistry {
    fn new() -> Self {
        Self {
            streams: Arc::new(std::sync::Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }
    fn create(&self) -> (String, Sender<String>) {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst).to_string();
        let (tx, _rx) = channel();
        self.streams
            .lock()
            .expect("registry lock")
            .insert(id.clone(), tx.clone());
        (id, tx)
    }
    fn next_stream_id(&self) -> String {
        self.next_id.fetch_add(1, Ordering::SeqCst).to_string()
    }
    fn get(&self, id: &str) -> Option<Sender<String>> {
        self.streams.lock().expect("registry lock").get(id).cloned()
    }
    fn remove(&self, id: &str) {
        self.streams.lock().expect("registry lock").remove(id);
    }
}
type StreamRegistry = Arc<SseRegistry>;

/// Running loopback MCP HTTP server.
#[derive(Debug)]
pub struct McpHttpServer {
    listener: TcpListener,
    config: HttpConfig,
    state: McpState,
    stop: Arc<AtomicBool>,
    registry: StreamRegistry,
}

impl McpHttpServer {
    /// Bind `127.0.0.1:port` and prepare to serve.
    pub fn bind(config: HttpConfig, state: McpState) -> std::io::Result<Self> {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, config.port))?;
        let _ = listener.set_nonblocking(false);
        Ok(Self {
            listener,
            config,
            state,
            stop: Arc::new(AtomicBool::new(false)),
            registry: Arc::new(SseRegistry::new()),
        })
    }

    /// Kernel-selected or configured port.
    pub fn local_port(&self) -> std::io::Result<u16> {
        Ok(self.listener.local_addr()?.port())
    }

    /// Shared stop flag for cooperative shutdown.
    #[must_use]
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }

    /// Shared stream registry for testing.
    #[must_use]
    pub fn registry(&self) -> StreamRegistry {
        Arc::clone(&self.registry)
    }

    /// Request shutdown of the accept loop.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Ok(port) = self.local_port() {
            let _ = TcpStream::connect(SocketAddrV4::new(Ipv4Addr::LOCALHOST, port));
        }
    }

    /// Accept and handle one connection.
    pub fn serve_one(&self) -> std::io::Result<()> {
        let (stream, _) = self.listener.accept()?;
        if self.stop.load(Ordering::SeqCst) {
            return Ok(());
        }
        let registry = Arc::clone(&self.registry);
        let stop = Arc::clone(&self.stop);
        handle_connection(
            stream,
            self.state,
            self.config.bearer_token.as_deref(),
            registry,
            stop,
        )
    }

    /// Serve until [`Self::request_stop`].
    pub fn run(&self) -> std::io::Result<()> {
        let mut reported_errors = 0_usize;
        while !self.stop.load(Ordering::SeqCst) {
            if let Err(err) = self.serve_one() {
                if self.stop.load(Ordering::SeqCst) {
                    break;
                }
                if reported_errors < MAX_REPORTED_SERVER_ERRORS {
                    eprintln!("MCP loopback HTTP serve error: {err}");
                } else if reported_errors == MAX_REPORTED_SERVER_ERRORS {
                    eprintln!(
                        "MCP loopback HTTP serve errors suppressed after {MAX_REPORTED_SERVER_ERRORS} reports"
                    );
                }
                reported_errors = reported_errors.saturating_add(1);
            }
        }
        Ok(())
    }
}

/// Handle one HTTP connection end-to-end.
pub fn handle_connection(
    stream: TcpStream,
    state: McpState,
    bearer_token: Option<&str>,
    streams: StreamRegistry,
    stop: Arc<AtomicBool>,
) -> std::io::Result<()> {
    handle_connection_with(stream, state, bearer_token, rpc::process, streams, stop)
}

fn handle_connection_with<F>(
    mut stream: TcpStream,
    state: McpState,
    bearer_token: Option<&str>,
    process: F,
    streams: StreamRegistry,
    stop: Arc<AtomicBool>,
) -> std::io::Result<()>
where
    F: Fn(McpState, &str) -> Option<crate::rpc::RpcResponse>,
{
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let raw = match read_request(&mut stream, MAX_REQUEST_SIZE) {
        ReadResult::Empty => return Ok(()),
        ReadResult::Incomplete => {
            return write_all(
                &mut stream,
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"incomplete request\"}",
            );
        }
        ReadResult::TooLarge => {
            return write_all(
                &mut stream,
                b"HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}",
            );
        }
        ReadResult::Request(raw) => raw,
    };

    let Ok(raw_text) = std::str::from_utf8(&raw) else {
        return write_all(
            &mut stream,
            b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"invalid utf-8\"}",
        );
    };

    let mut line_end = 0_usize;
    while line_end < raw_text.len() && raw_text.as_bytes()[line_end] != b'\n' {
        line_end += 1;
    }
    let request_line = raw_text[..line_end].trim_end_matches('\r');
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or("");
    let path_and_query = parts.next().unwrap_or("");
    let path = path_and_query.split('?').next().unwrap_or("");

    if !request_origin_is_allowed(&raw) {
        return write_all(
            &mut stream,
            b"HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: 26\r\nConnection: close\r\n\r\n{\"error\":\"invalid origin\"}",
        );
    }

    if let Some(token) = bearer_token
        && !has_bearer_token(raw_text, token)
    {
        return write_unauthorized(&mut stream, "unauthorized");
    }

    if method == "GET" && path == "/sse" {
        return serve_sse(&mut stream, &streams, &stop)?;
    }

    if method == "POST" && path == "/message" {
        return serve_message(&mut stream, state, &raw, body_raw(&raw), &process, &streams);
    }

    write_all(
        &mut stream,
        b"HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n",
    )
}

/// Extract the raw body bytes from an HTTP request line and headers.
fn body_raw(raw: &[u8]) -> Option<&[u8]> {
    let raw_str = std::str::from_utf8(raw).ok()?;
    let header_end = raw_str.find("\r\n\r\n")?;
    let body_start = header_end + 4;
    let body = &raw[body_start..];
    let len = raw_str[..header_end]
        .lines()
        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
        .and_then(|l| l.split_once(':'))
        .and_then(|(_, v)| v.trim().parse::<usize>().ok())?;
    if body.len() < len {
        return None;
    }
    Some(&body[..len])
}

/// Handle `GET /sse`: establish a persistent SSE stream.
fn serve_sse(
    stream: &mut TcpStream,
    streams: &StreamRegistry,
    stop: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    let stream_id = streams.next_stream_id();
    let (_id, _tx) = streams.create();

    let mut response = String::new();
    response.push_str("HTTP/1.1 200 OK\r\n");
    response.push_str("Content-Type: text/event-stream\r\n");
    response.push_str("Cache-Control: no-cache\r\n");
    response.push_str("Connection: keep-alive\r\n");
    response.push_str("\r\n");
    response.push_str(&format!(
        "data: {}\n\n",
        serde_json::json!({
            "endpoint": format!("/message?id={stream_id}")
        })
        .to_string()
    ));

    write_all(stream, response.as_bytes())?;

    // Keep the connection open until stop or peer disconnects.
    let mut buf = [0_u8; 4096];
    while !stop.load(Ordering::SeqCst) {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
    }
    streams.remove(&stream_id);
    Ok(())
}

/// Handle `POST /message?id={stream_id}`: publish JSON-RPC response as SSE event.
fn serve_message<F>(
    stream: &mut TcpStream,
    state: McpState,
    raw: &[u8],
    body: Option<&[u8]>,
    process: &F,
    streams: &StreamRegistry,
) -> std::io::Result<()>
where
    F: Fn(McpState, &str) -> Option<crate::rpc::RpcResponse>,
{
    let stream_id = std::str::from_utf8(raw)
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|l| l.to_ascii_lowercase().starts_with("request-target:"))
                .or_else(|| {
                    let request_line_end = text.find("\r\n").unwrap_or(text.len());
                    text[..request_line_end].split(' ').nth(1)
                })
        })
        .and_then(|target| {
            target.split('?').nth(1).and_then(|q| {
                q.split('&')
                    .find(|pair| pair.starts_with("id="))
                    .and_then(|pair| pair.strip_prefix("id="))
            })
        })
        .unwrap_or("");

    let body_text = body.unwrap_or_default();
    let body_str = std::str::from_utf8(body_text).unwrap_or("");

    if let Some(response) = process(state, body_str) {
        if let Some(sender) = streams.get(stream_id) {
            let _ = sender.send(serde_json::to_string(&response.0).unwrap_or_default());
        }
    }
    streams.remove(stream_id);

    write_all(
        stream,
        b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
    )?;
    Ok(())
}

/// Validate the `Origin` header against loopback to prevent DNS rebinding.
fn request_origin_is_allowed(raw: &[u8]) -> bool {
    let block = std::str::from_utf8(raw)
        .ok()
        .map(|text| {
            let header_end = text.find("\r\n\r\n").unwrap_or(text.len());
            &text[..header_end]
        })
        .unwrap_or("");
    let mut origins = block
        .split("\r\n")
        .skip(1)
        .take_while(|line| !line.is_empty())
        .filter_map(|line| line.split_once(':'))
        .filter_map(|(name, value)| {
            name.trim_matches([' ', '\t'])
                .eq_ignore_ascii_case("Origin")
                .then_some(value.trim_matches([' ', '\t']))
        });
    let Some(origin) = origins.next() else {
        return true;
    };
    origins.next().is_none() && is_allowed_loopback_origin(origin)
}

fn is_allowed_loopback_origin(origin: &str) -> bool {
    let Some(authority) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    if authority.is_empty()
        || authority
            .bytes()
            .any(|byte| matches!(byte, b'/' | b'?' | b'#' | b'@'))
    {
        return false;
    }
    let (host, port) = authority
        .split_once(':')
        .map_or((authority, None), |(host, port)| (host, Some(port)));
    if !matches!(host, "127.0.0.1" | "localhost") {
        return false;
    }
    port.is_none_or(|port| port.parse::<u16>().is_ok_and(|port| port != 0))
}

/// Spawn the HTTP server on a background thread; returns `(port, stop_handle)`.
#[must_use]
pub fn spawn_from_env(state: McpState) -> Option<(u16, Arc<AtomicBool>, thread::JoinHandle<()>)> {
    let config = HttpConfig::from_env();
    let server = McpHttpServer::bind(config, state).ok()?;
    let port = server.local_port().ok()?;
    let stop = server.stop_flag();
    let stop_clone = Arc::clone(&stop);
    let handle = thread::spawn(move || {
        let _ = server.run();
    });
    Some((port, stop_clone, handle))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpStream;
    use std::time::Duration;

    struct EnvOverrideCleanup;

    impl Drop for EnvOverrideCleanup {
        fn drop(&mut self) {
            env::clear_override(MCP_HTTP_PORT);
            env::clear_override(MCP_HTTP_TOKEN);
        }
    }

    fn with_server(token: Option<&str>, f: impl FnOnce(u16)) {
        let config = HttpConfig {
            port: 0,
            bearer_token: token.map(str::to_string),
        };
        let server = McpHttpServer::bind(config, McpState::new()).expect("bind");
        let port = server.local_port().expect("port");
        let stop = server.stop_flag();
        let stop_for_thread = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            for _ in 0..8 {
                if stop_for_thread.load(Ordering::SeqCst) {
                    break;
                }
                let _ = server.serve_one();
            }
        });
        thread::sleep(Duration::from_millis(30));
        f(port);
        stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(format!("127.0.0.1:{port}"));
        let _ = handle.join();
    }

    fn read_http(stream: &mut TcpStream) -> String {
        let mut out = Vec::new();
        let mut buf = [0_u8; 4096];
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => out.extend_from_slice(&buf[..n]),
            }
            if out.len() > 64 && out.windows(4).any(|w| w == b"\r\n\r\n") {
                if let Ok(text) = std::str::from_utf8(&out)
                    && let Some(cl) = text
                        .lines()
                        .find(|l| l.to_ascii_lowercase().starts_with("content-length:"))
                        .and_then(|l| l.split(':').nth(1))
                        .and_then(|v| v.trim().parse::<usize>().ok())
                {
                    if let Some(pos) = text.find("\r\n\r\n") {
                        let body_start = pos + 4;
                        if out.len() >= body_start + cl {
                            break;
                        }
                    }
                } else if out.len() > 256 {
                    break;
                }
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    #[test]
    fn environment_port_zero_and_invalid_values_fall_back_to_8080() {
        let _guard = env::lock_for_test();
        let _cleanup = EnvOverrideCleanup;
        for value in ["", "0", "-1", "65536", "not-a-port"] {
            env::set_override(MCP_HTTP_PORT, value);
            assert_eq!(HttpConfig::from_env().port, DEFAULT_HTTP_PORT, "{value:?}");
        }
        env::set_override(MCP_HTTP_PORT, "61234");
        assert_eq!(HttpConfig::from_env().port, 61_234);
    }

    #[test]
    fn direct_ephemeral_server_can_be_woken_on_its_actual_bound_port() {
        let server = McpHttpServer::bind(
            HttpConfig {
                port: 0,
                bearer_token: None,
            },
            McpState::new(),
        )
        .expect("bind ephemeral");
        let port = server.local_port().expect("actual port");
        assert_ne!(port, 0);
        let stop = server.stop_flag();
        let handle = thread::spawn(move || server.run());
        stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect((Ipv4Addr::LOCALHOST, port));
        handle
            .join()
            .expect("server thread joins")
            .expect("server stops cleanly");
    }

    #[test]
    fn post_message_ping_returns_sse_message() {
        use std::io::Write;
        with_server(None, |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            let body = r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#;
            let req = format!(
                "POST /message?id=test-stream HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.contains("202 Accepted"), "{resp}");
            assert!(resp.contains("Content-Length: 0"), "{resp}");
        });
    }

    #[test]
    fn notification_is_accepted_without_a_jsonrpc_response_body() {
        use std::io::Write;
        with_server(None, |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            let body = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
            let req = format!(
                "POST /message?id=test-stream HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.starts_with("HTTP/1.1 202 Accepted"), "{resp}");
            assert!(resp.contains("Content-Length: 0"), "{resp}");
        });
    }

    #[test]
    fn hostile_origin_is_rejected_before_auth_or_dispatch() {
        use std::io::Write;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind probe listener");
        let port = listener.local_addr().expect("probe address").port();
        let dispatched = Arc::new(AtomicBool::new(false));
        let dispatched_in_handler = Arc::clone(&dispatched);
        let handle = thread::spawn(move || {
            let (stream, _) = listener.accept().expect("accept probe request");
            let streams = Arc::new(SseRegistry::new());
            let stop = Arc::new(AtomicBool::new(false));
            handle_connection_with(
                stream,
                McpState::new(),
                Some("local-token"),
                move |_, _| {
                    dispatched_in_handler.store(true, Ordering::SeqCst);
                    None
                },
                streams,
                stop,
            )
            .expect("handle hostile origin");
        });

        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect");
        let body = r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#;
        let req = format!(
            "POST /message HTTP/1.1\r\nOrigin: https://attacker.example\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(req.as_bytes()).expect("write");
        let resp = read_http(&mut stream);
        handle.join().expect("origin probe joins");

        assert!(resp.starts_with("HTTP/1.1 403 Forbidden"), "{resp}");
        assert!(resp.contains("invalid origin"), "{resp}");
        assert!(
            !resp.contains("Unauthorized"),
            "origin must be checked before auth"
        );
        assert!(!resp.contains("\"id\":7"), "request must not be dispatched");
        assert!(
            !dispatched.load(Ordering::SeqCst),
            "hostile Origin reached the mutating dispatch seam"
        );
    }

    #[test]
    fn loopback_origins_are_narrowly_validated() {
        for origin in [
            "http://127.0.0.1",
            "http://127.0.0.1:8080",
            "https://localhost",
            "https://localhost:443",
        ] {
            assert!(is_allowed_loopback_origin(origin), "{origin}");
        }
        for origin in [
            "null",
            "file://localhost",
            "https://localhost.example",
            "https://127.0.0.1.attacker.example",
            "https://localhost@attacker.example",
            "https://localhost:0",
            "https://localhost:not-a-port",
        ] {
            assert!(!is_allowed_loopback_origin(origin), "{origin}");
        }
        assert!(!request_origin_is_allowed(
            b"GET /sse HTTP/1.1\r\nOrigin: http://localhost\r\nOrigin: https://attacker.example\r\n\r\n"
        ));
    }

    #[test]
    fn get_sse_announces_message_endpoint() {
        use std::io::Write;
        with_server(None, |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            stream
                .write_all(b"GET /sse HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                .expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.contains("200 OK"), "{resp}");
            assert!(resp.contains("text/event-stream"), "{resp}");
        });
    }

    #[test]
    fn bearer_token_required_when_configured() {
        use std::io::Write;
        with_server(Some("local-token"), |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
            let req = format!(
                "POST /message?id=test-stream HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.contains("401"), "{resp}");

            let mut stream2 = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect2");
            let req2 = format!(
                "POST /message?id=test-stream HTTP/1.1\r\nAuthorization: Bearer local-token\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream2.write_all(req2.as_bytes()).expect("write2");
            let resp2 = read_http(&mut stream2);
            assert!(resp2.contains("202 Accepted"), "{resp2}");
        });
    }

    #[test]
    fn wrong_sse_token_is_unauthorized() {
        use std::io::Write;
        with_server(Some("local-token"), |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            stream
                .write_all(
                    b"GET /sse HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer wrong\r\n\r\n",
                )
                .expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.contains("401"), "{resp}");
        });
    }

    #[test]
    fn malformed_and_empty_bearer_schemes_are_unauthorized() {
        use std::io::Write;
        with_server(Some("local-token"), |port| {
            let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
            for auth in [
                "Authorization: Bearer \r\n",
                "Authorization: Basic local-token\r\n",
                "Authorization: bearer local-token\r\n",
                "Authorization: Bearerlocal-token\r\n",
            ] {
                let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
                let req = format!(
                    "POST /message?id=test-stream HTTP/1.1\r\n{auth}Content-Length: {}\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(req.as_bytes()).expect("write");
                let resp = read_http(&mut stream);
                assert!(
                    resp.contains("401"),
                    "expected 401 for auth header {auth:?}, got {resp}"
                );
                assert!(
                    !resp.contains("local-token"),
                    "response must not echo the configured token: {resp}"
                );
            }
        });
    }
}
