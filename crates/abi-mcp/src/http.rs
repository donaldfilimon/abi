//! Loopback HTTP + SSE transport for MCP JSON-RPC.
//!
//! Ported from `src/mcp/http_transport.zig`. Listens on `127.0.0.1` only:
//! - `GET /sse` — opens an event-stream and announces `data: /message`
//! - `POST /message` — one JSON-RPC request body → one JSON-RPC response
//!
//! Optional bearer auth via `ABI_MCP_HTTP_TOKEN`. Default port 8080
//! (`ABI_MCP_HTTP_PORT`). Not a general web server; one request per connection.

use std::net::{Ipv4Addr, SocketAddrV4, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use abi_foundation::env::{self, MCP_HTTP_PORT, MCP_HTTP_TOKEN};
use abi_foundation::http::{
    MAX_REQUEST_SIZE, ReadResult, find_body, has_bearer_token, read_request, write_all,
    write_unauthorized,
};

use crate::rpc;
use crate::state::McpState;

/// Default MCP HTTP port when `ABI_MCP_HTTP_PORT` is unset.
pub const DEFAULT_HTTP_PORT: u16 = 8080;

/// Configuration for the HTTP/SSE listener.
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
            .unwrap_or(DEFAULT_HTTP_PORT);
        let bearer_token = env::get(MCP_HTTP_TOKEN).filter(|t| !t.is_empty());
        Self { port, bearer_token }
    }
}

/// Running loopback MCP HTTP server.
#[derive(Debug)]
pub struct McpHttpServer {
    listener: TcpListener,
    config: HttpConfig,
    state: McpState,
    stop: Arc<AtomicBool>,
}

impl McpHttpServer {
    /// Bind `127.0.0.1:port` and prepare to serve.
    pub fn bind(config: HttpConfig, state: McpState) -> std::io::Result<Self> {
        let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, config.port))?;
        // Short accept timeout so Ctrl-C / stop flags are observed.
        let _ = listener.set_nonblocking(false);
        Ok(Self {
            listener,
            config,
            state,
            stop: Arc::new(AtomicBool::new(false)),
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

    /// Request shutdown of the accept loop.
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake a blocking accept with a local connect.
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
        handle_connection(stream, self.state, self.config.bearer_token.as_deref())
    }

    /// Serve until [`Self::request_stop`].
    pub fn run(&self) -> std::io::Result<()> {
        while !self.stop.load(Ordering::SeqCst) {
            if let Err(err) = self.serve_one() {
                if self.stop.load(Ordering::SeqCst) {
                    break;
                }
                // Transient accept errors: log-equivalent discard for library use.
                let _ = err;
            }
        }
        Ok(())
    }
}

/// Handle one HTTP connection end-to-end.
pub fn handle_connection(
    mut stream: TcpStream,
    state: McpState,
    bearer_token: Option<&str>,
) -> std::io::Result<()> {
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
    let path = parts.next().unwrap_or("");

    if let Some(token) = bearer_token
        && !has_bearer_token(raw_text, token)
    {
        return write_unauthorized(&mut stream, "unauthorized");
    }

    if method == "GET" && path == "/sse" {
        return write_all(
            &mut stream,
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\nevent: endpoint\r\ndata: /message\r\n\r\n",
        );
    }

    if method == "POST" && path == "/message" {
        let Some(body) = find_body(&raw) else {
            return write_all(
                &mut stream,
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"no body\"}",
            );
        };
        if body.len() > MAX_REQUEST_SIZE {
            return write_all(
                &mut stream,
                b"HTTP/1.1 413 Payload Too Large\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"request too large\"}",
            );
        }
        let Ok(body_text) = std::str::from_utf8(body) else {
            return write_all(
                &mut stream,
                b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{\"error\":\"invalid utf-8 body\"}",
            );
        };
        let result_json = match rpc::process(state, body_text) {
            Some(response) => response.to_json_string(),
            None => {
                // Empty/notification: 202 with empty JSON object body (stdio would write nothing).
                serde_json::json!({"jsonrpc":"2.0","result":null}).to_string()
            }
        };
        let mut out = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            result_json.len()
        );
        out.push_str(&result_json);
        return write_all(&mut stream, out.as_bytes());
    }

    write_all(
        &mut stream,
        b"HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n",
    )
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
            // Serve a few connections then exit when stop is set.
            for _ in 0..8 {
                if stop_for_thread.load(Ordering::SeqCst) {
                    break;
                }
                let _ = server.serve_one();
            }
        });
        // Brief settle.
        thread::sleep(Duration::from_millis(30));
        f(port);
        stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(format!("127.0.0.1:{port}"));
        let _ = handle.join();
    }

    fn read_http(stream: &mut TcpStream) -> String {
        use std::io::Read;
        let mut out = Vec::new();
        let mut buf = [0_u8; 4096];
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => out.extend_from_slice(&buf[..n]),
            }
            // Enough for our test responses.
            if out.len() > 64 && out.windows(4).any(|w| w == b"\r\n\r\n") {
                // If Content-Length is present and satisfied, stop.
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
    fn post_message_ping_returns_jsonrpc_result() {
        use std::io::Write;
        with_server(None, |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            let body = r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#;
            let req = format!(
                "POST /message HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.contains("200 OK"), "{resp}");
            assert!(
                resp.contains(r#""id":7"#) || resp.contains(r#""result":{}"#),
                "{resp}"
            );
            assert!(resp.contains("jsonrpc"), "{resp}");
        });
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
            assert!(resp.contains("data: /message"), "{resp}");
        });
    }

    #[test]
    fn bearer_token_required_when_configured() {
        use std::io::Write;
        with_server(Some("local-token"), |port| {
            let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
            let body = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
            let req = format!(
                "POST /message HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).expect("write");
            let resp = read_http(&mut stream);
            assert!(resp.contains("401"), "{resp}");

            // Wake server for second connection.
            let mut stream2 = TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect2");
            let req2 = format!(
                "POST /message HTTP/1.1\r\nAuthorization: Bearer local-token\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream2.write_all(req2.as_bytes()).expect("write2");
            let resp2 = read_http(&mut stream2);
            assert!(resp2.contains("200 OK"), "{resp2}");
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
}
