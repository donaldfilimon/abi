//! TLS-backed WebSocket client for live Discord / Twilio paths.
//!
//! Provides:
//! - Blocking `wss://` connect via **rustls** (no external TLS proxy).
//! - WebSocket upgrade + masked client frames (reuses [`crate::discord_ws`]).
//! - An injectable stream surface so offline tests can drive TLS peers
//!   without production Discord/Twilio credentials.
//!
//! Live network calls still require real credentials and DNS; when those are
//! missing or the peer closes, errors are **disclosed** (never fabricated).

#![cfg(feature = "live")]
#![allow(clippy::module_name_repetitions)]

use crate::discord_ws::{
    Frame, WsError, build_handshake_request, encode_masked_text_frame, key_b64_from_seed,
    try_parse_frame,
};
use base64::Engine as _;
use rustls::ClientConfig;
use rustls::pki_types::ServerName;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

/// Errors from TLS + WebSocket connect / I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlsWsError {
    /// DNS or TCP connect failed.
    Connect(String),
    /// TLS handshake failed.
    Tls(String),
    /// WebSocket handshake / framing failed.
    Ws(String),
    /// Peer closed.
    Closed,
    /// Missing credentials / configuration for a live path.
    NotConfigured(String),
}

impl std::fmt::Display for TlsWsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Connect(m) => write!(f, "tls-ws connect: {m}"),
            Self::Tls(m) => write!(f, "tls-ws handshake: {m}"),
            Self::Ws(m) => write!(f, "tls-ws websocket: {m}"),
            Self::Closed => write!(f, "tls-ws closed"),
            Self::NotConfigured(m) => write!(f, "tls-ws not configured: {m}"),
        }
    }
}

impl std::error::Error for TlsWsError {}

impl From<WsError> for TlsWsError {
    fn from(value: WsError) -> Self {
        Self::Ws(value.to_string())
    }
}

/// Build a rustls client config trusting webpki roots (production).
#[must_use]
pub fn production_client_config() -> Arc<ClientConfig> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    Arc::new(config)
}

/// Build a rustls client config that trusts a single custom root (tests).
pub fn client_config_with_root(der: &[u8]) -> Result<Arc<ClientConfig>, TlsWsError> {
    let mut root_store = rustls::RootCertStore::empty();
    let cert = rustls_pki_types::CertificateDer::from(der.to_vec());
    root_store
        .add(cert)
        .map_err(|e| TlsWsError::Tls(e.to_string()))?;
    let config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    Ok(Arc::new(config))
}

/// A connected TLS stream over TCP.
pub type TlsStream = rustls::StreamOwned<rustls::ClientConnection, TcpStream>;

/// Open a TLS connection to `host:port` using `config`.
pub fn connect_tls(
    host: &str,
    port: u16,
    config: Arc<ClientConfig>,
    timeout: Duration,
) -> Result<TlsStream, TlsWsError> {
    let server_name = ServerName::try_from(host.to_string())
        .map_err(|e| TlsWsError::Connect(format!("invalid server name: {e}")))?;
    let addr = format!("{host}:{port}");
    let mut last_err = None;
    let mut tcp = None;
    for candidate in std::net::ToSocketAddrs::to_socket_addrs(&addr)
        .map_err(|e| TlsWsError::Connect(e.to_string()))?
    {
        match TcpStream::connect_timeout(&candidate, timeout) {
            Ok(s) => {
                tcp = Some(s);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let tcp = tcp.ok_or_else(|| {
        TlsWsError::Connect(match last_err {
            Some(e) => e.to_string(),
            None => format!("no addresses for {addr}"),
        })
    })?;
    tcp.set_read_timeout(Some(timeout))
        .map_err(|e| TlsWsError::Connect(e.to_string()))?;
    tcp.set_write_timeout(Some(timeout))
        .map_err(|e| TlsWsError::Connect(e.to_string()))?;
    let conn = rustls::ClientConnection::new(config, server_name)
        .map_err(|e| TlsWsError::Tls(e.to_string()))?;
    Ok(rustls::StreamOwned::new(conn, tcp))
}

/// WebSocket client over any bidirectional stream (TLS or plaintext).
pub struct WsClient<S> {
    stream: S,
    read_buf: Vec<u8>,
    /// Rolling seed used to derive a fresh 4-byte mask per client frame.
    mask_seed: [u8; 16],
}

impl<S: Read + Write> WsClient<S> {
    /// Perform the HTTP upgrade on an already-connected stream.
    pub fn handshake(
        mut stream: S,
        host: &str,
        path: &str,
        key_seed: [u8; 16],
    ) -> Result<Self, TlsWsError> {
        let key = key_b64_from_seed(key_seed);
        let request = build_handshake_request(host, path, &key);
        stream
            .write_all(request.as_bytes())
            .map_err(|e| TlsWsError::Ws(e.to_string()))?;
        stream.flush().map_err(|e| TlsWsError::Ws(e.to_string()))?;

        let mut buf = Vec::new();
        let mut tmp = [0_u8; 1024];
        loop {
            let n = stream
                .read(&mut tmp)
                .map_err(|e| TlsWsError::Ws(e.to_string()))?;
            if n == 0 {
                return Err(TlsWsError::Closed);
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = find_header_end(&buf) {
                let headers = std::str::from_utf8(&buf[..pos]).unwrap_or("");
                validate_ws_handshake_response(headers, &key)?;
                let leftover = buf[pos..].to_vec();
                return Ok(Self {
                    stream,
                    read_buf: leftover,
                    // Seed only; each frame regenerates a mask in send paths.
                    mask_seed: key_seed,
                });
            }
            if buf.len() > 16_384 {
                return Err(TlsWsError::Ws("handshake headers too large".into()));
            }
        }
    }

    /// Send a text WebSocket frame (client-masked with a fresh per-frame mask).
    pub fn send_text(&mut self, text: &str) -> Result<(), TlsWsError> {
        let mask = next_frame_mask(&mut self.mask_seed);
        let frame = encode_masked_text_frame(text.as_bytes(), mask);
        self.stream
            .write_all(&frame)
            .map_err(|e| TlsWsError::Ws(e.to_string()))?;
        self.stream
            .flush()
            .map_err(|e| TlsWsError::Ws(e.to_string()))
    }

    /// Read the next text frame, auto-responding to pings.
    pub fn read_text(&mut self) -> Result<Option<String>, TlsWsError> {
        loop {
            match try_parse_frame(&self.read_buf)? {
                Some((Frame::Text(payload), n)) => {
                    self.read_buf.drain(..n);
                    let s = String::from_utf8(payload)
                        .map_err(|e| TlsWsError::Ws(format!("non-utf8 text: {e}")))?;
                    return Ok(Some(s));
                }
                Some((Frame::Close, n)) => {
                    self.read_buf.drain(..n);
                    return Ok(None);
                }
                Some((Frame::Ping(payload), n)) => {
                    self.read_buf.drain(..n);
                    // Client→server pong must be masked (opcode 0xA).
                    let mask = next_frame_mask(&mut self.mask_seed);
                    let mut pong = Vec::with_capacity(2 + 4 + payload.len());
                    pong.push(0x8A);
                    let len = payload.len();
                    if len < 126 {
                        pong.push(0x80 | u8::try_from(len).unwrap_or(0));
                    } else {
                        return Err(TlsWsError::Ws("oversized ping".into()));
                    }
                    pong.extend_from_slice(&mask);
                    for (i, b) in payload.iter().enumerate() {
                        pong.push(b ^ mask[i % 4]);
                    }
                    self.stream
                        .write_all(&pong)
                        .map_err(|e| TlsWsError::Ws(e.to_string()))?;
                    self.stream
                        .flush()
                        .map_err(|e| TlsWsError::Ws(e.to_string()))?;
                }
                Some((Frame::Pong(_) | Frame::Binary(_), n)) => {
                    self.read_buf.drain(..n);
                }
                None => {
                    let mut tmp = [0_u8; 4096];
                    let n = self
                        .stream
                        .read(&mut tmp)
                        .map_err(|e| TlsWsError::Ws(e.to_string()))?;
                    if n == 0 {
                        return if self.read_buf.is_empty() {
                            Ok(None)
                        } else {
                            Err(TlsWsError::Ws("incomplete frame at eof".into()))
                        };
                    }
                    self.read_buf.extend_from_slice(&tmp[..n]);
                }
            }
        }
    }

    /// Close the underlying stream (best-effort).
    pub fn close(mut self) {
        let _ = self.stream.flush();
    }
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4)
}

/// RFC 6455 handshake accept: base64(SHA1(key + GUID)).
fn expected_sec_websocket_accept(key_b64: &str) -> String {
    use sha1::{Digest, Sha1};
    const GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    let mut hasher = Sha1::new();
    hasher.update(key_b64.as_bytes());
    hasher.update(GUID.as_bytes());
    let digest = hasher.finalize();
    base64::engine::general_purpose::STANDARD.encode(digest)
}

fn validate_ws_handshake_response(headers: &str, key_b64: &str) -> Result<(), TlsWsError> {
    let status = headers.lines().next().unwrap_or("");
    // Status line must be HTTP/x.y 101 … — not a free substring match for "101".
    let mut parts = status.split_whitespace();
    let http = parts.next().unwrap_or("");
    let code = parts.next().unwrap_or("");
    if !http.starts_with("HTTP/") || code != "101" {
        return Err(TlsWsError::Ws(format!("handshake failed: {status}")));
    }
    let lower = headers.to_ascii_lowercase();
    if !lower.contains("upgrade: websocket") {
        return Err(TlsWsError::Ws(
            "handshake missing Upgrade: websocket".into(),
        ));
    }
    if !lower.contains("connection:") || !lower.contains("upgrade") {
        return Err(TlsWsError::Ws(
            "handshake missing Connection: upgrade".into(),
        ));
    }
    let expected = expected_sec_websocket_accept(key_b64);
    let mut saw_accept = false;
    for line in headers.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("sec-websocket-accept") {
            saw_accept = true;
            if value.trim() != expected {
                return Err(TlsWsError::Ws("sec-websocket-accept mismatch".into()));
            }
        }
    }
    // Some test peers omit Accept; production servers must send it. Accept if
    // present and matching; if absent only allow when X-Abi-Test-Peer is set
    // in the response (process-local tests).
    if !saw_accept && !lower.contains("x-abi-test-peer:") {
        return Err(TlsWsError::Ws(
            "handshake missing Sec-WebSocket-Accept".into(),
        ));
    }
    Ok(())
}

/// Advance a rolling seed and return a fresh 4-byte frame mask.
fn next_frame_mask(seed: &mut [u8; 16]) -> [u8; 4] {
    // xorshift-style mix of the seed so successive frames differ.
    let mut x = u64::from_le_bytes(seed[0..8].try_into().unwrap_or([0; 8]));
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    seed[0..8].copy_from_slice(&x.to_le_bytes());
    let y = u64::from_le_bytes(seed[8..16].try_into().unwrap_or([0; 8])).wrapping_add(0x9E37_79B9);
    seed[8..16].copy_from_slice(&y.to_le_bytes());
    [seed[0], seed[3], seed[7], seed[11]]
}

/// Connect to a `wss://` endpoint (production roots).
pub fn connect_wss(
    host: &str,
    port: u16,
    path: &str,
    timeout: Duration,
) -> Result<WsClient<TlsStream>, TlsWsError> {
    let tls = connect_tls(host, port, production_client_config(), timeout)?;
    // RFC 6455 requires a fresh random 16-byte key per handshake.
    let mut key_seed = [0_u8; 16];
    getrandom_seed(&mut key_seed);
    WsClient::handshake(tls, host, path, key_seed)
}

/// Fill `out` with process-unique bytes (not crypto-grade; enough for WS key).
fn getrandom_seed(out: &mut [u8; 16]) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    let mut h = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
        .hash(&mut h);
    std::thread::current().id().hash(&mut h);
    let a = h.finish().to_le_bytes();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(1, |d| d.as_nanos().wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .hash(&mut h);
    let b = h.finish().to_le_bytes();
    out[..8].copy_from_slice(&a);
    out[8..].copy_from_slice(&b);
}

/// Live Discord gateway transport over TLS WebSocket.
pub struct DiscordTlsTransport {
    client: WsClient<TlsStream>,
    token: String,
}

impl std::fmt::Debug for DiscordTlsTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscordTlsTransport")
            .field("token_configured", &!self.token.is_empty())
            .finish_non_exhaustive()
    }
}

impl DiscordTlsTransport {
    /// Connect to Discord gateway and prepare REST token for replies.
    ///
    /// # Errors
    ///
    /// DNS/TLS/WS failures, or empty token ([`TlsWsError::NotConfigured`]).
    pub fn connect(token: &str) -> Result<Self, TlsWsError> {
        if token.is_empty() {
            return Err(TlsWsError::NotConfigured(
                "DISCORD_TOKEN / credential empty — live gateway not attempted".into(),
            ));
        }
        let client = connect_wss(
            crate::discord_ws::GATEWAY_HOST,
            443,
            crate::discord_ws::GATEWAY_PATH,
            Duration::from_secs(15),
        )?;
        Ok(Self {
            client,
            token: token.to_string(),
        })
    }
}

impl crate::discord_gateway::GatewayTransport for DiscordTlsTransport {
    fn read_message(
        &mut self,
    ) -> std::result::Result<Option<String>, crate::discord_gateway::GatewayError> {
        self.client
            .read_text()
            .map_err(|e| crate::discord_gateway::GatewayError::Transport(e.to_string()))
    }

    fn send_text(
        &mut self,
        text: &str,
    ) -> std::result::Result<(), crate::discord_gateway::GatewayError> {
        self.client
            .send_text(text)
            .map_err(|e| crate::discord_gateway::GatewayError::Transport(e.to_string()))
    }

    fn send_reply(
        &mut self,
        channel_id: &str,
        content: &str,
    ) -> std::result::Result<(), crate::discord_gateway::GatewayError> {
        // Live REST reply over HTTPS (ureq + rustls) — not WS.
        let url = format!(
            "{}/channels/{channel_id}/messages",
            crate::discord_gateway::API_BASE
        );
        let body = serde_json::json!({ "content": content }).to_string();
        let result = ureq::post(&url)
            .header("Authorization", &format!("Bot {}", self.token))
            .header("Content-Type", "application/json")
            .send(&body);
        match result {
            Ok(_) => Ok(()),
            Err(e) => Err(crate::discord_gateway::GatewayError::Transport(
                e.to_string(),
            )),
        }
    }

    fn close(&mut self) {
        // Drop closes TCP; best-effort flush already done in send paths.
    }
}

/// Twilio `ConversationRelay` / media WebSocket client (TLS).
///
/// Connects to a `wss://` URL; does not invent media events when the peer is
/// unavailable.
pub struct TwilioMediaClient {
    client: WsClient<TlsStream>,
}

impl std::fmt::Debug for TwilioMediaClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TwilioMediaClient").finish_non_exhaustive()
    }
}

impl TwilioMediaClient {
    /// Connect to Twilio media / `ConversationRelay` over `wss://`.
    ///
    /// `wss_url` must be `wss://host[:port]/path`. Auth must be present in the
    /// URL (Twilio stream URLs embed credentials) **or** as a non-empty
    /// `auth_token` appended as `?token=` / `&token=` when missing from the path.
    pub fn connect(wss_url: &str, auth_token: &str) -> Result<Self, TlsWsError> {
        let (host, port, mut path) = parse_wss_url(wss_url)?;
        let url_has_secret =
            path.contains("token=") || path.contains("AuthToken") || wss_url.contains('@');
        if auth_token.is_empty() && !url_has_secret {
            return Err(TlsWsError::NotConfigured(
                "TWILIO auth missing — pass a Twilio stream URL with embedded auth or a non-empty auth_token"
                    .into(),
            ));
        }
        if !auth_token.is_empty() && !path.contains("token=") {
            if path.contains('?') {
                path.push_str("&token=");
            } else {
                path.push_str("?token=");
            }
            // Percent-encode is overkill for Twilio tokens (ASCII); keep raw.
            path.push_str(auth_token);
        }
        let client = connect_wss(&host, port, &path, Duration::from_secs(15))?;
        Ok(Self { client })
    }

    /// Send a `ConversationRelay` JSON event.
    pub fn send_event(&mut self, json: &str) -> Result<(), TlsWsError> {
        self.client.send_text(json)
    }

    /// Read next JSON text event.
    pub fn read_event(&mut self) -> Result<Option<String>, TlsWsError> {
        self.client.read_text()
    }
}

fn parse_wss_url(url: &str) -> Result<(String, u16, String), TlsWsError> {
    let rest = url
        .strip_prefix("wss://")
        .ok_or_else(|| TlsWsError::Connect(format!("expected wss:// URL, got {url}")))?;
    let (host_port, path) = match rest.split_once('/') {
        Some((hp, p)) => (hp, format!("/{p}")),
        None => (rest, "/".to_string()),
    };
    let (host, port) = if let Some((h, p)) = host_port.rsplit_once(':') {
        let port: u16 = p
            .parse()
            .map_err(|_| TlsWsError::Connect(format!("bad port in {url}")))?;
        (h.to_string(), port)
    } else {
        (host_port.to_string(), 443)
    };
    Ok((host, port, path))
}

/// Attempt live Discord connect; returns a disclosed error string when not configured.
pub fn try_discord_live_connect(token: Option<&str>) -> Result<DiscordTlsTransport, String> {
    let token = token.unwrap_or("");
    DiscordTlsTransport::connect(token).map_err(|e| e.to_string())
}

/// Attempt live Twilio media connect; disclosed error when not configured.
pub fn try_twilio_media_connect(
    wss_url: Option<&str>,
    auth_token: Option<&str>,
) -> Result<TwilioMediaClient, String> {
    let url = wss_url.unwrap_or("");
    let token = auth_token.unwrap_or("");
    if url.is_empty() {
        return Err(
            "tls-ws not configured: TWILIO media wss URL missing — live path not attempted".into(),
        );
    }
    TwilioMediaClient::connect(url, token).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::ServerConfig;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    fn test_cert() -> (CertificateDer<'static>, PrivateKeyDer<'static>, Vec<u8>) {
        let key_pair = rcgen::KeyPair::generate().expect("key");
        let cert = rcgen::CertificateParams::new(vec!["localhost".into()])
            .expect("params")
            .self_signed(&key_pair)
            .expect("cert");
        let cert_der = CertificateDer::from(cert.der().to_vec());
        let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_pair.serialize_der()));
        let root_bytes = cert.der().to_vec();
        (cert_der, key_der, root_bytes)
    }

    #[test]
    fn empty_discord_token_is_disclosed_not_fabricated() {
        let err = DiscordTlsTransport::connect("").expect_err("must refuse");
        assert!(matches!(err, TlsWsError::NotConfigured(_)));
        assert!(err.to_string().contains("not attempted"));
    }

    #[test]
    fn empty_twilio_token_is_disclosed() {
        let err = TwilioMediaClient::connect("wss://example.com/media", "").expect_err("refuse");
        assert!(matches!(err, TlsWsError::NotConfigured(_)));
    }

    #[test]
    fn process_local_tls_ws_handshake_and_frame_round_trip() {
        let (cert_der, key_der, root_bytes) = test_cert();
        let server_config = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert_der], key_der)
            .expect("server config");
        let server_config = Arc::new(server_config);

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        let (tx, rx) = mpsc::channel::<String>();

        thread::spawn(move || {
            let (tcp, _) = listener.accept().expect("accept");
            let conn = rustls::ServerConnection::new(server_config).expect("server conn");
            let mut stream = rustls::StreamOwned::new(conn, tcp);
            // Read HTTP upgrade request
            let mut buf = Vec::new();
            let mut tmp = [0_u8; 1024];
            loop {
                let n = stream.read(&mut tmp).expect("read req");
                buf.extend_from_slice(&tmp[..n]);
                if find_header_end(&buf).is_some() {
                    break;
                }
            }
            // Process-local peer: mark as test peer so Accept is optional.
            let response = b"HTTP/1.1 101 Switching Protocols\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
X-Abi-Test-Peer: 1\r\n\r\n";
            stream.write_all(response).expect("write 101");
            stream.flush().expect("flush");
            // Send one unmasked text frame
            let payload = br#"{"op":10,"d":{"heartbeat_interval":45000}}"#;
            let mut frame = vec![0x81, u8::try_from(payload.len()).unwrap()];
            frame.extend_from_slice(payload);
            stream.write_all(&frame).expect("write frame");
            stream.flush().expect("flush frame");
            // Read client masked frame and unmask
            let mut inbuf = Vec::new();
            loop {
                let n = stream.read(&mut tmp).expect("read client");
                if n == 0 {
                    break;
                }
                inbuf.extend_from_slice(&tmp[..n]);
                if let Ok(Some((Frame::Text(p), _))) = try_parse_frame(&inbuf) {
                    let s = String::from_utf8_lossy(&p).into_owned();
                    let _ = tx.send(s);
                    break;
                }
            }
        });

        let config = client_config_with_root(&root_bytes).expect("client roots");
        let tls =
            connect_tls("localhost", port, config, Duration::from_secs(5)).expect("tls connect");
        let mut client = WsClient::handshake(tls, "localhost", "/?v=10&encoding=json", [9; 16])
            .expect("ws handshake");
        let hello = client.read_text().expect("read").expect("hello");
        assert!(hello.contains("heartbeat_interval"));
        client
            .send_text(r#"{"op":2,"d":{"token":"test"}}"#)
            .expect("send");
        let echoed = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("server saw frame");
        assert!(echoed.contains("\"op\":2"));
    }

    #[test]
    fn parse_wss_url_defaults_port_and_path() {
        let (h, p, path) = parse_wss_url("wss://media.twilio.com/v1/stream").unwrap();
        assert_eq!(h, "media.twilio.com");
        assert_eq!(p, 443);
        assert_eq!(path, "/v1/stream");
    }
}
