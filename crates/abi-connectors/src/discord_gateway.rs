//! Discord gateway bot loop with injectable transport.
//!
//! The loop is fully testable via [`FakeTransport`] with no network. Live
//! `wss://` uses [`crate::tls_ws::DiscordTlsTransport`] (rustls) when the
//! `live` feature is enabled — credentials missing → disclosed error.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::match_same_arms
)]

use crate::discord_routing::{MAX_MESSAGE_CONTENT_BYTES, route_discord_message, truncate};
use crate::payload::build_discord_identify_body;
use crate::{ConnectorError, Result};
use serde_json::Value;

/// Discord gateway WebSocket endpoint (v10, JSON encoding).
pub const GATEWAY_URL: &str = "wss://gateway.discord.gg/?v=10&encoding=json";
/// Discord REST API base.
pub const API_BASE: &str = "https://discord.com/api/v10";
/// Default gateway intents: guild messages | DMs | message content.
pub const DEFAULT_INTENTS: u32 = (1 << 9) | (1 << 12) | (1 << 15);

/// Gateway run configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayConfig {
    /// Bot token (printable ASCII, non-empty for identify).
    pub token: String,
    /// Whether a token is configured (drives `!status` without exposing it).
    pub token_configured: bool,
    /// Gateway intents bitmask.
    pub intents: u32,
    /// Command prefix (empty disables all replies).
    pub prefix: String,
}

impl GatewayConfig {
    /// Config with default intents and `!` prefix.
    #[must_use]
    pub fn new(token: impl Into<String>, token_configured: bool) -> Self {
        Self {
            token: token.into(),
            token_configured,
            intents: DEFAULT_INTENTS,
            prefix: "!".into(),
        }
    }
}

/// Outcome of a gateway run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GatewayStats {
    /// Gateway frames processed after Hello.
    pub message_events: usize,
    /// REST replies sent.
    pub replies_sent: usize,
    /// Heartbeat (op 1) frames sent.
    pub heartbeats_sent: usize,
    /// Whether Identify was sent at least once.
    pub identified: bool,
}

/// Errors specific to the gateway loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    /// Transport closed before Hello.
    ClosedEarly,
    /// Hello lacked `heartbeat_interval`.
    MissingHeartbeatInterval,
    /// Token failed validation before Identify.
    Authentication,
    /// Transport I/O failure.
    Transport(String),
}

impl std::fmt::Display for GatewayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ClosedEarly => write!(f, "gateway closed early"),
            Self::MissingHeartbeatInterval => write!(f, "missing heartbeat interval"),
            Self::Authentication => write!(f, "authentication error"),
            Self::Transport(msg) => write!(f, "gateway transport: {msg}"),
        }
    }
}

impl std::error::Error for GatewayError {}

/// Transport surface used by [`Gateway::run`].
pub trait GatewayTransport {
    /// Read the next JSON text message (owned), or `None` at end-of-stream.
    fn read_message(&mut self) -> std::result::Result<Option<String>, GatewayError>;
    /// Send a JSON text frame (gateway opcodes).
    fn send_text(&mut self, text: &str) -> std::result::Result<(), GatewayError>;
    /// Send a REST reply to `channel_id`.
    fn send_reply(
        &mut self,
        channel_id: &str,
        content: &str,
    ) -> std::result::Result<(), GatewayError>;
    /// Close the transport.
    fn close(&mut self);
}

/// The bot loop driver.
#[derive(Debug, Clone)]
pub struct Gateway {
    config: GatewayConfig,
}

impl Gateway {
    /// Construct a gateway with `config`.
    #[must_use]
    pub const fn new(config: GatewayConfig) -> Self {
        Self { config }
    }

    /// Run the gateway loop over `transport`.
    ///
    /// `max_message_events` bounds post-Hello frames processed (`None` = until
    /// the transport ends).
    pub fn run(
        &self,
        transport: &mut dyn GatewayTransport,
        max_message_events: Option<usize>,
    ) -> std::result::Result<GatewayStats, GatewayError> {
        let mut stats = GatewayStats::default();
        let mut seq: Option<i64> = None;

        let hello = transport.read_message()?.ok_or(GatewayError::ClosedEarly)?;
        parse_heartbeat_interval(&hello).ok_or(GatewayError::MissingHeartbeatInterval)?;

        let identify = build_identify(&self.config)?;
        transport.send_text(&identify)?;
        stats.identified = true;

        loop {
            if max_message_events.is_some_and(|m| stats.message_events >= m) {
                break;
            }
            let Some(msg) = transport.read_message()? else {
                break;
            };
            stats.message_events += 1;

            let Some(op) = parse_op(&msg) else {
                continue;
            };
            match op {
                1 => {
                    let hb = build_heartbeat(seq);
                    transport.send_text(&hb)?;
                    stats.heartbeats_sent += 1;
                }
                10 => {
                    let identify2 = build_identify(&self.config)?;
                    transport.send_text(&identify2)?;
                    stats.identified = true;
                }
                11 => {}
                0 => {
                    if let Some(s) = parse_seq(&msg) {
                        seq = Some(s);
                    }
                    if let Some(mc) = extract_message_create(&msg) {
                        if mc.author_bot {
                            continue;
                        }
                        let Some(reply) = route_discord_message(
                            &mc.content,
                            &self.config.prefix,
                            self.config.token_configured,
                        ) else {
                            continue;
                        };
                        let truncated = truncate(&reply, MAX_MESSAGE_CONTENT_BYTES);
                        transport.send_reply(&mc.channel_id, &truncated)?;
                        stats.replies_sent += 1;
                    }
                }
                _ => {}
            }
        }

        transport.close();
        Ok(stats)
    }
}

/// Offline transport for tests: scripted inbound frames + captured outbound.
#[derive(Debug, Default)]
pub struct FakeTransport {
    incoming: Vec<String>,
    /// JSON text frames sent via [`GatewayTransport::send_text`].
    pub outgoing: Vec<String>,
    /// REST replies `(channel_id, content)`.
    pub replies: Vec<(String, String)>,
    /// Whether [`GatewayTransport::close`] was called.
    pub closed: bool,
}

impl FakeTransport {
    /// Empty transport.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue an inbound JSON message.
    pub fn enqueue(&mut self, msg: impl Into<String>) {
        self.incoming.push(msg.into());
    }
}

impl GatewayTransport for FakeTransport {
    fn read_message(&mut self) -> std::result::Result<Option<String>, GatewayError> {
        if self.incoming.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.incoming.remove(0)))
        }
    }

    fn send_text(&mut self, text: &str) -> std::result::Result<(), GatewayError> {
        self.outgoing.push(text.to_string());
        Ok(())
    }

    fn send_reply(
        &mut self,
        channel_id: &str,
        content: &str,
    ) -> std::result::Result<(), GatewayError> {
        self.replies
            .push((channel_id.to_string(), content.to_string()));
        Ok(())
    }

    fn close(&mut self) {
        self.closed = true;
    }
}

/// Validate a Discord bot token (printable non-whitespace ASCII, non-empty).
pub fn validate_token(token: &str) -> Result<()> {
    if token.is_empty() {
        return Err(ConnectorError::AuthenticationError);
    }
    for b in token.bytes() {
        if b.is_ascii_whitespace() || !(0x21..=0x7e).contains(&b) {
            return Err(ConnectorError::AuthenticationError);
        }
    }
    Ok(())
}

/// Validate a Discord snowflake id (1–32 decimal digits).
pub fn validate_discord_id(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 32 || !id.bytes().all(|b| b.is_ascii_digit()) {
        return Err(ConnectorError::InvalidResponse);
    }
    Ok(())
}

/// Validate message content length (1–2000 bytes).
pub fn validate_message_content(content: &str) -> Result<()> {
    if content.is_empty() || content.len() > 2000 {
        return Err(ConnectorError::InvalidResponse);
    }
    Ok(())
}

fn build_identify(config: &GatewayConfig) -> std::result::Result<String, GatewayError> {
    validate_token(&config.token).map_err(|_| GatewayError::Authentication)?;
    Ok(build_discord_identify_body(&config.token, config.intents))
}

fn build_heartbeat(seq: Option<i64>) -> String {
    match seq {
        Some(s) => format!(r#"{{"op":1,"d":{s}}}"#),
        None => r#"{"op":1,"d":null}"#.to_string(),
    }
}

fn parse_op(msg: &str) -> Option<i64> {
    let v: Value = serde_json::from_str(msg).ok()?;
    v.get("op")?.as_i64()
}

fn parse_seq(msg: &str) -> Option<i64> {
    let v: Value = serde_json::from_str(msg).ok()?;
    v.get("s")?.as_i64()
}

fn parse_heartbeat_interval(msg: &str) -> Option<i64> {
    let v: Value = serde_json::from_str(msg).ok()?;
    let d = v.get("d")?;
    d.get("heartbeat_interval")
        .and_then(|x| x.as_i64().or_else(|| x.as_f64().map(|f| f as i64)))
}

struct MessageCreate {
    channel_id: String,
    content: String,
    author_bot: bool,
}

fn extract_message_create(msg: &str) -> Option<MessageCreate> {
    let v: Value = serde_json::from_str(msg).ok()?;
    if v.get("op")?.as_i64()? != 0 {
        return None;
    }
    if v.get("t")?.as_str()? != "MESSAGE_CREATE" {
        return None;
    }
    let d = v.get("d")?;
    let channel_id = d.get("channel_id")?.as_str()?.to_string();
    let content = d.get("content")?.as_str()?.to_string();
    let author_bot = d
        .get("author")
        .and_then(|a| a.get("bot"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Some(MessageCreate {
        channel_id,
        content,
        author_bot,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_loop_hello_identify_heartbeat_message_create() {
        let mut transport = FakeTransport::new();
        transport.enqueue(r#"{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}"#);
        transport.enqueue(r#"{"op":1,"d":null,"s":null,"t":null}"#);
        transport.enqueue(
            r#"{"op":0,"s":101,"t":"MESSAGE_CREATE","d":{"channel_id":"123","author":{"bot":false},"content":"!governance"}}"#,
        );
        transport.enqueue(
            r#"{"op":0,"s":102,"t":"MESSAGE_CREATE","d":{"channel_id":"999","author":{"bot":true},"content":"!help"}}"#,
        );

        let gw = Gateway::new(GatewayConfig::new("SECRET_TOKEN", true));
        let stats = gw.run(&mut transport, None).expect("run");
        assert!(stats.identified);
        assert_eq!(stats.heartbeats_sent, 1);
        assert_eq!(stats.replies_sent, 1);
        assert_eq!(transport.outgoing.len(), 2);
        assert_eq!(transport.replies[0].0, "123");
        assert!(!transport.replies[0].1.contains("SECRET_TOKEN"));
        assert!(transport.closed);
        assert!(transport.replies[0].1.contains("constitutional"));
    }

    #[test]
    fn gateway_loop_honors_max_message_events() {
        let mut transport = FakeTransport::new();
        transport.enqueue(r#"{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}"#);
        transport.enqueue(
            r#"{"op":0,"s":1,"t":"MESSAGE_CREATE","d":{"channel_id":"1","author":{"bot":false},"content":"!help"}}"#,
        );
        transport.enqueue(
            r#"{"op":0,"s":2,"t":"MESSAGE_CREATE","d":{"channel_id":"2","author":{"bot":false},"content":"!status"}}"#,
        );
        let gw = Gateway::new(GatewayConfig::new("T", false));
        let stats = gw.run(&mut transport, Some(2)).expect("run");
        assert_eq!(stats.message_events, 2);
    }

    #[test]
    fn identify_escapes_json_breaking_token_characters() {
        let mut transport = FakeTransport::new();
        transport.enqueue(r#"{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}"#);
        let gw = Gateway::new(GatewayConfig::new(r#"weird"token\value"#, true));
        let stats = gw.run(&mut transport, Some(1)).expect("run");
        assert!(stats.identified);
        let identify: Value = serde_json::from_str(&transport.outgoing[0]).expect("json");
        assert_eq!(
            identify["d"]["token"].as_str().unwrap(),
            r#"weird"token\value"#
        );
    }

    #[test]
    fn empty_token_rejects_before_sending() {
        let mut transport = FakeTransport::new();
        transport.enqueue(r#"{"op":10,"d":{"heartbeat_interval":45000},"s":null,"t":null}"#);
        let gw = Gateway::new(GatewayConfig::new("", false));
        let err = gw.run(&mut transport, None).unwrap_err();
        assert_eq!(err, GatewayError::Authentication);
        assert!(transport.outgoing.is_empty());
    }

    #[test]
    fn validate_token_and_snowflake() {
        assert!(validate_token("abc.DEF-123").is_ok());
        assert!(validate_token("").is_err());
        assert!(validate_token("has space").is_err());
        assert!(validate_discord_id("123456789012345678").is_ok());
        assert!(validate_discord_id("abc").is_err());
        assert!(validate_message_content("hi").is_ok());
        assert!(validate_message_content("").is_err());
    }
}
