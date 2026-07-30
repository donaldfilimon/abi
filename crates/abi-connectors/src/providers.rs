//! Provider clients: `OpenAI`, Anthropic, Grok, Discord, Twilio.
//!
//! Ported from `openai.zig`, `anthropic.zig`, `grok.zig`, `discord.zig` and
//! `twilio.zig`. Each Zig file repeated the same four-method shape — request,
//! request-live, stream, stream-live — differing only in path, auth header and
//! body builder. Here that shape lives once in [`Client`] and each provider
//! contributes a [`Provider`] describing its differences.
//!
//! The local/live split is preserved exactly, including the rule that a *local*
//! call on a config marked live returns [`ConnectorError::LiveTransportUnavailable`]
//! rather than quietly synthesizing a response and passing it off as the
//! provider's.

use crate::connector::{ConnectorConfig, ConnectorError, Response, Result, TransportMode};
use crate::payload;
use crate::transport::{Header, Method, Request, Transport, build_request, execute};
use crate::url::{basic_auth_header, bearer_header, bot_header};
use std::fmt::Write as _;

/// Which provider a [`Client`] talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// `OpenAI`, or anything speaking its chat-completions API.
    OpenAi,
    /// Anthropic's messages API.
    Anthropic,
    /// xAI Grok, which speaks the `OpenAI` chat-completions API.
    Grok,
}

impl Provider {
    /// The request path for a completion.
    #[must_use]
    pub const fn completion_path(self) -> &'static str {
        match self {
            Self::OpenAi | Self::Grok => "/v1/chat/completions",
            Self::Anthropic => "/v1/messages",
        }
    }

    /// The provider name used in log and diagnostic output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Grok => "grok",
        }
    }

    /// Authorization headers for this provider.
    ///
    /// Anthropic uses `x-api-key` plus a required `anthropic-version`; the others
    /// use `Authorization: Bearer`.
    #[must_use]
    fn auth_headers(self, api_key: &str) -> Vec<Header> {
        match self {
            Self::OpenAi | Self::Grok => {
                vec![Header::new("authorization", bearer_header(api_key))]
            }
            Self::Anthropic => vec![
                Header::new("x-api-key", api_key),
                Header::new("anthropic-version", ANTHROPIC_VERSION),
            ],
        }
    }
}

/// The Anthropic API version header value.
///
/// Required by that API; omitting it is a 400.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default `max_tokens` for Anthropic requests when the caller does not choose.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

/// A chat client for one provider.
#[derive(Debug, Clone)]
pub struct Client {
    provider: Provider,
    config: ConnectorConfig,
}

impl Client {
    /// A client for `provider` using `config`.
    #[must_use]
    pub fn new(provider: Provider, config: ConnectorConfig) -> Self {
        Self { provider, config }
    }

    /// An `OpenAI` client.
    #[must_use]
    pub fn openai(config: ConnectorConfig) -> Self {
        Self::new(Provider::OpenAi, config)
    }

    /// An Anthropic client.
    #[must_use]
    pub fn anthropic(config: ConnectorConfig) -> Self {
        Self::new(Provider::Anthropic, config)
    }

    /// A Grok client.
    #[must_use]
    pub fn grok(config: ConnectorConfig) -> Self {
        Self::new(Provider::Grok, config)
    }

    /// This client's provider.
    #[must_use]
    pub const fn provider(&self) -> Provider {
        self.provider
    }

    /// This client's configuration.
    #[must_use]
    pub const fn config(&self) -> &ConnectorConfig {
        &self.config
    }

    /// Refuse a local-transport call on a live-configured client.
    ///
    /// The rule from Zig: once a caller asks for live, a synthesized response
    /// must never be returned in its place, because it would be indistinguishable
    /// from a real one downstream.
    fn require_local(&self) -> Result<()> {
        self.config.validate()?;
        if self.config.transport == TransportMode::Live {
            return Err(ConnectorError::LiveTransportUnavailable);
        }
        Ok(())
    }

    /// A chat completion on the local transport, with no network call.
    ///
    /// `messages` must be a JSON array for `OpenAI` and Grok. For Anthropic it is
    /// treated as a plain prompt string.
    pub fn complete(&self, model: &str, messages: &str) -> Result<Response> {
        self.require_local()?;
        match self.provider {
            Provider::OpenAi | Provider::Grok => {
                let body = payload::build_openai_body(model, messages, false)?;
                Ok(Response::ok(payload::openai_local_response(
                    model,
                    messages,
                    body.len(),
                )))
            }
            Provider::Anthropic => {
                let body =
                    payload::build_anthropic_body(model, messages, DEFAULT_MAX_TOKENS, false);
                Ok(Response::ok(payload::anthropic_local_response(
                    model,
                    messages,
                    DEFAULT_MAX_TOKENS,
                    body.len(),
                )))
            }
        }
    }

    /// A streaming chat completion on the local transport.
    pub fn complete_stream(&self, model: &str, messages: &str) -> Result<Response> {
        self.require_local()?;
        match self.provider {
            Provider::OpenAi | Provider::Grok => {
                let body = payload::build_openai_body(model, messages, true)?;
                Ok(Response::ok(payload::openai_local_stream(
                    model,
                    messages,
                    body.len(),
                )))
            }
            Provider::Anthropic => {
                let body = payload::build_anthropic_body(model, messages, DEFAULT_MAX_TOKENS, true);
                Ok(Response::ok(payload::anthropic_local_stream(
                    model,
                    messages,
                    DEFAULT_MAX_TOKENS,
                    body.len(),
                )))
            }
        }
    }

    /// Build the live request for a completion, without issuing it.
    ///
    /// Exposed so the request can be asserted on in tests, which the Zig helpers
    /// made impossible.
    pub fn build_completion_request(
        &self,
        model: &str,
        messages: &str,
        stream: bool,
    ) -> Result<Request> {
        let body = match self.provider {
            Provider::OpenAi | Provider::Grok => {
                payload::build_openai_body(model, messages, stream)?
            }
            Provider::Anthropic => {
                payload::build_anthropic_body(model, messages, DEFAULT_MAX_TOKENS, stream)
            }
        };
        build_request(
            &self.config,
            Method::Post,
            self.provider.completion_path(),
            Some("application/json"),
            Some(body),
            self.provider.auth_headers(&self.config.api_key),
        )
    }

    /// A chat completion over the live transport.
    pub fn complete_live(
        &self,
        transport: &dyn Transport,
        model: &str,
        messages: &str,
    ) -> Result<Response> {
        let request = self.build_completion_request(model, messages, false)?;
        execute(transport, &request)
    }

    /// A streaming chat completion over the live transport.
    ///
    /// Returns the raw SSE body; feed it to [`crate::sse::parse_stream`].
    pub fn complete_stream_live(
        &self,
        transport: &dyn Transport,
        model: &str,
        messages: &str,
    ) -> Result<Response> {
        let request = self.build_completion_request(model, messages, true)?;
        execute(transport, &request)
    }
}

/// A Discord client.
#[derive(Debug, Clone)]
pub struct DiscordClient {
    config: ConnectorConfig,
}

impl DiscordClient {
    /// A Discord client. The `api_key` is the bot token.
    #[must_use]
    pub fn new(config: ConnectorConfig) -> Self {
        Self { config }
    }

    /// Acknowledge a message send on the local transport, without a network call.
    pub fn send_message(&self, channel_id: &str, content: &str) -> Result<Response> {
        self.config.validate()?;
        if self.config.transport == TransportMode::Live {
            return Err(ConnectorError::LiveTransportUnavailable);
        }
        Ok(Response::ok(payload::discord_local_ack(
            channel_id, content,
        )))
    }

    /// Build the live create-message request.
    pub fn build_send_request(&self, channel_id: &str, content: &str) -> Result<Request> {
        build_request(
            &self.config,
            Method::Post,
            &format!("/api/v10/channels/{channel_id}/messages"),
            Some("application/json"),
            Some(payload::build_discord_message_body(content)),
            vec![Header::new(
                "authorization",
                bot_header(&self.config.api_key),
            )],
        )
    }

    /// Send a message over the live transport.
    pub fn send_message_live(
        &self,
        transport: &dyn Transport,
        channel_id: &str,
        content: &str,
    ) -> Result<Response> {
        let request = self.build_send_request(channel_id, content)?;
        execute(transport, &request)
    }
}

/// A Twilio client.
///
/// Twilio authenticates with HTTP Basic (account SID as the username, auth token
/// as the password) and takes form-encoded bodies rather than JSON.
#[derive(Debug, Clone)]
pub struct TwilioClient {
    account_sid: String,
    auth_token: String,
    config: ConnectorConfig,
}

impl TwilioClient {
    /// A Twilio client.
    #[must_use]
    pub fn new(
        account_sid: impl Into<String>,
        auth_token: impl Into<String>,
        config: ConnectorConfig,
    ) -> Self {
        Self {
            account_sid: account_sid.into(),
            auth_token: auth_token.into(),
            config,
        }
    }

    /// The account SID.
    #[must_use]
    pub fn account_sid(&self) -> &str {
        &self.account_sid
    }

    /// Build the live send-message request.
    pub fn build_message_request(&self, to: &str, from: &str, body: &str) -> Result<Request> {
        if self.account_sid.is_empty() || self.auth_token.is_empty() {
            return Err(ConnectorError::AuthenticationError);
        }
        let form = format!(
            "To={}&From={}&Body={}",
            form_encode(to),
            form_encode(from),
            form_encode(body)
        );
        build_request(
            &self.config,
            Method::Post,
            &format!("/2010-04-01/Accounts/{}/Messages.json", self.account_sid),
            Some("application/x-www-form-urlencoded"),
            Some(form),
            vec![Header::new(
                "authorization",
                basic_auth_header(&self.account_sid, &self.auth_token),
            )],
        )
    }

    /// Send a message over the live transport.
    pub fn send_message_live(
        &self,
        transport: &dyn Transport,
        to: &str,
        from: &str,
        body: &str,
    ) -> Result<Response> {
        let request = self.build_message_request(to, from, body)?;
        execute(transport, &request)
    }
}

/// Percent-encode a form field value.
///
/// Only unreserved characters pass through. Notably `+` is encoded as `%2B`
/// rather than left alone: in `application/x-www-form-urlencoded` a literal `+`
/// decodes as a space, which would silently corrupt every E.164 phone number
/// Twilio is given.
#[must_use]
pub fn form_encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(char::from(byte));
            }
            _ => {
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::RecordingTransport;

    fn local(base: &str) -> ConnectorConfig {
        ConnectorConfig::new("test-key", base)
    }

    fn live() -> ConnectorConfig {
        ConnectorConfig::new("sk-test", "https://api.example.com").live()
    }

    #[test]
    fn local_openai_completion_synthesizes_a_parseable_response() {
        let client = Client::openai(local("http://local.invalid"));
        let response = client
            .complete("gpt-4", r#"[{"role":"user","content":"hi"}]"#)
            .expect("local completion");
        assert_eq!(response.status, 200);
        assert!(response.json().is_ok());
        assert!(response.body.contains("model=gpt-4"));
    }

    #[test]
    fn local_openai_completion_rejects_non_array_messages() {
        let client = Client::openai(local("http://local.invalid"));
        assert_eq!(
            client.complete("gpt-4", r#"{"not":"array"}"#).unwrap_err(),
            ConnectorError::InvalidResponse
        );
    }

    #[test]
    fn local_openai_stream_is_sse_shaped() {
        let client = Client::openai(local("http://local.invalid"));
        let response = client.complete_stream("gpt-4", "[]").expect("local stream");
        assert!(response.body.starts_with("data: "));
        assert!(response.body.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn a_live_config_refuses_local_synthesis() {
        // The critical rule: never pass a synthesized response off as the
        // provider's, because downstream cannot tell the difference.
        let client = Client::openai(live());
        assert_eq!(
            client.complete("gpt-4", "[]").unwrap_err(),
            ConnectorError::LiveTransportUnavailable
        );
        assert_eq!(
            client.complete_stream("gpt-4", "[]").unwrap_err(),
            ConnectorError::LiveTransportUnavailable
        );
    }

    #[test]
    fn a_local_call_still_validates_the_config() {
        let client = Client::openai(ConnectorConfig::new("", "http://local.invalid"));
        assert_eq!(
            client.complete("gpt-4", "[]").unwrap_err(),
            ConnectorError::AuthenticationError
        );
    }

    #[test]
    fn local_anthropic_completion_uses_the_anthropic_shape() {
        let client = Client::anthropic(local("http://local.invalid"));
        let response = client.complete("claude-opus-5", "hello").expect("local");
        let parsed = response.json().expect("valid JSON");
        assert_eq!(parsed["content"][0]["type"], "text");
        assert!(
            parsed["content"][0]["text"]
                .as_str()
                .expect("text")
                .contains("max_tokens=4096")
        );
    }

    #[test]
    fn grok_speaks_the_openai_dialect() {
        assert_eq!(
            Provider::Grok.completion_path(),
            Provider::OpenAi.completion_path()
        );
        let client = Client::grok(local("http://local.invalid"));
        let response = client.complete("grok-2", "[]").expect("local");
        assert!(response.body.contains("OpenAI-compatible local response"));
    }

    #[test]
    fn live_openai_request_targets_the_right_url_with_bearer_auth() {
        let transport = RecordingTransport::new().with_response(Ok(Response::ok("{}")));
        let client = Client::openai(live());
        client
            .complete_live(&transport, "gpt-4", "[]")
            .expect("live call");

        let request = transport.sole_request();
        assert_eq!(request.method, Method::Post);
        assert_eq!(request.url, "https://api.example.com/v1/chat/completions");
        assert_eq!(request.content_type, Some("application/json"));
        assert_eq!(
            request.headers,
            vec![Header::new("authorization", "Bearer sk-test")]
        );
        assert_eq!(
            request.body.as_deref(),
            Some(r#"{"model":"gpt-4","messages":[]}"#)
        );
    }

    #[test]
    fn live_anthropic_request_sends_x_api_key_and_a_version() {
        // Anthropic rejects a request with no anthropic-version header.
        let transport = RecordingTransport::new().with_response(Ok(Response::ok("{}")));
        let client = Client::anthropic(live());
        client
            .complete_live(&transport, "claude-opus-5", "hi")
            .expect("live call");

        let request = transport.sole_request();
        assert_eq!(request.url, "https://api.example.com/v1/messages");
        assert_eq!(
            request.headers,
            vec![
                Header::new("x-api-key", "sk-test"),
                Header::new("anthropic-version", ANTHROPIC_VERSION),
            ]
        );
        // Never Bearer for Anthropic.
        assert!(
            !request
                .headers
                .iter()
                .any(|h| h.name.eq_ignore_ascii_case("authorization"))
        );
    }

    #[test]
    fn a_live_stream_request_sets_the_stream_flag() {
        let transport = RecordingTransport::new().with_response(Ok(Response::ok("")));
        let client = Client::openai(live());
        client
            .complete_stream_live(&transport, "gpt-4", "[]")
            .expect("live stream");
        assert!(
            transport
                .sole_request()
                .body
                .expect("body")
                .contains(r#""stream":true"#)
        );
    }

    #[test]
    fn a_live_call_over_cleartext_is_refused_before_any_request() {
        let transport = RecordingTransport::new();
        let client = Client::openai(ConnectorConfig::new("sk", "http://api.example.com").live());
        assert_eq!(
            client.complete_live(&transport, "gpt-4", "[]").unwrap_err(),
            ConnectorError::InsecureBaseUrl
        );
        assert!(
            transport.requests().is_empty(),
            "nothing may be sent to a cleartext URL"
        );
    }

    #[test]
    fn a_provider_error_status_is_mapped() {
        let transport = RecordingTransport::new().with_response(Ok(Response {
            status: 401,
            body: String::new(),
        }));
        let client = Client::openai(live());
        assert_eq!(
            client.complete_live(&transport, "gpt-4", "[]").unwrap_err(),
            ConnectorError::AuthenticationError
        );
    }

    #[test]
    fn discord_local_send_acknowledges_without_a_network_call() {
        let client = DiscordClient::new(local("http://local.invalid"));
        let response = client.send_message("123", "hello").expect("local ack");
        let parsed = response.json().expect("valid JSON");
        assert_eq!(parsed["status"], "queued-local");
        assert_eq!(parsed["channel_id"], "123");
        assert_eq!(parsed["content_bytes"], 5);
    }

    #[test]
    fn discord_live_send_uses_bot_auth_and_the_v10_path() {
        let transport = RecordingTransport::new().with_response(Ok(Response::ok("{}")));
        let client = DiscordClient::new(ConnectorConfig::new("tok", "https://discord.com").live());
        client
            .send_message_live(&transport, "42", "hi")
            .expect("live send");

        let request = transport.sole_request();
        assert_eq!(
            request.url,
            "https://discord.com/api/v10/channels/42/messages"
        );
        assert_eq!(
            request.headers,
            vec![Header::new("authorization", "Bot tok")]
        );
    }

    #[test]
    fn twilio_live_send_uses_basic_auth_and_a_form_body() {
        let transport = RecordingTransport::new().with_response(Ok(Response::ok("{}")));
        let client = TwilioClient::new(
            "AC123",
            "secret",
            ConnectorConfig::new("unused", "https://api.twilio.com").live(),
        );
        client
            .send_message_live(&transport, "+15551234567", "+15559876543", "hi there")
            .expect("live send");

        let request = transport.sole_request();
        assert_eq!(
            request.url,
            "https://api.twilio.com/2010-04-01/Accounts/AC123/Messages.json"
        );
        assert_eq!(
            request.content_type,
            Some("application/x-www-form-urlencoded")
        );
        assert_eq!(
            request.headers,
            vec![Header::new("authorization", "Basic QUMxMjM6c2VjcmV0")]
        );
        assert_eq!(
            request.body.as_deref(),
            Some("To=%2B15551234567&From=%2B15559876543&Body=hi%20there")
        );
    }

    #[test]
    fn twilio_requires_both_sid_and_token() {
        let config = ConnectorConfig::new("unused", "https://api.twilio.com").live();
        assert_eq!(
            TwilioClient::new("", "secret", config.clone())
                .build_message_request("+1", "+2", "x")
                .unwrap_err(),
            ConnectorError::AuthenticationError
        );
        assert_eq!(
            TwilioClient::new("AC1", "", config)
                .build_message_request("+1", "+2", "x")
                .unwrap_err(),
            ConnectorError::AuthenticationError
        );
    }

    #[test]
    fn form_encode_escapes_plus_so_phone_numbers_survive() {
        // A literal '+' in a form body decodes as a space, which would turn
        // "+15551234567" into " 15551234567" and break every E.164 number.
        assert_eq!(form_encode("+15551234567"), "%2B15551234567");
        assert_eq!(form_encode("hi there"), "hi%20there");
        assert_eq!(form_encode("a&b=c"), "a%26b%3Dc");
        assert_eq!(form_encode("safe-._~AZaz09"), "safe-._~AZaz09");
    }

    #[test]
    fn form_encode_handles_multibyte_utf8() {
        // Percent-encoding is per byte, so a multibyte char becomes several.
        assert_eq!(form_encode("é"), "%C3%A9");
    }

    #[test]
    fn provider_names_are_stable() {
        assert_eq!(Provider::OpenAi.name(), "openai");
        assert_eq!(Provider::Anthropic.name(), "anthropic");
        assert_eq!(Provider::Grok.name(), "grok");
    }
}
