//! The live HTTP transport, behind a trait.
//!
//! Ported from the request-issuing half of `src/connectors/http.zig`.
//!
//! ## Why a trait
//!
//! The Zig helpers called `std.http.Client` directly, so nothing about the live
//! path could be tested — every live test would have needed a real network. The
//! `Transport` trait lets provider clients be exercised against
//! [`RecordingTransport`], which asserts on the URL, headers and body that
//! *would* have been sent. That is the difference between "the live path
//! compiles" and "the live path sends the right request".
//!
//! ## Why `ureq`
//!
//! Rust's std has no HTTP client. `ureq` is blocking, which matches the
//! scheduler's synchronous one-shot model, and uses rustls rather than linking
//! OpenSSL. It is behind the default-on `live` feature so a build can drop the
//! networking dependency entirely and keep only the local transport.

use crate::connector::{ConnectorConfig, ConnectorError, Response, Result, TransportMode};
use crate::url::{join_url, map_status};

/// One HTTP header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Header name.
    pub name: String,
    /// Header value.
    pub value: String,
}

impl Header {
    /// A header from a name and value.
    #[must_use]
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

/// A request the transport should issue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// HTTP method.
    pub method: Method,
    /// Absolute URL.
    pub url: String,
    /// `Content-Type`, if the request has a body.
    pub content_type: Option<&'static str>,
    /// Extra headers, typically authorization.
    pub headers: Vec<Header>,
    /// Request body.
    pub body: Option<String>,
    /// Request deadline.
    pub timeout: std::time::Duration,
}

/// The HTTP methods the connectors use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `GET`.
    Get,
    /// `POST`.
    Post,
}

impl Method {
    /// The uppercase method name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
        }
    }
}

/// Issues HTTP requests on behalf of a connector.
pub trait Transport {
    /// Issue `request` and return the response.
    ///
    /// Implementations must **not** follow redirects: a redirect from an HTTPS
    /// base URL to a cleartext location would defeat
    /// [`crate::url::require_https_base_url`] and leak the API key.
    fn execute(&self, request: &Request) -> Result<Response>;
}

/// Build a request for a connector call, validating the config and URL.
pub fn build_request(
    config: &ConnectorConfig,
    method: Method,
    path: &str,
    content_type: Option<&'static str>,
    body: Option<String>,
    headers: Vec<Header>,
) -> Result<Request> {
    if config.transport != TransportMode::Live {
        return Err(ConnectorError::LiveTransportUnavailable);
    }
    config.validate()?;
    Ok(Request {
        method,
        url: join_url(&config.base_url, path)?,
        content_type,
        headers,
        body,
        timeout: config.timeout(),
    })
}

/// A transport that records requests and replays canned responses.
///
/// For testing provider clients without a network. Not `cfg(test)`: downstream
/// crates (`abi-ai`, `abi-mcp`) need it for their own tests.
#[derive(Debug, Default)]
pub struct RecordingTransport {
    requests: std::sync::Mutex<Vec<Request>>,
    responses: std::sync::Mutex<std::collections::VecDeque<Result<Response>>>,
}

impl RecordingTransport {
    /// A transport that answers every request with `200 OK` and an empty body.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a response to return, in order.
    #[must_use]
    pub fn with_response(self, response: Result<Response>) -> Self {
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push_back(response);
        self
    }

    /// Every request issued so far, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<Request> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// The single request issued, panicking if there was not exactly one.
    #[must_use]
    pub fn sole_request(&self) -> Request {
        let requests = self.requests();
        assert_eq!(requests.len(), 1, "expected exactly one request");
        requests.into_iter().next().expect("length checked")
    }
}

impl Transport for RecordingTransport {
    fn execute(&self, request: &Request) -> Result<Response> {
        self.requests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(request.clone());
        self.responses
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop_front()
            .unwrap_or_else(|| Ok(Response::ok(String::new())))
    }
}

/// A transport that always reports the live path as unavailable.
///
/// The default when the `live` feature is off, so a call fails loudly rather than
/// silently returning synthesized data as if it came from the provider.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnavailableTransport;

impl Transport for UnavailableTransport {
    fn execute(&self, _request: &Request) -> Result<Response> {
        Err(ConnectorError::LiveTransportUnavailable)
    }
}

#[cfg(feature = "live")]
mod live {
    use super::{Method, Request, Response, Transport};
    use crate::connector::{ConnectorError, Result};
    use crate::url::map_status;

    /// The real blocking HTTP transport.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct HttpTransport;

    impl HttpTransport {
        /// A transport issuing real requests.
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }

    impl Transport for HttpTransport {
        fn execute(&self, request: &Request) -> Result<Response> {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_global(Some(request.timeout))
                // Redirects are disabled deliberately: following one from an
                // HTTPS base URL to an http:// location would send the API key
                // in cleartext, defeating require_https_base_url.
                .max_redirects(0)
                .build()
                .into();

            // GET and POST builders are distinct types in ureq
            // (`RequestBuilder<WithoutBody>` vs `<WithBody>`), so the two paths
            // cannot share a variable — each applies headers and dispatches
            // separately.
            let outcome = match request.method {
                Method::Get => {
                    let mut builder = agent.get(&request.url);
                    if let Some(content_type) = request.content_type {
                        builder = builder.header("Content-Type", content_type);
                    }
                    for header in &request.headers {
                        builder = builder.header(header.name.as_str(), header.value.as_str());
                    }
                    builder.call()
                }
                Method::Post => {
                    let mut builder = agent.post(&request.url);
                    if let Some(content_type) = request.content_type {
                        builder = builder.header("Content-Type", content_type);
                    }
                    for header in &request.headers {
                        builder = builder.header(header.name.as_str(), header.value.as_str());
                    }
                    match &request.body {
                        Some(body) => builder.send(body.as_str()),
                        None => builder.send_empty(),
                    }
                }
            };

            let mut response = match outcome {
                Ok(response) => response,
                Err(ureq::Error::StatusCode(status)) => {
                    // Surface the provider's status through the same mapping a
                    // successful-but-error response would take.
                    map_status(status)?;
                    return Err(ConnectorError::InvalidResponse);
                }
                Err(ureq::Error::Timeout(_)) => return Err(ConnectorError::Timeout),
                Err(_) => return Err(ConnectorError::ConnectionFailed),
            };

            let status = response.status().as_u16();
            map_status(status)?;
            let body = response
                .body_mut()
                .read_to_string()
                .map_err(|_| ConnectorError::ReadFailed)?;
            Ok(Response { status, body })
        }
    }
}

#[cfg(feature = "live")]
pub use live::HttpTransport;

/// The transport used when a caller does not supply one.
///
/// [`HttpTransport`] with the `live` feature, [`UnavailableTransport`] without.
#[cfg(feature = "live")]
pub type DefaultTransport = HttpTransport;

/// The transport used when a caller does not supply one.
#[cfg(not(feature = "live"))]
pub type DefaultTransport = UnavailableTransport;

/// Issue a request through `transport`, mapping the status.
pub fn execute(transport: &dyn Transport, request: &Request) -> Result<Response> {
    let response = transport.execute(request)?;
    map_status(response.status)?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_config() -> ConnectorConfig {
        ConnectorConfig::new("sk-test", "https://api.example.com").live()
    }

    #[test]
    fn build_request_refuses_the_local_transport() {
        let config = ConnectorConfig::new("sk-test", "https://api.example.com");
        assert_eq!(
            build_request(&config, Method::Post, "/v1", None, None, Vec::new()).unwrap_err(),
            ConnectorError::LiveTransportUnavailable
        );
    }

    #[test]
    fn build_request_validates_the_config() {
        let mut config = live_config();
        config.api_key.clear();
        assert_eq!(
            build_request(&config, Method::Post, "/v1", None, None, Vec::new()).unwrap_err(),
            ConnectorError::AuthenticationError
        );
    }

    #[test]
    fn build_request_rejects_a_cleartext_base_url() {
        let config = ConnectorConfig::new("sk-test", "http://api.example.com").live();
        assert_eq!(
            build_request(&config, Method::Post, "/v1", None, None, Vec::new()).unwrap_err(),
            ConnectorError::InsecureBaseUrl
        );
    }

    #[test]
    fn build_request_composes_url_headers_and_timeout() {
        let config = live_config().with_timeout_ms(1_500);
        let request = build_request(
            &config,
            Method::Post,
            "/v1/chat/completions",
            Some("application/json"),
            Some("{}".to_string()),
            vec![Header::new("authorization", "Bearer sk-test")],
        )
        .expect("builds");

        assert_eq!(request.method, Method::Post);
        assert_eq!(request.url, "https://api.example.com/v1/chat/completions");
        assert_eq!(request.content_type, Some("application/json"));
        assert_eq!(request.body.as_deref(), Some("{}"));
        assert_eq!(request.timeout, std::time::Duration::from_millis(1_500));
        assert_eq!(request.headers[0].name, "authorization");
    }

    #[test]
    fn recording_transport_captures_requests_and_replays_responses() {
        let transport = RecordingTransport::new()
            .with_response(Ok(Response::ok("first")))
            .with_response(Ok(Response::ok("second")));

        let request = build_request(&live_config(), Method::Get, "/a", None, None, Vec::new())
            .expect("builds");
        assert_eq!(execute(&transport, &request).expect("ok").body, "first");
        assert_eq!(execute(&transport, &request).expect("ok").body, "second");
        // Exhausted queue falls back to an empty 200.
        assert_eq!(execute(&transport, &request).expect("ok").body, "");
        assert_eq!(transport.requests().len(), 3);
    }

    #[test]
    fn execute_maps_an_error_status_from_the_transport() {
        let transport = RecordingTransport::new().with_response(Ok(Response {
            status: 429,
            body: String::new(),
        }));
        let request = build_request(&live_config(), Method::Get, "/a", None, None, Vec::new())
            .expect("builds");
        assert_eq!(
            execute(&transport, &request).unwrap_err(),
            ConnectorError::RateLimited
        );
    }

    #[test]
    fn unavailable_transport_fails_loudly() {
        let request = build_request(&live_config(), Method::Get, "/a", None, None, Vec::new())
            .expect("builds");
        assert_eq!(
            execute(&UnavailableTransport, &request).unwrap_err(),
            ConnectorError::LiveTransportUnavailable
        );
    }

    #[test]
    fn method_names_are_uppercase() {
        assert_eq!(Method::Get.name(), "GET");
        assert_eq!(Method::Post.name(), "POST");
    }
}
