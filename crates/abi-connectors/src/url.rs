//! Base-URL validation, URL joining, and authorization headers.
//!
//! Ported from the URL and header half of `src/connectors/http.zig`.
//!
//! [`require_https_base_url`] is the security-relevant function here (the Zig
//! source ties it to threat-model item TM-007): a live connector sends an API key,
//! so its base URL must be HTTPS. Loopback is exempted so local integration tests
//! and on-device servers work without TLS.
//!
//! The subtlety is parsing the destination, not checking a textual host prefix.
//! Userinfo such as `127.0.0.1:443@evil.com` can look loopback-shaped while
//! directing a cleartext request and its API key to the remote host.

use crate::connector::{ConnectorError, Result};
use base64::Engine as _;

/// Require that a live connector's base URL cannot leak credentials.
///
/// Accepts any `https://` URL, plus `http://127.0.0.1` and `http://localhost`
/// (optionally with a port or path). Everything else is
/// [`ConnectorError::InsecureBaseUrl`].
pub fn require_https_base_url(base_url: &str) -> Result<()> {
    // ureq also parses requests as http::Uri. Use the same authority semantics
    // before allowing cleartext, so userinfo cannot disguise a remote host.
    let uri: http::Uri = base_url
        .parse()
        .map_err(|_| ConnectorError::InsecureBaseUrl)?;
    let scheme = uri.scheme_str().ok_or(ConnectorError::InsecureBaseUrl)?;
    let authority = uri.authority().ok_or(ConnectorError::InsecureBaseUrl)?;
    if scheme.eq_ignore_ascii_case("https") {
        return Ok(());
    }
    if scheme.eq_ignore_ascii_case("http")
        && !authority.as_str().contains('@')
        && (authority.host().eq_ignore_ascii_case("127.0.0.1")
            || authority.host().eq_ignore_ascii_case("localhost"))
    {
        return Ok(());
    }
    Err(ConnectorError::InsecureBaseUrl)
}

/// Join a base URL and a path with exactly one separating slash.
///
/// Validates the base URL first, so a URL that would leak a key cannot be built
/// even by a caller that forgot to check.
pub fn join_url(base_url: &str, path: &str) -> Result<String> {
    if base_url.is_empty() || path.is_empty() {
        return Err(ConnectorError::ConnectionFailed);
    }
    require_https_base_url(base_url)?;

    let base_slash = base_url.ends_with('/');
    let path_slash = path.starts_with('/');
    Ok(match (base_slash, path_slash) {
        (true, true) => format!("{base_url}{}", &path[1..]),
        (false, false) => format!("{base_url}/{path}"),
        _ => format!("{base_url}{path}"),
    })
}

/// `Authorization: Bearer <api_key>` — `OpenAI`, Grok.
#[must_use]
pub fn bearer_header(api_key: &str) -> String {
    format!("Bearer {api_key}")
}

/// `Authorization: Bot <token>` — Discord.
#[must_use]
pub fn bot_header(token: &str) -> String {
    format!("Bot {token}")
}

/// `Authorization: Basic <base64(user:pass)>` — Twilio.
#[must_use]
pub fn basic_auth_header(username: &str, password: &str) -> String {
    let combined = format!("{username}:{password}");
    let encoded = base64::engine::general_purpose::STANDARD.encode(combined.as_bytes());
    format!("Basic {encoded}")
}

/// Map an HTTP status onto the connector error set.
///
/// 2xx succeeds; 401/403 is authentication; 408/504 is timeout; 429 is rate
/// limiting; anything else is an invalid response.
pub fn map_status(status: u16) -> Result<()> {
    match status {
        200..=299 => Ok(()),
        401 | 403 => Err(ConnectorError::AuthenticationError),
        408 | 504 => Err(ConnectorError::Timeout),
        429 => Err(ConnectorError::RateLimited),
        _ => Err(ConnectorError::InvalidResponse),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_urls_are_accepted() {
        for url in [
            "https://api.openai.com",
            "https://api.anthropic.com/v1",
            "HTTPS://API.EXAMPLE.COM",
            "https://x",
        ] {
            assert!(require_https_base_url(url).is_ok(), "{url} should pass");
        }
    }

    #[test]
    fn loopback_http_is_exempted() {
        for url in [
            "http://127.0.0.1",
            "http://127.0.0.1:8080",
            "http://127.0.0.1/v1",
            "http://localhost",
            "http://localhost:11434",
            "http://localhost/v1/chat",
            "HTTP://LOCALHOST:1234",
        ] {
            assert!(require_https_base_url(url).is_ok(), "{url} should pass");
        }
    }

    #[test]
    fn cleartext_remote_urls_are_rejected() {
        for url in [
            "http://api.openai.com",
            "ftp://example.com",
            "example.com",
            "",
            "//api.example.com",
        ] {
            assert_eq!(
                require_https_base_url(url).unwrap_err(),
                ConnectorError::InsecureBaseUrl,
                "{url} should be rejected"
            );
        }
    }

    #[test]
    fn loopback_lookalike_hostnames_do_not_bypass_the_check() {
        // Each of these resolves to a remote host while looking loopback-shaped.
        for url in [
            "http://127.0.0.1.evil.com",
            "http://127.0.0.1@evil.com",
            "http://127.0.0.1:443@evil.com",
            "http://localhost.evil.com",
            "http://localhost@evil.com",
            "http://localhost:80@evil.com",
            "http://127.0.0.10",
            "http://localhosts",
            "http://127.0.0.1x",
        ] {
            assert_eq!(
                require_https_base_url(url).unwrap_err(),
                ConnectorError::InsecureBaseUrl,
                "{url} must not be treated as loopback"
            );
        }
    }

    #[test]
    fn join_url_inserts_exactly_one_slash() {
        assert_eq!(
            join_url("https://api.example.com", "/v1/chat").unwrap(),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url("https://api.example.com/", "/v1/chat").unwrap(),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url("https://api.example.com", "v1/chat").unwrap(),
            "https://api.example.com/v1/chat"
        );
        assert_eq!(
            join_url("https://api.example.com/", "v1/chat").unwrap(),
            "https://api.example.com/v1/chat"
        );
    }

    #[test]
    fn join_url_rejects_empty_parts() {
        assert_eq!(
            join_url("", "/v1").unwrap_err(),
            ConnectorError::ConnectionFailed
        );
        assert_eq!(
            join_url("https://x", "").unwrap_err(),
            ConnectorError::ConnectionFailed
        );
    }

    #[test]
    fn join_url_enforces_https_even_if_the_caller_forgot() {
        // Defence in depth: a URL that would leak a key cannot be constructed.
        assert_eq!(
            join_url("http://api.example.com", "/v1").unwrap_err(),
            ConnectorError::InsecureBaseUrl
        );
    }

    #[test]
    fn auth_headers_have_the_provider_expected_shapes() {
        assert_eq!(bearer_header("sk-abc"), "Bearer sk-abc");
        assert_eq!(bot_header("tok"), "Bot tok");
        // Twilio: base64 of "user:pass".
        assert_eq!(basic_auth_header("user", "pass"), "Basic dXNlcjpwYXNz");
    }

    #[test]
    fn basic_auth_encodes_bytes_that_need_padding() {
        assert_eq!(
            basic_auth_header("AC123", "secret"),
            "Basic QUMxMjM6c2VjcmV0"
        );
        assert_eq!(basic_auth_header("a", "b"), "Basic YTpi");
    }

    #[test]
    fn status_mapping_matches_the_zig_table() {
        assert!(map_status(200).is_ok());
        assert!(map_status(204).is_ok());
        assert!(map_status(299).is_ok());

        assert_eq!(
            map_status(401).unwrap_err(),
            ConnectorError::AuthenticationError
        );
        assert_eq!(
            map_status(403).unwrap_err(),
            ConnectorError::AuthenticationError
        );
        assert_eq!(map_status(408).unwrap_err(), ConnectorError::Timeout);
        assert_eq!(map_status(504).unwrap_err(), ConnectorError::Timeout);
        assert_eq!(map_status(429).unwrap_err(), ConnectorError::RateLimited);

        for status in [300u16, 400, 404, 500, 502] {
            assert_eq!(
                map_status(status).unwrap_err(),
                ConnectorError::InvalidResponse,
                "status {status}"
            );
        }
    }
}
