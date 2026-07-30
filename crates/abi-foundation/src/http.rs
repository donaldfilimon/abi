//! Minimal HTTP/1.1 request and response handling.
//!
//! Ported from `src/foundation/http/`. This is not a general HTTP library — it is
//! exactly what the MCP HTTP transport and the WDBX REST/cluster listeners need:
//! read one request into a fixed buffer, check a bearer token, write one response.
//!
//! Two changes from the Zig shape:
//!
//! - The read/write helpers took a `std.Io.net.Stream`, so nothing could be tested
//!   without a live socket. Here they are generic over [`std::io::Read`] /
//!   [`std::io::Write`], and the parser is exercised against in-memory buffers.
//! - `bind_loopback` used a raw `getsockname` extern to recover the
//!   kernel-assigned port; `TcpListener::local_addr` does that without FFI.

use std::io::{Read, Write};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};

/// Largest request this module will buffer: 64 KiB.
///
/// A request larger than this is rejected rather than grown into, which is what
/// bounds a hostile `Content-Length` to a fixed allocation.
pub const MAX_REQUEST_SIZE: usize = 64 * 1024;

/// The header/body separator.
const HEADER_TERMINATOR: &[u8] = b"\r\n\r\n";

/// Outcome of reading one request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadResult {
    /// A complete request, headers and declared body.
    Request(Vec<u8>),
    /// The peer closed without sending anything.
    Empty,
    /// Headers or body exceeded [`MAX_REQUEST_SIZE`].
    TooLarge,
    /// The connection ended before the declared body arrived.
    ///
    /// Distinct from [`ReadResult::Empty`] on purpose: callers must **not**
    /// dispatch a truncated payload, because a partial JSON-RPC body can parse
    /// into a different, valid request.
    Incomplete,
}

/// Parse a `Content-Length` header out of a header block.
///
/// The request line is skipped, header names are matched case-insensitively, and
/// values are trimmed of spaces and tabs. A malformed value yields `None` rather
/// than an error, so the caller treats it as absent.
#[must_use]
pub fn parse_content_length(header_block: &str) -> Option<usize> {
    header_lines(header_block)
        .find_map(|(name, value)| name.eq_ignore_ascii_case("Content-Length").then_some(value))
        .and_then(|value| value.parse().ok())
}

/// Read a header's value from a raw request or response.
///
/// Only the header block is searched, so a matching byte sequence in the body
/// cannot be mistaken for a header.
#[must_use]
pub fn header_value<'a>(raw: &'a str, name: &str) -> Option<&'a str> {
    let block = raw.find("\r\n\r\n").map_or(raw, |index| &raw[..index]);
    header_lines(block).find_map(|(key, value)| key.eq_ignore_ascii_case(name).then_some(value))
}

/// Iterate `(name, value)` header pairs, skipping the request/status line.
fn header_lines(block: &str) -> impl Iterator<Item = (&str, &str)> {
    block
        .split("\r\n")
        .skip(1)
        .take_while(|line| !line.is_empty())
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((
                name.trim_matches([' ', '\t']),
                value.trim_matches([' ', '\t']),
            ))
        })
}

/// The total byte count a request should reach, or `None` if it will not fit.
///
/// Guards against an overflow that a naive `header_end + declared_body_len` would
/// hit: a `Content-Length` of `usize::MAX` would wrap and produce a small,
/// plausible-looking target. Each comparison is done against the remaining
/// capacity instead.
#[must_use]
pub fn request_target_within_buffer(
    header_end: usize,
    declared_body_len: usize,
    capacity: usize,
) -> Option<usize> {
    if header_end > capacity {
        return None;
    }
    let remaining = capacity - header_end;
    if declared_body_len > remaining {
        return None;
    }
    Some(header_end + declared_body_len)
}

/// The body of a raw request or response, if it has one.
#[must_use]
pub fn find_body(raw: &[u8]) -> Option<&[u8]> {
    let index = find_subslice(raw, HEADER_TERMINATOR)?;
    let start = index + HEADER_TERMINATOR.len();
    if start >= raw.len() {
        return None;
    }
    Some(&raw[start..])
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Read one request from `reader`, bounded by `max_size`.
pub fn read_request<R: Read>(reader: &mut R, max_size: usize) -> ReadResult {
    let mut buf = vec![0u8; max_size];
    let mut total = 0usize;
    let mut header_end: Option<usize> = None;
    let mut want_total: Option<usize> = None;

    loop {
        if header_end.is_none()
            && let Some(index) = find_subslice(&buf[..total], HEADER_TERMINATOR)
        {
            let end = index + HEADER_TERMINATOR.len();
            header_end = Some(end);
            // A header block that is not UTF-8 cannot carry a Content-Length we
            // would honour; treat it as absent rather than failing the read.
            let declared = std::str::from_utf8(&buf[..end])
                .ok()
                .and_then(parse_content_length)
                .unwrap_or(0);
            match request_target_within_buffer(end, declared, buf.len()) {
                Some(target) => want_total = Some(target),
                None => return ReadResult::TooLarge,
            }
        }

        if let Some(want) = want_total
            && total >= want
        {
            break;
        }

        if total >= buf.len() {
            return ReadResult::TooLarge;
        }

        match reader.read(&mut buf[total..]) {
            Ok(0) | Err(_) => break,
            Ok(n) => total += n,
        }
    }

    if total == 0 {
        return ReadResult::Empty;
    }
    match (header_end, want_total) {
        (None, _) => ReadResult::Incomplete,
        (Some(_), Some(want)) if total < want => ReadResult::Incomplete,
        (Some(_), _) => {
            buf.truncate(total);
            ReadResult::Request(buf)
        }
    }
}

/// Read a whole response from `reader` until EOF, bounded by `max_size`.
pub fn read_response<R: Read>(reader: &mut R, max_size: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    reader.take(max_size as u64).read_to_end(&mut buf)?;
    Ok(buf)
}

/// Write `bytes` and flush.
pub fn write_all<W: Write>(writer: &mut W, bytes: &[u8]) -> std::io::Result<()> {
    writer.write_all(bytes)?;
    writer.flush()
}

/// The reason phrase for a status code.
///
/// Unknown codes render as `"OK"`, matching the Zig fallback. That is
/// questionable — a 503 would be labelled `OK` — but it is observable in existing
/// response bytes, so changing it belongs with the MCP/WDBX port where the
/// response contract is tested, not here.
#[must_use]
pub const fn reason_phrase(status: u16) -> &'static str {
    match status {
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

/// Write a `401 Unauthorized` with a JSON error body.
pub fn write_unauthorized<W: Write>(writer: &mut W, error_msg: &str) -> std::io::Result<()> {
    let body = format!(
        "{{\"error\":\"{}\"}}",
        crate::json::escape_json_string_raw(error_msg)
    );
    let response = format!(
        "HTTP/1.1 401 Unauthorized\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         WWW-Authenticate: Bearer\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    );
    write_all(writer, response.as_bytes())
}

/// Whether the request carries `Authorization: Bearer <token>`.
///
/// Comparison goes through [`fixed_work_eq`]; see that function for why.
///
/// An empty `token` never authorizes. The Zig version compared unconditionally,
/// so a listener started with an unset token would accept a bare
/// `Authorization: Bearer ` header — the empty presented credential compared
/// equal to the empty expected one. Callers should still refuse to start without
/// a token; this makes the failure closed rather than open if one slips through.
#[must_use]
pub fn has_bearer_token(raw: &str, token: &str) -> bool {
    if token.is_empty() {
        return false;
    }
    let Some(value) = header_value(raw, "Authorization") else {
        return false;
    };
    let Some(presented) = value.strip_prefix("Bearer ") else {
        return false;
    };
    fixed_work_eq(presented.as_bytes(), token.as_bytes())
}

/// Compare two byte strings in work independent of their contents *and* lengths.
///
/// Ported from the Zig `fixedWorkEql`. Three properties matter, and plain `==`
/// has none of them:
///
/// - it short-circuits on the first differing byte, leaking a prefix-match length
///   through timing;
/// - it compares lengths first, so a wrong-length token is rejected faster than a
///   right-length one;
/// - it may vectorize into an early exit.
///
/// Both operands are walked to `max(a.len, b.len)`, treating out-of-range bytes as
/// zero, and the length difference is folded into the same accumulator. The
/// accumulator passes through [`std::hint::black_box`] each iteration so the
/// optimizer cannot reintroduce an early exit once it proves the result is already
/// non-zero.
///
/// This is a hardened comparison, not a formally constant-time one — that would
/// need a crate such as `subtle`, whose slice `ct_eq` checks length first and so
/// would not preserve the length-independence this relies on.
#[must_use]
pub fn fixed_work_eq(a: &[u8], b: &[u8]) -> bool {
    let max_len = a.len().max(b.len());
    let mut diff: usize = a.len() ^ b.len();
    for i in 0..max_len {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        diff |= usize::from(av ^ bv);
        diff = std::hint::black_box(diff);
    }
    diff == 0
}

/// Bind a TCP listener to an ephemeral port on loopback.
///
/// Returns the listener and the port the kernel assigned. Loopback-only is
/// deliberate: the MCP HTTP transport and WDBX REST surface are not
/// authenticated strongly enough to face a network, and binding `0.0.0.0` would
/// silently expose them.
pub fn bind_loopback() -> std::io::Result<(TcpListener, u16)> {
    let addr = SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0));
    let listener = TcpListener::bind(addr)?;
    let port = listener.local_addr()?.port();
    if port == 0 {
        return Err(std::io::Error::other("kernel assigned port 0"));
    }
    Ok((listener, port))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_content_length_case_insensitively_with_padding() {
        assert_eq!(
            parse_content_length("POST / HTTP/1.1\r\nContent-Length: 42\r\n\r\n"),
            Some(42)
        );
        assert_eq!(
            parse_content_length("POST / HTTP/1.1\r\nHost: x\r\ncontent-length:   7  \r\n\r\n"),
            Some(7)
        );
    }

    #[test]
    fn absent_or_malformed_content_length_is_none() {
        assert_eq!(
            parse_content_length("POST / HTTP/1.1\r\nHost: x\r\n\r\n"),
            None
        );
        assert_eq!(
            parse_content_length("POST / HTTP/1.1\r\nContent-Length: abc\r\n\r\n"),
            None
        );
        assert_eq!(
            parse_content_length("POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n"),
            None
        );
    }

    #[test]
    fn content_length_in_the_request_line_is_not_matched() {
        // The request line is skipped, so a path that looks like a header is safe.
        assert_eq!(
            parse_content_length("GET /Content-Length:99 HTTP/1.1\r\nHost: x\r\n\r\n"),
            None
        );
    }

    #[test]
    fn request_target_rejects_oversize_without_overflowing() {
        assert_eq!(request_target_within_buffer(40, 8, 64), Some(48));
        assert_eq!(request_target_within_buffer(40, 25, 64), None);
        assert_eq!(request_target_within_buffer(40, usize::MAX, 64), None);
        assert_eq!(request_target_within_buffer(65, 0, 64), None);
        assert_eq!(request_target_within_buffer(64, 0, 64), Some(64));
    }

    #[test]
    fn header_value_reads_only_the_header_block() {
        let raw = "POST / HTTP/1.1\r\nHost: h\r\nX-Test: v\r\n\r\nX-Test: body-value";
        assert_eq!(header_value(raw, "x-test"), Some("v"));
        assert_eq!(header_value(raw, "Host"), Some("h"));
        assert_eq!(header_value(raw, "Missing"), None);
    }

    #[test]
    fn find_body_returns_none_when_there_is_no_body() {
        assert_eq!(find_body(b"GET / HTTP/1.1\r\n\r\n"), None);
        assert_eq!(find_body(b"GET / HTTP/1.1\r\n"), None);
        assert_eq!(find_body(b"POST / HTTP/1.1\r\n\r\n{}"), Some(&b"{}"[..]));
    }

    #[test]
    fn reads_a_complete_request() {
        let raw = b"POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}";
        let result = read_request(&mut &raw[..], MAX_REQUEST_SIZE);
        assert_eq!(result, ReadResult::Request(raw.to_vec()));
    }

    #[test]
    fn reads_a_request_with_no_body() {
        let raw = b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n";
        assert_eq!(
            read_request(&mut &raw[..], MAX_REQUEST_SIZE),
            ReadResult::Request(raw.to_vec())
        );
    }

    #[test]
    fn a_closed_connection_reads_as_empty() {
        assert_eq!(
            read_request(&mut &b""[..], MAX_REQUEST_SIZE),
            ReadResult::Empty
        );
    }

    #[test]
    fn a_truncated_body_reads_as_incomplete_not_a_request() {
        // The critical case: dispatching this partial body would execute a
        // different request than the peer sent.
        let raw = b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}";
        assert_eq!(
            read_request(&mut &raw[..], MAX_REQUEST_SIZE),
            ReadResult::Incomplete
        );
    }

    #[test]
    fn headers_with_no_terminator_read_as_incomplete() {
        let raw = b"POST / HTTP/1.1\r\nContent-Length: 2\r\n";
        assert_eq!(
            read_request(&mut &raw[..], MAX_REQUEST_SIZE),
            ReadResult::Incomplete
        );
    }

    #[test]
    fn an_oversize_declared_body_reads_as_too_large() {
        let raw = b"POST / HTTP/1.1\r\nContent-Length: 99999\r\n\r\n{}";
        assert_eq!(read_request(&mut &raw[..], 128), ReadResult::TooLarge);
    }

    #[test]
    fn headers_that_fill_the_buffer_read_as_too_large() {
        let padding = "x".repeat(200);
        let raw = format!("POST / HTTP/1.1\r\nX-Pad: {padding}\r\n");
        assert_eq!(read_request(&mut raw.as_bytes(), 64), ReadResult::TooLarge);
    }

    #[test]
    fn bearer_token_is_accepted_and_mismatches_rejected() {
        let raw = "POST / HTTP/1.1\r\n\
                   Host: 127.0.0.1\r\n\
                   authorization:   Bearer local-token  \r\n\
                   Content-Length: 2\r\n\r\n{}";
        assert!(has_bearer_token(raw, "local-token"));
        assert!(!has_bearer_token(raw, "wrong-token"));
        // Prefix and suffix of the real token must both fail.
        assert!(!has_bearer_token(raw, "local-tok"));
        assert!(!has_bearer_token(raw, "local-token-extra"));
    }

    #[test]
    fn a_missing_or_non_bearer_authorization_is_rejected() {
        assert!(!has_bearer_token(
            "POST / HTTP/1.1\r\n\r\n{}",
            "local-token"
        ));
        assert!(!has_bearer_token(
            "POST / HTTP/1.1\r\nAuthorization: Basic nope\r\n\r\n{}",
            "local-token"
        ));
        // Lowercase scheme is not a Bearer credential.
        assert!(!has_bearer_token(
            "POST / HTTP/1.1\r\nAuthorization: bearer local-token\r\n\r\n{}",
            "local-token"
        ));
    }

    #[test]
    fn an_empty_configured_token_never_authorizes_a_bearer_header() {
        assert!(!has_bearer_token(
            "POST / HTTP/1.1\r\nAuthorization: Bearer real\r\n\r\n{}",
            ""
        ));
        // The case the Zig version got wrong: empty presented vs empty expected
        // compared equal, so an unset token authorized a bare Bearer header.
        assert!(!has_bearer_token(
            "POST / HTTP/1.1\r\nAuthorization: Bearer \r\n\r\n{}",
            ""
        ));
    }

    #[test]
    fn fixed_work_eq_matches_equal_and_rejects_unequal() {
        assert!(fixed_work_eq(b"abc", b"abc"));
        assert!(!fixed_work_eq(b"abc", b"abd"));
        assert!(!fixed_work_eq(b"abc", b"ab"));
        assert!(!fixed_work_eq(b"ab", b"abc"));
        assert!(fixed_work_eq(b"", b""));
        assert!(!fixed_work_eq(b"", b"a"));
    }

    #[test]
    fn fixed_work_eq_is_not_fooled_by_zero_padding() {
        // Out-of-range bytes read as 0, so "a\0" must not equal "a".
        assert!(!fixed_work_eq(b"a\0", b"a"));
        assert!(!fixed_work_eq(b"a", b"a\0"));
    }

    #[test]
    fn unauthorized_response_is_well_formed_and_escapes_its_message() {
        let mut out = Vec::new();
        write_unauthorized(&mut out, "missing token").expect("write");
        let text = String::from_utf8(out).expect("utf8");
        assert!(text.starts_with("HTTP/1.1 401 Unauthorized\r\n"));
        assert!(text.contains("WWW-Authenticate: Bearer\r\n"));
        assert!(text.ends_with("{\"error\":\"missing token\"}"));

        let declared = parse_content_length(&text).expect("Content-Length present");
        let body = find_body(text.as_bytes()).expect("body present");
        assert_eq!(declared, body.len(), "Content-Length must match the body");
    }

    #[test]
    fn unauthorized_message_cannot_break_out_of_the_json_body() {
        let mut out = Vec::new();
        write_unauthorized(&mut out, "bad \"quote\" and \\slash").expect("write");
        let text = String::from_utf8(out).expect("utf8");
        let body = find_body(text.as_bytes()).expect("body");
        let parsed: serde_json::Value =
            serde_json::from_slice(body).expect("body must stay valid JSON");
        assert_eq!(parsed["error"], "bad \"quote\" and \\slash");
    }

    #[test]
    fn reason_phrases_cover_the_codes_the_listeners_emit() {
        assert_eq!(reason_phrase(200), "OK");
        assert_eq!(reason_phrase(400), "Bad Request");
        assert_eq!(reason_phrase(401), "Unauthorized");
        assert_eq!(reason_phrase(404), "Not Found");
        assert_eq!(reason_phrase(405), "Method Not Allowed");
        assert_eq!(reason_phrase(429), "Too Many Requests");
        assert_eq!(reason_phrase(500), "Internal Server Error");
    }

    #[test]
    fn read_response_is_bounded_by_max_size() {
        let raw = b"HTTP/1.1 200 OK\r\n\r\nbody";
        let got = read_response(&mut &raw[..], 8).expect("read");
        assert_eq!(got.len(), 8);
        let got = read_response(&mut &raw[..], MAX_REQUEST_SIZE).expect("read");
        assert_eq!(got, raw);
    }

    #[test]
    fn bind_loopback_returns_a_usable_loopback_port() {
        let (listener, port) = bind_loopback().expect("bind");
        assert!(port > 0);
        let addr = listener.local_addr().expect("local_addr");
        assert_eq!(addr.port(), port);
        assert!(addr.ip().is_loopback(), "must not bind a routable address");
    }
}
