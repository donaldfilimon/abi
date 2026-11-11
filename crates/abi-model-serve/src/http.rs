//! Bounded HTTP/1.1 request reading and response writing for the loopback
//! listener. Every connection carries one request and is then closed.

use std::io::{Read, Write};

/// Largest accepted request line plus headers: 16 KiB.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;

/// One parsed request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// Upper-case method token.
    pub method: String,
    /// Request target without its query string.
    pub path: String,
    /// Header pairs with lower-cased names, in arrival order.
    pub headers: Vec<(String, String)>,
    /// Body bytes, exactly `Content-Length` long.
    pub body: Vec<u8>,
}

impl Request {
    /// First value of a header, by lower-case name.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// Why a request could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HttpError {
    /// The peer closed or stalled before a full request arrived.
    Incomplete,
    /// The request line or a header was not valid HTTP/1.1.
    Malformed,
    /// The header block exceeded [`MAX_HEADER_BYTES`].
    HeadersTooLarge,
    /// A body-carrying request had no usable `Content-Length`.
    LengthRequired,
    /// The declared body exceeded the configured ceiling.
    BodyTooLarge,
}

impl HttpError {
    /// Status code and reason phrase for this failure.
    #[must_use]
    pub const fn status(self) -> (u16, &'static str) {
        match self {
            Self::Incomplete => (408, "Request Timeout"),
            Self::Malformed => (400, "Bad Request"),
            Self::HeadersTooLarge => (431, "Request Header Fields Too Large"),
            Self::LengthRequired => (411, "Length Required"),
            Self::BodyTooLarge => (413, "Content Too Large"),
        }
    }
}

/// Read one request, refusing oversized headers and bodies before buffering
/// them.
pub fn read_request(stream: &mut impl Read, max_body_bytes: usize) -> Result<Request, HttpError> {
    let mut buffer = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let head_end = loop {
        if let Some(position) = find(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(HttpError::HeadersTooLarge);
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return Err(HttpError::Incomplete),
            Ok(read) => buffer.extend_from_slice(&chunk[..read]),
        }
    };
    if head_end > MAX_HEADER_BYTES {
        return Err(HttpError::HeadersTooLarge);
    }
    let head = std::str::from_utf8(&buffer[..head_end]).map_err(|_| HttpError::Malformed)?;
    let mut lines = head.split("\r\n");
    let mut request_line = lines.next().unwrap_or_default().split(' ');
    let (Some(method), Some(target), Some(version), None) = (
        request_line.next(),
        request_line.next(),
        request_line.next(),
        request_line.next(),
    ) else {
        return Err(HttpError::Malformed);
    };
    if method.is_empty() || !target.starts_with('/') || !version.starts_with("HTTP/1.") {
        return Err(HttpError::Malformed);
    }
    let mut headers = Vec::new();
    for line in lines {
        let (name, value) = line.split_once(':').ok_or(HttpError::Malformed)?;
        if name.is_empty() || name.contains(' ') {
            return Err(HttpError::Malformed);
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_owned()));
    }
    let mut request = Request {
        method: method.to_ascii_uppercase(),
        path: target.split('?').next().unwrap_or(target).to_owned(),
        headers,
        body: Vec::new(),
    };
    if request.header("transfer-encoding").is_some() {
        return Err(HttpError::LengthRequired);
    }
    let length = match request.header("content-length") {
        Some(value) => value.parse::<usize>().map_err(|_| HttpError::Malformed)?,
        None if request.method == "POST" => return Err(HttpError::LengthRequired),
        None => 0,
    };
    if length > max_body_bytes {
        return Err(HttpError::BodyTooLarge);
    }
    let mut body = buffer.split_off(head_end + 4);
    if body.len() > length {
        return Err(HttpError::Malformed);
    }
    while body.len() < length {
        let wanted = (length - body.len()).min(chunk.len());
        match stream.read(&mut chunk[..wanted]) {
            Ok(0) | Err(_) => return Err(HttpError::Incomplete),
            Ok(read) => body.extend_from_slice(&chunk[..read]),
        }
    }
    request.body = body;
    Ok(request)
}

/// Write one complete response and flush it.
pub(crate) fn write_response(
    stream: &mut impl Write,
    status: (u16, &str),
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        status.0,
        status.1,
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()
}

/// Write the head of a server-sent-event response. The body is delimited by
/// connection close.
pub(crate) fn write_sse_head(stream: &mut impl Write) -> std::io::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
    )?;
    stream.flush()
}

/// Write one `data:` event and flush it so the client sees it immediately.
pub(crate) fn write_sse_event(stream: &mut impl Write, payload: &str) -> std::io::Result<()> {
    write!(stream, "data: {payload}\n\n")?;
    stream.flush()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::{HttpError, MAX_HEADER_BYTES, read_request};

    fn read(raw: &[u8], max_body: usize) -> Result<super::Request, HttpError> {
        read_request(&mut &raw[..], max_body)
    }

    #[test]
    fn a_post_is_parsed_with_lowercased_headers_and_an_exact_body() {
        let request = read(
            b"POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-LENGTH: 4\r\n\r\nbody",
            64,
        )
        .expect("valid request");
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/chat/completions");
        assert_eq!(request.header("content-length"), Some("4"));
        assert_eq!(request.body, b"body");
    }

    #[test]
    fn framing_failures_are_typed() {
        assert_eq!(
            read(b"GET /health HTTP/1.1\r\n", 8),
            Err(HttpError::Incomplete)
        );
        assert_eq!(read(b"GET\r\n\r\n", 8), Err(HttpError::Malformed));
        assert_eq!(
            read(b"GET health HTTP/1.1\r\n\r\n", 8),
            Err(HttpError::Malformed)
        );
        assert_eq!(
            read(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n", 8),
            Err(HttpError::LengthRequired)
        );
        assert_eq!(
            read(b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", 8),
            Err(HttpError::LengthRequired)
        );
        assert_eq!(
            read(b"POST / HTTP/1.1\r\nContent-Length: 9\r\n\r\n", 8),
            Err(HttpError::BodyTooLarge)
        );
        assert_eq!(
            read(b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\nab", 8),
            Err(HttpError::Incomplete)
        );
        assert_eq!(
            read(b"POST / HTTP/1.1\r\nContent-Length: 1\r\n\r\nab", 8),
            Err(HttpError::Malformed)
        );
        let mut huge = b"GET /health HTTP/1.1\r\nX-Pad: ".to_vec();
        huge.resize(MAX_HEADER_BYTES + 2048, b'a');
        assert_eq!(read(&huge, 8), Err(HttpError::HeadersTooLarge));
    }
}
