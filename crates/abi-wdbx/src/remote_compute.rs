//! Reference loopback transport for the WDBX remote-compute DOT operation.

use std::fmt::Write as _;
use std::io::Write as _;
use std::net::{TcpListener, TcpStream};

use crate::compute::{Backend, ComputeError, dot, select};
use crate::net_line::{LineError, read_line, write_line};

/// Environment variable containing report-only `host:port` endpoint metadata.
pub const ENDPOINT_ENV: &str = "ABI_REMOTE_COMPUTE_ENDPOINT";
/// Maximum accepted request frame.
pub const MAX_MESSAGE_SIZE: usize = 64 * 1024;

/// Remote DOT protocol failure.
#[derive(Debug)]
pub enum RemoteError {
    /// Underlying socket or framing failure.
    Io(std::io::Error),
    /// A bounded frame was too large.
    FrameTooLong,
    /// The request did not match `DOT <n> <a...> <b...>`.
    MalformedRequest,
    /// The response was not a finite `f32`.
    MalformedResponse,
    /// Input vectors have different dimensions.
    DimensionMismatch,
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::FrameTooLong => formatter.write_str("remote compute frame too long"),
            Self::MalformedRequest => formatter.write_str("malformed remote compute request"),
            Self::MalformedResponse => formatter.write_str("malformed remote compute response"),
            Self::DimensionMismatch => formatter.write_str("vector dimensions differ"),
        }
    }
}

impl std::error::Error for RemoteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RemoteError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<LineError> for RemoteError {
    fn from(error: LineError) -> Self {
        match error {
            LineError::Io(error) => Self::Io(error),
            LineError::TooLong => Self::FrameTooLong,
        }
    }
}

impl From<ComputeError> for RemoteError {
    fn from(error: ComputeError) -> Self {
        match error {
            ComputeError::DimensionMismatch => Self::DimensionMismatch,
        }
    }
}

/// Local reference product a correct endpoint must reproduce.
pub fn local_dot(a: &[f32], b: &[f32]) -> Result<f32, RemoteError> {
    dot(select(Backend::CpuScalar), a, b).map_err(Into::into)
}

/// Current report-only endpoint metadata.
#[must_use]
pub fn endpoint() -> Option<String> {
    std::env::var(ENDPOINT_ENV).ok()
}

/// Parse the port from `host:port`, returning `None` for malformed metadata.
#[must_use]
pub fn parse_endpoint_port(endpoint: &str) -> Option<u16> {
    endpoint.rsplit_once(':')?.1.parse().ok()
}

/// Try the configured loopback reference transport, falling back to local CPU.
pub fn dot_or_local(a: &[f32], b: &[f32]) -> Result<f32, RemoteError> {
    dot_or_local_at(endpoint().as_deref(), a, b)
}

/// Try explicit endpoint metadata, falling back locally when absent or unreachable.
pub fn dot_or_local_at(endpoint: Option<&str>, a: &[f32], b: &[f32]) -> Result<f32, RemoteError> {
    if a.len() != b.len() {
        return Err(RemoteError::DimensionMismatch);
    }
    if let Some(port) = endpoint.and_then(parse_endpoint_port)
        && let Ok(Some(stream)) = dial_dot(port, a, b)
        && let Ok(result) = read_dot_reply(stream)
    {
        return Ok(result);
    }
    local_dot(a, b)
}

/// Accept one request, evaluate DOT, and respond.
pub fn serve_once(listener: &TcpListener) -> Result<(), RemoteError> {
    let (mut stream, _) = listener.accept()?;
    let mut buffer = vec![0; MAX_MESSAGE_SIZE].into_boxed_slice();
    let line = read_line(&mut stream, &mut buffer)?;
    let text = std::str::from_utf8(line).map_err(|_| RemoteError::MalformedRequest)?;
    let payload = text
        .strip_prefix("DOT ")
        .ok_or(RemoteError::MalformedRequest)?;
    let mut tokens = payload.split(' ');
    let dimension: usize = tokens
        .next()
        .ok_or(RemoteError::MalformedRequest)?
        .parse()
        .map_err(|_| RemoteError::MalformedRequest)?;
    let total = dimension
        .checked_mul(2)
        .ok_or(RemoteError::MalformedRequest)?;
    if total > MAX_MESSAGE_SIZE {
        return Err(RemoteError::MalformedRequest);
    }
    let mut values = Vec::with_capacity(total);
    for _ in 0..total {
        let value = tokens
            .next()
            .ok_or(RemoteError::MalformedRequest)?
            .parse::<f32>()
            .map_err(|_| RemoteError::MalformedRequest)?;
        if !value.is_finite() {
            return Err(RemoteError::MalformedRequest);
        }
        values.push(value);
    }
    let result = local_dot(&values[..dimension], &values[dimension..])?;
    write_line(&mut stream, format!("{result}\n").as_bytes())?;
    Ok(())
}

/// Connect to the loopback reference endpoint and send one DOT request.
///
/// An unreachable port returns `Ok(None)` so callers can choose the CPU fallback.
pub fn dial_dot(port: u16, a: &[f32], b: &[f32]) -> Result<Option<TcpStream>, RemoteError> {
    if a.len() != b.len() {
        return Err(RemoteError::DimensionMismatch);
    }
    let mut message = String::new();
    write!(&mut message, "DOT {}", a.len()).expect("writing to String cannot fail");
    for value in a.iter().chain(b) {
        write!(&mut message, " {value}").expect("writing to String cannot fail");
    }
    message.push('\n');

    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port)) else {
        return Ok(None);
    };
    stream.write_all(message.as_bytes())?;
    stream.flush()?;
    Ok(Some(stream))
}

/// Read and parse one DOT response.
pub fn read_dot_reply(mut stream: TcpStream) -> Result<f32, RemoteError> {
    let mut buffer = [0; 64];
    let line = read_line(&mut stream, &mut buffer)?;
    let text = std::str::from_utf8(line).map_err(|_| RemoteError::MalformedResponse)?;
    let result = text
        .parse::<f32>()
        .map_err(|_| RemoteError::MalformedResponse)?;
    if !result.is_finite() {
        return Err(RemoteError::MalformedResponse);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::net::Shutdown;
    use std::thread;

    use super::*;

    #[test]
    fn endpoint_port_parsing_matches_the_oracle() {
        assert_eq!(parse_endpoint_port("127.0.0.1:8080"), Some(8080));
        assert_eq!(parse_endpoint_port("[::1]:65535"), Some(65535));
        assert_eq!(parse_endpoint_port("missing-port:"), None);
        assert_eq!(parse_endpoint_port("not-an-endpoint"), None);
    }

    #[test]
    fn loopback_dispatch_matches_the_local_reference() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || serve_once(&listener));
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [0.5, -1.0, 2.0, 0.25];
        let stream = dial_dot(port, &a, &b).expect("dial").expect("connected");
        let remote = read_dot_reply(stream).expect("reply");
        server.join().expect("server thread").expect("served");
        assert!((remote - local_dot(&a, &b).expect("local")).abs() < 1e-4);
    }

    #[test]
    fn malformed_request_is_rejected() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || serve_once(&listener));
        let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        client
            .write_all(b"GARBAGE not a dot request\n")
            .expect("send");
        client.shutdown(Shutdown::Write).expect("shutdown write");
        assert!(matches!(
            server.join().expect("server thread"),
            Err(RemoteError::MalformedRequest)
        ));
    }

    #[test]
    fn unreachable_or_malformed_endpoint_falls_back_locally() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        let expected = local_dot(&a, &b).expect("local");
        let unreachable =
            dot_or_local_at(Some("127.0.0.1:0"), &a, &b).expect("unreachable fallback");
        let malformed = dot_or_local_at(Some("malformed"), &a, &b).expect("malformed fallback");
        assert!((unreachable - expected).abs() < f32::EPSILON);
        assert!((malformed - expected).abs() < f32::EPSILON);
        assert!(matches!(
            dot_or_local_at(None, &a, &[1.0]),
            Err(RemoteError::DimensionMismatch)
        ));
    }

    #[test]
    fn oversized_and_non_finite_inputs_are_not_mis_served() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || serve_once(&listener));
        let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        client.write_all(b"DOT 1 NaN 1\n").expect("send");
        assert!(matches!(
            server.join().expect("server thread"),
            Err(RemoteError::MalformedRequest)
        ));
    }
}
