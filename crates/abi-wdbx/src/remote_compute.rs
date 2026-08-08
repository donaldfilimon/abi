//! Reference loopback transport for the WDBX remote-compute DOT operation.

use std::fmt::Write as _;
use std::io::Write as _;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::compute::{
    Accelerator, Backend, CapabilityEvidence, CapabilityState, ComputeError, CpuBackend, dot,
    select,
};
use crate::net_line::{LineError, read_line, write_line};

/// Environment variable containing report-only `host:port` endpoint metadata.
pub const ENDPOINT_ENV: &str = "ABI_REMOTE_COMPUTE_ENDPOINT";
/// Environment variable containing the bearer token for remote compute.
pub const TOKEN_ENV: &str = "ABI_REMOTE_COMPUTE_TOKEN";
/// Maximum accepted request frame.
pub const MAX_MESSAGE_SIZE: usize = 64 * 1024;
/// Maximum accepted bearer-token length.
pub const MAX_TOKEN_SIZE: usize = 256;
/// Connect/read/write deadline for the reference transport.
pub const IO_TIMEOUT: Duration = Duration::from_millis(500);

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
    /// A bearer token was missing, malformed, or did not match.
    AuthenticationFailed,
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
            Self::AuthenticationFailed => {
                formatter.write_str("remote compute authentication failed")
            }
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

/// Current bearer token, when configured and syntactically bounded.
#[must_use]
pub fn token() -> Option<String> {
    std::env::var(TOKEN_ENV)
        .ok()
        .filter(|value| valid_token(value))
}

fn valid_token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_TOKEN_SIZE
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn tokens_equal(left: &str, right: &str) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..MAX_TOKEN_SIZE {
        let left = left.as_bytes().get(index).copied().unwrap_or(0);
        let right = right.as_bytes().get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

fn listener_auth_allowed(address: SocketAddr, expected_token: Option<&str>) -> bool {
    address.ip().is_loopback() || expected_token.is_some_and(valid_token)
}

/// Parse the port from `host:port`, returning `None` for malformed metadata.
#[must_use]
pub fn parse_endpoint_port(endpoint: &str) -> Option<u16> {
    endpoint.rsplit_once(':')?.1.parse().ok()
}

/// Try the configured loopback reference transport, falling back to local CPU.
pub fn dot_or_local(a: &[f32], b: &[f32]) -> Result<f32, RemoteError> {
    dot_or_local_at_with_token(endpoint().as_deref(), token().as_deref(), a, b)
}

/// Try explicit endpoint metadata, falling back locally when absent or unreachable.
pub fn dot_or_local_at(endpoint: Option<&str>, a: &[f32], b: &[f32]) -> Result<f32, RemoteError> {
    dot_or_local_at_with_token(endpoint, token().as_deref(), a, b)
}

/// Try an authenticated endpoint, falling back to deterministic local CPU.
pub fn dot_or_local_at_with_token(
    endpoint: Option<&str>,
    bearer_token: Option<&str>,
    a: &[f32],
    b: &[f32],
) -> Result<f32, RemoteError> {
    RemoteAccelerator::new(
        endpoint.map(ToOwned::to_owned),
        bearer_token.map(ToOwned::to_owned),
    )
    .dot(a, b)
    .map_err(Into::into)
}

/// Accept one request, evaluate DOT, and respond.
pub fn serve_once(listener: &TcpListener) -> Result<(), RemoteError> {
    serve_once_with_token(listener, token().as_deref())
}

/// Accept one request, enforce the optional bearer token, and respond.
pub fn serve_once_with_token(
    listener: &TcpListener,
    expected_token: Option<&str>,
) -> Result<(), RemoteError> {
    if !listener_auth_allowed(listener.local_addr()?, expected_token) {
        return Err(RemoteError::AuthenticationFailed);
    }
    let (mut stream, _) = listener.accept()?;
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
    let mut buffer = vec![0; MAX_MESSAGE_SIZE].into_boxed_slice();
    let line = read_line(&mut stream, &mut buffer)?;
    let text = std::str::from_utf8(line).map_err(|_| RemoteError::MalformedRequest)?;
    let payload = text
        .strip_prefix("DOT ")
        .ok_or(RemoteError::MalformedRequest)?;
    let mut tokens = payload.split(' ');
    let presented_token = tokens.next().ok_or(RemoteError::MalformedRequest)?;
    match expected_token {
        Some(expected) if valid_token(expected) && tokens_equal(presented_token, expected) => {}
        None if presented_token == "-" => {}
        _ => return Err(RemoteError::AuthenticationFailed),
    }
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
    if tokens.next().is_some() {
        return Err(RemoteError::MalformedRequest);
    }
    let result = local_dot(&values[..dimension], &values[dimension..])?;
    write_line(&mut stream, format!("{result}\n").as_bytes())?;
    Ok(())
}

/// Connect to the loopback reference endpoint and send one DOT request.
///
/// An unreachable port returns `Ok(None)` so callers can choose the CPU fallback.
pub fn dial_dot(port: u16, a: &[f32], b: &[f32]) -> Result<Option<TcpStream>, RemoteError> {
    dial_dot_inner(port, None, a, b)
}

/// Connect and send one authenticated DOT request.
pub fn dial_dot_with_token(
    port: u16,
    bearer_token: &str,
    a: &[f32],
    b: &[f32],
) -> Result<Option<TcpStream>, RemoteError> {
    if !valid_token(bearer_token) {
        return Err(RemoteError::AuthenticationFailed);
    }
    dial_dot_inner(port, Some(bearer_token), a, b)
}

fn dial_dot_inner(
    port: u16,
    bearer_token: Option<&str>,
    a: &[f32],
    b: &[f32],
) -> Result<Option<TcpStream>, RemoteError> {
    if a.len() != b.len() {
        return Err(RemoteError::DimensionMismatch);
    }
    let mut message = String::new();
    write!(
        &mut message,
        "DOT {} {}",
        bearer_token.unwrap_or("-"),
        a.len()
    )
    .expect("writing to String cannot fail");
    for value in a.iter().chain(b) {
        write!(&mut message, " {value}").expect("writing to String cannot fail");
    }
    message.push('\n');
    if message.len() > MAX_MESSAGE_SIZE {
        return Err(RemoteError::FrameTooLong);
    }

    let address = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&address, IO_TIMEOUT) else {
        return Ok(None);
    };
    stream.set_read_timeout(Some(IO_TIMEOUT))?;
    stream.set_write_timeout(Some(IO_TIMEOUT))?;
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

/// Authenticated remote accelerator with deterministic CPU fallback.
///
/// Endpoint and bearer-token configuration is not availability evidence. The
/// capability advances to initialized/executed only after an authenticated
/// response, and to runtime-verified only after matching the CPU oracle.
pub struct RemoteAccelerator {
    endpoint: Option<String>,
    bearer_token: Option<String>,
    cpu: CpuBackend,
    executed: AtomicBool,
    verified: AtomicBool,
}

impl std::fmt::Debug for RemoteAccelerator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RemoteAccelerator")
            .field("endpoint_configured", &self.endpoint.is_some())
            .field("bearer_token", &"[REDACTED]")
            .field("cpu", &self.cpu)
            .field("executed", &self.executed.load(Ordering::Relaxed))
            .field("verified", &self.verified.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for RemoteAccelerator {
    fn default() -> Self {
        Self::from_env()
    }
}

impl RemoteAccelerator {
    /// Construct from the remote endpoint and bearer-token environment values.
    #[must_use]
    pub fn from_env() -> Self {
        Self::new(endpoint(), token())
    }

    /// Construct from explicit configuration.
    #[must_use]
    pub fn new(endpoint: Option<String>, bearer_token: Option<String>) -> Self {
        Self {
            endpoint,
            bearer_token: bearer_token.filter(|value| valid_token(value)),
            cpu: CpuBackend::new(),
            executed: AtomicBool::new(false),
            verified: AtomicBool::new(false),
        }
    }

    fn configured(&self) -> bool {
        self.endpoint
            .as_deref()
            .and_then(parse_endpoint_port)
            .is_some()
            && self.bearer_token.as_deref().is_some_and(valid_token)
    }
}

impl Accelerator for RemoteAccelerator {
    fn backend(&self) -> Backend {
        Backend::TpuRemote
    }

    fn capability(&self) -> CapabilityState {
        let executed = self.executed.load(Ordering::Relaxed);
        let verified = self.verified.load(Ordering::Relaxed);
        CapabilityState::new(
            Backend::TpuRemote,
            CapabilityEvidence::new()
                .with_compiled(true)
                .with_available(executed)
                .with_initialized(executed)
                .with_executed(executed)
                .with_runtime_verified(verified),
            if verified {
                "authenticated remote DOT execution matched deterministic CPU oracle"
            } else if executed {
                "authenticated remote DOT executed but did not match CPU oracle; fallback active"
            } else if self.configured() {
                "authenticated remote endpoint configured but runtime execution is unverified"
            } else {
                "authenticated remote endpoint/token not configured; CPU fallback active"
            },
        )
        .expect("remote evidence ladder is valid")
    }

    fn dot(&self, left: &[f32], right: &[f32]) -> Result<f32, ComputeError> {
        let oracle = self.cpu.dot(left, right)?;
        let Some(port) = self.endpoint.as_deref().and_then(parse_endpoint_port) else {
            return Ok(oracle);
        };
        let Some(bearer_token) = self.bearer_token.as_deref() else {
            return Ok(oracle);
        };
        let Ok(Some(stream)) = dial_dot_with_token(port, bearer_token, left, right) else {
            return Ok(oracle);
        };
        let Ok(remote) = read_dot_reply(stream) else {
            return Ok(oracle);
        };
        self.executed.store(true, Ordering::Relaxed);
        let verified = (remote - oracle).abs() <= 1e-3 * remote.abs().max(oracle.abs()).max(1.0);
        if verified {
            self.verified.store(true, Ordering::Relaxed);
        }
        Ok(if verified { remote } else { oracle })
    }
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
        let server = thread::spawn(move || serve_once_with_token(&listener, None));
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
        let server = thread::spawn(move || serve_once_with_token(&listener, None));
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
        let server = thread::spawn(move || serve_once_with_token(&listener, None));
        let mut client = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        client.write_all(b"DOT - 1 NaN 1\n").expect("send");
        assert!(matches!(
            server.join().expect("server thread"),
            Err(RemoteError::MalformedRequest)
        ));
    }

    #[test]
    fn authenticated_adapter_executes_and_matches_cpu_oracle() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || serve_once_with_token(&listener, Some("test-token")));
        let adapter = RemoteAccelerator::new(
            Some(format!("127.0.0.1:{port}")),
            Some("test-token".to_owned()),
        );
        assert!(!adapter.capability().available());
        assert_eq!(adapter.dot(&[1.0, 2.0], &[3.0, 4.0]), Ok(11.0));
        server.join().expect("server thread").expect("served");
        let state = adapter.capability();
        assert!(state.available());
        assert!(state.initialized());
        assert!(state.executed());
        assert!(state.runtime_verified());
    }

    #[test]
    fn wrong_or_missing_token_fails_closed_to_cpu() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || serve_once_with_token(&listener, Some("right-token")));
        let adapter = RemoteAccelerator::new(
            Some(format!("127.0.0.1:{port}")),
            Some("wrong-token".to_owned()),
        );
        assert_eq!(adapter.dot(&[1.0, 2.0], &[3.0, 4.0]), Ok(11.0));
        assert!(matches!(
            server.join().expect("server thread"),
            Err(RemoteError::AuthenticationFailed)
        ));
        assert!(!adapter.capability().executed());

        let unconfigured = RemoteAccelerator::new(Some(format!("127.0.0.1:{port}")), None);
        assert_eq!(unconfigured.dot(&[1.0], &[2.0]), Ok(2.0));
        assert!(!unconfigured.capability().available());
    }

    #[test]
    fn malformed_tokens_and_oversized_client_frames_are_bounded() {
        assert!(!valid_token(""));
        assert!(!valid_token("contains space"));
        assert!(!valid_token(&"x".repeat(MAX_TOKEN_SIZE + 1)));
        assert!(tokens_equal("same-token", "same-token"));
        assert!(!tokens_equal("short", "longer"));
        assert!(!tokens_equal("same-token", "same-tokem"));
        assert!(matches!(
            dial_dot_with_token(1, "contains space", &[1.0], &[1.0]),
            Err(RemoteError::AuthenticationFailed)
        ));
        let oversized = vec![1.234_567_9_f32; MAX_MESSAGE_SIZE];
        assert!(matches!(
            dial_dot_with_token(1, "token", &oversized, &oversized),
            Err(RemoteError::FrameTooLong)
        ));
    }

    #[test]
    fn non_loopback_listener_policy_requires_authentication() {
        let non_loopback = SocketAddr::from(([0, 0, 0, 0], 4000));
        let loopback = SocketAddr::from(([127, 0, 0, 1], 4000));
        assert!(listener_auth_allowed(loopback, None));
        assert!(!listener_auth_allowed(non_loopback, None));
        assert!(!listener_auth_allowed(non_loopback, Some("bad token")));
        assert!(listener_auth_allowed(non_loopback, Some("bounded-token")));
    }

    #[test]
    fn incorrect_or_slow_remote_results_fall_back_without_verification() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buffer = [0_u8; 256];
            let _ = read_line(&mut stream, &mut buffer).expect("request");
            write_line(&mut stream, b"999\n").expect("wrong reply");
        });
        let adapter = RemoteAccelerator::new(
            Some(format!("127.0.0.1:{port}")),
            Some("test-token".to_owned()),
        );
        assert_eq!(adapter.dot(&[1.0, 2.0], &[3.0, 4.0]), Ok(11.0));
        server.join().expect("server thread");
        assert!(adapter.capability().executed());
        assert!(!adapter.capability().runtime_verified());

        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("listener");
        let port = listener.local_addr().expect("local address").port();
        let server = thread::spawn(move || {
            let (_stream, _) = listener.accept().expect("accept");
            thread::sleep(IO_TIMEOUT + Duration::from_millis(100));
        });
        let adapter = RemoteAccelerator::new(
            Some(format!("127.0.0.1:{port}")),
            Some("test-token".to_owned()),
        );
        assert_eq!(adapter.dot(&[1.0], &[2.0]), Ok(2.0));
        assert!(!adapter.capability().executed());
        server.join().expect("server thread");
    }
}
