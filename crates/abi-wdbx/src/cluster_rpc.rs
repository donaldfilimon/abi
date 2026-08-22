//! Reference-scoped TCP transport for the WDBX consensus demo.
//!
//! One bounded newline frame is exchanged per connection. Loopback operation
//! may be unauthenticated for compatibility; non-loopback listeners require a
//! shared secret. This is not TLS, production multi-host Raft, or sharding.

use crate::cluster::{AppendReply, Node, VoteReply, apply_append, apply_vote};
use crate::net_line::{LineError, MAX_LINE_SIZE, read_line, write_line};
use abi_foundation::env::{WDBX_CLUSTER_PEERS, WDBX_CLUSTER_TOKEN};
use abi_foundation::http::fixed_work_eq;
use std::collections::BTreeSet;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

mod data;

use data::{ClusterDataPlane, encode_response};
pub use data::{
    ClusterDataRequest, ClusterDataResponse, ClusterKvState, dial_data, read_data_reply,
};

/// Connect/read/write deadline for the bounded reference transport.
pub const CLUSTER_RPC_TIMEOUT: Duration = Duration::from_secs(2);
/// Largest authenticated exact-transaction frame accepted by the data plane.
pub const MAX_CLUSTER_DATA_FRAME_SIZE: usize = 64 * 1024;

/// Shared-secret configuration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClusterAuth {
    token: Option<String>,
}

impl ClusterAuth {
    /// Construct a disabled policy.
    #[must_use]
    pub const fn disabled() -> Self {
        Self { token: None }
    }

    /// Construct an enabled policy after validating frame safety.
    pub fn new(token: impl Into<String>) -> Result<Self, RpcError> {
        let token = token.into();
        if token.is_empty() || token.bytes().any(|byte| byte.is_ascii_whitespace()) {
            return Err(RpcError::InvalidToken);
        }
        Ok(Self { token: Some(token) })
    }

    /// Whether a token is required.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.token.is_some()
    }

    fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    fn matches(&self, supplied: Option<&str>) -> bool {
        match (self.token(), supplied) {
            (None, None) => true,
            (Some(expected), Some(got)) => fixed_work_eq(expected.as_bytes(), got.as_bytes()),
            _ => false,
        }
    }
}

/// Authentication plus optional numeric peer allowlist.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClusterPolicy {
    /// Shared-secret policy.
    pub auth: ClusterAuth,
    peers: Option<BTreeSet<u32>>,
}

impl ClusterPolicy {
    /// Build a policy from operator-facing strings.
    pub fn from_values(token: Option<&str>, peers: Option<&str>) -> Result<Self, RpcError> {
        let auth = token.map(ClusterAuth::new).transpose()?.unwrap_or_default();
        let peers = peers.map(parse_peers).transpose()?;
        Ok(Self { auth, peers })
    }

    /// Load the canonical cluster token and peer-list environment variables.
    pub fn from_env() -> Result<Self, RpcError> {
        let token = std::env::var(WDBX_CLUSTER_TOKEN).ok();
        let peers = std::env::var(WDBX_CLUSTER_PEERS).ok();
        Self::from_values(token.as_deref(), peers.as_deref())
    }

    /// Whether a peer id is admitted.
    #[must_use]
    pub fn allows_peer(&self, id: u32) -> bool {
        self.peers.as_ref().is_none_or(|peers| peers.contains(&id))
    }

    /// Replace the allowlist without restarting a listener.
    #[must_use]
    pub fn with_peers(mut self, peers: Option<BTreeSet<u32>>) -> Self {
        self.peers = peers;
        self
    }
}

fn parse_peers(raw: &str) -> Result<BTreeSet<u32>, RpcError> {
    let mut peers = BTreeSet::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(RpcError::InvalidPeers);
        }
        peers.insert(part.parse().map_err(|_| RpcError::InvalidPeers)?);
    }
    if peers.is_empty() {
        return Err(RpcError::InvalidPeers);
    }
    Ok(peers)
}

/// Cluster RPC or policy failure.
#[derive(Debug)]
pub enum RpcError {
    /// Socket failure.
    Io(io::Error),
    /// Request frame has invalid syntax.
    MalformedRequest,
    /// Response frame has invalid syntax.
    MalformedResponse,
    /// Frame exceeds the fixed transport bound.
    LineTooLong,
    /// Host is not a supported numeric address or `localhost`.
    InvalidHost,
    /// A routable listener was requested without authentication.
    NonLoopbackRequiresToken,
    /// Token is empty or contains frame-delimiting whitespace.
    InvalidToken,
    /// Peer list is empty, malformed, or contains a non-`u32` id.
    InvalidPeers,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "{error}"),
            Self::MalformedRequest => formatter.write_str("malformed cluster request"),
            Self::MalformedResponse => formatter.write_str("malformed cluster response"),
            Self::LineTooLong => formatter.write_str("cluster frame too long"),
            Self::InvalidHost => formatter.write_str("invalid cluster host"),
            Self::NonLoopbackRequiresToken => {
                formatter.write_str("non-loopback cluster bind requires a token")
            }
            Self::InvalidToken => formatter.write_str("invalid cluster token"),
            Self::InvalidPeers => formatter.write_str("invalid cluster peer list"),
        }
    }
}

impl std::error::Error for RpcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for RpcError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<LineError> for RpcError {
    fn from(error: LineError) -> Self {
        match error {
            LineError::Io(error) => Self::Io(error),
            LineError::TooLong => Self::LineTooLong,
        }
    }
}

enum Request<'line> {
    Vote {
        term: u64,
        candidate: u32,
        supplied: Option<&'line str>,
    },
    Append {
        term: u64,
        leader: Option<u32>,
        data: &'line str,
        supplied: Option<&'line str>,
    },
    Shutdown {
        supplied: Option<&'line str>,
    },
    Data {
        body: &'line str,
        supplied: Option<&'line str>,
    },
    Unknown,
}

fn parse_request(line: &str) -> Result<Request<'_>, RpcError> {
    if let Some(rest) = line.strip_prefix("AUTH ") {
        let (token, command) = rest.split_once(' ').ok_or(RpcError::MalformedRequest)?;
        if let Some(vote) = command.strip_prefix("VOTE ") {
            let (term, candidate) = vote.split_once(' ').ok_or(RpcError::MalformedRequest)?;
            return Ok(Request::Vote {
                term: term.parse().map_err(|_| RpcError::MalformedRequest)?,
                candidate: candidate.parse().map_err(|_| RpcError::MalformedRequest)?,
                supplied: Some(token),
            });
        }
        if let Some(append) = command.strip_prefix("APPEND ") {
            let mut fields = append.splitn(3, ' ');
            let term = fields
                .next()
                .ok_or(RpcError::MalformedRequest)?
                .parse()
                .map_err(|_| RpcError::MalformedRequest)?;
            let leader_field = fields.next().unwrap_or_default();
            let leader = leader_field.parse().ok();
            let data = fields.next().unwrap_or_default();
            return Ok(Request::Append {
                term,
                leader,
                data,
                supplied: Some(token),
            });
        }
        if command == "SHUTDOWN" {
            return Ok(Request::Shutdown {
                supplied: Some(token),
            });
        }
        if let Some(body) = command.strip_prefix("DATA ") {
            return Ok(Request::Data {
                body,
                supplied: Some(token),
            });
        }
        return Ok(Request::Unknown);
    }
    if let Some(vote) = line.strip_prefix("VOTE ") {
        let (term, candidate) = vote.split_once(' ').ok_or(RpcError::MalformedRequest)?;
        return Ok(Request::Vote {
            term: term.parse().map_err(|_| RpcError::MalformedRequest)?,
            candidate: candidate.parse().map_err(|_| RpcError::MalformedRequest)?,
            supplied: None,
        });
    }
    if let Some(append) = line.strip_prefix("APPEND ") {
        let (term, data) = append.split_once(' ').unwrap_or((append, ""));
        return Ok(Request::Append {
            term: term.parse().map_err(|_| RpcError::MalformedRequest)?,
            leader: None,
            data,
            supplied: None,
        });
    }
    Ok(Request::Unknown)
}

/// One-request-per-connection consensus listener.
#[derive(Debug)]
pub struct ClusterRpcServer {
    listener: TcpListener,
    node: Node,
    policy: ClusterPolicy,
    data_plane: Option<ClusterDataPlane>,
    shutdown_requested: bool,
}

impl ClusterRpcServer {
    /// Bind an explicit host, enforcing authentication before routable exposure.
    pub fn bind(
        host: &str,
        port: u16,
        node: Node,
        policy: ClusterPolicy,
    ) -> Result<Self, RpcError> {
        if !is_loopback_host(host) && !policy.auth.enabled() {
            return Err(RpcError::NonLoopbackRequiresToken);
        }
        let listener = TcpListener::bind(SocketAddr::new(parse_host(host)?, port))?;
        Ok(Self {
            listener,
            node,
            policy,
            data_plane: None,
            shutdown_requested: false,
        })
    }

    /// Bind a listener with an isolated writable v2 data plane.
    pub fn bind_with_store(
        host: &str,
        port: u16,
        node: Node,
        policy: ClusterPolicy,
        store: crate::VersionedStore,
    ) -> Result<Self, RpcError> {
        let mut server = Self::bind(host, port, node, policy)?;
        server.data_plane = Some(ClusterDataPlane::new(store));
        Ok(server)
    }

    /// Bound port, including a kernel-selected ephemeral port.
    pub fn local_port(&self) -> Result<u16, RpcError> {
        Ok(self.listener.local_addr()?.port())
    }

    /// Current node state.
    #[must_use]
    pub const fn node(&self) -> &Node {
        &self.node
    }

    /// Whether an authenticated shutdown request was accepted.
    #[must_use]
    pub const fn shutdown_requested(&self) -> bool {
        self.shutdown_requested
    }

    /// Replace peer membership for subsequent requests.
    pub fn set_peers(&mut self, peers: Option<BTreeSet<u32>>) {
        self.policy = self.policy.clone().with_peers(peers);
    }

    /// Accept, authorize, apply and answer one RPC.
    pub fn serve_one(&mut self) -> Result<(), RpcError> {
        let (mut stream, _) = self.listener.accept()?;
        stream.set_read_timeout(Some(CLUSTER_RPC_TIMEOUT))?;
        stream.set_write_timeout(Some(CLUSTER_RPC_TIMEOUT))?;
        let mut buffer = vec![0; MAX_CLUSTER_DATA_FRAME_SIZE];
        let line = read_line(&mut stream, &mut buffer)?;
        let line = std::str::from_utf8(line).map_err(|_| RpcError::MalformedRequest)?;
        let request = parse_request(line)?;
        if line.len() >= MAX_LINE_SIZE && !matches!(&request, Request::Data { .. }) {
            return Err(RpcError::LineTooLong);
        }
        let response = match request {
            Request::Vote {
                term,
                candidate,
                supplied,
            } => {
                if !self.policy.auth.matches(supplied) || !self.policy.allows_peer(candidate) {
                    format!("DENIED {}\n", self.node.term)
                } else {
                    let granted = apply_vote(&mut self.node, term, candidate);
                    format!(
                        "{} {}\n",
                        if granted { "GRANTED" } else { "DENIED" },
                        self.node.term
                    )
                }
            }
            Request::Append {
                term,
                leader,
                data,
                supplied,
            } => {
                let authorized = self.policy.auth.matches(supplied)
                    && (supplied.is_none() || leader.is_some())
                    && self
                        .policy
                        .peers
                        .as_ref()
                        .is_none_or(|_| leader.is_some_and(|id| self.policy.allows_peer(id)));
                if authorized {
                    let ack = apply_append(&mut self.node, term, data.as_bytes());
                    format!("{} {}\n", if ack { "ACK" } else { "NACK" }, self.node.term)
                } else {
                    format!("NACK {}\n", self.node.term)
                }
            }
            Request::Shutdown { supplied } => {
                if self.policy.auth.enabled() && self.policy.auth.matches(supplied) {
                    self.shutdown_requested = true;
                    format!("BYE {}\n", self.node.term)
                } else {
                    format!("DENIED {}\n", self.node.term)
                }
            }
            Request::Data { body, supplied } => {
                if !self.policy.auth.enabled() || !self.policy.auth.matches(supplied) {
                    format!("DENIED {}\n", self.node.term)
                } else if let Some(data_plane) = &mut self.data_plane {
                    encode_response(&data_plane.handle(body))
                } else {
                    encode_response(&ClusterDataResponse::Error {
                        message: "data plane is unavailable".into(),
                    })
                }
            }
            Request::Unknown => "ERR 0\n".to_string(),
        };
        write_line(&mut stream, response.as_bytes())?;
        Ok(())
    }

    /// Serve until interrupted.
    pub fn run(&mut self) -> Result<(), RpcError> {
        while !self.shutdown_requested {
            if let Err(error) = self.serve_one() {
                eprintln!("cluster RPC serve error: {error}");
            }
        }
        Ok(())
    }
}

/// Send an optional-auth `RequestVote` and return its open reply stream.
pub fn dial_vote(
    host: &str,
    port: u16,
    term: u64,
    candidate: u32,
    token: Option<&str>,
) -> Result<Option<TcpStream>, RpcError> {
    let message = token.map_or_else(
        || format!("VOTE {term} {candidate}\n"),
        |token| format!("AUTH {token} VOTE {term} {candidate}\n"),
    );
    dial(host, port, &message)
}

/// Send an optional-auth `AppendEntries` and return its open reply stream.
pub fn dial_append(
    host: &str,
    port: u16,
    term: u64,
    leader: Option<u32>,
    data: &str,
    token: Option<&str>,
) -> Result<Option<TcpStream>, RpcError> {
    let message = match token {
        Some(token) => {
            let leader = leader.ok_or(RpcError::MalformedRequest)?;
            format!("AUTH {token} APPEND {term} {leader} {data}\n")
        }
        None => format!("APPEND {term} {data}\n"),
    };
    dial(host, port, &message)
}

/// Send an authenticated graceful-shutdown request.
pub fn dial_shutdown(host: &str, port: u16, token: &str) -> Result<Option<TcpStream>, RpcError> {
    if token.is_empty() {
        return Err(RpcError::InvalidToken);
    }
    dial(host, port, &format!("AUTH {token} SHUTDOWN\n"))
}

fn dial(host: &str, port: u16, message: &str) -> Result<Option<TcpStream>, RpcError> {
    let address = match parse_host(host) {
        Ok(address) => SocketAddr::new(address, port),
        Err(_) => return Ok(None),
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&address, CLUSTER_RPC_TIMEOUT) else {
        return Ok(None);
    };
    stream.set_read_timeout(Some(CLUSTER_RPC_TIMEOUT))?;
    stream.set_write_timeout(Some(CLUSTER_RPC_TIMEOUT))?;
    write_line(&mut stream, message.as_bytes())?;
    Ok(Some(stream))
}

/// Read a vote reply.
pub fn read_vote_reply(mut stream: TcpStream) -> Result<VoteReply, RpcError> {
    let (verb, term) = read_reply(&mut stream)?;
    Ok(VoteReply {
        granted: verb == "GRANTED",
        term,
    })
}

/// Read an append reply.
pub fn read_append_reply(mut stream: TcpStream) -> Result<AppendReply, RpcError> {
    let (verb, term) = read_reply(&mut stream)?;
    Ok(AppendReply {
        ack: verb == "ACK",
        term,
    })
}

/// Read an authenticated graceful-shutdown acknowledgement.
pub fn read_shutdown_reply(mut stream: TcpStream) -> Result<bool, RpcError> {
    let (verb, _) = read_reply(&mut stream)?;
    Ok(verb == "BYE")
}

fn read_reply(stream: &mut TcpStream) -> Result<(String, u64), RpcError> {
    let mut buffer = [0; 64];
    let line = read_line(stream, &mut buffer)?;
    let line = std::str::from_utf8(line).map_err(|_| RpcError::MalformedResponse)?;
    let (verb, term) = line.split_once(' ').ok_or(RpcError::MalformedResponse)?;
    Ok((
        verb.to_string(),
        term.parse().map_err(|_| RpcError::MalformedResponse)?,
    ))
}

fn parse_host(host: &str) -> Result<IpAddr, RpcError> {
    if host == "localhost" {
        return Ok(IpAddr::V4(Ipv4Addr::LOCALHOST));
    }
    host.parse().map_err(|_| RpcError::InvalidHost)
}

fn is_loopback_host(host: &str) -> bool {
    host == "localhost"
        || host == "::1"
        || host.starts_with("127.")
        || parse_host(host).is_ok_and(|address| address.is_loopback())
}

#[cfg(test)]
mod tests;
