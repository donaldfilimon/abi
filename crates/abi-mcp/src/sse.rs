//! Persistent MCP 2024-11-05 HTTP+SSE sessions for the loopback listener.
//!
//! `GET /sse` registers a session under an unguessable random (v4) UUID, answers with
//! an `endpoint` event naming `/message?sessionId=<id>`, and hands the socket
//! to a dedicated thread so the accept loop stays free for the POSTs that
//! publish to it. That thread forwards each queued JSON-RPC response as an SSE
//! `message` event, whose data is the exact line stdio would write. It exits,
//! and deregisters the session, when the client disconnects, a write fails or
//! times out, or the server stop flag is set. Sessions are capped at
//! [`MAX_SESSIONS`]; an extra `GET /sse` gets `503`, never an unbounded thread.
//!
//! This is the 2024-11-05 transport, superseded upstream by Streamable HTTP
//! (2025-03-26). It stays loopback-only and bearer-gated like the rest of
//! [`crate::http`].

use std::collections::HashMap;
use std::io::{ErrorKind, Read};
use std::net::{Shutdown, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;
use std::time::Duration;

use abi_foundation::http::write_all;

/// Upper bound on concurrently open SSE sessions (one thread each).
pub const MAX_SESSIONS: usize = 16;

/// How often a session thread checks the stop flag and peer liveness.
const POLL_INTERVAL: Duration = Duration::from_millis(250);
/// Idle time before a `:` keepalive comment is written.
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);
/// A client that stops reading cannot wedge a session thread past this.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// Bounds the liveness probe; data the client sends on the stream is ignored.
const LIVENESS_PROBE: Duration = Duration::from_millis(1);

/// Open SSE sessions, keyed by session id.
#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: Mutex<HashMap<String, Sender<String>>>,
}

impl SessionRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of open sessions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether no session is open.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Sender<String>>> {
        // The map holds no invariant a panicking holder could break halfway.
        self.sessions.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn register(&self) -> Option<(String, Receiver<String>)> {
        let mut sessions = self.lock();
        if sessions.len() >= MAX_SESSIONS {
            return None;
        }
        let id = uuid::Uuid::new_v4().simple().to_string();
        let (tx, rx) = channel();
        sessions.insert(id.clone(), tx);
        Some((id, rx))
    }

    /// Queue one JSON-RPC response line for a session's stream.
    ///
    /// Returns `false` when no such session is open (never registered, or its
    /// client already went away), so the caller can answer `404` instead of
    /// acknowledging a response nobody will receive.
    pub(crate) fn publish(&self, id: &str, line: String) -> bool {
        self.lock()
            .get(id)
            .is_some_and(|sender| sender.send(line).is_ok())
    }

    /// Whether `id` names an open session.
    pub(crate) fn contains(&self, id: &str) -> bool {
        self.lock().contains_key(id)
    }

    fn remove(&self, id: &str) {
        self.lock().remove(id);
    }
}

/// Serve `GET /sse`: register a session, announce its endpoint, and move the
/// stream to a session thread. Returns once the thread owns the socket.
pub(crate) fn open(
    mut stream: TcpStream,
    sessions: &Arc<SessionRegistry>,
    stop: &Arc<AtomicBool>,
) -> std::io::Result<()> {
    let Some((id, rx)) = sessions.register() else {
        return write_all(
            &mut stream,
            b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 29\r\nConnection: close\r\n\r\n{\"error\":\"too many sessions\"}",
        );
    };
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\nevent: endpoint\ndata: /message?sessionId={id}\n\n"
    );
    let prepared = stream
        .set_write_timeout(Some(WRITE_TIMEOUT))
        .and_then(|()| stream.set_read_timeout(Some(LIVENESS_PROBE)))
        .and_then(|()| write_all(&mut stream, head.as_bytes()));
    if let Err(error) = prepared {
        sessions.remove(&id);
        return Err(error);
    }
    let sessions = Arc::clone(sessions);
    let stop = Arc::clone(stop);
    thread::spawn(move || {
        pump(&mut stream, &rx, &stop);
        sessions.remove(&id);
        let _ = stream.shutdown(Shutdown::Both);
    });
    Ok(())
}

/// Forward queued responses until the peer leaves or the server stops.
fn pump(stream: &mut TcpStream, rx: &Receiver<String>, stop: &AtomicBool) {
    let mut idle = Duration::ZERO;
    while !stop.load(Ordering::SeqCst) {
        match rx.recv_timeout(POLL_INTERVAL) {
            Ok(line) => {
                let event = format!("event: message\ndata: {line}\n\n");
                if write_all(stream, event.as_bytes()).is_err() {
                    return;
                }
                idle = Duration::ZERO;
            }
            Err(RecvTimeoutError::Timeout) => {
                if !peer_is_open(stream) {
                    return;
                }
                idle += POLL_INTERVAL;
                if idle >= KEEPALIVE_INTERVAL {
                    if write_all(stream, b":\n\n").is_err() {
                        return;
                    }
                    idle = Duration::ZERO;
                }
            }
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

/// A read that returns end-of-stream means the client closed its side. The
/// read timeout is [`LIVENESS_PROBE`], so a quiet but open peer costs ~1 ms.
fn peer_is_open(stream: &mut TcpStream) -> bool {
    let mut scratch = [0_u8; 256];
    match stream.read(&mut scratch) {
        Ok(0) => false,
        Ok(_) => true,
        Err(error) => matches!(
            error.kind(),
            ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_caps_sessions_and_frees_a_slot_on_remove() {
        let registry = SessionRegistry::new();
        let mut ids = Vec::new();
        for _ in 0..MAX_SESSIONS {
            let (id, _rx) = registry.register().expect("below the cap");
            ids.push(id);
        }
        assert!(registry.register().is_none(), "cap must refuse");
        registry.remove(&ids[0]);
        assert!(registry.register().is_some(), "a freed slot is reusable");
    }

    #[test]
    fn session_ids_are_distinct_uuids() {
        let registry = SessionRegistry::new();
        let (a, _ra) = registry.register().expect("first");
        let (b, _rb) = registry.register().expect("second");
        assert_ne!(a, b);
        assert_eq!(a.len(), 32, "simple (unhyphenated) UUID form: {a}");
        assert!(a.bytes().all(|byte| byte.is_ascii_hexdigit()), "{a}");
    }

    #[test]
    fn publish_reaches_the_receiver_and_fails_for_unknown_or_gone_sessions() {
        let registry = SessionRegistry::new();
        let (id, rx) = registry.register().expect("register");
        assert!(registry.publish(&id, "line".to_string()));
        assert_eq!(rx.recv().expect("queued line"), "line");
        assert!(!registry.publish("not-a-session", "x".to_string()));
        drop(rx);
        assert!(
            !registry.publish(&id, "x".to_string()),
            "a session whose thread dropped its receiver must not accept lines"
        );
    }
}
