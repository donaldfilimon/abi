//! Bounded request workers keep an incomplete HTTP read off the accept loop.

use super::{McpHttpServer, handle_connection};
use abi_foundation::http::DeadlineReader;
use std::io;
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Each request can hold a worker until its read deadline or tool dispatch ends.
const MAX_REQUEST_WORKERS: usize = 16;
const RESPONSE_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
/// Give a complete queued request a brief chance to drain before returning 503.
const OVERLOAD_READ_DEADLINE: Duration = Duration::from_millis(25);
/// A broken peer must not create an unbounded stderr stream.
const MAX_REPORTED_SERVER_ERRORS: usize = 8;
const OVERLOADED: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 29\r\nConnection: close\r\n\r\n{\"error\":\"server overloaded\"}";

struct ActiveRequest(Arc<AtomicUsize>);

impl Drop for ActiveRequest {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(super) fn run(server: &McpHttpServer) {
    let mut workers: Vec<JoinHandle<io::Result<()>>> = Vec::new();
    let mut reported_errors = 0_usize;
    while !server.stop.load(Ordering::SeqCst) {
        let accepted = server.listener.accept();
        if server.stop.load(Ordering::SeqCst) {
            break;
        }
        reap_finished(&mut workers, &mut reported_errors);
        let (mut stream, _) = match accepted {
            Ok(connection) => connection,
            Err(error) => {
                report_error(&error, &mut reported_errors);
                continue;
            }
        };
        if workers.len() >= MAX_REQUEST_WORKERS
            || server.active_requests.load(Ordering::Acquire) >= MAX_REQUEST_WORKERS
        {
            if let Err(error) = reject_overloaded(&mut stream, &server.stop) {
                report_error(&error, &mut reported_errors);
            }
            continue;
        }
        if let Err(error) = stream.set_write_timeout(Some(RESPONSE_WRITE_TIMEOUT)) {
            report_error(&error, &mut reported_errors);
            continue;
        }
        let state = server.state;
        let token = server.config.bearer_token.clone();
        let sessions = Arc::clone(&server.sessions);
        let stop = Arc::clone(&server.stop);
        let active_requests = Arc::clone(&server.active_requests);
        active_requests.fetch_add(1, Ordering::AcqRel);
        match thread::Builder::new()
            .name("mcp-http-request".into())
            .spawn(move || {
                let _active = ActiveRequest(active_requests);
                handle_connection(stream, state, token.as_deref(), &sessions, &stop)
            }) {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                server.active_requests.fetch_sub(1, Ordering::AcqRel);
                report_error(&error, &mut reported_errors);
            }
        }
    }
    // Shutdown interrupts incomplete reads. Join workers so stdio EOF cannot
    // silently abandon an in-flight tool mutation.
    for worker in workers {
        report_worker(
            worker,
            &mut reported_errors,
            server.stop.load(Ordering::SeqCst),
        );
    }
}

fn reject_overloaded(
    stream: &mut TcpStream,
    stop: &std::sync::atomic::AtomicBool,
) -> io::Result<()> {
    // Closing with unread request bytes can reset the connection before the
    // client sees the 503. Consume a complete queued request if available,
    // but cap this accept-loop work independently of the normal 30 s reader.
    let _ = abi_foundation::http::read_request(
        &mut DeadlineReader::with_stop(stream, stop, OVERLOAD_READ_DEADLINE),
        abi_foundation::http::MAX_REQUEST_SIZE,
    );
    stream.set_write_timeout(Some(RESPONSE_WRITE_TIMEOUT))?;
    abi_foundation::http::write_all(stream, OVERLOADED)
}

fn reap_finished(workers: &mut Vec<JoinHandle<io::Result<()>>>, reported_errors: &mut usize) {
    let mut index = 0;
    while index < workers.len() {
        if workers[index].is_finished() {
            let worker = workers.swap_remove(index);
            report_worker(worker, reported_errors, false);
        } else {
            index += 1;
        }
    }
}

fn report_worker(worker: JoinHandle<io::Result<()>>, reported_errors: &mut usize, stopping: bool) {
    let result = worker
        .join()
        .unwrap_or_else(|_| Err(io::Error::other("request worker panicked")));
    if let Err(error) = result
        && !stopping
    {
        report_error(&error, reported_errors);
    }
}

fn report_error(error: &io::Error, reported_errors: &mut usize) {
    if *reported_errors < MAX_REPORTED_SERVER_ERRORS {
        eprintln!("MCP loopback HTTP serve error: {error}");
    } else if *reported_errors == MAX_REPORTED_SERVER_ERRORS {
        eprintln!(
            "MCP loopback HTTP serve errors suppressed after {MAX_REPORTED_SERVER_ERRORS} reports"
        );
    }
    *reported_errors = reported_errors.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::HttpConfig;
    use crate::state::McpState;
    use std::io::{Read, Write};
    use std::net::Ipv4Addr;
    use std::time::Instant;

    fn wait_for_workers(active: &AtomicUsize, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while active.load(Ordering::Acquire) != expected && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(active.load(Ordering::Acquire), expected);
    }

    fn slow_request(port: u16) -> TcpStream {
        let mut stream =
            TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect slow peer");
        stream
            .write_all(b"POST /message HTTP/1.1\r\nContent-Length: 10\r\n\r\nx")
            .expect("write incomplete request");
        stream
    }

    fn ping(port: u16) -> String {
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect ping");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set response deadline");
        let body = r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#;
        write!(
            stream,
            "POST /message HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .expect("write ping");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        response
    }

    fn start() -> (
        u16,
        Arc<AtomicUsize>,
        Arc<std::sync::atomic::AtomicBool>,
        JoinHandle<io::Result<()>>,
    ) {
        let server = McpHttpServer::bind(
            HttpConfig {
                port: 0,
                bearer_token: None,
            },
            McpState::new(),
        )
        .expect("bind server");
        let port = server.local_port().expect("bound port");
        let active = Arc::clone(&server.active_requests);
        let stop = server.stop_flag();
        let handle = thread::spawn(move || server.run());
        (port, active, stop, handle)
    }

    fn stop(
        port: u16,
        flag: &AtomicUsize,
        stop: &std::sync::atomic::AtomicBool,
        handle: JoinHandle<io::Result<()>>,
    ) {
        stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect((Ipv4Addr::LOCALHOST, port));
        handle.join().expect("server joins").expect("server stops");
        assert_eq!(flag.load(Ordering::Acquire), 0);
    }

    #[test]
    fn slow_peer_does_not_block_another_request_or_shutdown() {
        let (port, active, stop_flag, handle) = start();
        let _slow = slow_request(port);
        wait_for_workers(&active, 1);
        assert!(ping(port).starts_with("HTTP/1.1 200 OK"));
        stop(port, &active, &stop_flag, handle);
    }

    #[test]
    fn full_worker_set_returns_503_before_dispatch() {
        let (port, active, stop_flag, handle) = start();
        let slow: Vec<_> = (0..MAX_REQUEST_WORKERS)
            .map(|_| slow_request(port))
            .collect();
        wait_for_workers(&active, MAX_REQUEST_WORKERS);
        assert!(ping(port).starts_with("HTTP/1.1 503 Service Unavailable"));
        stop(port, &active, &stop_flag, handle);
        drop(slow);
    }
}
