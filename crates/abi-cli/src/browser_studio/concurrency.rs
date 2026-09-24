//! Bounded connection workers for the loopback studio.

use super::{REQUEST_DEADLINE, handle_connection_with_stop};
use abi_foundation::http::{DeadlineReader, MAX_REQUEST_SIZE, read_request, write_all};
use std::io;
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

const MAX_REQUEST_WORKERS: usize = 16;
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(5);
const OVERLOAD_READ_DEADLINE: Duration = Duration::from_millis(25);
const RESPONSE_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const OVERLOAD_WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const OVERLOADED: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: 29\r\nConnection: close\r\n\r\n{\"error\":\"server overloaded\"}";

fn reap_workers(workers: &mut Vec<thread::JoinHandle<io::Result<()>>>) {
    let mut index = 0;
    while index < workers.len() {
        if workers[index].is_finished() {
            if let Err(error) = workers
                .swap_remove(index)
                .join()
                .unwrap_or_else(|_| Err(io::Error::other("studio request worker panicked")))
            {
                eprintln!("browser studio: {error}");
            }
        } else {
            index += 1;
        }
    }
}

fn reject_overloaded(stream: &mut TcpStream, stop: &AtomicBool) -> io::Result<()> {
    let _ = read_request(
        &mut DeadlineReader::with_stop(stream, stop, OVERLOAD_READ_DEADLINE),
        MAX_REQUEST_SIZE,
    );
    stream.set_write_timeout(Some(OVERLOAD_WRITE_TIMEOUT))?;
    write_all(stream, OVERLOADED)
}

pub(super) fn serve_studio_loop(
    listener: &TcpListener,
    token: Option<String>,
    stop: &Arc<AtomicBool>,
) -> io::Result<()> {
    listener.set_nonblocking(true)?;
    let token = token.map(Arc::<str>::from);
    let mut workers = Vec::new();
    let result = loop {
        if stop.load(Ordering::SeqCst) {
            break Ok(());
        }
        reap_workers(&mut workers);
        match listener.accept() {
            Ok((mut stream, _)) if workers.len() >= MAX_REQUEST_WORKERS => {
                if let Err(error) = reject_overloaded(&mut stream, stop) {
                    eprintln!("browser studio overload response failed: {error}");
                }
            }
            Ok((mut stream, _)) => {
                if let Err(error) = stream.set_write_timeout(Some(RESPONSE_WRITE_TIMEOUT)) {
                    eprintln!("browser studio: {error}");
                    continue;
                }
                let token = token.clone();
                let stop = Arc::clone(stop);
                let worker = thread::Builder::new()
                    .name("browser-studio-request".into())
                    .spawn(move || {
                        handle_connection_with_stop(
                            &mut stream,
                            token.as_deref(),
                            REQUEST_DEADLINE,
                            Some(&stop),
                        )
                    });
                match worker {
                    Ok(worker) => workers.push(worker),
                    Err(error) => break Err(error),
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_POLL_INTERVAL);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => break Err(error),
        }
    };
    stop.store(true, Ordering::SeqCst);
    for worker in workers {
        if let Err(error) = worker
            .join()
            .unwrap_or_else(|_| Err(io::Error::other("studio request worker panicked")))
        {
            eprintln!("browser studio: {error}");
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::{Ipv4Addr, Shutdown};

    fn exchange(port: u16, request: &[u8]) -> String {
        let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect studio");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set response timeout");
        stream.write_all(request).expect("write request");
        stream.shutdown(Shutdown::Write).expect("finish request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        response
    }

    #[test]
    fn slow_studio_peer_does_not_hold_accept_and_shutdown_is_prompt() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind studio");
        let port = listener.local_addr().expect("studio address").port();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || serve_studio_loop(&listener, None, &server_stop));

        let mut slow = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect slow peer");
        slow.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n")
            .expect("send partial request");
        thread::sleep(Duration::from_millis(50));
        let started = std::time::Instant::now();
        let healthy = exchange(port, b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(healthy.starts_with("HTTP/1.1 200 OK"), "{healthy}");
        assert!(started.elapsed() < Duration::from_secs(3));

        stop.store(true, Ordering::SeqCst);
        let stopping = std::time::Instant::now();
        handle.join().expect("studio joins").expect("studio stops");
        assert!(stopping.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn saturated_studio_workers_get_503() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).expect("bind studio");
        let port = listener.local_addr().expect("studio address").port();
        let stop = Arc::new(AtomicBool::new(false));
        let server_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || serve_studio_loop(&listener, None, &server_stop));

        let mut slow_peers = Vec::new();
        for _ in 0..MAX_REQUEST_WORKERS {
            let mut stream =
                TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect slow peer");
            stream
                .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n")
                .expect("send partial request");
            slow_peers.push(stream);
        }
        thread::sleep(Duration::from_millis(100));
        let overloaded = exchange(port, b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n");
        assert!(
            overloaded.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{overloaded}"
        );
        stop.store(true, Ordering::SeqCst);
        handle.join().expect("studio joins").expect("studio stops");
        drop(slow_peers);
    }
}
