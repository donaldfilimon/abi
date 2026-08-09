//! `abi-mcp`: stdio plus a best-effort custom loopback HTTP entry point.
//!
//! Ported from `src/mcp/main.zig`. Stdio remains the primary transport; the
//! one-request-per-connection HTTP compatibility endpoint is spawned on a
//! background thread when the loopback bind succeeds.

use std::sync::atomic::Ordering;

fn main() {
    let state = abi_mcp::McpState::new();

    // Always attempt the loopback compatibility listener; failures stay stdio-only.
    let http_handle = match abi_mcp::McpHttpServer::bind(abi_mcp::HttpConfig::from_env(), state) {
        Ok(server) => match server.local_port() {
            Ok(port) => {
                let stop = server.stop_flag();
                eprintln!(
                    "MCP loopback HTTP compatibility transport listening on http://127.0.0.1:{port}/sse and /message"
                );
                let handle = std::thread::spawn(move || {
                    let _ = server.run();
                });
                Some((port, stop, handle))
            }
            Err(err) => {
                eprintln!(
                    "MCP HTTP transport could not resolve its bound port ({err}); continuing with stdio only"
                );
                None
            }
        },
        Err(err) => {
            eprintln!("MCP HTTP transport unavailable ({err}); continuing with stdio only");
            None
        }
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    abi_mcp::stdio::run_loop(state, stdin.lock(), stdout.lock());

    if let Some((port, stop, handle)) = http_handle {
        stop.store(true, Ordering::SeqCst);
        // Wake the actual bound listener. Re-reading the environment is wrong
        // for an internally requested ephemeral port and previously left the
        // accept thread blocked forever on shutdown.
        let _ = std::net::TcpStream::connect(format!("127.0.0.1:{port}"));
        let _ = handle.join();
    }
}
