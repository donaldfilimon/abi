//! `abi-mcp`: stdio + optional loopback HTTP/SSE MCP server entry point.
//!
//! Ported from `src/mcp/main.zig`. Stdio remains the primary transport; HTTP/SSE
//! is spawned on a background thread when the loopback bind succeeds.

use std::sync::atomic::Ordering;

fn main() {
    let state = abi_mcp::McpState::new();

    // Optional loopback HTTP/SSE (Zig always attempted; failures stay stdio-only).
    let http_handle = match abi_mcp::McpHttpServer::bind(abi_mcp::HttpConfig::from_env(), state) {
        Ok(server) => {
            let port = server.local_port().unwrap_or(0);
            let stop = server.stop_flag();
            eprintln!(
                "MCP HTTP+SSE transport listening on http://127.0.0.1:{port}/sse and /message"
            );
            let handle = std::thread::spawn(move || {
                let _ = server.run();
            });
            Some((stop, handle))
        }
        Err(err) => {
            eprintln!("MCP HTTP transport unavailable ({err}); continuing with stdio only");
            None
        }
    };

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    abi_mcp::stdio::run_loop(state, stdin.lock(), stdout.lock());

    if let Some((stop, handle)) = http_handle {
        stop.store(true, Ordering::SeqCst);
        // Wake accept on the configured (or default) port.
        let port = std::env::var("ABI_MCP_HTTP_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
            .unwrap_or(abi_mcp::DEFAULT_HTTP_PORT);
        let _ = std::net::TcpStream::connect(format!("127.0.0.1:{port}"));
        let _ = handle.join();
    }
}
