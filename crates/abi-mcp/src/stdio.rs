//! The stdio JSON-RPC transport: one line-delimited request per line, one
//! response line back.
//!
//! Ported from `src/mcp/stdio_transport.zig`. Reads byte-by-byte (rather than
//! `BufRead::read_line`) so an overlong line is caught and dropped — with a
//! `-32700 Parse error` response — before it grows past
//! [`crate::protocol::MAX_REQUEST_SIZE`], instead of buffering an unbounded
//! line from untrusted input.

use std::io::{Read, Write};

use crate::protocol::MAX_REQUEST_SIZE;
use crate::rpc::{self, RpcResponse};
use crate::state::McpState;

fn write_response<W: Write>(writer: &mut W, response: &RpcResponse) -> std::io::Result<()> {
    writeln!(writer, "{}", response.to_json_string())
}

/// Run the stdio request/response loop until `reader` reaches EOF.
///
/// Each complete line (bytes up to `\n`, with a trailing `\r` trimmed) is
/// handed to [`rpc::process`]; the response, if any, is written as one JSON
/// line. A read error ends the loop, matching Zig's `catch break`.
pub fn run_loop<R: Read, W: Write>(state: McpState, mut reader: R, mut writer: W) {
    let mut read_buf = [0_u8; 4096];
    let mut line: Vec<u8> = Vec::new();
    let mut discarding_overlong_line = false;

    loop {
        let n = match reader.read(&mut read_buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };

        for &byte in &read_buf[..n] {
            if discarding_overlong_line {
                if byte == b'\n' {
                    discarding_overlong_line = false;
                }
                continue;
            }
            if byte == b'\n' {
                if !process_line(state, &mut writer, &line) {
                    return;
                }
                line.clear();
                continue;
            }
            if line.len() >= MAX_REQUEST_SIZE {
                if write_response(&mut writer, &rpc::parse_error_response()).is_err() {
                    return;
                }
                line.clear();
                discarding_overlong_line = true;
                continue;
            }
            line.push(byte);
        }
    }

    if !discarding_overlong_line && !line.is_empty() {
        let _ = process_line(state, &mut writer, &line);
    }
}

fn process_line<W: Write>(state: McpState, writer: &mut W, raw: &[u8]) -> bool {
    let trimmed = trim_trailing_cr(raw);
    let Ok(text) = std::str::from_utf8(trimmed) else {
        return write_response(writer, &rpc::parse_error_response()).is_ok();
    };
    if let Some(response) = rpc::process(state, text) {
        return write_response(writer, &response).is_ok();
    }
    true
}

fn trim_trailing_cr(line: &[u8]) -> &[u8] {
    if line.last() == Some(&b'\r') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn run_bytes(input: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        run_loop(McpState::new(), Cursor::new(input), &mut out);
        String::from_utf8(out)
            .expect("valid utf8")
            .lines()
            .map(ToString::to_string)
            .collect()
    }

    fn run(input: &str) -> Vec<String> {
        run_bytes(input.as_bytes())
    }

    #[test]
    fn processes_one_line_request() {
        let lines = run("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"result\":{}"));
    }

    #[test]
    fn processes_multiple_lines_and_a_trailing_unterminated_one() {
        let lines = run(
            "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}",
        );
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"id\":1"));
        assert!(lines[1].contains("\"id\":2"));
    }

    #[test]
    fn trims_trailing_carriage_return() {
        let lines = run("{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\r\n");
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"result\":{}"));
    }

    #[test]
    fn invalid_utf8_returns_one_parse_error_and_recovers_at_the_next_frame() {
        let mut input = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\",\"bad\":\"".to_vec();
        input.push(0xff);
        input.extend_from_slice(b"\"}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n");

        let lines = run_bytes(&input);
        assert_eq!(lines.len(), 2);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&lines[0]).expect("parse error is json"),
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": {"code": -32700, "message": "Parse error"},
            })
        );
        assert!(lines[1].contains("\"id\":2"));
        assert!(lines[1].contains("\"result\":{}"));
    }

    #[test]
    fn drops_the_entire_overlong_line_and_recovers_at_the_next_newline() {
        let oversized_prefix = "x".repeat(MAX_REQUEST_SIZE + 1);
        let executable_suffix = r#"{"jsonrpc":"2.0","id":99,"method":"ping"}"#;
        let next_line = r#"{"jsonrpc":"2.0","id":100,"method":"ping"}"#;
        let lines = run(&format!(
            "{oversized_prefix}{executable_suffix}\n{next_line}\n"
        ));
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"code\":-32700"));
        assert!(!lines.iter().any(|line| line.contains("\"id\":99")));
        assert!(lines[1].contains("\"id\":100"));
    }

    #[test]
    fn overlong_unterminated_line_is_not_dispatched_at_eof() {
        let oversized_prefix = "x".repeat(MAX_REQUEST_SIZE + 1);
        let executable_suffix = r#"{"jsonrpc":"2.0","id":99,"method":"ping"}"#;
        let lines = run(&format!("{oversized_prefix}{executable_suffix}"));
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"code\":-32700"));
        assert!(!lines[0].contains("\"id\":99"));
    }

    #[test]
    fn empty_lines_produce_no_response() {
        let lines = run("\n\n");
        assert_eq!(lines, [] as [std::string::String; 0]);
    }

    struct FailWriter;

    impl Write for FailWriter {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("sink closed"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn write_error_ends_the_stdio_loop() {
        let input = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"ping\"}\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"ping\"}\n";
        run_loop(McpState::new(), Cursor::new(input.as_bytes()), FailWriter);
    }

    #[test]
    fn notifications_produce_no_bytes_and_do_not_hide_the_next_response() {
        let lines = run(
            "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}\n{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"ping\"}\n",
        );
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("\"id\":7"));
    }
}
