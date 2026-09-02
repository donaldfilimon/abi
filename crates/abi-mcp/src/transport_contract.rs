//! Shared JSON-RPC boundary contracts for stdio and loopback HTTP.

use std::io::{Cursor, Read, Write};
use std::net::{Ipv4Addr, Shutdown, TcpStream};
use std::thread;

use serde_json::{Value, json};

use crate::http::{HttpConfig, McpHttpServer};
use crate::protocol::{MAX_JSON_DEPTH, MAX_REQUEST_SIZE};
use crate::{McpState, stdio};

fn padded_ping(id: u64, len: usize) -> String {
    let base = format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"ping"}}"#);
    assert!(base.len() <= len);
    let padding = " ".repeat(len - base.len());
    base + &padding
}

fn stdio_exchange(input: &[u8]) -> Vec<Value> {
    let mut output = Vec::new();
    stdio::run_loop(McpState::new(), Cursor::new(input), &mut output);
    String::from_utf8(output)
        .expect("stdio response is UTF-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("stdio response is JSON"))
        .collect()
}

fn http_request(body: &str) -> Vec<u8> {
    format!(
        "POST /message HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

fn http_request_with_total_len(total_len: usize, id: u64) -> Vec<u8> {
    let base = format!(r#"{{"jsonrpc":"2.0","id":{id},"method":"ping"}}"#);
    let mut body_len = total_len.saturating_sub(http_request(&base).len() - base.len());
    loop {
        let request = http_request(&padded_ping(id, body_len));
        match request.len().cmp(&total_len) {
            std::cmp::Ordering::Equal => return request,
            std::cmp::Ordering::Less => body_len += total_len - request.len(),
            std::cmp::Ordering::Greater => body_len -= request.len() - total_len,
        }
    }
}

fn with_http_server(request_count: usize, test: impl FnOnce(u16)) {
    let server = McpHttpServer::bind(
        HttpConfig {
            port: 0,
            bearer_token: None,
        },
        McpState::new(),
    )
    .expect("bind ephemeral HTTP server");
    let port = server.local_port().expect("ephemeral port");
    let handle = thread::spawn(move || {
        for _ in 0..request_count {
            server.serve_one().expect("serve HTTP request");
        }
    });
    test(port);
    handle.join().expect("HTTP server joins");
}

fn send_http(port: u16, request: &[u8]) -> Vec<u8> {
    let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port)).expect("connect HTTP");
    stream.write_all(request).expect("write HTTP request");
    stream
        .shutdown(Shutdown::Write)
        .expect("finish HTTP request");
    let mut response = Vec::new();
    let mut buffer = [0_u8; 4096];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => response.extend_from_slice(&buffer[..count]),
            Err(_) if !response.is_empty() => break,
            Err(error) => panic!("read HTTP response: {error}"),
        }
    }
    response
}

fn http_exchange(body: &str) -> Vec<u8> {
    let mut response = Vec::new();
    with_http_server(1, |port| response = send_http(port, &http_request(body)));
    response
}

fn http_status(response: &[u8]) -> &str {
    let end = response
        .windows(2)
        .position(|window| window == b"\r\n")
        .expect("HTTP status line");
    std::str::from_utf8(&response[..end]).expect("HTTP status is UTF-8")
}

fn http_body(response: &[u8]) -> &[u8] {
    let start = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("HTTP header terminator")
        + 4;
    &response[start..]
}

fn shared_response(body: &str) -> (Value, Value) {
    let mut stdio_frame = body.as_bytes().to_vec();
    stdio_frame.push(b'\n');
    let stdio = stdio_exchange(&stdio_frame);
    assert_eq!(stdio.len(), 1);

    let http = http_exchange(body);
    assert_eq!(http_status(&http), "HTTP/1.1 200 OK");
    let http_json = serde_json::from_slice(http_body(&http)).expect("HTTP body is JSON");
    (stdio.into_iter().next().expect("stdio response"), http_json)
}

fn nested_ping(depth: usize) -> String {
    assert!(depth >= 1);
    let arrays = depth - 1;
    format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"ping","params":{}0{}}}"#,
        "[".repeat(arrays),
        "]".repeat(arrays)
    )
}

#[test]
fn maximum_valid_frames_dispatch_on_both_transports() {
    let stdio_body = padded_ping(1, MAX_REQUEST_SIZE);
    let mut stdio_frame = stdio_body.into_bytes();
    stdio_frame.push(b'\n');
    assert_eq!(
        stdio_exchange(&stdio_frame),
        [json!({"jsonrpc":"2.0","id":1,"result":{}})]
    );

    let request = http_request_with_total_len(MAX_REQUEST_SIZE, 1);
    assert_eq!(request.len(), MAX_REQUEST_SIZE);
    with_http_server(1, |port| {
        let response = send_http(port, &request);
        assert_eq!(http_status(&response), "HTTP/1.1 200 OK");
        assert_eq!(
            serde_json::from_slice::<Value>(http_body(&response)).expect("HTTP JSON"),
            json!({"jsonrpc":"2.0","id":1,"result":{}})
        );
    });
}

#[test]
fn depth_ids_and_unknown_methods_have_shared_semantics() {
    let at_limit = nested_ping(MAX_JSON_DEPTH);
    let (stdio, http) = shared_response(&at_limit);
    assert_eq!(stdio, http);
    assert_eq!(stdio["result"], json!({}));

    let over_limit = nested_ping(MAX_JSON_DEPTH + 1);
    let (stdio, http) = shared_response(&over_limit);
    assert_eq!(stdio, http);
    assert_eq!(stdio["error"]["code"], json!(-32700));

    for request in [
        r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#,
        r#"{"jsonrpc":"2.0","id":1.5,"method":"ping"}"#,
    ] {
        let (stdio, http) = shared_response(request);
        assert_eq!(stdio, http);
        assert_eq!(stdio["id"], Value::Null);
        assert_eq!(stdio["error"]["code"], json!(-32600));
    }

    let (stdio, http) = shared_response(r#"{"jsonrpc":"2.0","id":9,"method":"unknown"}"#);
    assert_eq!(stdio, http);
    assert_eq!(stdio["error"]["code"], json!(-32601));
}

#[test]
fn notifications_use_each_transports_exact_no_response_contract() {
    let body = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
    let mut frame = body.as_bytes().to_vec();
    frame.push(b'\n');
    assert_eq!(stdio_exchange(&frame), [] as [Value; 0]);

    let response = http_exchange(body);
    assert_eq!(http_status(&response), "HTTP/1.1 202 Accepted");
    assert_eq!(http_body(&response), b"");
}

#[test]
fn oversized_frames_use_transport_specific_errors_and_both_recover() {
    let oversized = padded_ping(1, MAX_REQUEST_SIZE + 1);
    let recovery = r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#;
    let stdio_input = format!("{oversized}\n{recovery}\n");
    let responses = stdio_exchange(stdio_input.as_bytes());
    assert_eq!(responses.len(), 2);
    assert_eq!(responses[0]["error"]["code"], json!(-32700));
    assert_eq!(responses[1]["id"], json!(2));
    assert_eq!(responses[1]["result"], json!({}));

    let oversized_request = http_request_with_total_len(MAX_REQUEST_SIZE + 1, 1);
    with_http_server(2, |port| {
        let rejected = send_http(port, &oversized_request);
        assert_eq!(http_status(&rejected), "HTTP/1.1 413 Payload Too Large");
        let recovered = send_http(port, &http_request(recovery));
        assert_eq!(http_status(&recovered), "HTTP/1.1 200 OK");
        let value: Value = serde_json::from_slice(http_body(&recovered)).expect("recovery JSON");
        assert_eq!(value["id"], json!(2));
        assert_eq!(value["result"], json!({}));
    });
}
