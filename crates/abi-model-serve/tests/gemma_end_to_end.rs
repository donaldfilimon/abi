//! The real stack end to end: a verified scratch Gemma 3 decoder loaded by
//! `abi-model-runtime`, served on loopback, and read back through both
//! response modes. The weights are random, so only shape, accounting, and
//! agreement between the two modes are asserted.

#[path = "../../abi-model-runtime/tests/support/gemma.rs"]
mod scratch;

use abi_model_serve::{ServeConfig, Server};
use scratch::{MODEL_ID, OUTPUT_TOKENS, fixture};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn post(address: std::net::SocketAddr, stream: bool) -> String {
    let body = format!(
        r#"{{"model":"{MODEL_ID}","stream":{stream},"messages":[{{"role":"system","content":"be terse"}},{{"role":"user","content":"hello world"}}]}}"#
    );
    let mut connection = TcpStream::connect(address).expect("connect");
    write!(
        connection,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .expect("send request");
    let mut response = String::new();
    connection
        .read_to_string(&mut response)
        .expect("read response");
    assert!(response.starts_with("HTTP/1.1 200 OK\r\n"), "{response}");
    response
        .split_once("\r\n\r\n")
        .expect("response head")
        .1
        .to_owned()
}

#[test]
fn a_loaded_gemma_decoder_answers_buffered_and_streamed_requests_identically() {
    let fixture = fixture();
    let provider = fixture.load().expect("scratch Gemma loads");
    let server =
        Server::bind(0, Arc::new(provider), ServeConfig::new(MODEL_ID)).expect("bind loopback");
    let address = server.local_addr().expect("bound address");
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let thread = std::thread::spawn(move || server.serve(&thread_stop).expect("serve loop"));

    let buffered: Value = serde_json::from_str(&post(address, false)).expect("completion JSON");
    let streamed_body = post(address, true);

    stop.store(true, Ordering::SeqCst);
    thread.join().expect("server joins");

    let text = buffered["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant text");
    let generated = buffered["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion tokens");
    // <bos>, one user turn carrying the folded system text, the open model turn.
    assert_eq!(buffered["usage"]["prompt_tokens"], 10);
    assert!(generated <= u64::from(OUTPUT_TOKENS));
    assert_eq!(buffered["choices"][0]["finish_reason"], "stop");

    let events: Vec<&str> = streamed_body
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    let streamed_text: String = events[..events.len() - 1]
        .iter()
        .map(|event| serde_json::from_str::<Value>(event).expect("chunk JSON"))
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect();
    // Greedy selection over a reset cache: both modes must agree exactly.
    assert_eq!(streamed_text, text);
}
