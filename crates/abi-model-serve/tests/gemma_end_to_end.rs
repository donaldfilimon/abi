//! The real stack end to end: a verified scratch Gemma 3 decoder loaded by
//! `abi-model-runtime`, served on loopback, and read back through both
//! response modes. The weights are random, so only shape, accounting, and
//! agreement between the two modes are asserted.

#[path = "../../abi-model-runtime/tests/support/gemma.rs"]
mod scratch;
#[path = "../../abi-model-runtime/tests/support/gemma4.rs"]
mod scratch4;

use abi_model_serve::{ServeConfig, Server};
use scratch::{MODEL_ID, OUTPUT_TOKENS, fixture};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn post(address: std::net::SocketAddr, model: &str, system: &str, stream: bool) -> String {
    let body = format!(
        r#"{{"model":"{model}","stream":{stream},"messages":[{{"role":"system","content":"{system}"}},{{"role":"user","content":"hello world"}}]}}"#
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

/// Serve `provider`, then read one buffered and one streamed completion.
fn both_modes(
    provider: abi_model_runtime::LocalModelProvider,
    model: &str,
    system: &str,
) -> (Value, String) {
    let tools = provider.supports_tools();
    let config = ServeConfig::new(model).with_tools(tools);
    let server = Server::bind(0, Arc::new(provider), config).expect("bind loopback");
    let address = server.local_addr().expect("bound address");
    let stop = Arc::new(AtomicBool::new(false));
    let thread_stop = Arc::clone(&stop);
    let thread = std::thread::spawn(move || server.serve(&thread_stop).expect("serve loop"));

    let buffered = serde_json::from_str(&post(address, model, system, false)).expect("JSON");
    let streamed = post(address, model, system, true);

    stop.store(true, Ordering::SeqCst);
    thread.join().expect("server joins");
    (buffered, streamed)
}

/// Concatenate the content deltas of a `[DONE]`-terminated stream.
fn streamed_text(body: &str) -> String {
    let events: Vec<&str> = body
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    events[..events.len() - 1]
        .iter()
        .map(|event| serde_json::from_str::<Value>(event).expect("chunk JSON"))
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_owned)
        })
        .collect()
}

#[test]
fn a_loaded_gemma_decoder_answers_buffered_and_streamed_requests_identically() {
    let provider = fixture().load().expect("scratch Gemma loads");
    let (buffered, streamed) = both_modes(provider, MODEL_ID, "be terse");

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
    // Greedy selection over a reset cache: both modes must agree exactly.
    assert_eq!(streamed_text(&streamed), text);
}

#[test]
fn a_loaded_gemma4_decoder_answers_buffered_and_streamed_requests_identically() {
    let fixture = scratch4::fixture();
    let provider = fixture.load().expect("scratch Gemma 4 loads");
    assert!(provider.supports_tools());
    assert_eq!(
        provider.load_report().tensor_count,
        fixture.language_model.0
    );
    let (buffered, streamed) = both_modes(provider, scratch4::MODEL_ID, "terse");

    let text = buffered["choices"][0]["message"]["content"]
        .as_str()
        .expect("assistant text");
    // <bos>, a real system turn, the user turn, the open model turn.
    assert_eq!(buffered["usage"]["prompt_tokens"], 12);
    assert_eq!(buffered["choices"][0]["finish_reason"], "stop");
    assert_eq!(streamed_text(&streamed), text);
}
