//! End-to-end loopback tests over deterministic runtime test providers.

use abi_agent_runtime::{
    EchoProvider, ModelProvider, RunBudget, RuntimeError, ScriptedProvider, ToolCall,
};
use abi_model_serve::{ServeConfig, Server};
use serde_json::Value;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

const MODEL: &str = "scripted-model";

struct Running {
    address: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Running {
    fn start(provider: Arc<dyn ModelProvider>, config: ServeConfig) -> Self {
        let server = Server::bind(0, provider, config).expect("bind loopback");
        let address = server.local_addr().expect("bound address");
        assert!(address.ip().is_loopback());
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = std::thread::spawn(move || server.serve(&thread_stop).expect("serve loop"));
        Self {
            address,
            stop,
            thread: Some(thread),
        }
    }

    /// Send raw bytes and return `(status, body)`.
    fn raw(&self, request: &[u8]) -> (u16, String) {
        let mut stream = TcpStream::connect(self.address).expect("connect");
        stream.write_all(request).expect("send request");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read response");
        let (head, body) = response.split_once("\r\n\r\n").expect("response head");
        let status = head
            .split(' ')
            .nth(1)
            .and_then(|code| code.parse().ok())
            .expect("status code");
        (status, body.to_owned())
    }

    fn get(&self, path: &str) -> (u16, String) {
        self.raw(format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n").as_bytes())
    }

    fn post(&self, body: &str, token: Option<&str>) -> (u16, String) {
        let authorization = token.map_or_else(String::new, |token| {
            format!("Authorization: Bearer {token}\r\n")
        });
        self.raw(
            format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n{authorization}Content-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn scripted() -> Arc<dyn ModelProvider> {
    Arc::new(
        ScriptedProvider::new("scripted")
            .with_text("hello ")
            .with_text("world")
            .with_input_tokens(5)
            .with_output_tokens_per_event(1),
    )
}

fn chat(stream: bool) -> String {
    format!(
        r#"{{"model":"{MODEL}","stream":{stream},"messages":[{{"role":"user","content":"hi"}}]}}"#
    )
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).expect("JSON body")
}

/// Parse an SSE body into its `data:` payloads.
fn events(body: &str) -> Vec<&str> {
    body.split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .collect()
}

#[test]
fn health_and_model_listing_name_the_one_loaded_model() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL));

    let (status, body) = server.get("/health");
    assert_eq!(status, 200);
    assert_eq!(json(&body)["model"], MODEL);

    let (status, body) = server.get("/v1/models");
    assert_eq!(status, 200);
    assert_eq!(json(&body)["data"][0]["id"], MODEL);
    assert_eq!(json(&body)["data"].as_array().map(Vec::len), Some(1));
}

#[test]
fn buffered_completion_returns_text_finish_reason_and_usage() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL));

    let (status, body) = server.post(&chat(false), None);
    let document = json(&body);

    assert_eq!(status, 200);
    assert_eq!(document["object"], "chat.completion");
    assert_eq!(document["model"], MODEL);
    assert_eq!(document["choices"][0]["message"]["role"], "assistant");
    assert_eq!(document["choices"][0]["message"]["content"], "hello world");
    assert_eq!(document["choices"][0]["finish_reason"], "stop");
    assert_eq!(document["usage"]["prompt_tokens"], 5);
    assert_eq!(document["usage"]["completion_tokens"], 2);
    assert_eq!(document["usage"]["total_tokens"], 7);
}

#[test]
fn streamed_completion_is_role_then_deltas_then_finish_then_done() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL));

    let (status, body) = server.post(&chat(true), None);
    let events = events(&body);

    assert_eq!(status, 200);
    assert_eq!(events.len(), 5);
    assert_eq!(json(events[0])["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(json(events[1])["choices"][0]["delta"]["content"], "hello ");
    assert_eq!(json(events[2])["choices"][0]["delta"]["content"], "world");
    assert!(json(events[2])["choices"][0]["finish_reason"].is_null());
    let last = json(events[3]);
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["usage"]["completion_tokens"], 2);
    assert_eq!(events[4], "[DONE]");
}

#[test]
fn an_output_budget_reports_length() {
    let config =
        ServeConfig::new(MODEL).with_budget(RunBudget::unlimited().with_max_output_tokens(1));
    let server = Running::start(scripted(), config);

    let (status, body) = server.post(&chat(false), None);

    assert_eq!(status, 200);
    assert_eq!(json(&body)["choices"][0]["finish_reason"], "length");
}

#[test]
fn the_echo_provider_sees_the_parsed_conversation() {
    let server = Running::start(Arc::new(EchoProvider::new()), ServeConfig::new(MODEL));

    let (status, body) = server.post(
        &format!(
            r#"{{"model":"{MODEL}","messages":[{{"role":"system","content":"s"}},{{"role":"user","content":"repeat these words"}}]}}"#
        ),
        None,
    );

    assert_eq!(status, 200);
    assert_eq!(
        json(&body)["choices"][0]["message"]["content"],
        "repeat these words"
    );
}

#[test]
fn another_model_name_is_never_served() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL));

    let (status, body) = server.post(
        r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#,
        None,
    );

    assert_eq!(status, 404);
    assert_eq!(json(&body)["error"]["code"], "model_not_found");
}

#[test]
fn tools_are_refused_with_a_stable_code() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL));

    let (status, body) = server.post(
        &format!(
            r#"{{"model":"{MODEL}","tools":[{{"type":"function","function":{{"name":"recall"}}}}],"messages":[{{"role":"user","content":"hi"}}]}}"#
        ),
        None,
    );

    assert_eq!(status, 400);
    assert_eq!(json(&body)["error"]["code"], "tools_unsupported");
    assert_eq!(json(&body)["error"]["type"], "invalid_request_error");
}

fn calling() -> Arc<dyn ModelProvider> {
    Arc::new(
        ScriptedProvider::new("scripted")
            .with_tool_call(ToolCall::new("call_1", "recall", r#"{"query":"tea"}"#))
            .with_tool_call(ToolCall::new("call_2", "list_facts", "{}")),
    )
}

fn tooled(stream: bool) -> String {
    format!(
        r#"{{"model":"{MODEL}","stream":{stream},"tools":[{{"type":"function","function":{{"name":"recall"}}}},{{"type":"function","function":{{"name":"list_facts"}}}}],"messages":[{{"role":"user","content":"hi"}}]}}"#
    )
}

#[test]
fn generated_tool_calls_are_returned_buffered_with_their_finish_reason() {
    let server = Running::start(calling(), ServeConfig::new(MODEL).with_tools(true));

    let (status, body) = server.post(&tooled(false), None);
    let document = json(&body);
    let message = &document["choices"][0]["message"];

    assert_eq!(status, 200);
    assert!(message["content"].is_null());
    assert_eq!(message["tool_calls"].as_array().map(Vec::len), Some(2));
    assert_eq!(message["tool_calls"][0]["id"], "call_1");
    assert_eq!(message["tool_calls"][0]["type"], "function");
    assert_eq!(message["tool_calls"][0]["function"]["name"], "recall");
    assert_eq!(
        message["tool_calls"][0]["function"]["arguments"],
        r#"{"query":"tea"}"#
    );
    assert!(message["tool_calls"][0].get("index").is_none());
    assert_eq!(document["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn generated_tool_calls_are_streamed_as_indexed_deltas() {
    let server = Running::start(calling(), ServeConfig::new(MODEL).with_tools(true));

    let (status, body) = server.post(&tooled(true), None);
    let events = events(&body);

    assert_eq!(status, 200);
    assert_eq!(events.len(), 5);
    let first = &json(events[1])["choices"][0]["delta"]["tool_calls"][0];
    let second = &json(events[2])["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(first["index"], 0);
    assert_eq!(first["function"]["name"], "recall");
    assert_eq!(second["index"], 1);
    assert_eq!(second["id"], "call_2");
    assert_eq!(json(events[3])["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(events[4], "[DONE]");
}

#[test]
fn a_tool_round_trip_is_accepted_when_tools_are_enabled() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL).with_tools(true));

    let (status, body) = server.post(
        &format!(
            r#"{{"model":"{MODEL}","tools":[{{"type":"function","function":{{"name":"recall"}}}}],"messages":[{{"role":"user","content":"hi"}},{{"role":"assistant","content":null,"tool_calls":[{{"id":"c1","type":"function","function":{{"name":"recall","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"c1","content":"likes tea"}}]}}"#
        ),
        None,
    );

    assert_eq!(status, 200);
    assert_eq!(json(&body)["choices"][0]["finish_reason"], "stop");
}

#[test]
fn a_configured_bearer_token_guards_v1_but_not_health() {
    let server = Running::start(
        scripted(),
        ServeConfig::new(MODEL).with_bearer_token("secret"),
    );

    assert_eq!(server.get("/health").0, 200);
    assert_eq!(server.get("/v1/models").0, 401);
    let (status, body) = server.post(&chat(false), Some("wrong"));
    assert_eq!(status, 401);
    assert_eq!(json(&body)["error"]["code"], "invalid_api_key");
    assert_eq!(server.post(&chat(false), Some("secret")).0, 200);
}

#[test]
fn provider_failures_are_reported_buffered_and_in_band_when_streaming() {
    let failing: Arc<dyn ModelProvider> = Arc::new(ScriptedProvider::new("scripted").with_failure(
        RuntimeError::Provider {
            provider: "scripted".to_owned(),
            message: "weights unavailable".to_owned(),
        },
    ));
    let server = Running::start(failing, ServeConfig::new(MODEL));

    let (status, body) = server.post(&chat(false), None);
    assert_eq!(status, 500);
    assert_eq!(json(&body)["error"]["code"], "provider_error");
    assert!(body.contains("weights unavailable"));

    let (status, body) = server.post(&chat(true), None);
    let events = events(&body);
    assert_eq!(status, 200);
    assert_eq!(
        json(events[events.len() - 1])["error"]["code"],
        "provider_error"
    );
    assert!(!body.contains("[DONE]"));
}

#[test]
fn routing_and_framing_refusals_use_typed_statuses() {
    let server = Running::start(scripted(), ServeConfig::new(MODEL).with_max_body_bytes(64));

    assert_eq!(server.get("/nope").0, 404);
    assert_eq!(server.get("/v1/chat/completions").0, 405);
    assert_eq!(server.raw(b"garbage\r\n\r\n").0, 400);
    assert_eq!(
        server
            .raw(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n")
            .0,
        411
    );
    let (status, _) = server.post(&"x".repeat(65), None);
    assert_eq!(status, 413);
    let (status, body) = server.post("{", None);
    assert_eq!(status, 400);
    assert_eq!(json(&body)["error"]["code"], "invalid_json");
    // The listener survives every refusal above.
    assert_eq!(server.get("/health").0, 200);
}
