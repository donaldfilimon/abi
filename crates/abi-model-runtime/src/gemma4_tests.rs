//! Prompt-rendering and output-grammar tests that need no decoder.

use super::{OutputFilter, Tokens, ToolTokens, render_chat};
use crate::gemma4_format::spell;
use crate::provider::Step;
use abi_agent_runtime::{Message, ModelRequest, Role, ToolSpec};
use serde_json::json;

const EOS: u32 = 1;
const TURN_END: u32 = 3;
const CALL_START: u32 = 6;
const CALL_END: u32 = 7;
const RESPONSE_START: u32 = 8;
const QUOTE: u32 = 10;
const CHANNEL_OPEN: u32 = 11;
const CHANNEL_CLOSE: u32 = 12;

fn tokens() -> Tokens {
    Tokens {
        bos: 0,
        eos: EOS,
        turn_start: 2,
        turn_end: TURN_END,
        tools: Some(ToolTokens {
            tool_start: 4,
            tool_end: 5,
            call_start: CALL_START,
            call_end: CALL_END,
            response_start: RESPONSE_START,
            response_end: 9,
            quote: QUOTE,
        }),
        channel: Some((CHANNEL_OPEN, CHANNEL_CLOSE)),
    }
}

fn filter(offered: &[&str]) -> OutputFilter {
    OutputFilter::new(
        tokens(),
        offered.iter().map(|name| (*name).to_owned()).collect(),
    )
}

/// Feed tokens; a call body always decodes to `body`.
fn run(filter: &mut OutputFilter, stream: &[u32], body: &str) -> Result<Vec<Step>, String> {
    stream
        .iter()
        .map(|token| filter.accept(*token, |_| Ok(body.to_owned())))
        .collect()
}

fn rendered(request: &ModelRequest) -> String {
    spell(&render_chat("m", request).expect("renders"))
}

fn tool_message(value: &serde_json::Value) -> Message {
    Message::new(Role::Tool, value.to_string())
}

#[test]
fn plain_text_passes_and_the_turn_end_stops() {
    let mut filter = filter(&[]);
    assert_eq!(
        run(&mut filter, &[100, 101, TURN_END], ""),
        Ok(vec![Step::Text(100), Step::Text(101), Step::Stop])
    );
    assert_eq!(run(&mut filter, &[EOS], ""), Ok(vec![Step::Stop]));
}

#[test]
fn a_thought_channel_is_consumed_but_never_shown() {
    let mut filter = filter(&[]);
    assert_eq!(
        run(
            &mut filter,
            &[CHANNEL_OPEN, 100, 101, CHANNEL_CLOSE, 102, TURN_END],
            ""
        ),
        Ok(vec![
            Step::Hidden(CHANNEL_OPEN),
            Step::Hidden(100),
            Step::Hidden(101),
            Step::Hidden(CHANNEL_CLOSE),
            Step::Text(102),
            Step::Stop,
        ])
    );
}

#[test]
fn a_closed_call_becomes_a_tool_call_and_the_response_token_stops() {
    let mut filter = filter(&["recall", "remember_fact"]);
    let steps = run(
        &mut filter,
        &[CALL_START, 100, QUOTE, 101, QUOTE, CALL_END, RESPONSE_START],
        "call:remember_fact{fact:<|\"|>likes tea<|\"|>}",
    )
    .expect("valid call");

    assert_eq!(steps.len(), 7);
    assert!(
        steps[..5]
            .iter()
            .all(|step| matches!(step, Step::Hidden(_)))
    );
    let Step::Call(token, call) = &steps[5] else {
        panic!("expected a call, got {:?}", steps[5]);
    };
    assert_eq!(*token, CALL_END);
    assert_eq!(call.id, "call_1");
    assert_eq!(call.name, "remember_fact");
    assert_eq!(call.input, r#"{"fact":"likes tea"}"#);
    assert_eq!(steps[6], Step::Stop);
}

#[test]
fn consecutive_calls_get_distinct_ids() {
    let mut filter = filter(&["recall"]);
    let steps = run(
        &mut filter,
        &[CALL_START, 100, CALL_END, CALL_START, 100, CALL_END],
        "call:recall{}",
    )
    .expect("two calls");
    let ids: Vec<_> = steps
        .iter()
        .filter_map(|step| match step {
            Step::Call(_, call) => Some(call.id.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(ids, ["call_1", "call_2"]);
}

#[test]
fn unusable_calls_fail_closed() {
    let unoffered = run(
        &mut filter(&["recall"]),
        &[CALL_START, 100, CALL_END],
        "call:purge{}",
    );
    assert!(
        unoffered
            .expect_err("unoffered tool")
            .contains("not offered")
    );

    // With no tools offered, a stray call token is hidden, not an error.
    assert_eq!(
        run(&mut filter(&[]), &[CALL_START, 100], ""),
        Ok(vec![Step::Hidden(CALL_START), Step::Text(100)])
    );

    let unclosed = run(&mut filter(&["recall"]), &[CALL_START, 100, TURN_END], "");
    assert!(unclosed.expect_err("unclosed call").contains("not closed"));

    let malformed = run(
        &mut filter(&["recall"]),
        &[CALL_START, 100, CALL_END],
        "recall{}",
    );
    assert!(malformed.expect_err("malformed call").contains("call:"));
}

#[test]
fn a_tokenizer_without_tool_tokens_treats_those_ids_as_text() {
    let mut plain = Tokens {
        tools: None,
        channel: None,
        ..tokens()
    };
    plain.eos = EOS;
    let mut filter = OutputFilter::new(plain, Vec::new());
    assert_eq!(
        run(&mut filter, &[CALL_START, CHANNEL_OPEN, RESPONSE_START], ""),
        Ok(vec![
            Step::Text(CALL_START),
            Step::Text(CHANNEL_OPEN),
            Step::Text(RESPONSE_START),
        ])
    );
}

#[test]
fn a_plain_conversation_renders_turns_and_an_open_model_turn() {
    let request = ModelRequest::new("m")
        .with_system("Be terse.")
        .with_user("  Hello.  ")
        .with_message(Message::assistant("Hi."))
        .with_user("Again?");
    assert_eq!(
        rendered(&request),
        concat!(
            "<bos><|turn>system\nBe terse.<turn|>\n",
            "<|turn>user\nHello.<turn|>\n",
            "<|turn>model\nHi.<turn|>\n",
            "<|turn>user\nAgain?<turn|>\n",
            "<|turn>model\n"
        )
    );
    assert_eq!(
        rendered(&ModelRequest::new("m").with_user("Hello.")),
        "<bos><|turn>user\nHello.<turn|>\n<|turn>model\n"
    );
}

#[test]
fn tools_are_declared_in_a_system_turn_even_without_system_text() {
    let request = ModelRequest::new("m")
        .with_user("What do you know?")
        .with_tool(
            ToolSpec::new("recall")
                .with_description("Recall facts.")
                .with_input_schema(
                    json!({"type":"object","properties":{"query":{"type":"string"}}}).to_string(),
                ),
        )
        .with_tool(ToolSpec::new("list_facts"));
    assert_eq!(
        rendered(&request),
        concat!(
            "<bos><|turn>system\n",
            "<|tool>declaration:recall{description:<|\"|>Recall facts.<|\"|>,",
            "parameters:{properties:{query:{type:<|\"|>STRING<|\"|>}},",
            "type:<|\"|>OBJECT<|\"|>}}<tool|>",
            "<|tool>declaration:list_facts{description:<|\"|><|\"|>}<tool|><turn|>\n",
            "<|turn>user\nWhat do you know?<turn|>\n",
            "<|turn>model\n"
        )
    );
}

#[test]
fn trailing_tool_results_leave_the_model_turn_open() {
    let request = ModelRequest::new("m")
        .with_user("Weather?")
        .with_message(tool_message(&json!({
            "call_id": "call_1",
            "tool": "get_current_weather",
            "status": "ok",
            "input": r#"{"location":"London"}"#,
            "payload": r#"{"temperature":15,"weather":"sunny"}"#
        })))
        .with_message(tool_message(&json!({
            "call_id": "call_2", "tool": "recall", "status": "ok", "payload": "three facts"
        })));
    assert_eq!(
        rendered(&request),
        concat!(
            "<bos><|turn>user\nWeather?<turn|>\n<|turn>model\n",
            "<|tool_call>call:get_current_weather{location:<|\"|>London<|\"|>}<tool_call|>",
            "<|tool_response>response:get_current_weather{temperature:15,",
            "weather:<|\"|>sunny<|\"|>}<tool_response|>",
            "<|tool_call>call:recall{}<tool_call|>",
            "<|tool_response>response:recall{value:<|\"|>three facts<|\"|>}<tool_response|>"
        )
    );
}

#[test]
fn an_answer_after_tool_results_closes_the_same_model_turn() {
    let failed = tool_message(&json!({
        "call_id": "c", "tool": "recall", "status": "error", "payload": "store offline"
    }));
    let request = ModelRequest::new("m")
        .with_user("Recall.")
        .with_message(failed.clone())
        .with_message(Message::assistant("I could not."))
        .with_user("Try again.");
    let text = rendered(&request);
    assert!(text.contains(concat!(
        "<|turn>model\n<|tool_call>call:recall{}<tool_call|>",
        "<|tool_response>response:recall{error:<|\"|>store offline<|\"|>}<tool_response|>",
        "I could not.<turn|>\n<|turn>user\nTry again.<turn|>\n<|turn>model\n"
    )));

    // A user turn directly after tool results closes the open model turn first.
    let direct = rendered(
        &ModelRequest::new("m")
            .with_user("Recall.")
            .with_message(failed)
            .with_user("Never mind."),
    );
    assert!(
        direct
            .ends_with("<tool_response|><turn|>\n<|turn>user\nNever mind.<turn|>\n<|turn>model\n")
    );
}

#[test]
fn unrenderable_tools_and_results_are_refused() {
    let bad_name = ModelRequest::new("m")
        .with_user("x")
        .with_tool(ToolSpec::new("bad{name"));
    assert!(
        render_chat("m", &bad_name)
            .expect_err("name")
            .to_string()
            .contains("grammar-safe")
    );

    let bad_schema = ModelRequest::new("m")
        .with_user("x")
        .with_tool(ToolSpec::new("t").with_input_schema("{not json"));
    assert!(
        render_chat("m", &bad_schema)
            .expect_err("schema")
            .to_string()
            .contains("not JSON")
    );

    for content in [
        "plain text",
        r#"{"payload":"no tool"}"#,
        r#"{"tool":"a b","payload":""}"#,
    ] {
        let request = ModelRequest::new("m")
            .with_user("x")
            .with_message(Message::new(Role::Tool, content));
        assert!(render_chat("m", &request).is_err(), "{content}");
    }
}
