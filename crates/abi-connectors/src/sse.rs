//! Server-Sent Events parsing for streaming completions.
//!
//! Ported from `src/connectors/sse_stream.zig`. Handles both provider dialects:
//!
//! - **`OpenAI`**: `data: {"choices":[{"delta":{"content":"tok"},"finish_reason":null}]}`,
//!   terminated by `data: [DONE]`.
//! - **Anthropic**: `event: content_block_delta` / `data: {"delta":{"text":"tok"}}`,
//!   terminated by `event: message_stop`.
//!
//! The Zig version passed a `*const fn (ctx: *anyopaque, ...)` plus an untyped
//! context pointer — the usual closure workaround. Here the sink is a `FnMut`, so
//! captured state needs no manual threading.

use crate::connector::{ConnectorError, Result};

/// One token delta from a stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamChunk {
    /// The token text. Empty when `done` is set.
    pub delta: String,
    /// Whether this marks the end of the stream.
    pub done: bool,
}

impl StreamChunk {
    /// A content delta.
    #[must_use]
    pub fn delta(text: impl Into<String>) -> Self {
        Self {
            delta: text.into(),
            done: false,
        }
    }

    /// The terminal marker.
    #[must_use]
    pub fn done() -> Self {
        Self {
            delta: String::new(),
            done: true,
        }
    }
}

/// Parse a complete SSE body, invoking `on_chunk` per delta.
///
/// Returns the concatenated token text. A `done` chunk is always delivered last,
/// even if the body carried no explicit terminator — callers rely on it to know
/// the stream finished rather than stalled.
///
/// `on_chunk` returning an error aborts parsing and propagates.
pub fn parse_stream<F>(body: &str, mut on_chunk: F) -> Result<String>
where
    F: FnMut(StreamChunk) -> Result<()>,
{
    let mut accumulated = String::new();
    let mut data_buffer = String::new();

    for raw_line in body.split('\n') {
        // Tolerate CRLF framing.
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);

        if line.is_empty() {
            if !data_buffer.is_empty() {
                process_event(&data_buffer, &mut on_chunk, &mut accumulated)?;
                data_buffer.clear();
            }
            continue;
        }

        if let Some(rest) = line.strip_prefix("data:") {
            let data = rest.trim_matches([' ', '\t']);
            if data == "[DONE]" {
                on_chunk(StreamChunk::done())?;
                continue;
            }
            data_buffer.push_str(data);
            data_buffer.push('\n');
        }
        // `event:` and comment lines carry no payload; the JSON shape alone
        // distinguishes the two dialects, exactly as in Zig.
    }

    if !data_buffer.is_empty() {
        process_event(&data_buffer, &mut on_chunk, &mut accumulated)?;
    }

    on_chunk(StreamChunk::done())?;
    Ok(accumulated)
}

/// Interpret one accumulated `data:` payload.
///
/// Presence of `choices` selects the `OpenAI` shape; otherwise a top-level `delta`
/// selects Anthropic's. This is the discriminator the Zig version used.
fn process_event<F>(event_data: &str, on_chunk: &mut F, accumulated: &mut String) -> Result<()>
where
    F: FnMut(StreamChunk) -> Result<()>,
{
    let trimmed = event_data.trim_matches([' ', '\t', '\n', '\r']);
    if trimmed.is_empty() {
        return Ok(());
    }

    let value: serde_json::Value =
        serde_json::from_str(trimmed).map_err(|_| ConnectorError::InvalidResponse)?;
    let Some(root) = value.as_object() else {
        return Err(ConnectorError::InvalidResponse);
    };

    // Anthropic: no `choices`, a top-level `delta.text`.
    if !root.contains_key("choices") {
        if let Some(text) = root
            .get("delta")
            .and_then(|d| d.get("text"))
            .and_then(serde_json::Value::as_str)
            && !text.is_empty()
        {
            accumulated.push_str(text);
            on_chunk(StreamChunk::delta(text))?;
        }
        return Ok(());
    }

    // OpenAI: choices[0].delta.content, with finish_reason terminating.
    let Some(choice) = root
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return Ok(());
    };
    let Some(content) = choice
        .get("delta")
        .and_then(|d| d.get("content"))
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(());
    };

    // A present-but-null finish_reason means "still going"; any other value ends
    // the stream. Absent entirely also means still going.
    let finished = choice
        .get("finish_reason")
        .is_some_and(|reason| !reason.is_null());

    if !content.is_empty() {
        accumulated.push_str(content);
        on_chunk(StreamChunk::delta(content))?;
    }
    if finished {
        on_chunk(StreamChunk::done())?;
    }
    Ok(())
}

/// Parse an SSE body, collecting deltas without a caller-supplied sink.
pub fn collect_stream(body: &str) -> Result<Vec<StreamChunk>> {
    let mut chunks = Vec::new();
    parse_stream(body, |chunk| {
        chunks.push(chunk);
        Ok(())
    })?;
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deltas(body: &str) -> Vec<String> {
        collect_stream(body)
            .expect("parses")
            .into_iter()
            .filter(|c| !c.done)
            .map(|c| c.delta)
            .collect()
    }

    #[test]
    fn parses_an_openai_stream() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let text = parse_stream(body, |_| Ok(())).expect("parses");
        assert_eq!(text, "Hello world");
        assert_eq!(deltas(body), ["Hello", " world"]);
    }

    #[test]
    fn parses_an_anthropic_stream() {
        let body = concat!(
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"Hi\"}}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\" there\"}}\n\n",
            "event: message_stop\n\n"
        );
        let text = parse_stream(body, |_| Ok(())).expect("parses");
        assert_eq!(text, "Hi there");
    }

    #[test]
    fn always_delivers_a_terminal_done_chunk() {
        // Callers use this to distinguish "finished" from "stalled".
        for body in [
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
            "data: [DONE]\n\n",
            "",
        ] {
            let chunks = collect_stream(body).expect("parses");
            assert!(
                chunks.last().is_some_and(|c| c.done),
                "no terminal done chunk for {body:?}"
            );
        }
    }

    #[test]
    fn finish_reason_null_does_not_end_the_stream() {
        // A present-but-null finish_reason is the normal mid-stream state.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"},\"finish_reason\":null}]}\n\n"
        );
        let chunks = collect_stream(body).expect("parses");
        let done_count = chunks.iter().filter(|c| c.done).count();
        assert_eq!(
            done_count, 1,
            "only the synthetic terminator should be done"
        );
        assert_eq!(deltas(body), ["a", "b"]);
    }

    #[test]
    fn a_non_null_finish_reason_ends_the_stream() {
        let body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n";
        let chunks = collect_stream(body).expect("parses");
        assert_eq!(chunks.iter().filter(|c| c.done).count(), 2);
    }

    #[test]
    fn handles_crlf_framing() {
        let body =
            "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\r\n\r\ndata: [DONE]\r\n\r\n";
        assert_eq!(parse_stream(body, |_| Ok(())).expect("parses"), "x");
    }

    #[test]
    fn tolerates_extra_whitespace_after_the_field_name() {
        let body = "data:{\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata:   {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n";
        assert_eq!(parse_stream(body, |_| Ok(())).expect("parses"), "ab");
    }

    #[test]
    fn ignores_empty_content_deltas() {
        // Providers emit a role-only opening frame with empty content.
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"real\"}}]}\n\n"
        );
        assert_eq!(deltas(body), ["real"]);
    }

    #[test]
    fn ignores_frames_with_no_content_field() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"kept\"}}]}\n\n"
        );
        assert_eq!(deltas(body), ["kept"]);
    }

    #[test]
    fn a_malformed_data_frame_is_an_invalid_response() {
        let body = "data: {not json}\n\n";
        assert_eq!(
            parse_stream(body, |_| Ok(())).unwrap_err(),
            ConnectorError::InvalidResponse
        );
    }

    #[test]
    fn a_non_object_data_frame_is_an_invalid_response() {
        assert_eq!(
            parse_stream("data: [1,2,3]\n\n", |_| Ok(())).unwrap_err(),
            ConnectorError::InvalidResponse
        );
    }

    #[test]
    fn a_trailing_frame_without_a_blank_line_is_still_processed() {
        // Real streams get cut off; the last frame must not be dropped.
        let body = "data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}";
        assert_eq!(parse_stream(body, |_| Ok(())).expect("parses"), "tail");
    }

    #[test]
    fn a_sink_error_aborts_parsing() {
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n"
        );
        let mut seen = 0;
        let result = parse_stream(body, |_| {
            seen += 1;
            Err(ConnectorError::ReadFailed)
        });
        assert_eq!(result.unwrap_err(), ConnectorError::ReadFailed);
        assert_eq!(seen, 1, "parsing must stop at the first sink error");
    }

    #[test]
    fn round_trips_the_synthesized_local_streams() {
        // The local transport's output must be parseable by this parser.
        let openai = crate::payload::openai_local_stream("gpt-4", "[]", 30);
        let text = parse_stream(&openai, |_| Ok(())).expect("parses openai local stream");
        assert!(text.starts_with("OpenAI-compatible local stream"));

        let anthropic = crate::payload::anthropic_local_stream("m", "p", 16, 20);
        let text = parse_stream(&anthropic, |_| Ok(())).expect("parses anthropic local stream");
        assert!(text.starts_with("Anthropic-compatible local stream"));
    }
}
