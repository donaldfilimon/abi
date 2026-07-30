//! Iterative persona/template generation with per-step emission.
//!
//! Ported from `src/features/ai/incremental.zig`.
//!
//! The response is built by appending planned segments — prefix tokens, input
//! word tokens, suffix tokens — and the sink is invoked after **each** append.
//! Emission happens during generation, not by slicing a finished string
//! afterwards.
//!
//! ## Honest scope
//!
//! This is genuine incremental decode of the local persona/template model, at
//! word/token granularity. It is **not** a neural language model or a token
//! sampler. The network-incremental baselines are live Anthropic SSE and the
//! local bridge's OpenAI-compatible SSE, both in `abi-connectors`.

use crate::identity::{AgentProfile, profile_contract};

/// How a caller chunks a streamed response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// Word/token iterative emit during generation.
    Incremental,
    /// Reserved label for callers that still post-hoc-chunk a finished buffer.
    PostHoc,
}

impl StreamMode {
    /// The label Zig's `streamModeLabel` produced.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Incremental => "incremental",
            Self::PostHoc => "post-hoc",
        }
    }
}

/// One emitted step of a generated response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamChunk<'a> {
    /// The newly appended text. Empty only on the final `done` chunk.
    pub delta: &'a str,
    /// Whether generation has finished.
    pub done: bool,
}

/// Receives generation steps as they are produced.
///
/// Zig passed a `*const fn (ctx: *anyopaque, chunk: StreamChunk) anyerror!void`
/// plus an opaque context pointer. A closure captures its own context, so the
/// `*anyopaque` pairing — and Zig's `error.MissingStreamContext` for a callback
/// supplied without one — has no analogue here: the pair cannot be split.
pub type StreamSink<'a> = &'a mut dyn FnMut(StreamChunk<'_>);

/// Generate `profile`'s response to `input`, emitting each step to `sink`.
///
/// The returned string is identical whether or not a sink is supplied, which is
/// what keeps the one-shot and streaming paths byte-for-byte consistent.
#[must_use]
pub fn generate_profile_incremental(
    profile: AgentProfile,
    input: &str,
    sink: Option<StreamSink<'_>>,
) -> String {
    // Collapsing `None` into a no-op sink keeps one code path, so the streamed
    // and one-shot strings cannot diverge — reborrowing an `Option<&mut dyn _>`
    // across three calls does not satisfy the borrow checker anyway.
    let mut discard = |_: StreamChunk<'_>| {};
    let sink: &mut dyn FnMut(StreamChunk<'_>) = match sink {
        Some(sink) => sink,
        None => &mut discard,
    };

    let contract = profile_contract(profile);
    let mut out = String::with_capacity(
        contract.response_prefix.len() + input.len() + contract.response_suffix.len(),
    );

    emit_text_tokens(&mut out, contract.response_prefix, &mut *sink);
    emit_text_tokens(&mut out, input, &mut *sink);
    emit_text_tokens(&mut out, contract.response_suffix, &mut *sink);

    sink(StreamChunk {
        delta: "",
        done: true,
    });
    out
}

/// Append `piece` and emit it as one step.
fn emit_append(out: &mut String, piece: &str, sink: &mut dyn FnMut(StreamChunk<'_>)) {
    if piece.is_empty() {
        return;
    }
    let start = out.len();
    out.push_str(piece);
    sink(StreamChunk {
        delta: &out[start..],
        done: false,
    });
}

/// Emit whitespace-separated tokens of `text` one at a time.
///
/// Whitespace is emitted as its own step so the concatenation of every delta
/// reconstructs `text` exactly.
fn emit_text_tokens(out: &mut String, text: &str, sink: &mut dyn FnMut(StreamChunk<'_>)) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // ASCII whitespace bytes never appear inside a multi-byte UTF-8
        // sequence, so slicing at these indices is always on a char boundary.
        if bytes[i].is_ascii_whitespace() {
            emit_append(out, &text[i..=i], &mut *sink);
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < bytes.len() && !bytes[j].is_ascii_whitespace() {
            j += 1;
        }
        emit_append(out, &text[i..j], &mut *sink);
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_matches_the_one_shot_persona_string() {
        for (profile, input) in [
            (AgentProfile::Abbey, "hello world"),
            (AgentProfile::Aviva, "execute"),
            (AgentProfile::Abi, "route"),
        ] {
            let contract = profile_contract(profile);
            let expected = format!(
                "{}{}{}",
                contract.response_prefix, input, contract.response_suffix
            );
            assert_eq!(generate_profile_incremental(profile, input, None), expected);
        }
    }

    #[test]
    fn sink_fires_during_generation_and_deltas_reconstruct_the_output() {
        let mut deltas = Vec::new();
        let mut saw_done = false;
        let output = {
            let mut sink = |chunk: StreamChunk<'_>| {
                if chunk.done {
                    saw_done = true;
                    return;
                }
                assert!(!chunk.delta.is_empty(), "non-final deltas are never empty");
                deltas.push(chunk.delta.to_string());
            };
            generate_profile_incremental(AgentProfile::Abbey, "alpha beta", Some(&mut sink))
        };

        assert!(saw_done);
        // prefix tokens + "alpha" + space + "beta" + suffix tokens: many steps,
        // not one post-hoc dump.
        assert!(deltas.len() >= 4, "got {} steps", deltas.len());
        assert_eq!(
            deltas.concat(),
            output,
            "deltas must reconstruct the output"
        );
    }

    #[test]
    fn empty_input_still_yields_prefix_and_suffix() {
        let contract = profile_contract(AgentProfile::Abbey);
        assert_eq!(
            generate_profile_incremental(AgentProfile::Abbey, "", None),
            format!("{}{}", contract.response_prefix, contract.response_suffix)
        );
    }

    #[test]
    fn multibyte_input_is_not_split_mid_character() {
        // The suffix itself contains U+2019, and the input here is entirely
        // multi-byte; slicing on a non-boundary would panic.
        let output = generate_profile_incremental(AgentProfile::Abbey, "日本語 テスト", None);
        assert!(output.contains("日本語 テスト"));
        assert!(output.contains('\u{2019}'));
    }

    #[test]
    fn whitespace_runs_are_preserved_exactly() {
        let output = generate_profile_incremental(AgentProfile::Aviva, "a  b\tc", None);
        assert!(output.contains("a  b\tc"), "got {output}");
    }

    #[test]
    fn stream_mode_labels_match_zig() {
        assert_eq!(StreamMode::Incremental.label(), "incremental");
        assert_eq!(StreamMode::PostHoc.label(), "post-hoc");
    }
}
