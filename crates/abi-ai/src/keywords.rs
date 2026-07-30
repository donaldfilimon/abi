//! The sentiment keyword table driving keyword-based persona routing.
//!
//! Ported verbatim from `src/features/ai/router_keywords.zig`. The scores are
//! contract: they decide which persona answers, and `ai_run`'s golden output
//! depends on `"hello world"` scoring no keywords at all and thus falling back
//! to the Abbey prior.

/// One keyword and its per-profile affinity.
#[derive(Debug, Clone, Copy)]
pub struct SentimentKeyword {
    /// The stem to match, compared case-insensitively as a **prefix** of a token.
    pub word: &'static str,
    /// Affinity toward Abbey.
    pub abbey_score: f32,
    /// Affinity toward Aviva.
    pub aviva_score: f32,
    /// Affinity toward ABI.
    pub abi_score: f32,
}

/// Build one table entry.
macro_rules! kw {
    ($word:literal, $abbey:literal, $aviva:literal, $abi:literal) => {
        SentimentKeyword {
            word: $word,
            abbey_score: $abbey,
            aviva_score: $aviva,
            abi_score: $abi,
        }
    };
}

/// The keyword table, in Zig's declaration order.
///
/// Order does not affect the result — every match adds into the same
/// accumulator — but it is preserved so the two tables diff cleanly.
///
/// Every entry must be a **single word**: matching runs per whitespace-split
/// token, so a multi-word phrase could never match. Zig carries the same note.
pub const SENTIMENT_KEYWORDS: [SentimentKeyword; 29] = [
    // Abbey is the broad, primary personality: analytical, creative,
    // empathetic, explanatory, and collaborative cues all strengthen Abbey.
    kw!("analyze", 0.9, 0.5, 0.2),
    kw!("structure", 0.9, 0.6, 0.2),
    kw!("logical", 0.85, 0.7, 0.2),
    kw!("compare", 0.8, 0.7, 0.2),
    kw!("explain", 0.95, 0.4, 0.2),
    kw!("creative", 0.95, 0.3, 0.1),
    kw!("imagine", 0.95, 0.3, 0.1),
    kw!("explore", 0.9, 0.4, 0.2),
    kw!("brainstorm", 0.95, 0.3, 0.1),
    kw!("help", 0.95, 0.4, 0.2),
    kw!("learn", 0.95, 0.3, 0.2),
    kw!("frustrated", 0.95, 0.2, 0.1),
    // Aviva is the direct expert mode for urgent, terse execution cues.
    kw!("run", 0.3, 0.95, 0.2),
    kw!("execute", 0.3, 0.95, 0.2),
    kw!("deploy", 0.3, 0.95, 0.2),
    kw!("build", 0.5, 0.9, 0.2),
    kw!("fix", 0.5, 0.95, 0.2),
    kw!("quick", 0.3, 0.95, 0.1),
    kw!("direct", 0.3, 0.95, 0.1),
    kw!("concise", 0.3, 0.95, 0.1),
    // ABI is selected for explicit orchestration/governance work rather than
    // ordinary user-facing execution.
    kw!("orchestrate", 0.2, 0.2, 0.95),
    kw!("routing", 0.2, 0.2, 0.95),
    kw!("governance", 0.3, 0.3, 0.95),
    kw!("policy", 0.3, 0.4, 0.9),
    kw!("profile", 0.3, 0.3, 0.9),
    kw!("safe", 0.8, 0.5, 0.5),
    kw!("risk", 0.8, 0.6, 0.6),
    kw!("design", 0.9, 0.5, 0.3),
    kw!("pattern", 0.85, 0.5, 0.3),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_matches_the_zig_entry_count() {
        assert_eq!(SENTIMENT_KEYWORDS.len(), 29);
    }

    #[test]
    fn every_entry_is_a_single_lowercase_word() {
        for entry in &SENTIMENT_KEYWORDS {
            assert!(
                !entry.word.contains(' '),
                "{} is multi-word and could never match",
                entry.word
            );
            assert!(
                !entry.word.is_empty()
                    && entry.word == entry.word.to_lowercase()
                    && entry.word.is_ascii(),
                "{} must be a lowercase ASCII stem",
                entry.word
            );
        }
    }

    #[test]
    fn keywords_are_unique() {
        let mut words: Vec<&str> = SENTIMENT_KEYWORDS.iter().map(|k| k.word).collect();
        let count = words.len();
        words.sort_unstable();
        words.dedup();
        assert_eq!(words.len(), count, "a duplicated stem would double-count");
    }
}
