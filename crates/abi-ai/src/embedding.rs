//! Deterministic text embeddings via signed feature hashing.
//!
//! Ported from `src/features/ai/helpers.zig`.
//!
//! Each character n-gram (unigram, bigram, trigram) hashes to a bucket **and a
//! sign**, so strings sharing n-grams land on overlapping buckets with consistent
//! signs and score high cosine similarity. Trigrams are weighted above bigrams
//! above unigrams, because a shared rare trigram is stronger evidence of
//! similarity than an incidental shared character.
//!
//! ## Honest scope
//!
//! This is a classical, non-learned embedding. It carries real lexical signal —
//! materially better than a per-position character-frequency hash — but it has no
//! trained semantics, no subword vocabulary, and no notion of meaning beyond
//! surface form. It is not a sentence-transformer.
//!
//! ## Why the hash is not interchangeable
//!
//! Both the bucket and the sign come from [`abi_foundation::wyhash`], which is a
//! deliberate port of Zig's `std.hash.Wyhash` rather than the Rust `wyhash`
//! crate. The vectors produced here are persisted into the user's WDBX store next
//! to vectors Zig wrote; a different hash would make cosine search across old and
//! new records silently wrong. See that module for the measured divergence.

use abi_foundation::wyhash;

/// Embedding dimensionality.
///
/// Wide enough that cosine similarity carries real lexical signal, and no wider
/// than the HNSW padding width, so the index stores it directly.
pub const EMBED_DIM: usize = 32;

/// n-gram widths and their weights, in the order Zig applied them.
///
/// The order does not affect the result — every gram adds into the same
/// accumulator — but it is preserved so the two implementations diff cleanly.
const GRAMS: [(usize, f32); 3] = [(1, 0.5), (2, 1.0), (3, 1.5)];

/// Embed `input` into a unit vector.
///
/// An empty input, or one whose signed features cancel exactly, maps to the fixed
/// unit vector `e₀` — never to a zero vector, which would make cosine similarity
/// undefined.
#[must_use]
pub fn text_embedding(input: &str) -> [f32; EMBED_DIM] {
    text_embedding_bytes(input.as_bytes())
}

/// Embed raw bytes, for callers holding a non-UTF-8 buffer.
///
/// The n-gram window is byte-oriented in both ports: a multi-byte character
/// contributes its individual bytes rather than one code point. That is a
/// property of the original, preserved deliberately — changing it would alter
/// every persisted vector.
#[must_use]
pub fn text_embedding_bytes(input: &[u8]) -> [f32; EMBED_DIM] {
    let mut out = [0.0_f32; EMBED_DIM];
    if input.is_empty() {
        out[0] = 1.0;
        return out;
    }

    let mut window = [0_u8; 3];
    for (n, weight) in GRAMS {
        let mut i = 0;
        while i + n <= input.len() {
            for k in 0..n {
                window[k] = input[i + k].to_ascii_lowercase();
            }
            // Seeding by `n` keeps the unigram, bigram, and trigram feature
            // spaces distinct, so "ab" cannot collide with the unigram "a".
            let h = wyhash::hash(n as u64, &window[..n]);
            // EMBED_DIM is 32, so the remainder always fits in usize on every target.
            #[allow(clippy::cast_possible_truncation)]
            let bucket = (h % EMBED_DIM as u64) as usize;
            // Sign from the top bit: this is what makes shared n-grams
            // constructive and unrelated ones cancel.
            out[bucket] += if (h >> 63) & 1 == 0 { weight } else { -weight };
            i += 1;
        }
    }

    let norm: f32 = out.iter().map(|v| v * v).sum();
    if norm == 0.0 {
        out[0] = 1.0;
        return out;
    }
    let scale = norm.sqrt();
    for value in &mut out {
        *value /= scale;
    }
    out
}

/// Derive a response vector from a query vector.
///
/// A small deterministic per-dimension perturbation, so the stored response
/// vector is distinct from but related to the query vector.
///
/// Note this does **not** re-normalize — the result's norm is very slightly off
/// unity. That is Zig's behavior, and the values are persisted, so it is
/// preserved rather than "fixed".
#[must_use]
pub fn response_embedding(query: &[f32; EMBED_DIM]) -> [f32; EMBED_DIM] {
    let mut out = [0.0_f32; EMBED_DIM];
    for (i, (slot, value)) in out.iter_mut().zip(query).enumerate() {
        *slot = value * if i % 2 == 0 { 0.99 } else { 1.01 };
    }
    out
}

/// Count lines that hold something other than whitespace.
///
/// Ported from `helpers.countNonEmptyLines`; the dataset record count in
/// `ai_train` is defined by it.
#[must_use]
pub fn count_non_empty_lines(data: &str) -> usize {
    data.split('\n')
        .filter(|line| !line.trim_matches([' ', '\t', '\r']).is_empty())
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(vector: &[f32; EMBED_DIM]) -> f32 {
        vector.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    fn cosine(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn embeddings_are_unit_length() {
        for input in ["a", "hello world", "the quick brown fox", "日本語"] {
            let vector = text_embedding(input);
            assert!(
                (norm(&vector) - 1.0).abs() < 0.001,
                "{input} has norm {}",
                norm(&vector)
            );
        }
    }

    #[test]
    fn empty_input_maps_to_the_first_basis_vector() {
        let vector = text_embedding("");
        assert_eq!(vector[0].to_bits(), 1.0_f32.to_bits());
        assert!(vector[1..].iter().all(|v| v.to_bits() == 0.0_f32.to_bits()));
    }

    #[test]
    fn embedding_is_deterministic() {
        let a = text_embedding("hello world");
        let b = text_embedding("hello world");
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn embedding_is_case_insensitive() {
        let a = text_embedding("Hello World");
        let b = text_embedding("hello world");
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn shared_ngrams_raise_cosine_similarity() {
        // The property the whole scheme exists for.
        let base = text_embedding("the quick brown fox jumps");
        let near = text_embedding("the quick brown fox leaps");
        let far = text_embedding("zzzz qqqq vvvv");
        assert!(
            cosine(&base, &near) > cosine(&base, &far),
            "near={} far={}",
            cosine(&base, &near),
            cosine(&base, &far)
        );
    }

    #[test]
    fn distinct_inputs_give_distinct_vectors() {
        let a = text_embedding("alpha");
        let b = text_embedding("beta");
        assert!(a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()));
    }

    #[test]
    fn short_inputs_below_each_gram_width_are_handled() {
        // "a" yields one unigram and no bigram or trigram; the loops must simply
        // not run rather than reading out of bounds.
        for input in ["a", "ab", "abc"] {
            let vector = text_embedding(input);
            assert!((norm(&vector) - 1.0).abs() < 0.001, "{input}");
        }
    }

    #[test]
    fn response_embedding_perturbs_without_reordering() {
        let query = text_embedding("hello world");
        let response = response_embedding(&query);
        assert!(
            query
                .iter()
                .zip(&response)
                .any(|(x, y)| x.to_bits() != y.to_bits())
        );
        // Alternating 0.99/1.01 scaling keeps it close to the query.
        assert!(cosine(&query, &response) > 0.99);
        for (i, (q, r)) in query.iter().zip(&response).enumerate() {
            let factor = if i % 2 == 0 { 0.99 } else { 1.01 };
            assert!((r - q * factor).abs() < f32::EPSILON, "dimension {i}");
        }
    }

    #[test]
    fn non_utf8_bytes_are_accepted() {
        // The byte-oriented window means invalid UTF-8 is still embeddable.
        let vector = text_embedding_bytes(&[0xff, 0xfe, 0x00, 0x41]);
        assert!((norm(&vector) - 1.0).abs() < 0.001);
    }

    #[test]
    fn count_non_empty_lines_ignores_whitespace_only_lines() {
        assert_eq!(count_non_empty_lines("hello\nworld\n"), 2);
        assert_eq!(count_non_empty_lines(""), 0);
        assert_eq!(count_non_empty_lines("  \n\t\n\r\n"), 0);
        assert_eq!(count_non_empty_lines("a\n \nb"), 2);
    }
}
