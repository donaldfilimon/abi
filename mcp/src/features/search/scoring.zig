//! BM25 Scoring
//!
//! Okapi BM25 relevance scoring with configurable k1 and b parameters.
//! Uses the Lucene IDF variant (always non-negative for small corpora).

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

pub fn bm25Score(
    tf: u32, // term frequency in document
    df: u32, // document frequency (how many docs contain term)
    total_docs: u64,
    doc_len: u32, // terms in document
    avg_doc_len: f64,
) f32 {
    if (df == 0 or total_docs == 0) return 0;

    const n = @as(f64, @floatFromInt(total_docs));
    const df_f = @as(f64, @floatFromInt(df));
    const tf_f = @as(f64, @floatFromInt(tf));
    const dl = @as(f64, @floatFromInt(doc_len));

    // IDF (Lucene variant, always non-negative for small corpora)
    // = log(1 + (N - df + 0.5) / (df + 0.5))
    const idf = @log(1.0 + (n - df_f + 0.5) / (df_f + 0.5));

    // TF component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
    const tf_component = (tf_f * (BM25_K1 + 1.0)) /
        (tf_f + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / avg_doc_len));

    return @floatCast(idf * tf_component);
}
