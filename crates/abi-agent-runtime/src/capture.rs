//! Bounded text capture.
//!
//! Capture is how a run keeps a copy of streamed text for the caller. It is
//! always bounded: a provider that streams unbounded output must never be able
//! to grow a caller's buffer without limit, and the truncation must be
//! disclosed rather than silent.

/// Default ceiling on bytes retained by a single [`CapturedText`] buffer.
///
/// This caps the bytes a run *captures*, not the bytes a provider may emit.
/// Every event is still delivered to the [`EventSink`]; only the retained copy
/// is bounded. 64 `KiB` is large enough for a normal reply and small enough
/// that a runaway stream cannot exhaust memory through capture alone.
///
/// [`EventSink`]: crate::EventSink
pub const MAX_CAPTURED_BYTES: usize = 64 * 1024;

/// Text retained from a stream, truncated at a byte ceiling on a `UTF-8`
/// character boundary.
///
/// Truncation never splits a `char`: the last partially-fitting character is
/// dropped whole and counted in [`CapturedText::dropped_bytes`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapturedText {
    text: String,
    limit: usize,
    dropped_bytes: u64,
}

impl Default for CapturedText {
    fn default() -> Self {
        Self::with_limit(MAX_CAPTURED_BYTES)
    }
}

impl CapturedText {
    /// Create an empty buffer bounded by [`MAX_CAPTURED_BYTES`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty buffer bounded by an explicit byte ceiling.
    #[must_use]
    pub fn with_limit(limit: usize) -> Self {
        Self {
            text: String::new(),
            limit,
            dropped_bytes: 0,
        }
    }

    /// Capture `input` under the default ceiling.
    #[must_use]
    pub fn capture(input: &str) -> Self {
        let mut captured = Self::new();
        captured.push_str(input);
        captured
    }

    /// Capture `input` under an explicit ceiling.
    #[must_use]
    pub fn capture_with_limit(input: &str, limit: usize) -> Self {
        let mut captured = Self::with_limit(limit);
        captured.push_str(input);
        captured
    }

    /// Append `chunk`, retaining only what fits under the ceiling.
    ///
    /// Bytes that do not fit are counted, not stored.
    pub fn push_str(&mut self, chunk: &str) {
        let remaining = self.limit.saturating_sub(self.text.len());
        if chunk.len() <= remaining {
            self.text.push_str(chunk);
            return;
        }
        let cut = floor_char_boundary(chunk, remaining);
        self.text.push_str(&chunk[..cut]);
        self.dropped_bytes = self.dropped_bytes.saturating_add(to_u64(chunk.len() - cut));
    }

    /// Retained text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Byte ceiling this buffer was built with.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Whether any byte was dropped because of the ceiling.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.dropped_bytes > 0
    }

    /// Number of bytes dropped because of the ceiling.
    #[must_use]
    pub fn dropped_bytes(&self) -> u64 {
        self.dropped_bytes
    }

    /// Consume the buffer and return the retained text.
    #[must_use]
    pub fn into_text(self) -> String {
        self.text
    }
}

/// Largest index at or below `index` that is a `UTF-8` character boundary.
fn floor_char_boundary(text: &str, index: usize) -> usize {
    if index >= text.len() {
        return text.len();
    }
    let mut candidate = index;
    while candidate > 0 && !text.is_char_boundary(candidate) {
        candidate -= 1;
    }
    candidate
}

/// Widen a length without an `as` cast, saturating instead of truncating.
fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{CapturedText, MAX_CAPTURED_BYTES, floor_char_boundary};

    #[test]
    fn keeps_everything_under_the_limit() {
        let captured = CapturedText::capture("hello");
        assert_eq!(captured.text(), "hello");
        assert!(!captured.truncated());
        assert_eq!(captured.dropped_bytes(), 0);
        assert_eq!(captured.limit(), MAX_CAPTURED_BYTES);
    }

    #[test]
    fn truncates_on_a_char_boundary_not_mid_char() {
        // "héllo" is 6 bytes: h(1) é(2) l(1) l(1) o(1). A 2-byte ceiling lands
        // inside 'é', so the whole character must be dropped.
        let captured = CapturedText::capture_with_limit("héllo", 2);
        assert_eq!(captured.text(), "h");
        assert!(captured.truncated());
        assert_eq!(captured.dropped_bytes(), 5);
    }

    #[test]
    fn truncates_exactly_at_a_boundary_when_one_lands_there() {
        let captured = CapturedText::capture_with_limit("héllo", 3);
        assert_eq!(captured.text(), "hé");
        assert_eq!(captured.dropped_bytes(), 3);
    }

    #[test]
    fn later_chunks_are_counted_but_not_stored() {
        let mut captured = CapturedText::with_limit(3);
        captured.push_str("abc");
        captured.push_str("defg");
        assert_eq!(captured.text(), "abc");
        assert_eq!(captured.dropped_bytes(), 4);
        assert!(captured.truncated());
    }

    #[test]
    fn zero_limit_stores_nothing() {
        let captured = CapturedText::capture_with_limit("é", 0);
        assert_eq!(captured.text(), "");
        assert_eq!(captured.dropped_bytes(), 2);
    }

    #[test]
    fn boundary_helper_saturates_at_the_end() {
        assert_eq!(floor_char_boundary("abc", 99), 3);
        assert_eq!(floor_char_boundary("é", 1), 0);
    }

    #[test]
    fn into_text_returns_the_retained_bytes() {
        assert_eq!(CapturedText::capture("abc").into_text(), "abc");
    }
}
