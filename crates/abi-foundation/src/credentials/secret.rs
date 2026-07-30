//! A string that is zeroed on drop and redacted in debug output.
//!
//! Ported from the secret-hygiene half of `src/foundation/credentials_types.zig`,
//! which paired every owned key with `std.crypto.secureZero` before freeing.
//!
//! Two reasons this is a type rather than a helper function:
//!
//! 1. Zig's `wipeAndFree` had to be called explicitly at each site, so a missed
//!    `deinit` leaked a plaintext secret into freed heap. `Drop` makes the wipe
//!    unconditional.
//! 2. Zig's `Credentials` held raw slices, so anything that formatted the struct
//!    printed live secrets. [`Secret`]'s `Debug` prints `Secret(<redacted>)`.
//!
//! The zeroing itself goes through `zeroize`, which uses a volatile write and a
//! compiler fence. A hand-written `for b in buf { *b = 0 }` is dead-store
//! eliminable — the buffer is never read afterwards — which is the precise
//! failure `secureZero` existed to prevent.

use std::fmt;
use zeroize::Zeroize as _;

/// An owned secret string, zeroed when dropped.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wrap `value` as a secret.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the plaintext.
    ///
    /// Every call is a place a secret can escape into a log or an error message;
    /// keep the borrow as short as possible.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// The length of the plaintext in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the plaintext is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// A redacted form safe to display: first 3 characters, then `…`.
    ///
    /// Used by `abi auth status`, which shows that a credential is present
    /// without revealing it. Secrets shorter than 8 bytes are fully masked,
    /// since a prefix of a short secret leaks proportionally much more.
    #[must_use]
    pub fn redacted(&self) -> String {
        if self.0.len() < 8 {
            return "…".to_string();
        }
        let prefix: String = self.0.chars().take(3).collect();
        format!("{prefix}…")
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for Secret {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_the_plaintext() {
        let secret = Secret::new("sk-test-key");
        assert_eq!(secret.expose(), "sk-test-key");
        assert_eq!(secret.len(), 11);
        assert!(!secret.is_empty());
    }

    #[test]
    fn debug_never_prints_the_plaintext() {
        let secret = Secret::new("sk-super-secret-value");
        let rendered = format!("{secret:?}");
        assert_eq!(rendered, "Secret(<redacted>)");
        assert!(!rendered.contains("super"));
    }

    #[test]
    fn redacted_shows_a_short_prefix_only() {
        assert_eq!(Secret::new("sk-abcdefghij").redacted(), "sk-…");
        assert_eq!(Secret::new("AC1234567890").redacted(), "AC1…");
    }

    #[test]
    fn redacted_fully_masks_short_secrets() {
        // A 3-char prefix of a 4-char secret would leak most of it.
        assert_eq!(Secret::new("abcd").redacted(), "…");
        assert_eq!(Secret::new("").redacted(), "…");
        assert_eq!(Secret::new("1234567").redacted(), "…");
    }

    #[test]
    fn equality_compares_plaintext() {
        assert_eq!(Secret::new("same"), Secret::new("same"));
        assert_ne!(Secret::new("same"), Secret::new("different"));
    }

    #[test]
    fn clone_produces_an_independent_secret() {
        let original = Secret::new("sk-clone-me");
        let cloned = original.clone();
        drop(original);
        // The clone must survive the original's zeroing wipe.
        assert_eq!(cloned.expose(), "sk-clone-me");
    }

    #[test]
    fn from_conversions_are_available() {
        assert_eq!(Secret::from("str").expose(), "str");
        assert_eq!(Secret::from("owned".to_string()).expose(), "owned");
    }
}
