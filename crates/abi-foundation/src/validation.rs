//! Input validation predicates.
//!
//! Ported from `src/foundation/validation.zig`. The Zig version returned bare
//! error values from an inferred error set; here each check returns a
//! [`ValidationError`] so callers can discriminate without string matching.

use std::fmt;

/// Why an input failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// The input was empty but a value was required.
    EmptySlice,
    /// The input contained an interior NUL byte.
    NullByte,
    /// None of the required patterns were present.
    MissingRequiredPattern,
    /// A forbidden pattern was present.
    ForbiddenPatternFound {
        /// The pattern that was matched.
        pattern: String,
    },
    /// The input length was outside the permitted range.
    InvalidLength {
        /// The observed length.
        actual: usize,
        /// A description of what was expected, e.g. `"at least 3"`.
        expected: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySlice => f.write_str("value must not be empty"),
            Self::NullByte => f.write_str("value must not contain NUL bytes"),
            Self::MissingRequiredPattern => f.write_str("value is missing a required pattern"),
            Self::ForbiddenPatternFound { pattern } => {
                write!(f, "value contains forbidden pattern {pattern:?}")
            }
            Self::InvalidLength { actual, expected } => {
                write!(f, "invalid length {actual}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Shorthand for a validation outcome.
pub type Result<T = ()> = std::result::Result<T, ValidationError>;

/// Reject an empty input.
pub fn non_empty(src: &[u8]) -> Result {
    if src.is_empty() {
        return Err(ValidationError::EmptySlice);
    }
    Ok(())
}

/// Reject input containing an interior NUL byte.
///
/// Relevant wherever a value crosses into a C API or a filesystem path.
pub fn no_null_bytes(src: &[u8]) -> Result {
    if src.contains(&0) {
        return Err(ValidationError::NullByte);
    }
    Ok(())
}

/// Require that at least one of `patterns` appears in `src`.
pub fn contains_one_of(src: &str, patterns: &[&str]) -> Result {
    if patterns.iter().any(|pattern| src.contains(pattern)) {
        return Ok(());
    }
    Err(ValidationError::MissingRequiredPattern)
}

/// Require that none of `forbidden` appears in `src`.
pub fn does_not_contain(src: &str, forbidden: &[&str]) -> Result {
    for pattern in forbidden {
        if src.contains(pattern) {
            return Err(ValidationError::ForbiddenPatternFound {
                pattern: (*pattern).to_string(),
            });
        }
    }
    Ok(())
}

/// Require at least `min_len` bytes.
pub fn min_length(src: &[u8], min_len: usize) -> Result {
    if src.len() < min_len {
        return Err(ValidationError::InvalidLength {
            actual: src.len(),
            expected: format!("at least {min_len}"),
        });
    }
    Ok(())
}

/// Require at most `max_len` bytes.
pub fn max_length(src: &[u8], max_len: usize) -> Result {
    if src.len() > max_len {
        return Err(ValidationError::InvalidLength {
            actual: src.len(),
            expected: format!("at most {max_len}"),
        });
    }
    Ok(())
}

/// Require exactly `exact_len` bytes.
pub fn exact_length(src: &[u8], exact_len: usize) -> Result {
    if src.len() != exact_len {
        return Err(ValidationError::InvalidLength {
            actual: src.len(),
            expected: format!("exactly {exact_len}"),
        });
    }
    Ok(())
}

/// Whether `byte` is an ASCII hexadecimal digit.
#[must_use]
pub const fn is_hex_digit(byte: u8) -> bool {
    byte.is_ascii_hexdigit()
}

/// Whether `src` is non-empty and consists solely of ASCII digits.
///
/// An empty input is *not* all-digits, matching the Zig behaviour.
#[must_use]
pub fn is_all_digits(src: &[u8]) -> bool {
    !src.is_empty() && src.iter().all(u8::is_ascii_digit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_hex_digit_covers_all_three_ranges() {
        for &byte in b"09afAF" {
            assert!(is_hex_digit(byte), "{} should be hex", byte as char);
        }
        for &byte in b"gz -" {
            assert!(!is_hex_digit(byte), "{} should not be hex", byte as char);
        }
    }

    #[test]
    fn is_all_digits_rejects_empty_and_mixed() {
        assert!(is_all_digits(b"12345"));
        assert!(is_all_digits(b"0"));
        assert!(!is_all_digits(b"12a45"));
        assert!(!is_all_digits(b""));
    }

    #[test]
    fn non_empty_rejects_only_empty() {
        assert_eq!(non_empty(b""), Err(ValidationError::EmptySlice));
        assert!(non_empty(b"non-empty").is_ok());
    }

    #[test]
    fn no_null_bytes_finds_interior_nul() {
        assert_eq!(
            no_null_bytes(b"hello\0world"),
            Err(ValidationError::NullByte)
        );
        assert!(no_null_bytes(b"hello world").is_ok());
    }

    #[test]
    fn contains_one_of_needs_a_single_match() {
        assert_eq!(
            contains_one_of("hello world", &["foo", "bar"]),
            Err(ValidationError::MissingRequiredPattern)
        );
        assert!(contains_one_of("hello world", &["hello", "xyz"]).is_ok());
    }

    #[test]
    fn does_not_contain_names_the_offending_pattern() {
        let err = does_not_contain("hello world", &["world"]).unwrap_err();
        assert_eq!(
            err,
            ValidationError::ForbiddenPatternFound {
                pattern: "world".to_string()
            }
        );
        assert!(does_not_contain("hello foo", &["world", "bar"]).is_ok());
    }

    #[test]
    fn length_checks_report_actual_and_expected() {
        let err = min_length(b"hi", 3).unwrap_err();
        assert_eq!(
            err,
            ValidationError::InvalidLength {
                actual: 2,
                expected: "at least 3".to_string()
            }
        );
        assert!(min_length(b"hello", 3).is_ok());

        assert!(max_length(b"hello world", 5).is_err());
        assert!(max_length(b"hello", 5).is_ok());

        assert!(exact_length(b"hi", 3).is_err());
        assert!(exact_length(b"hello world", 5).is_err());
        assert!(exact_length(b"hello", 5).is_ok());
    }

    #[test]
    fn errors_render_readable_messages() {
        assert_eq!(
            ValidationError::EmptySlice.to_string(),
            "value must not be empty"
        );
        assert_eq!(
            min_length(b"hi", 3).unwrap_err().to_string(),
            "invalid length 2, expected at least 3"
        );
    }
}
