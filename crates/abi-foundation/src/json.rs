//! JSON string escaping.
//!
//! Ported from `src/foundation/json.zig`. Most of the framework's JSON now goes
//! through `serde_json`, but the CLI's `help-json` and several MCP payloads
//! assemble JSON by hand, and their output is contract-tested byte for byte.
//! These helpers reproduce the Zig escaping rules exactly:
//!
//! - `"`, `\`, `\n`, `\r`, `\t`, `\b` (0x08) and `\f` (0x0c) use short escapes;
//! - all other control bytes below 0x20 become `\u00XX` with **lowercase** hex;
//! - every other byte passes through unchanged.
//!
//! The lowercase hex matters: Zig's `writeJsonString` used a `0123456789abcdef`
//! table, so `\u001f` — not `\u001F` — is what the contract tests expect.

use std::fmt::Write as _;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Append the escaped *body* of `value` to `out`, without surrounding quotes.
///
/// Iterates `char`s, not bytes. Escaping per byte and rebuilding with
/// `char::from(u8)` would reinterpret each UTF-8 continuation byte as a Latin-1
/// codepoint and mangle any non-ASCII text — only ASCII needs escaping, and
/// every byte that does is single-byte by construction.
pub fn append_escaped_body(out: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let byte = c as u32;
                out.push_str("\\u00");
                out.push(char::from(HEX[(byte >> 4) as usize]));
                out.push(char::from(HEX[(byte & 0x0f) as usize]));
            }
            c => out.push(c),
        }
    }
}

/// Append `value` to `out` as a quoted, escaped JSON string.
pub fn append_json_string(out: &mut String, value: &str) {
    out.push('"');
    append_escaped_body(out, value);
    out.push('"');
}

/// Return `value` as a quoted, escaped JSON string.
#[must_use]
pub fn escape_json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    append_json_string(&mut out, value);
    out
}

/// Return the escaped body of `value` with no surrounding quotes.
#[must_use]
pub fn escape_json_string_raw(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    append_escaped_body(&mut out, value);
    out
}

/// Write `value` to `writer` as a quoted, escaped JSON string.
pub fn write_json_string<W: std::fmt::Write>(writer: &mut W, value: &str) -> std::fmt::Result {
    writer.write_char('"')?;
    let mut body = String::with_capacity(value.len());
    append_escaped_body(&mut body, value);
    writer.write_str(&body)?;
    writer.write_char('"')
}

/// Write `value` to a byte sink as a quoted, escaped JSON string.
pub fn write_json_string_io<W: std::io::Write>(writer: &mut W, value: &str) -> std::io::Result<()> {
    writer.write_all(escape_json_string(value).as_bytes())
}

/// Borrow a JSON value as a string, or `None` if it is not a string.
///
/// Mirrors Zig's `strField`, which existed because that `std.json.Value` union
/// needed an explicit switch at every access site.
#[must_use]
pub fn str_field(value: &serde_json::Value) -> Option<&str> {
    value.as_str()
}

/// Look up `key` on a JSON object and borrow it as a string.
#[must_use]
pub fn str_member<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str()
}

/// Render a `f64` the way the Zig side did, so numeric payloads stay stable.
///
/// Zig printed non-finite values as JSON `null`, because JSON has no encoding
/// for them and emitting `NaN` would produce a document that no parser accepts.
#[must_use]
pub fn number_to_json(value: f64) -> String {
    if value.is_finite() {
        let mut out = String::new();
        let _ = write!(out, "{value}");
        out
    } else {
        "null".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_metacharacters_and_control_chars() {
        let mut out = String::new();
        append_json_string(&mut out, "a\"b\\c\n\t");
        assert_eq!(out, "\"a\\\"b\\\\c\\n\\t\"");
    }

    #[test]
    fn escapes_low_control_bytes_as_lowercase_unicode() {
        // The Zig table was lowercase; the contract tests depend on it.
        assert_eq!(escape_json_string("\u{1}"), "\"\\u0001\"");
        assert_eq!(escape_json_string("\u{1f}"), "\"\\u001f\"");
        assert_eq!(escape_json_string("\u{b}"), "\"\\u000b\"");
        assert_eq!(escape_json_string("\u{e}"), "\"\\u000e\"");
    }

    #[test]
    fn uses_short_escapes_for_backspace_and_formfeed() {
        assert_eq!(escape_json_string("\u{8}"), "\"\\b\"");
        assert_eq!(escape_json_string("\u{c}"), "\"\\f\"");
    }

    #[test]
    fn raw_variant_omits_the_quotes() {
        assert_eq!(escape_json_string_raw("hello\nworld"), "hello\\nworld");
        assert_eq!(escape_json_string("hello\nworld"), "\"hello\\nworld\"");
    }

    #[test]
    fn multibyte_utf8_passes_through_unchanged() {
        assert_eq!(escape_json_string("héllo → 世界"), "\"héllo → 世界\"");
    }

    #[test]
    fn output_is_parseable_by_serde() {
        let tricky = "quote\" back\\slash\nnewline\ttab\u{1}ctl";
        let encoded = escape_json_string(tricky);
        let decoded: String = serde_json::from_str(&encoded).expect("valid JSON string");
        assert_eq!(decoded, tricky);
    }

    #[test]
    fn matches_serde_semantics_across_all_control_bytes() {
        // Different escape *spelling* is fine; a different decoded value is not.
        for byte in 0u8..0x20 {
            let source = String::from_utf8(vec![byte]).expect("ascii");
            let ours = escape_json_string(&source);
            let decoded: String = serde_json::from_str(&ours)
                .unwrap_or_else(|e| panic!("byte {byte:#04x} produced invalid JSON: {e}"));
            assert_eq!(decoded, source, "byte {byte:#04x} round-trip");
        }
    }

    #[test]
    fn write_json_string_matches_the_allocating_form() {
        let mut out = String::new();
        write_json_string(&mut out, "a\"b\n").expect("write to String cannot fail");
        assert_eq!(out, escape_json_string("a\"b\n"));

        let mut bytes = Vec::new();
        write_json_string_io(&mut bytes, "a\"b\n").expect("write to Vec cannot fail");
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            escape_json_string("a\"b\n")
        );
    }

    #[test]
    fn str_field_and_str_member_read_strings_only() {
        let value = serde_json::json!({ "name": "abi", "count": 3 });
        assert_eq!(str_member(&value, "name"), Some("abi"));
        assert_eq!(str_member(&value, "count"), None);
        assert_eq!(str_member(&value, "missing"), None);
        assert_eq!(str_field(&serde_json::json!("plain")), Some("plain"));
        assert_eq!(str_field(&serde_json::json!(7)), None);
    }

    #[test]
    fn non_finite_numbers_become_null() {
        assert_eq!(number_to_json(1.5), "1.5");
        assert_eq!(number_to_json(f64::NAN), "null");
        assert_eq!(number_to_json(f64::INFINITY), "null");
        assert_eq!(number_to_json(f64::NEG_INFINITY), "null");
    }
}
