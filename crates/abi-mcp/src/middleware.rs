//! Centralized security validation for MCP `tools/call` arguments.
//!
//! Ported from `src/mcp/middleware.zig`. Every tool dispatch passes through
//! [`validate_arguments`] before the tool body runs. Validation policy is
//! declared per tool as a slice of [`FieldSpec`] on each tool descriptor, so
//! the rules live next to the advertised input schema instead of being
//! scattered through the individual handlers.

use serde_json::{Map, Value};

use crate::handlers::ToolError;

/// Upper bound applied to every string field unless a descriptor overrides it.
///
/// The transport already caps a whole request at 64 KB
/// ([`crate::protocol::MAX_REQUEST_SIZE`]); per-field caps stop a single field
/// from consuming the entire budget.
pub const DEFAULT_MAX_FIELD_LEN: usize = 16 * 1024;

/// How a string field is interpreted for validation purposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// Arbitrary text (prompts, queries). Null bytes and over-length rejected.
    FreeText,
    /// A short identifier (profile, plugin name). Same checks as `FreeText`
    /// today, kept distinct so identifier-specific rules can tighten later.
    Identifier,
    /// A filesystem path. Adds path-traversal rejection on top of `FreeText`.
    FilePath,
    /// A value constrained to `choices`.
    EnumChoice,
}

/// Declarative validation rule for one argument of a tool.
#[derive(Debug, Clone, Copy)]
pub struct FieldSpec {
    /// The argument name, as it appears in `arguments`.
    pub name: &'static str,
    /// Whether the field must be present.
    pub required: bool,
    /// How the field's string value is interpreted.
    pub kind: FieldKind,
    /// Error returned when a required field is missing or has the wrong JSON
    /// type. Mirrors the handler's historical per-field error so the frozen
    /// MCP error contract is preserved.
    pub missing_error: ToolError,
    /// Error returned when an `EnumChoice` value is not one of `choices`.
    pub invalid_error: ToolError,
    /// Upper bound on the field's byte length.
    pub max_len: usize,
    /// Valid values for an `EnumChoice` field.
    pub choices: &'static [&'static str],
}

impl FieldSpec {
    /// A `FreeText` field with the default length cap and no enum choices.
    #[must_use]
    pub const fn new(name: &'static str, required: bool, missing_error: ToolError) -> Self {
        Self {
            name,
            required,
            kind: FieldKind::FreeText,
            missing_error,
            invalid_error: ToolError::InvalidFieldValue,
            max_len: DEFAULT_MAX_FIELD_LEN,
            choices: &[],
        }
    }

    /// Set [`Self::kind`].
    #[must_use]
    pub const fn kind(mut self, kind: FieldKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set [`Self::choices`].
    #[must_use]
    pub const fn choices(mut self, choices: &'static [&'static str]) -> Self {
        self.choices = choices;
        self
    }

    /// Set [`Self::invalid_error`].
    #[must_use]
    pub const fn invalid_error(mut self, error: ToolError) -> Self {
        self.invalid_error = error;
        self
    }

    /// Set [`Self::max_len`].
    #[must_use]
    pub const fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }
}

/// Validates the `arguments` object of a `tools/call` request against
/// `fields`. Returns the first violation; returns cleanly when the tool
/// declares no fields (status / no-argument tools, which never read args).
pub fn validate_arguments(
    fields: &[FieldSpec],
    params_obj: &Map<String, Value>,
) -> Result<(), ToolError> {
    if fields.is_empty() {
        return Ok(());
    }

    let Some(Value::Object(args)) = params_obj.get("arguments") else {
        return Err(ToolError::MissingArguments);
    };

    // Pass 1: presence and type for every field. A missing or ill-typed
    // *required* field must win over a *content* rejection (enum/path/length)
    // on any other field, mirroring the historical handlers, which extracted
    // all required strings before running semantic checks.
    for field in fields {
        match args.get(field.name) {
            None => {
                if field.required {
                    return Err(field.missing_error);
                }
            }
            Some(Value::String(_)) => {}
            Some(_) => return Err(field.missing_error),
        }
    }

    // Pass 2: content checks (NUL bytes, length, path traversal, enum
    // membership) on the present string fields.
    for field in fields {
        if let Some(Value::String(s)) = args.get(field.name) {
            validate_string(field, s)?;
        }
    }

    Ok(())
}

fn validate_string(field: &FieldSpec, s: &str) -> Result<(), ToolError> {
    if s.contains('\0') {
        return Err(ToolError::InvalidFieldEncoding);
    }
    if s.len() > field.max_len {
        return Err(ToolError::FieldTooLong);
    }
    match field.kind {
        FieldKind::FreeText | FieldKind::Identifier => Ok(()),
        FieldKind::FilePath => validate_path(s),
        FieldKind::EnumChoice => {
            if field.choices.contains(&s) {
                Ok(())
            } else {
                Err(field.invalid_error)
            }
        }
    }
}

/// Rejects path traversal (`..` components). Absolute paths are still allowed
/// at the middleware layer for local single-user use.
fn validate_path(path: &str) -> Result<(), ToolError> {
    for segment in path.split(['/', '\\']) {
        if segment == ".." {
            return Err(ToolError::PathTraversal);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn args_object(json_text: &str) -> Value {
        serde_json::from_str(json_text).expect("valid json")
    }

    fn obj(value: &Value) -> &Map<String, Value> {
        value.as_object().expect("object")
    }

    #[test]
    fn passes_valid_free_text_and_enum() {
        let fields = [
            FieldSpec::new("service", true, ToolError::MissingConnectorService)
                .kind(FieldKind::EnumChoice)
                .invalid_error(ToolError::UnknownConnector)
                .choices(&["openai", "grok"]),
            FieldSpec::new("input", true, ToolError::MissingInput),
        ];
        let parsed = args_object(r#"{"arguments":{"service":"openai","input":"hello"}}"#);
        validate_arguments(&fields, obj(&parsed)).expect("valid");
    }

    #[test]
    fn empty_fields_needs_no_arguments() {
        let parsed = json!({});
        validate_arguments(&[], obj(&parsed)).expect("valid");
    }

    #[test]
    fn reports_the_fields_missing_error() {
        let fields = [FieldSpec::new("input", true, ToolError::MissingInput)];
        let parsed = args_object(r#"{"arguments":{}}"#);
        assert_eq!(
            validate_arguments(&fields, obj(&parsed)),
            Err(ToolError::MissingInput)
        );
    }

    #[test]
    fn requires_an_arguments_object_when_fields_exist() {
        let fields = [FieldSpec::new("input", true, ToolError::MissingInput)];
        let parsed = args_object(r#"{"name":"ai_run"}"#);
        assert_eq!(
            validate_arguments(&fields, obj(&parsed)),
            Err(ToolError::MissingArguments)
        );
    }

    #[test]
    fn rejects_unknown_enum_value() {
        let fields = [
            FieldSpec::new("service", true, ToolError::MissingConnectorService)
                .kind(FieldKind::EnumChoice)
                .invalid_error(ToolError::UnknownConnector)
                .choices(&["openai", "grok"]),
        ];
        let parsed = args_object(r#"{"arguments":{"service":"pirate"}}"#);
        assert_eq!(
            validate_arguments(&fields, obj(&parsed)),
            Err(ToolError::UnknownConnector)
        );
    }

    #[test]
    fn missing_required_field_wins_over_invalid_enum_on_another_field() {
        let fields = [
            FieldSpec::new("service", true, ToolError::MissingConnectorService)
                .kind(FieldKind::EnumChoice)
                .invalid_error(ToolError::UnknownConnector)
                .choices(&["openai", "grok"]),
            FieldSpec::new("input", true, ToolError::MissingInput),
        ];
        let parsed = args_object(r#"{"arguments":{"service":"pirate"}}"#);
        assert_eq!(
            validate_arguments(&fields, obj(&parsed)),
            Err(ToolError::MissingInput)
        );
    }

    #[test]
    fn rejects_null_bytes() {
        let fields = [FieldSpec::new("input", true, ToolError::MissingInput)];
        let parsed = args_object("{\"arguments\":{\"input\":\"ab\\u0000cd\"}}");
        assert_eq!(
            validate_arguments(&fields, obj(&parsed)),
            Err(ToolError::InvalidFieldEncoding)
        );
    }

    #[test]
    fn rejects_over_length_field() {
        let fields = [FieldSpec::new("input", true, ToolError::MissingInput).max_len(4)];
        let parsed = args_object(r#"{"arguments":{"input":"toolong"}}"#);
        assert_eq!(
            validate_arguments(&fields, obj(&parsed)),
            Err(ToolError::FieldTooLong)
        );
    }

    #[test]
    fn rejects_path_traversal_but_allows_plain_and_absolute_paths() {
        let fields = [
            FieldSpec::new("dataset", true, ToolError::MissingDataset).kind(FieldKind::FilePath)
        ];

        let traversal = args_object(r#"{"arguments":{"dataset":"../../etc/passwd"}}"#);
        assert_eq!(
            validate_arguments(&fields, obj(&traversal)),
            Err(ToolError::PathTraversal)
        );

        let plain = args_object(r#"{"arguments":{"dataset":"data/train.jsonl"}}"#);
        validate_arguments(&fields, obj(&plain)).expect("plain path is valid");

        let absolute = args_object(r#"{"arguments":{"dataset":"/home/test/train.jsonl"}}"#);
        validate_arguments(&fields, obj(&absolute)).expect("absolute path is valid");
    }

    #[test]
    fn skips_absent_optional_fields() {
        let fields = [
            FieldSpec::new("name", true, ToolError::MissingPluginName).kind(FieldKind::Identifier),
            FieldSpec {
                required: false,
                ..FieldSpec::new("input", false, ToolError::MissingInput)
            },
        ];
        let parsed = args_object(r#"{"arguments":{"name":"example-plugin"}}"#);
        validate_arguments(&fields, obj(&parsed)).expect("valid");
    }
}
