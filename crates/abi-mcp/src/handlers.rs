//! MCP tool definitions, dispatch, and error normalization.
//!
//! Ported from `src/mcp/handlers.zig`. Declares the frozen 12 tools as a
//! table with pre-parsed JSON schemas and declarative middleware validation
//! rules, matching `tests/golden/mcp-tools-list.json` byte for byte (property
//! order included).

use serde_json::{Map, Value, json};

use crate::middleware::{FieldKind, FieldSpec, validate_arguments};
use crate::state::McpState;

/// Every error [`crate::rpc::process`] can normalize to a client-facing
/// message. Mirrors `handlers.errorMessage` in Zig: whatever the specific
/// variant, `tools/call` always reports JSON-RPC code `-32603`, and only the
/// message differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolError {
    /// `tools/call` params were absent or not an object.
    MissingParams,
    /// `params.name` was absent or not a string.
    MissingToolName,
    /// `params.arguments` was absent or not an object, for a tool that
    /// declares required fields.
    MissingArguments,
    /// The `input` field was absent or not a string.
    MissingInput,
    /// The `model` field was present but not a string.
    MissingModel,
    /// The `profile` field was absent or not a string.
    MissingProfile,
    /// The `dataset` field was absent or not a string.
    MissingDataset,
    /// The `query` field was absent or not a string.
    MissingQuery,
    /// The `service` field was absent or not a string.
    MissingConnectorService,
    /// The `name` field was absent or not a string.
    MissingPluginName,
    /// The `artifact_dir` field was present but not a string.
    MissingArtifactDir,
    /// `service` was a string but not one of the known connectors.
    UnknownConnector,
    /// `format` was a string but not one of `jsonl`/`csv`/`text`.
    InvalidDatasetFormat,
    /// An `EnumChoice` field held a value outside its choices.
    InvalidFieldValue,
    /// A string field contained a NUL byte.
    InvalidFieldEncoding,
    /// A string field exceeded its length cap.
    FieldTooLong,
    /// A `FilePath` field contained a `..` traversal component.
    PathTraversal,
    /// `params.name` did not match any of the 12 tools.
    UnknownTool,
    /// The tool's arguments validated, but its backing feature has not been
    /// ported to Rust yet. Not a Zig error — the Rust port's own honest-stub
    /// convention (see `abi-cli`'s "not yet ported" handlers).
    NotYetPorted,
    /// A backing service (scheduler, WDBX store) failed unexpectedly.
    Internal,
}

impl ToolError {
    /// The stable, non-leaking client message for this error. Both transports
    /// route through this so neither exposes raw internal error identifiers.
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::MissingParams => "Missing params",
            Self::MissingToolName => "Missing tool name",
            Self::MissingArguments => "Missing arguments",
            Self::MissingInput => "Missing input",
            Self::MissingModel => "Missing model",
            Self::MissingProfile => "Missing profile",
            Self::MissingDataset => "Missing dataset",
            Self::MissingQuery => "Missing query",
            Self::MissingConnectorService => "Missing connector service",
            Self::MissingPluginName => "Missing plugin name",
            Self::MissingArtifactDir => "Missing artifact dir",
            Self::UnknownConnector => "Invalid connector service",
            Self::InvalidDatasetFormat => "Invalid dataset format",
            Self::InvalidFieldValue => "Invalid field value",
            Self::InvalidFieldEncoding => "Invalid field encoding",
            Self::FieldTooLong => "Field too long",
            Self::PathTraversal => "Invalid path",
            Self::UnknownTool => "Method not found",
            Self::NotYetPorted => "Internal error: Rust handler not yet ported",
            Self::Internal => "Internal error",
        }
    }
}

struct ToolDescriptor {
    name: &'static str,
    description: &'static str,
    /// Pre-parsed so property order in the literal string is preserved (the
    /// `serde_json` `preserve_order` feature keeps object insertion order).
    input_schema: fn() -> Value,
    fields: &'static [FieldSpec],
}

fn schema(text: &str) -> Value {
    serde_json::from_str(text).expect("tool input_schema literals are valid JSON")
}

macro_rules! tool_schema {
    ($name:ident, $text:expr) => {
        fn $name() -> Value {
            schema($text)
        }
    };
}

tool_schema!(
    ai_run_schema,
    r#"{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}"#
);
tool_schema!(
    ai_complete_schema,
    r#"{"type":"object","properties":{"input":{"type":"string"},"model":{"type":"string"}},"required":["input"]}"#
);
tool_schema!(
    ai_learn_schema,
    r#"{"type":"object","properties":{"input":{"type":"string"},"model":{"type":"string"},"evidence_limit":{"type":"integer"}},"required":["input"]}"#
);
tool_schema!(
    ai_train_schema,
    r#"{"type":"object","properties":{"profile":{"type":"string"},"dataset":{"type":"string"},"format":{"type":"string","enum":["jsonl","csv","text"]},"artifact_dir":{"type":"string"}},"required":["profile","dataset"]}"#
);
tool_schema!(
    wdbx_query_schema,
    r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#
);
tool_schema!(
    no_args_schema,
    r#"{"type":"object","properties":{},"required":[]}"#
);
tool_schema!(
    connector_test_schema,
    r#"{"type":"object","properties":{"service":{"type":"string","enum":["openai","anthropic","discord","twilio","grok"]},"input":{"type":"string"}},"required":["service","input"]}"#
);
tool_schema!(
    plugin_run_schema,
    r#"{"type":"object","properties":{"name":{"type":"string"},"input":{"type":"string"}},"required":["name"]}"#
);

const AI_RUN_FIELDS: &[FieldSpec] = &[FieldSpec::new("input", true, ToolError::MissingInput)];
const AI_COMPLETE_FIELDS: &[FieldSpec] = &[
    FieldSpec::new("input", true, ToolError::MissingInput),
    FieldSpec::new("model", false, ToolError::MissingModel),
];
const AI_LEARN_FIELDS: &[FieldSpec] = AI_COMPLETE_FIELDS;
const AI_TRAIN_FIELDS: &[FieldSpec] = &[
    FieldSpec::new("profile", true, ToolError::MissingProfile).kind(FieldKind::Identifier),
    FieldSpec::new("dataset", true, ToolError::MissingDataset).kind(FieldKind::FilePath),
    FieldSpec::new("format", false, ToolError::InvalidDatasetFormat)
        .kind(FieldKind::EnumChoice)
        .invalid_error(ToolError::InvalidDatasetFormat)
        .choices(&["jsonl", "csv", "text"]),
    FieldSpec::new("artifact_dir", false, ToolError::MissingArtifactDir).kind(FieldKind::FilePath),
];
const WDBX_QUERY_FIELDS: &[FieldSpec] = &[FieldSpec::new("query", true, ToolError::MissingQuery)];
const CONNECTOR_TEST_FIELDS: &[FieldSpec] = &[
    FieldSpec::new("service", true, ToolError::MissingConnectorService)
        .kind(FieldKind::EnumChoice)
        .invalid_error(ToolError::UnknownConnector)
        .choices(&["openai", "anthropic", "discord", "twilio", "grok"]),
    FieldSpec::new("input", true, ToolError::MissingInput),
];
const PLUGIN_RUN_FIELDS: &[FieldSpec] = &[
    FieldSpec::new("name", true, ToolError::MissingPluginName).kind(FieldKind::Identifier),
    FieldSpec::new("input", false, ToolError::MissingInput),
];

/// The frozen 12 tools, in the order `tools/list` emits them — matching
/// `tests/golden/mcp-tools-list.json`, not source-tidy alphabetical order.
const TOOLS: &[ToolDescriptor] = &[
    ToolDescriptor {
        name: "ai_run",
        description: "Run AI inference with internal profile routing",
        input_schema: ai_run_schema,
        fields: AI_RUN_FIELDS,
    },
    ToolDescriptor {
        name: "ai_complete",
        description: "Run local model completion and record metadata in the MCP WDBX store when available",
        input_schema: ai_complete_schema,
        fields: AI_COMPLETE_FIELDS,
    },
    ToolDescriptor {
        name: "ai_learn",
        description: "Run the SEA self-learning completion: evidence-augmented completion with persona-router adaptation against the MCP WDBX store",
        input_schema: ai_learn_schema,
        fields: AI_LEARN_FIELDS,
    },
    ToolDescriptor {
        name: "ai_train",
        description: "Train an agent profile",
        input_schema: ai_train_schema,
        fields: AI_TRAIN_FIELDS,
    },
    ToolDescriptor {
        name: "wdbx_query",
        description: "Query the vector store",
        input_schema: wdbx_query_schema,
        fields: WDBX_QUERY_FIELDS,
    },
    ToolDescriptor {
        name: "scheduler_stats",
        description: "Report live scheduler task counts and status for the MCP server",
        input_schema: no_args_schema,
        fields: &[],
    },
    ToolDescriptor {
        name: "scheduler_info",
        description: "Compatibility alias for scheduler_stats",
        input_schema: no_args_schema,
        fields: &[],
    },
    ToolDescriptor {
        name: "connector_test",
        description: "Exercise a connector through its deterministic local path",
        input_schema: connector_test_schema,
        fields: CONNECTOR_TEST_FIELDS,
    },
    ToolDescriptor {
        name: "gpu_status",
        description: "Report current GPU backend, acceleration mode, and capabilities",
        input_schema: no_args_schema,
        fields: &[],
    },
    ToolDescriptor {
        name: "plugin_list",
        description: "List bundled plugin metadata loaded through the plugin manager",
        input_schema: no_args_schema,
        fields: &[],
    },
    ToolDescriptor {
        name: "wdbx_stats",
        description: "Report statistics for the long-lived MCP WDBX store (kv, vectors, blocks, spatial)",
        input_schema: no_args_schema,
        fields: &[],
    },
    ToolDescriptor {
        name: "plugin_run",
        description: "Execute a registered plugin",
        input_schema: plugin_run_schema,
        fields: PLUGIN_RUN_FIELDS,
    },
];

/// The `initialize` response `result` object.
#[must_use]
pub fn handle_initialize() -> Value {
    json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "abi-mcp", "version": "0.2.0"},
    })
}

/// The `tools/list` response `result` object.
#[must_use]
pub fn handle_tools_list() -> Value {
    let tools: Vec<Value> = TOOLS
        .iter()
        .map(|tool| {
            let mut obj = Map::new();
            obj.insert("name".to_string(), Value::String(tool.name.to_string()));
            obj.insert(
                "description".to_string(),
                Value::String(tool.description.to_string()),
            );
            obj.insert("inputSchema".to_string(), (tool.input_schema)());
            Value::Object(obj)
        })
        .collect();
    json!({ "tools": tools })
}

fn tool_arguments(params_obj: &Map<String, Value>) -> Result<&Map<String, Value>, ToolError> {
    match params_obj.get("arguments") {
        Some(Value::Object(obj)) => Ok(obj),
        _ => Err(ToolError::MissingArguments),
    }
}

fn object_string<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    missing_error: ToolError,
) -> Result<&'a str, ToolError> {
    match obj.get(key) {
        Some(Value::String(s)) => Ok(s.as_str()),
        _ => Err(missing_error),
    }
}

/// The `tools/call` response `result` object: dispatches to the tool named in
/// `params.name`, after validating `params.arguments` against that tool's
/// declared field policy.
pub fn handle_tools_call(state: McpState, params: Option<&Value>) -> Result<Value, ToolError> {
    let Some(Value::Object(params_obj)) = params else {
        return Err(ToolError::MissingParams);
    };

    let tool_name = match params_obj.get("name") {
        Some(Value::String(s)) => s.as_str(),
        _ => return Err(ToolError::MissingToolName),
    };

    let descriptor = TOOLS.iter().find(|t| t.name == tool_name);
    if let Some(descriptor) = descriptor {
        validate_arguments(descriptor.fields, params_obj)?;
    }

    match tool_name {
        "ai_run" | "ai_complete" | "ai_learn" | "ai_train" | "wdbx_query" | "gpu_status"
        | "plugin_list" | "plugin_run" => Err(ToolError::NotYetPorted),
        "scheduler_stats" | "scheduler_info" => Ok(text_result(&state.scheduler_stats_text())),
        "connector_test" => {
            let args = tool_arguments(params_obj)?;
            let service = object_string(args, "service", ToolError::MissingConnectorService)?;
            let input = object_string(args, "input", ToolError::MissingInput)?;
            match crate::connector_tools::run_connector_test(service, input) {
                Ok(text) => Ok(text_result(&text)),
                Err(crate::connector_tools::ConnectorToolError::NotYetPorted) => {
                    Err(ToolError::NotYetPorted)
                }
                Err(crate::connector_tools::ConnectorToolError::Failed) => Err(ToolError::Internal),
            }
        }
        "wdbx_stats" => match state.wdbx_stats_text() {
            Ok(text) => Ok(text_result(&text)),
            Err(_) => Err(ToolError::Internal),
        },
        _ => Err(ToolError::UnknownTool),
    }
}

fn text_result(text: &str) -> Value {
    json!({ "content": [{ "type": "text", "text": text }] })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pins the middleware<->catalog invariant: every field a tool validates
    // must also be advertised in that tool's input_schema.
    #[test]
    fn every_validated_field_is_advertised_in_the_tool_input_schema() {
        for descriptor in TOOLS {
            let schema_text = (descriptor.input_schema)().to_string();
            for field in descriptor.fields {
                assert!(
                    schema_text.contains(field.name),
                    "tool '{}' validates field '{}' absent from its input_schema",
                    descriptor.name,
                    field.name
                );
            }
        }
    }

    #[test]
    fn exactly_twelve_tools_in_the_frozen_order() {
        let names: Vec<&str> = TOOLS.iter().map(|t| t.name).collect();
        assert_eq!(
            names,
            [
                "ai_run",
                "ai_complete",
                "ai_learn",
                "ai_train",
                "wdbx_query",
                "scheduler_stats",
                "scheduler_info",
                "connector_test",
                "gpu_status",
                "plugin_list",
                "wdbx_stats",
                "plugin_run",
            ]
        );
    }
}
