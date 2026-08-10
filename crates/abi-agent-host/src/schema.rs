//! Compiled startup-owned JSON Schema catalogue.

use crate::HostError;
use abi_agent_runtime::{ToolCall, ToolSpec};
use jsonschema::Validator;
use serde_json::Value;
use std::collections::HashMap;

pub(crate) struct CompiledTool {
    pub(crate) spec: ToolSpec,
    validator: Validator,
}

pub(crate) struct ToolSchemas {
    tools: HashMap<String, CompiledTool>,
    ordered_specs: Vec<ToolSpec>,
}

impl ToolSchemas {
    pub(crate) fn compile(specs: Vec<ToolSpec>) -> Result<Self, HostError> {
        let mut tools = HashMap::with_capacity(specs.len());
        for spec in &specs {
            if tools.contains_key(&spec.name) {
                return Err(HostError::DuplicateToolName {
                    name: spec.name.clone(),
                });
            }
            let schema: Value = if spec.input_schema.trim().is_empty() {
                Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&spec.input_schema).map_err(|error| {
                    HostError::InvalidToolSchema {
                        tool: spec.name.clone(),
                        message: bounded(error.to_string()),
                    }
                })?
            };
            let validator = jsonschema::validator_for(&schema).map_err(|error| {
                HostError::InvalidToolSchema {
                    tool: spec.name.clone(),
                    message: bounded(error.to_string()),
                }
            })?;
            tools.insert(
                spec.name.clone(),
                CompiledTool {
                    spec: spec.clone(),
                    validator,
                },
            );
        }
        Ok(Self {
            tools,
            ordered_specs: specs,
        })
    }

    pub(crate) fn specs(&self) -> &[ToolSpec] {
        &self.ordered_specs
    }

    pub(crate) fn validate(&self, call: &ToolCall) -> Result<&CompiledTool, HostError> {
        let tool = self
            .tools
            .get(&call.name)
            .ok_or_else(|| HostError::UnknownTool {
                call_id: call.id.clone(),
                tool: call.name.clone(),
            })?;
        let input: Value =
            serde_json::from_str(&call.input).map_err(|error| HostError::MalformedToolInput {
                call_id: call.id.clone(),
                message: bounded(error.to_string()),
            })?;
        tool.validator
            .validate(&input)
            .map_err(|error| HostError::ToolSchemaViolation {
                call_id: call.id.clone(),
                message: bounded(error.to_string()),
            })?;
        Ok(tool)
    }
}

pub(crate) fn bounded(message: String) -> String {
    const MAX_ERROR_BYTES: usize = 1_024;
    if message.len() <= MAX_ERROR_BYTES {
        return message;
    }
    let mut end = MAX_ERROR_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::ToolSchemas;
    use crate::HostError;
    use abi_agent_runtime::{ToolCall, ToolSpec};

    fn spec() -> ToolSpec {
        ToolSpec::new("read").with_input_schema(
            r#"{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}"#,
        )
    }

    #[test]
    fn compiles_and_validates_the_full_schema() {
        let schemas = ToolSchemas::compile(vec![spec()]).expect("schema compiles");
        assert_eq!(schemas.specs(), [spec()]);
        schemas
            .validate(&ToolCall::new("c1", "read", r#"{"path":"a"}"#))
            .expect("input validates");
        assert!(matches!(
            schemas.validate(&ToolCall::new("c2", "read", r#"{"path":1}"#)),
            Err(HostError::ToolSchemaViolation { .. })
        ));
    }

    #[test]
    fn empty_schema_allows_any_json_but_not_malformed_json() {
        let schemas = ToolSchemas::compile(vec![ToolSpec::new("read")]).expect("empty compiles");
        schemas
            .validate(&ToolCall::new("c1", "read", "[]"))
            .expect("empty schema accepts JSON");
        assert!(matches!(
            schemas.validate(&ToolCall::new("c2", "read", "no")),
            Err(HostError::MalformedToolInput { .. })
        ));
    }

    #[test]
    fn duplicate_names_and_invalid_schemas_fail_at_startup() {
        assert!(matches!(
            ToolSchemas::compile(vec![ToolSpec::new("x"), ToolSpec::new("x")]),
            Err(HostError::DuplicateToolName { .. })
        ));
        assert!(matches!(
            ToolSchemas::compile(vec![ToolSpec::new("x").with_input_schema("{")]),
            Err(HostError::InvalidToolSchema { .. })
        ));
        assert!(matches!(
            ToolSchemas::compile(vec![
                ToolSpec::new("x")
                    .with_input_schema(r#"{"$ref":"https://example.invalid/schema"}"#)
            ]),
            Err(HostError::InvalidToolSchema { .. })
        ));
    }
}
