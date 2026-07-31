//! Record types for the WDBX on-disk format.

use crate::hash::{FormatError, Hash, Result};

/// A key/value entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvRecord {
    /// Entry key.
    pub key: String,
    /// Entry value, an **opaque** string.
    ///
    /// Frequently holds JSON encoded as a string, so it is double-encoded.
    /// Parsing it here would fail on the values that are not JSON.
    pub value: String,
}

/// A stored vector.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorRecord {
    /// Vector id.
    pub id: u64,
    /// Components. Dimensionality is a property of the data, not the format.
    pub values: Vec<f32>,
}

/// An audit-chain block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRecord {
    /// This block's hash.
    pub hash: Hash,
    /// The predecessor's hash, or [`Hash::GENESIS`].
    pub prev_hash: Hash,
    /// Unix milliseconds.
    pub timestamp_ms: i64,
    /// Position in the chain.
    pub sequence: u64,
    /// The AI profile that produced the entry.
    pub profile: String,
    /// Query vector id.
    pub query_id: u64,
    /// Response vector id.
    pub response_id: u64,
    /// Opaque metadata string.
    pub metadata: String,
}

/// A 3-D spatial record.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialRecord {
    /// Record id.
    pub id: u64,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Z coordinate.
    pub z: f32,
    /// Opaque payload.
    pub payload: String,
}

/// A temporal-graph node or edge.
///
/// The live store contains none, but the Zig serializer emits them and its parser
/// accepts them, so a segment written by a build that used the temporal graph must
/// still load. The payload is kept as raw JSON rather than modelled, because there
/// is no sample to validate a stricter shape against — inventing one risks
/// rejecting real data.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalRecord {
    /// Whether this is a node or an edge.
    pub kind: TemporalKind,
    /// The record's fields, verbatim.
    pub fields: serde_json::Map<String, serde_json::Value>,
}

/// Which kind of temporal record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalKind {
    /// A graph node.
    Node,
    /// A graph edge.
    Edge,
}

/// One record from a segment.
#[derive(Debug, Clone, PartialEq)]
pub enum Record {
    /// A key/value entry.
    Kv(KvRecord),
    /// A vector.
    Vector(VectorRecord),
    /// An audit-chain block.
    Block(BlockRecord),
    /// A spatial record.
    Spatial(SpatialRecord),
    /// A temporal node or edge.
    Temporal(TemporalRecord),
}

impl Record {
    /// The `type` discriminator this record serializes with.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        match self {
            Self::Kv(_) => "kv",
            Self::Vector(_) => "vector",
            Self::Block(_) => "block",
            Self::Spatial(_) => "spatial",
            Self::Temporal(record) => match record.kind {
                TemporalKind::Node => "temporal_node",
                TemporalKind::Edge => "temporal_edge",
            },
        }
    }

    /// Parse one JSONL record.
    pub fn parse(line: &str) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_str(line).map_err(|e| FormatError::InvalidField {
                record: "record",
                field: "<line>",
                reason: e.to_string(),
            })?;
        let object = value.as_object().ok_or_else(|| FormatError::InvalidField {
            record: "record",
            field: "<line>",
            reason: "not a JSON object".to_string(),
        })?;

        let type_name = object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or(FormatError::MissingField {
                record: "record",
                field: "type",
            })?;

        match type_name {
            "kv" => Ok(Self::Kv(KvRecord {
                key: string_field(object, "kv", "key")?,
                value: string_field(object, "kv", "value")?,
            })),
            "vector" => {
                let items = object
                    .get("values")
                    .and_then(serde_json::Value::as_array)
                    .ok_or(FormatError::MissingField {
                        record: "vector",
                        field: "values",
                    })?;
                let mut values = Vec::with_capacity(items.len());
                for item in items {
                    let n = item.as_f64().ok_or_else(|| FormatError::InvalidField {
                        record: "vector",
                        field: "values",
                        reason: format!("element {item} is not a number"),
                    })?;
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "the format stores f32; f64 is only the JSON transport type"
                    )]
                    values.push(n as f32);
                }
                Ok(Self::Vector(VectorRecord {
                    id: u64_field(object, "vector", "id")?,
                    values,
                }))
            }
            "block" => Ok(Self::Block(BlockRecord {
                hash: Hash::from_json(
                    object.get("hash").ok_or(FormatError::MissingField {
                        record: "block",
                        field: "hash",
                    })?,
                    "hash",
                )?,
                prev_hash: Hash::from_json(
                    object.get("prev_hash").ok_or(FormatError::MissingField {
                        record: "block",
                        field: "prev_hash",
                    })?,
                    "prev_hash",
                )?,
                timestamp_ms: i64_field(object, "block", "timestamp_ms")?,
                sequence: u64_field(object, "block", "sequence")?,
                profile: string_field(object, "block", "profile")?,
                query_id: u64_field(object, "block", "query_id")?,
                response_id: u64_field(object, "block", "response_id")?,
                metadata: string_field(object, "block", "metadata")?,
            })),
            "spatial" => Ok(Self::Spatial(SpatialRecord {
                id: u64_field(object, "spatial", "id")?,
                x: f32_field(object, "spatial", "x")?,
                y: f32_field(object, "spatial", "y")?,
                z: f32_field(object, "spatial", "z")?,
                payload: string_field(object, "spatial", "payload")?,
            })),
            "temporal_node" | "temporal_edge" => Ok(Self::Temporal(TemporalRecord {
                kind: if type_name == "temporal_node" {
                    TemporalKind::Node
                } else {
                    TemporalKind::Edge
                },
                fields: object.clone(),
            })),
            other => Err(FormatError::UnknownRecordType {
                found: other.to_string(),
            }),
        }
    }
}

fn string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<String> {
    object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not a string".to_string(),
        })
}

fn u64_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<u64> {
    object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_u64()
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not a non-negative integer".to_string(),
        })
}

fn i64_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<i64> {
    object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_i64()
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not an integer".to_string(),
        })
}

fn f32_field(
    object: &serde_json::Map<String, serde_json::Value>,
    record: &'static str,
    field: &'static str,
) -> Result<f32> {
    let n = object
        .get(field)
        .ok_or(FormatError::MissingField { record, field })?
        .as_f64()
        .ok_or_else(|| FormatError::InvalidField {
            record,
            field,
            reason: "not a number".to_string(),
        })?;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the format stores f32; f64 is only the JSON transport type"
    )]
    Ok(n as f32)
}
