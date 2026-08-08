//! Training configuration, dataset inspection, and the store-facing train path.
//!
//! Ported from `src/features/ai/training.zig` and `training_support.zig`, scoped
//! to what MCP `ai_train` needs. The optional `PointNeuralNetwork` autoencoder
//! path is not ported yet — when a dataset is available we still accept the
//! training metadata and store profile vectors; the message discloses
//! `model weights unchanged` rather than claiming a trained net.

use std::path::{Component, Path, PathBuf};

use crate::embedding::{EMBED_DIM, count_non_empty_lines, response_embedding, text_embedding};
use crate::identity::{AgentProfile, KNOWN_PROFILES};

/// Dataset file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DatasetFormat {
    /// Newline-delimited free text.
    #[default]
    Text,
    /// CSV with a header line that is not counted.
    Csv,
    /// JSON Lines; each non-empty line must parse as JSON.
    Jsonl,
}

impl DatasetFormat {
    /// Parse a format name (`text`/`csv`/`jsonl`).
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "text" => Self::Text,
            "csv" => Self::Csv,
            "jsonl" => Self::Jsonl,
            _ => return None,
        })
    }

    /// Canonical format name.
    #[must_use]
    pub const fn text(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Csv => "csv",
            Self::Jsonl => "jsonl",
        }
    }
}

/// Dataset location + format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetSpec {
    /// Path to the dataset file.
    pub path: String,
    /// Record format.
    pub format: DatasetFormat,
}

/// Training request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingConfig {
    /// Persona profile label (`abbey`/`aviva`/`abi`).
    pub profile: String,
    /// Dataset to inspect.
    pub dataset: DatasetSpec,
    /// Directory for optional weight artifacts (not yet written by this port).
    pub artifact_dir: String,
}

/// Outcome of a train call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingResult {
    /// Whether training was accepted.
    pub accepted: bool,
    /// Profile that was trained.
    pub profile: String,
    /// Dataset path echoed back.
    pub dataset_path: String,
    /// Artifact directory echoed back.
    pub artifact_dir: String,
    /// Human-readable message for the MCP report tail.
    pub message: String,
    /// Records accounted as stored (1 after a successful store write).
    pub records_stored: usize,
    /// Acceleration backend name. Honest: no Rust GPU is linked → `"cpu"`.
    pub acceleration_backend: &'static str,
    /// Stored query vector id, when the store path ran.
    pub query_vector_id: Option<String>,
    /// Stored response vector id, when the store path ran.
    pub response_vector_id: Option<String>,
}

/// Summary of a dataset inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DatasetSummary {
    /// Whether the file was readable.
    pub available: bool,
    /// Counted records.
    pub records: usize,
    /// Byte length of the file contents, or the path length when missing.
    pub bytes: usize,
}

/// Training validation / I/O errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingError {
    /// Profile label is empty or not one of the three personas.
    InvalidProfile,
    /// Dataset path is empty.
    InvalidDatasetPath,
    /// Artifact directory is empty.
    InvalidArtifactPath,
    /// Path contains a `..` segment.
    PathTraversal,
    /// Absolute path escapes the training data root.
    PathOutsideRoot,
    /// Dataset JSONL line was not valid JSON.
    InvalidJsonl,
    /// I/O failure reading the dataset.
    Io(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProfile => f.write_str("invalid training profile"),
            Self::InvalidDatasetPath => f.write_str("invalid dataset path"),
            Self::InvalidArtifactPath => f.write_str("invalid artifact path"),
            Self::PathTraversal => f.write_str("path traversal"),
            Self::PathOutsideRoot => f.write_str("path outside training data root"),
            Self::InvalidJsonl => f.write_str("invalid jsonl record"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Parse a persona profile label.
///
/// # Errors
///
/// Returns [`TrainingError::InvalidProfile`] for unknown names.
pub fn parse_agent_profile(name: &str) -> Result<AgentProfile, TrainingError> {
    KNOWN_PROFILES
        .iter()
        .copied()
        .find(|p| p.label() == name)
        .ok_or(TrainingError::InvalidProfile)
}

/// Profile + non-empty path checks.
///
/// # Errors
///
/// Returns a validation error when any required field is empty or the profile
/// is unknown.
pub fn validate_training_config(config: &TrainingConfig) -> Result<(), TrainingError> {
    if config.profile.is_empty() {
        return Err(TrainingError::InvalidProfile);
    }
    parse_agent_profile(&config.profile)?;
    if config.dataset.path.is_empty() {
        return Err(TrainingError::InvalidDatasetPath);
    }
    if config.artifact_dir.is_empty() {
        return Err(TrainingError::InvalidArtifactPath);
    }
    Ok(())
}

/// Reject `..` path segments.
fn reject_dot_dot(path: &str) -> Result<(), TrainingError> {
    for component in Path::new(path).components() {
        if matches!(component, Component::ParentDir) {
            return Err(TrainingError::PathTraversal);
        }
    }
    Ok(())
}

/// Whether `path` is under `root` (lexical, with a boundary check).
fn path_is_under_root(path: &Path, root: &Path) -> bool {
    let Ok(stripped) = path.strip_prefix(root) else {
        return false;
    };
    // strip_prefix succeeds for exact root and descendants.
    let _ = stripped;
    true
}

/// Confine `path` under `root`. Absolute paths outside the root fail; relative
/// paths are joined with the root first.
///
/// # Errors
///
/// Returns traversal / outside-root errors.
pub fn confine_training_path(path: &str, root: &Path) -> Result<PathBuf, TrainingError> {
    if path.is_empty() {
        return Err(TrainingError::InvalidDatasetPath);
    }
    reject_dot_dot(path)?;
    let candidate = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        root.join(path)
    };
    // Prefer canonical resolution when the path exists.
    if let Ok(resolved) = candidate.canonicalize() {
        let root_canon = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        if !path_is_under_root(&resolved, &root_canon) {
            return Err(TrainingError::PathOutsideRoot);
        }
        return Ok(resolved);
    }
    // Not yet created (common for artifact_dir): lexical check.
    if !path_is_under_root(&candidate, root) && Path::new(path).is_absolute() {
        return Err(TrainingError::PathOutsideRoot);
    }
    Ok(candidate)
}

/// Inspect a dataset file, counting records by format.
///
/// Missing files yield `available=false` rather than an error — matching Zig.
///
/// # Errors
///
/// Returns [`TrainingError::InvalidJsonl`] when a JSONL line is not JSON, or
/// an I/O error for unreadable-but-present files.
pub fn inspect_dataset(dataset: &DatasetSpec) -> Result<DatasetSummary, TrainingError> {
    let data = match std::fs::read_to_string(&dataset.path) {
        Ok(data) => data,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(DatasetSummary {
                available: false,
                records: 0,
                bytes: dataset.path.len(),
            });
        }
        Err(err) => return Err(TrainingError::Io(err.to_string())),
    };
    let records = count_dataset_records(dataset.format, &data)?;
    Ok(DatasetSummary {
        available: true,
        records,
        bytes: data.len(),
    })
}

fn count_dataset_records(format: DatasetFormat, data: &str) -> Result<usize, TrainingError> {
    match format {
        DatasetFormat::Text => Ok(count_non_empty_lines(data)),
        DatasetFormat::Csv => {
            let lines = count_non_empty_lines(data);
            Ok(lines.saturating_sub(1))
        }
        DatasetFormat::Jsonl => {
            let mut records = 0_usize;
            for line in data.split('\n') {
                let trimmed = line.trim_matches(|c| matches!(c, ' ' | '\t' | '\r'));
                if trimmed.is_empty() {
                    continue;
                }
                serde_json::from_str::<serde_json::Value>(trimmed)
                    .map_err(|_| TrainingError::InvalidJsonl)?;
                records += 1;
            }
            Ok(records)
        }
    }
}

/// Embedding derived from the persona label — shares the feature space of every
/// other stored vector.
#[must_use]
pub fn profile_embedding(agent: AgentProfile) -> [f32; EMBED_DIM] {
    text_embedding(agent.label())
}

/// Pure train result before store persistence: validate, inspect dataset, build
/// the metadata-accepted message. Does not write anything.
///
/// # Errors
///
/// Validation / dataset inspection failures.
pub fn train_inspect(
    config: &TrainingConfig,
) -> Result<(TrainingResult, DatasetSummary), TrainingError> {
    validate_training_config(config)?;
    let summary = inspect_dataset(&config.dataset)?;
    // Neural-net training is not yet ported; disclose that weights are unchanged.
    let net_part = "; model weights unchanged";
    let message = format!(
        "training metadata accepted; dataset_available={}; records={}; bytes={}{net_part}",
        if summary.available { "true" } else { "false" },
        summary.records,
        summary.bytes,
    );
    Ok((
        TrainingResult {
            accepted: true,
            profile: config.profile.clone(),
            dataset_path: config.dataset.path.clone(),
            artifact_dir: config.artifact_dir.clone(),
            message,
            records_stored: summary.records,
            acceleration_backend: "cpu",
            query_vector_id: None,
            response_vector_id: None,
        },
        summary,
    ))
}

/// Persist profile vectors + training metadata into a store, matching Zig's
/// `trainWithStore` after the inspect step.
///
/// Returns the updated result with `records_stored=1` and vector ids filled.
#[must_use]
pub fn apply_store_persistence(
    result: TrainingResult,
    query_id: impl std::fmt::Display,
    response_id: impl std::fmt::Display,
) -> TrainingResult {
    let dataset_records = result.records_stored;
    TrainingResult {
        records_stored: 1,
        query_vector_id: Some(query_id.to_string()),
        response_vector_id: Some(response_id.to_string()),
        message: format!("training metadata recorded in wdbx; dataset_records={dataset_records}"),
        ..result
    }
}

/// The KV value written under `agent:{profile}:training`.
#[must_use]
pub fn training_store_value(
    config: &TrainingConfig,
    query_id: impl std::fmt::Display,
    response_id: impl std::fmt::Display,
    backend: &str,
) -> String {
    format!(
        "profile={};dataset={};artifact_dir={};accelerator={backend};shader=zig_kernel;mlir=linalg;query_id={query_id};response_id={response_id};status=accepted",
        config.profile, config.dataset.path, config.artifact_dir,
    )
}

/// KV key for a profile's training record.
#[must_use]
pub fn training_store_key(profile: &str) -> String {
    format!("agent:{profile}:training")
}

/// Build the profile query/response embedding pair for store persistence.
#[must_use]
pub fn training_vectors(profile: AgentProfile) -> ([f32; EMBED_DIM], [f32; EMBED_DIM]) {
    let query = profile_embedding(profile);
    let response = response_embedding(&query);
    (query, response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_profiles() {
        assert_eq!(parse_agent_profile("abbey").unwrap(), AgentProfile::Abbey);
        assert_eq!(parse_agent_profile("aviva").unwrap(), AgentProfile::Aviva);
        assert_eq!(parse_agent_profile("abi").unwrap(), AgentProfile::Abi);
        assert!(parse_agent_profile("nobody").is_err());
    }

    #[test]
    fn validates_required_fields() {
        let ok = TrainingConfig {
            profile: "abi".into(),
            dataset: DatasetSpec {
                path: "demo".into(),
                format: DatasetFormat::Text,
            },
            artifact_dir: "zig-cache/agents".into(),
        };
        assert!(validate_training_config(&ok).is_ok());
        let bad = TrainingConfig {
            profile: "nobody".into(),
            ..ok.clone()
        };
        assert!(validate_training_config(&bad).is_err());
    }

    #[test]
    fn reject_dot_dot_segments() {
        assert!(reject_dot_dot("../etc/passwd").is_err());
        assert!(reject_dot_dot("safe/path").is_ok());
    }

    #[test]
    fn inspect_missing_dataset_is_unavailable() {
        let summary = inspect_dataset(&DatasetSpec {
            path: "/no/such/dataset/for/abi/train".into(),
            format: DatasetFormat::Text,
        })
        .unwrap();
        assert!(!summary.available);
        assert_eq!(summary.records, 0);
    }

    #[test]
    fn count_text_and_csv_records() {
        assert_eq!(
            count_dataset_records(DatasetFormat::Text, "a\nb\n\nc\n").unwrap(),
            3
        );
        assert_eq!(
            count_dataset_records(DatasetFormat::Csv, "h\na\nb\n").unwrap(),
            2
        );
        assert_eq!(
            count_dataset_records(DatasetFormat::Jsonl, "{\"a\":1}\n\n{\"b\":2}\n").unwrap(),
            2
        );
    }

    #[test]
    fn train_inspect_discloses_cpu_and_unchanged_weights() {
        let config = TrainingConfig {
            profile: "abi".into(),
            dataset: DatasetSpec {
                path: "/no/such/file".into(),
                format: DatasetFormat::Text,
            },
            artifact_dir: "out".into(),
        };
        let (result, _) = train_inspect(&config).unwrap();
        assert!(result.accepted);
        assert_eq!(result.acceleration_backend, "cpu");
        assert!(result.message.contains("model weights unchanged"));
        assert!(result.message.contains("dataset_available=false"));
    }

    #[test]
    fn apply_store_persistence_sets_ids_and_rewrites_message() {
        let result = TrainingResult {
            accepted: true,
            profile: "abi".into(),
            dataset_path: "demo".into(),
            artifact_dir: "out".into(),
            message: "old".into(),
            records_stored: 0,
            acceleration_backend: "cpu",
            query_vector_id: None,
            response_vector_id: None,
        };
        let updated = apply_store_persistence(result, 10, 11);
        assert_eq!(updated.records_stored, 1);
        assert_eq!(updated.query_vector_id.as_deref(), Some("10"));
        assert_eq!(updated.response_vector_id.as_deref(), Some("11"));
        assert_eq!(
            updated.message,
            "training metadata recorded in wdbx; dataset_records=0"
        );
    }
}
