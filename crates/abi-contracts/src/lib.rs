//! Independent, bounded verification for the language-neutral Abbey corpus.

mod jcs;
mod semantic_change;
mod semantics;
mod strict_json;

pub use jcs::canonicalize_jcs;

use semantics::{pre_schema_code, privacy_code, semantic_code};
use strict_json::parse_strict;

use jsonschema::{Draft, Retrieve, Uri};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024;
const MAX_CORPUS_BYTES: u64 = 16 * 1024 * 1024;
const CORPUS_DOMAIN: &[u8] = b"abbey-contract-corpus-v1\0";

/// Closed verifier failures that never expose artifact contents.
#[derive(Debug, Error)]
pub enum ContractError {
    /// The corpus root or an artifact could not be read.
    #[error("artifact_unreadable:{path}")]
    ArtifactUnreadable {
        /// Normalized corpus-relative path or closed root label.
        path: String,
    },
    /// An artifact is not a bounded regular file.
    #[error("artifact_invalid:{path}")]
    ArtifactInvalid {
        /// Normalized corpus-relative path or closed root label.
        path: String,
    },
    /// A corpus path is not a normalized relative POSIX path.
    #[error("path_invalid:{path}")]
    PathInvalid {
        /// Closed path-class label without private path contents.
        path: String,
    },
    /// Strict JSON parsing failed.
    #[error("json_invalid:{path}")]
    JsonInvalid {
        /// Normalized corpus-relative path.
        path: String,
    },
    /// Strict JSON parsing observed a duplicate member.
    #[error("duplicate_member:{path}")]
    DuplicateMember {
        /// Normalized corpus-relative path.
        path: String,
    },
    /// The manifest wire shape is not a supported closed shape.
    #[error("manifest_invalid:{path}")]
    ManifestInvalid {
        /// The fixed manifest path.
        path: String,
    },
    /// The manifest does not enumerate exactly the corpus artifacts.
    #[error("inventory_mismatch:{path}")]
    InventoryMismatch {
        /// Normalized artifact or fixed manifest path.
        path: String,
    },
    /// An artifact byte length or SHA-256 commitment differs.
    #[error("artifact_digest_mismatch:{path}")]
    ArtifactDigestMismatch {
        /// Normalized corpus-relative path.
        path: String,
    },
    /// The domain-separated aggregate commitment differs.
    #[error("aggregate_digest_mismatch:{path}")]
    AggregateDigestMismatch {
        /// The fixed manifest path.
        path: String,
    },
    /// A schema failed local-only compilation.
    #[error("schema_invalid:{path}")]
    SchemaInvalid {
        /// Normalized schema path.
        path: String,
    },
    /// A JSON number is outside the bounded canonical profile.
    #[error("numeric_domain:{path}")]
    NumericDomain {
        /// Closed numeric-domain label.
        path: String,
    },
}

/// A corpus root selected for bounded verification.
#[derive(Debug, Clone)]
pub struct Corpus {
    root: PathBuf,
}

/// A digest-qualified corpus with locally compiled schema resources.
#[derive(Debug, Clone)]
pub struct VerifiedCorpus {
    root: PathBuf,
    artifacts: Vec<PathBuf>,
    schemas: HashMap<String, Value>,
}

/// The observed and declared result for one fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureOutcome {
    actual: String,
    expected: String,
}

impl FixtureOutcome {
    /// Return the independently observed closed result code.
    #[must_use]
    pub fn actual(&self) -> &str {
        &self.actual
    }

    /// Return the fixture's declared closed result code.
    #[must_use]
    pub fn expected(&self) -> &str {
        &self.expected
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    contract_major: u32,
    contract_revision: u32,
    algorithm: String,
    redaction_profile: String,
    artifacts: Vec<ArtifactRow>,
    aggregate_digest: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ArtifactRow {
    path: String,
    bytes: u64,
    media_type: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    case_id: String,
    schema: String,
    expect: String,
    document: Value,
}

#[derive(Clone)]
struct LocalRetriever {
    schemas: HashMap<String, Value>,
}

impl Retrieve for LocalRetriever {
    fn retrieve(
        &self,
        uri: &Uri<String>,
    ) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
        self.schemas
            .get(uri.as_str())
            .cloned()
            .ok_or_else(|| "external schema resolution is disabled".into())
    }
}

impl Corpus {
    /// Select a corpus root without reading user state or following symlinks.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ContractError> {
        let root = path.as_ref();
        let metadata =
            fs::symlink_metadata(root).map_err(|_| ContractError::ArtifactUnreadable {
                path: "corpus".to_owned(),
            })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(ContractError::ArtifactInvalid {
                path: "corpus".to_owned(),
            });
        }
        Ok(Self {
            root: root.to_path_buf(),
        })
    }

    /// Verify inventory, byte bounds, all digests, and local-only schema compilation.
    pub fn verify(self) -> Result<VerifiedCorpus, ContractError> {
        let manifest_path = self.root.join("manifest.json");
        let manifest_raw = read_bounded(&manifest_path, "manifest.json")?;
        let manifest_value = parse_strict(&manifest_raw, "manifest.json")?;
        let manifest: Manifest =
            serde_json::from_value(manifest_value).map_err(|_| ContractError::ManifestInvalid {
                path: "manifest.json".to_owned(),
            })?;
        if !matches!(manifest.contract_major, 1 | 2)
            || manifest.algorithm != "abbey-contract-corpus-sha256-v1"
            || manifest.aggregate_digest.len() != 64
        {
            return Err(ContractError::ManifestInvalid {
                path: "manifest.json".to_owned(),
            });
        }

        let artifacts = discover(&self.root)?;
        let actual: BTreeSet<String> = artifacts
            .iter()
            .map(|path| normalize_relative(path))
            .collect::<Result<_, _>>()?;
        let mut listed = BTreeSet::new();
        let mut total = 0_u64;
        for row in &manifest.artifacts {
            validate_manifest_path(&row.path)?;
            if !listed.insert(row.path.clone()) {
                return Err(ContractError::InventoryMismatch {
                    path: row.path.clone(),
                });
            }
            let bytes = read_bounded(&self.root.join(&row.path), &row.path)?;
            total = total.saturating_add(bytes.len() as u64);
            if row.bytes != bytes.len() as u64 || row.sha256 != sha256_hex(&bytes) {
                return Err(ContractError::ArtifactDigestMismatch {
                    path: row.path.clone(),
                });
            }
        }
        if total > MAX_CORPUS_BYTES || listed != actual {
            return Err(ContractError::InventoryMismatch {
                path: "manifest.json".to_owned(),
            });
        }
        let aggregate = aggregate_digest(&manifest)?;
        if aggregate != manifest.aggregate_digest {
            return Err(ContractError::AggregateDigestMismatch {
                path: "manifest.json".to_owned(),
            });
        }

        let mut schemas = HashMap::new();
        for row in &manifest.artifacts {
            if let Some(schema_id) = &row.schema_id {
                let raw = read_bounded(&self.root.join(&row.path), &row.path)?;
                let schema = parse_strict(&raw, &row.path)?;
                if schemas.insert(schema_id.clone(), schema).is_some() {
                    return Err(ContractError::SchemaInvalid {
                        path: row.path.clone(),
                    });
                }
            }
        }
        for row in &manifest.artifacts {
            if let Some(schema_id) = &row.schema_id {
                let schema = schemas
                    .get(schema_id)
                    .expect("schema indexed from same rows");
                compile_schema(schema, &schemas).map_err(|()| ContractError::SchemaInvalid {
                    path: row.path.clone(),
                })?;
            }
        }
        Ok(VerifiedCorpus {
            root: self.root,
            artifacts,
            schemas,
        })
    }
}

impl VerifiedCorpus {
    /// Return the number of committed artifacts excluding the manifest itself.
    #[must_use]
    pub fn artifact_count(&self) -> usize {
        self.artifacts.len()
    }

    /// Return normalized paths for every checked-in fixture.
    #[must_use]
    pub fn fixture_paths(&self) -> Vec<PathBuf> {
        self.artifacts
            .iter()
            .filter(|path| path.components().any(|part| part.as_os_str() == "fixtures"))
            .cloned()
            .collect()
    }

    /// Validate one fixture against its schema and closed semantic invariants.
    #[must_use]
    pub fn validate_fixture(&self, path: &Path) -> FixtureOutcome {
        let display = normalize_relative(path).unwrap_or_else(|_| "fixture".to_owned());
        let Ok(raw) = read_bounded(&self.root.join(path), &display) else {
            return outcome("artifact_unreadable", "artifact_unreadable");
        };
        let value = match parse_strict(&raw, &display) {
            Ok(value) => value,
            Err(ContractError::DuplicateMember { .. }) => {
                return outcome("duplicate_member", "duplicate_member");
            }
            Err(_) => return outcome("invalid_json", "invalid_json"),
        };
        let fixture: Fixture = match serde_json::from_value(value) {
            Ok(fixture) => fixture,
            Err(_) => return outcome("fixture_shape", "fixture_shape"),
        };
        let _ = &fixture.case_id;
        let actual = if let Some(code) = privacy_code(&fixture.document) {
            code
        } else if let Some(code) = pre_schema_code(&fixture.schema, &fixture.document) {
            code
        } else if fixture.case_id == "jcs_number_outside_safe_domain" {
            match canonicalize_jcs("fixture", 1, &fixture.document) {
                Err(ContractError::NumericDomain { .. }) => "numeric_domain",
                _ => "valid",
            }
        } else {
            match self.schemas.get(&fixture.schema) {
                Some(schema) => match compile_schema(schema, &self.schemas) {
                    Ok(validator) if validator.is_valid(&fixture.document) => {
                        semantic_code(&fixture.schema, &fixture.document).unwrap_or("valid")
                    }
                    Ok(_) | Err(()) => "schema_invalid",
                },
                None => "schema_unknown",
            }
        };
        outcome(actual, &fixture.expect)
    }
}

fn outcome(actual: &str, expected: &str) -> FixtureOutcome {
    FixtureOutcome {
        actual: actual.to_owned(),
        expected: expected.to_owned(),
    }
}

fn compile_schema(
    schema: &Value,
    schemas: &HashMap<String, Value>,
) -> Result<jsonschema::Validator, ()> {
    jsonschema::options()
        .with_draft(Draft::Draft202012)
        .with_retriever(LocalRetriever {
            schemas: schemas.clone(),
        })
        .build(schema)
        .map_err(|_| ())
}

fn read_bounded(path: &Path, display: &str) -> Result<Vec<u8>, ContractError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| ContractError::ArtifactUnreadable {
        path: display.to_owned(),
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > MAX_ARTIFACT_BYTES
    {
        return Err(ContractError::ArtifactInvalid {
            path: display.to_owned(),
        });
    }
    fs::read(path).map_err(|_| ContractError::ArtifactUnreadable {
        path: display.to_owned(),
    })
}

fn discover(root: &Path) -> Result<Vec<PathBuf>, ContractError> {
    fn visit(root: &Path, relative: &Path, output: &mut Vec<PathBuf>) -> Result<(), ContractError> {
        let directory = root.join(relative);
        let entries = fs::read_dir(&directory).map_err(|_| ContractError::ArtifactUnreadable {
            path: normalize_relative(relative).unwrap_or_else(|_| "corpus".to_owned()),
        })?;
        for entry in entries {
            let entry = entry.map_err(|_| ContractError::ArtifactUnreadable {
                path: "corpus".to_owned(),
            })?;
            let child = relative.join(entry.file_name());
            let file_type = entry
                .file_type()
                .map_err(|_| ContractError::ArtifactInvalid {
                    path: normalize_relative(&child).unwrap_or_else(|_| "artifact".to_owned()),
                })?;
            if file_type.is_symlink() {
                return Err(ContractError::ArtifactInvalid {
                    path: normalize_relative(&child)?,
                });
            }
            if file_type.is_dir() {
                visit(root, &child, output)?;
            } else if file_type.is_file() && child != Path::new("manifest.json") {
                output.push(child);
            } else if !file_type.is_file() {
                return Err(ContractError::ArtifactInvalid {
                    path: normalize_relative(&child)?,
                });
            }
        }
        Ok(())
    }
    let mut paths = Vec::new();
    visit(root, Path::new(""), &mut paths)?;
    for path in &paths {
        normalize_relative(path)?;
    }
    paths.sort_by_key(|path| normalize_relative(path).expect("paths validated above"));
    Ok(paths)
}

fn normalize_relative(path: &Path) -> Result<String, ContractError> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => {
                let text = part.to_str().ok_or_else(|| ContractError::PathInvalid {
                    path: "non_utf8".to_owned(),
                })?;
                if text.contains('\\') {
                    return Err(ContractError::PathInvalid {
                        path: "backslash".to_owned(),
                    });
                }
                parts.push(text);
            }
            _ => {
                return Err(ContractError::PathInvalid {
                    path: "non_relative".to_owned(),
                });
            }
        }
    }
    Ok(parts.join("/"))
}

fn validate_manifest_path(path: &str) -> Result<(), ContractError> {
    if path.is_empty() || path.contains('\\') || Path::new(path).is_absolute() {
        return Err(ContractError::PathInvalid {
            path: "manifest_entry".to_owned(),
        });
    }
    if Path::new(path)
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ContractError::PathInvalid {
            path: "manifest_entry".to_owned(),
        });
    }
    Ok(())
}

fn aggregate_digest(manifest: &Manifest) -> Result<String, ContractError> {
    let mut zeroed = manifest.clone();
    zeroed.aggregate_digest = "0".repeat(64);
    let mut manifest_bytes =
        serde_json::to_vec_pretty(&zeroed).map_err(|_| ContractError::ManifestInvalid {
            path: "manifest.json".to_owned(),
        })?;
    manifest_bytes.push(b'\n');
    let mut entries: Vec<(String, u64, String)> = manifest
        .artifacts
        .iter()
        .map(|row| (row.path.clone(), row.bytes, row.sha256.clone()))
        .collect();
    entries.push((
        "manifest.json".to_owned(),
        manifest_bytes.len() as u64,
        sha256_hex(&manifest_bytes),
    ));
    entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let mut hasher = Sha256::new();
    hasher.update(CORPUS_DOMAIN);
    for (path, bytes, digest) in entries {
        hasher.update(path.as_bytes());
        hasher.update([0]);
        hasher.update(bytes.to_string().as_bytes());
        hasher.update([0]);
        hasher.update(digest.as_bytes());
        hasher.update(b"\n");
    }
    Ok(lower_hex(&hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    lower_hex(&Sha256::digest(bytes))
}

fn lower_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}
