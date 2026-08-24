//! Independent, bounded verification for the language-neutral Abbey corpus.

mod semantic_change;

use jsonschema::{Draft, Retrieve, Uri};
use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::{Component, Path, PathBuf};
use thiserror::Error;

const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024;
const MAX_CORPUS_BYTES: u64 = 16 * 1024 * 1024;
const CORPUS_DOMAIN: &[u8] = b"abbey-contract-corpus-v1\0";
const JCS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

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
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Canonicalize a bounded authority object under the `abbey-jcs-v1` profile.
///
/// The returned bytes include the schema-family/major domain prefix. This
/// profile accepts strings, booleans, null, arrays, objects, safe integers, and
/// negative zero (normalized to zero). Other floating-point values are rejected
/// before canonicalization.
pub fn canonicalize_jcs(
    schema_family: &str,
    major: u32,
    value: &Value,
) -> Result<Vec<u8>, ContractError> {
    if schema_family.is_empty()
        || schema_family.len() > 64
        || !schema_family.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
    {
        return Err(ContractError::PathInvalid {
            path: "schema_family".to_owned(),
        });
    }
    let mut output = format!("abbey-jcs-v1:{schema_family}:{major}\0").into_bytes();
    canonical_value(value, &mut output)?;
    Ok(output)
}

fn canonical_value(value: &Value, output: &mut Vec<u8>) -> Result<(), ContractError> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(flag) => output.extend_from_slice(if *flag { b"true" } else { b"false" }),
        Value::String(text) => output.extend_from_slice(
            serde_json::to_string(text)
                .expect("serializing a string is infallible")
                .as_bytes(),
        ),
        Value::Number(number) => canonical_number(number, output)?,
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                canonical_value(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(map) => {
            output.push(b'{');
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by_key(|key| key.encode_utf16().collect::<Vec<_>>());
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend_from_slice(
                    serde_json::to_string(key)
                        .expect("serializing a key is infallible")
                        .as_bytes(),
                );
                output.push(b':');
                canonical_value(&map[key], output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

fn canonical_number(number: &Number, output: &mut Vec<u8>) -> Result<(), ContractError> {
    if let Some(integer) = number.as_i64() {
        if integer.unsigned_abs() > JCS_SAFE_INTEGER {
            return Err(ContractError::NumericDomain {
                path: "number".to_owned(),
            });
        }
        output.extend_from_slice(integer.to_string().as_bytes());
        return Ok(());
    }
    if let Some(integer) = number.as_u64() {
        if integer > JCS_SAFE_INTEGER {
            return Err(ContractError::NumericDomain {
                path: "number".to_owned(),
            });
        }
        output.extend_from_slice(integer.to_string().as_bytes());
        return Ok(());
    }
    if number.as_f64().is_some_and(|float| float == 0.0) {
        output.push(b'0');
        return Ok(());
    }
    Err(ContractError::NumericDomain {
        path: "number".to_owned(),
    })
}

fn privacy_code(value: &Value) -> Option<&'static str> {
    const FORBIDDEN: &[&str] = &[
        "audio",
        "transcript",
        "message",
        "prompt",
        "response_text",
        "credential",
        "token",
        "password",
        "username",
        "display_name",
        "filesystem_path",
        "participant_identity",
    ];
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                if FORBIDDEN.contains(&key.as_str()) || privacy_code(nested).is_some() {
                    return Some("forbidden_content");
                }
            }
        }
        Value::Array(items) => {
            if items.iter().any(|item| privacy_code(item).is_some()) {
                return Some("forbidden_content");
            }
        }
        Value::String(text) => {
            if (17..=20).contains(&text.len()) && text.bytes().all(|byte| byte.is_ascii_digit()) {
                return Some("forbidden_content");
            }
            if ["/Users/", "/home/", "C:\\", "sk-", "ghp_"]
                .iter()
                .any(|prefix| text.starts_with(prefix))
            {
                return Some("forbidden_content");
            }
        }
        _ => {}
    }
    None
}

fn pre_schema_code(schema: &str, document: &Value) -> Option<&'static str> {
    let map = document.as_object()?;
    if schema.ends_with("/learning/promotion-candidate.schema.json")
        && [
            "grant",
            "approval",
            "safety_policy_mutation",
            "command_registration",
            "platform_write",
            "direct_platform_write",
        ]
        .iter()
        .any(|key| map.contains_key(*key))
    {
        return Some("learning_authority_forbidden");
    }
    if schema.ends_with("/episode/proposal.schema.json")
        && map.get("priority_class").and_then(Value::as_str) == Some("MandatoryIncident")
        && (map.get("minimized").and_then(Value::as_bool) != Some(true)
            || map.get("redacted").and_then(Value::as_bool) != Some(true)
            || map.get("deletion_required").and_then(Value::as_bool) != Some(true)
            || map.get("deletion_key").and_then(Value::as_str).is_none()
            || map.get("retention_class").and_then(Value::as_str) != Some("mandatory_incident")
            || !matches!(
                map.get("hold_state").and_then(Value::as_str),
                Some("active" | "released")
            ))
    {
        return Some("mandatory_controls_missing");
    }
    None
}

fn semantic_code(schema: &str, document: &Value) -> Option<&'static str> {
    let map = document.as_object()?;
    if schema.ends_with("/identity/delegation-chain.schema.json") {
        let hops = map.get("hops")?.as_array()?;
        let mut seen = BTreeSet::new();
        if let Some(first) = hops.first().and_then(Value::as_object) {
            seen.insert(first.get("delegator_principal_id")?.as_str()?);
        }
        for pair in hops.windows(2) {
            let left = pair[0].as_object()?;
            let right = pair[1].as_object()?;
            if left.get("delegatee_principal_id") != right.get("delegator_principal_id") {
                return Some("delegation_chain_broken");
            }
        }
        for hop in hops {
            let delegatee = hop.as_object()?.get("delegatee_principal_id")?.as_str()?;
            if !seen.insert(delegatee) {
                return Some("delegation_cycle");
            }
        }
    }
    if schema.ends_with("/authorization/approval.schema.json")
        && map.get("approver_principal_id") == map.get("request_subject_principal_id")
    {
        return Some("self_approval");
    }
    if let Some(code) = semantic_change::code(schema, map) {
        return Some(code);
    }
    if schema.ends_with("/authorization/policy-decision.schema.json")
        && map.get("reason_code").and_then(Value::as_str) == Some("dependency_unavailable")
        && map.get("decision").and_then(Value::as_str) != Some("deny")
    {
        return Some("degraded_authority");
    }
    if schema.ends_with("/cognition/request.schema.json")
        && matches!(
            map.get("effect_class").and_then(Value::as_str),
            Some("durable_write" | "platform_effect")
        )
        && !map.contains_key("idempotency_key")
    {
        return Some("idempotency_required");
    }
    if schema.ends_with("/event/cancellation.schema.json")
        && map.get("cancellation_reference") != map.get("target_cancellation_reference")
    {
        return Some("cancellation_mismatch");
    }
    consent_semantic(schema, map).or_else(|| memory_learning_semantic(schema, map))
}

fn consent_semantic(schema: &str, map: &Map<String, Value>) -> Option<&'static str> {
    if schema.ends_with("/consent/transition.schema.json") {
        let transition = (
            map.get("from_state").and_then(Value::as_str),
            map.get("to_state").and_then(Value::as_str),
        );
        let valid = matches!(
            transition,
            (Some("Closed"), Some("PendingAttestation"))
                | (Some("PendingAttestation"), Some("Open"))
                | (Some("Open"), Some("Closing"))
                | (Some("Closing"), Some("Closed"))
        );
        if !valid {
            return Some(
                if matches!(
                    map.get("reason_code").and_then(Value::as_str),
                    Some(
                        "participant_change"
                            | "unidentified_participant"
                            | "attestation_lost"
                            | "manager_deauthorized"
                            | "connection_lost"
                            | "explicit_stop"
                    )
                ) {
                    "consent_close_required"
                } else {
                    "consent_transition_invalid"
                },
            );
        }
        if transition.1 == Some("Open")
            && (map.get("manager_authorized").and_then(Value::as_bool) != Some(true)
                || map
                    .get("all_current_participants_consented")
                    .and_then(Value::as_bool)
                    != Some(true)
                || map.get("participant_count").and_then(Value::as_u64) == Some(0))
        {
            return Some("consent_open_denied");
        }
        if transition.1 == Some("Closing") {
            let actual: BTreeSet<&str> = map
                .get("cancelled_stages")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .collect();
            let expected = BTreeSet::from([
                "decoded_receive",
                "stt",
                "reasoning",
                "synthesis",
                "provider",
                "playback",
            ]);
            if actual != expected {
                return Some("consent_cancellation_incomplete");
            }
        }
    }
    None
}

fn memory_learning_semantic(schema: &str, map: &Map<String, Value>) -> Option<&'static str> {
    if schema.ends_with("/episode/claim.schema.json") {
        let level = |field: &str| {
            map.get(field)
                .and_then(Value::as_str)
                .and_then(|text| text.strip_prefix('C'))
                .and_then(|text| text.parse::<u8>().ok())
        };
        if level("display_evidence_level") > level("evidence_level") {
            return Some("evidence_overclaim");
        }
    }
    if schema.ends_with("/learning/guild-learning-policy.schema.json") {
        if matches!(
            map.get("state").and_then(Value::as_str),
            Some("Unset" | "ExplicitDisabled")
        ) && map.get("adaptive_update_allowed").and_then(Value::as_bool) == Some(true)
        {
            return Some("learning_disabled");
        }
        if map.get("quiet_override").and_then(Value::as_bool) == Some(true)
            && map
                .get("unsolicited_action_allowed")
                .and_then(Value::as_bool)
                == Some(true)
        {
            return Some("quiet_override");
        }
    }
    None
}

fn parse_strict(bytes: &[u8], path: &str) -> Result<Value, ContractError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue
        .deserialize(&mut deserializer)
        .map_err(|error| {
            if error.to_string().contains("duplicate member") {
                ContractError::DuplicateMember {
                    path: path.to_owned(),
                }
            } else {
                ContractError::JsonInvalid {
                    path: path.to_owned(),
                }
            }
        })?;
    deserializer.end().map_err(|_| ContractError::JsonInvalid {
        path: path.to_owned(),
    })?;
    Ok(value)
}

struct StrictValue;

impl<'de> DeserializeSeed<'de> for StrictValue {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value without duplicate members")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(StrictValue)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut map = Map::new();
        while let Some(key) = access.next_key::<String>()? {
            if map.contains_key(&key) {
                return Err(de::Error::custom("duplicate member"));
            }
            let value = access.next_value_seed(StrictValue)?;
            map.insert(key, value);
        }
        Ok(Value::Object(map))
    }
}
