//! Rust port of `tools/abbey_contracts.py` and `tools/vendor_abbey_contracts.py`.
//! Reuses `abi-contracts` for corpus verification; vendor logic preserves
//! exact-byte semantics, atomic publish, and error codes.

use abi_contracts::{ContractError, Corpus};
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
};
use serde_json::{Map, Number, Value};
use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

const MAX_ARTIFACT_BYTES: u64 = 1024 * 1024;
const LOCK_NAME: &str = "abbey-contracts.lock.json";
const CORPUS_DIRECTORY: &str = "corpus";
const SOURCE_REPOSITORY: &str = "https://github.com/donaldfilimon/abi";

/// Closed vendor failures that never expose artifact contents.
#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum VendorError {
    #[error("source_revision_invalid")]
    SourceRevisionInvalid,
    #[error("source_symlink")]
    SourceSymlink,
    #[error("source_unreadable")]
    SourceUnreadable,
    #[error("source_not_directory")]
    SourceNotDirectory,
    #[error("source_corpus_invalid:{path}")]
    SourceCorpusInvalid { path: String },
    #[error("source_inventory_mismatch")]
    SourceInventoryMismatch,
    #[error("source_changed_during_copy")]
    SourceChangedDuringCopy,
    #[error("destination_lock_invalid:{path}")]
    DestinationLockInvalid { path: String },
    #[error("destination_corpus_invalid:{path}")]
    DestinationCorpusInvalid { path: String },
    #[error("destination_inventory_mismatch")]
    DestinationInventoryMismatch,
    #[error("unmanaged_destination")]
    UnmanagedDestination,
    #[error("destination_missing")]
    DestinationMissing,
    #[error("destination_symlink")]
    DestinationSymlink,
    #[error("destination_symlink:{path}")]
    DestinationSymlinkPath { path: String },
    #[error("destination_lock_mismatch:{path}")]
    DestinationLockMismatch { path: String },
    #[error("destination_byte_unreadable:{path}")]
    DestinationByteUnreadable { path: String },
    #[error("destination_byte_mismatch:{path}")]
    DestinationByteMismatch { path: String },
    #[error("destination_parent_invalid")]
    DestinationParentInvalid,
    #[error("destination_inside_source")]
    DestinationInsideSource,
    #[error("copy_failed:{path}")]
    CopyFailed { path: String },
    #[error("publication_failed")]
    PublicationFailed,
    #[error("lock_value_invalid")]
    LockValueInvalid,
    #[error("artifact_unreadable:{path}")]
    ArtifactUnreadable { path: String },
    #[error("artifact_invalid:{path}")]
    ArtifactInvalid { path: String },
}

/// Evidence returned by successful vendor write or check.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct VendorReport {
    /// Domain-separated aggregate digest.
    pub aggregate_digest: String,
    /// Number of committed artifacts.
    pub artifact_count: usize,
    /// Total bytes of all artifacts.
    pub total_bytes: u64,
    /// Whether the corpus was written.
    pub wrote: bool,
}

/// Verification report for `abbey verify` - mirrors Python `VerificationReport`.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    /// Domain-separated aggregate digest.
    pub aggregate_digest: String,
    /// Number of artifacts.
    pub artifact_count: usize,
    /// Total bytes.
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Manifest {
    contract_major: u32,
    contract_revision: u32,
    algorithm: String,
    redaction_profile: String,
    artifacts: Vec<ArtifactRow>,
    aggregate_digest: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ArtifactRow {
    path: String,
    bytes: u64,
    media_type: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Lock {
    source_repository: String,
    source_revision: String,
    contract_major: u32,
    contract_revision: u32,
    aggregate_digest: String,
}

fn truncate_display(text: &str) -> String {
    let count = text.chars().count();
    if count <= 256 {
        text.to_owned()
    } else {
        let truncated: String = text.chars().take(253).collect();
        format!("{truncated}...")
    }
}

fn normalize_path(path: &Path) -> String {
    truncate_display(&path.to_string_lossy().replace('\\', "/"))
}

// --- Strict JSON (duplicate + non-finite rejection) copied from abi-contracts ---

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

fn parse_strict_for_vendor(bytes: &[u8], path: &str) -> Result<Value, VendorError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictValue.deserialize(&mut deserializer).map_err(|e| {
        if e.to_string().contains("duplicate member") {
            VendorError::ArtifactInvalid {
                path: truncate_display(path),
            }
        } else {
            VendorError::ArtifactUnreadable {
                path: truncate_display(path),
            }
        }
    })?;
    deserializer
        .end()
        .map_err(|_| VendorError::ArtifactUnreadable {
            path: truncate_display(path),
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

// --- Shared helpers ---

fn json_bytes(value: &Value) -> Result<Vec<u8>, VendorError> {
    let mut bytes = serde_json::to_string_pretty(value)
        .map_err(|_| VendorError::LockValueInvalid)?
        .into_bytes();
    bytes.push(b'\n');
    Ok(bytes)
}

fn lock_json_bytes(lock: &Lock) -> Result<Vec<u8>, VendorError> {
    let value = serde_json::to_value(lock).map_err(|_| VendorError::LockValueInvalid)?;
    json_bytes(&value)
}

fn load_json_strict_vendor(path: &Path) -> Result<Value, VendorError> {
    let display = normalize_path(path);
    // lstat check via symlink_metadata
    let metadata = fs::symlink_metadata(path).map_err(|_| VendorError::ArtifactUnreadable {
        path: display.clone(),
    })?;
    if metadata.file_type().is_symlink() {
        return Err(VendorError::ArtifactInvalid { path: display });
    }
    if !metadata.is_file() {
        return Err(VendorError::ArtifactInvalid { path: display });
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return Err(VendorError::ArtifactInvalid { path: display });
    }
    let raw = fs::read(path).map_err(|_| VendorError::ArtifactUnreadable {
        path: display.clone(),
    })?;
    // utf8 check
    std::str::from_utf8(&raw).map_err(|_| VendorError::ArtifactUnreadable {
        path: display.clone(),
    })?;
    parse_strict_for_vendor(&raw, &display)
}

fn validate_revision(source_revision: &str) -> Result<(), VendorError> {
    if source_revision.len() != 40
        || !source_revision
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
    {
        return Err(VendorError::SourceRevisionInvalid);
    }
    Ok(())
}

fn absolute_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        let current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        current.join(path)
    }
}

fn closed_manifest(source: &Path) -> Result<(Manifest, VerificationReport), VendorError> {
    if fs::symlink_metadata(source).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(VendorError::SourceSymlink);
    }
    let resolved = fs::canonicalize(source).map_err(|_| VendorError::SourceUnreadable)?;
    if !resolved.is_dir() {
        return Err(VendorError::SourceNotDirectory);
    }
    // Verify via abi-contracts
    let corpus = Corpus::open(&resolved).map_err(|e| VendorError::SourceCorpusInvalid {
        path: truncate_display(&format!("{e}")),
    })?;
    let verified = corpus
        .verify()
        .map_err(|e| VendorError::SourceCorpusInvalid {
            path: truncate_display(&format!("{e}")),
        })?;
    // Check inventory mismatch is already handled as error in verify; but python checks report.unlisted etc
    // If verify succeeded, then inventory matches.

    let manifest_path = resolved.join("manifest.json");
    let manifest_value =
        load_json_strict_vendor(&manifest_path).map_err(|_| VendorError::SourceCorpusInvalid {
            path: "manifest.json".to_owned(),
        })?;
    let manifest: Manifest =
        serde_json::from_value(manifest_value).map_err(|_| VendorError::SourceCorpusInvalid {
            path: "manifest.json".to_owned(),
        })?;
    // Build report from manifest
    let total: u64 = manifest.artifacts.iter().map(|a| a.bytes).sum();
    let report = VerificationReport {
        aggregate_digest: manifest.aggregate_digest.clone(),
        artifact_count: verified.artifact_count(),
        total_bytes: total,
    };
    // Also need to ensure unlisted/missing/duplicates not present - already verified.
    Ok((manifest, report))
}

fn artifact_paths(manifest: &Manifest) -> Vec<String> {
    manifest
        .artifacts
        .iter()
        .map(|row| row.path.clone())
        .collect()
}

fn expected_lock(manifest: &Manifest, source_revision: &str) -> Lock {
    Lock {
        source_repository: SOURCE_REPOSITORY.to_owned(),
        source_revision: source_revision.to_owned(),
        contract_major: manifest.contract_major,
        contract_revision: manifest.contract_revision,
        aggregate_digest: manifest.aggregate_digest.clone(),
    }
}

fn load_lock(destination: &Path) -> Result<Lock, VendorError> {
    let lock_path = destination.join(LOCK_NAME);
    let value =
        load_json_strict_vendor(&lock_path).map_err(|_| VendorError::DestinationLockInvalid {
            path: LOCK_NAME.to_owned(),
        })?;
    // Check keys
    let map = value
        .as_object()
        .ok_or_else(|| VendorError::DestinationLockInvalid {
            path: LOCK_NAME.to_owned(),
        })?;
    let expected_keys: BTreeSet<&str> = [
        "source_repository",
        "source_revision",
        "contract_major",
        "contract_revision",
        "aggregate_digest",
    ]
    .into_iter()
    .collect();
    let actual_keys: BTreeSet<&str> = map.keys().map(String::as_str).collect();
    if actual_keys != expected_keys {
        return Err(VendorError::DestinationLockInvalid {
            path: LOCK_NAME.to_owned(),
        });
    }
    let lock: Lock =
        serde_json::from_value(value).map_err(|_| VendorError::DestinationLockInvalid {
            path: LOCK_NAME.to_owned(),
        })?;
    if lock.source_repository != SOURCE_REPOSITORY {
        return Err(VendorError::DestinationLockInvalid {
            path: LOCK_NAME.to_owned(),
        });
    }
    validate_revision(&lock.source_revision).map_err(|_| VendorError::DestinationLockInvalid {
        path: LOCK_NAME.to_owned(),
    })?;
    if lock.aggregate_digest.len() != 64
        || !lock
            .aggregate_digest
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
    {
        return Err(VendorError::DestinationLockInvalid {
            path: LOCK_NAME.to_owned(),
        });
    }
    Ok(lock)
}

fn validate_managed_destination(
    destination: &Path,
) -> Result<(Lock, VerificationReport), VendorError> {
    if fs::symlink_metadata(destination).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(VendorError::DestinationSymlink);
    }
    if !destination.exists() || !destination.is_dir() {
        return Err(VendorError::DestinationMissing);
    }
    let entries: BTreeSet<String> = fs::read_dir(destination)
        .map_err(|_| VendorError::DestinationMissing)?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    let expected_entries: BTreeSet<String> = [LOCK_NAME, CORPUS_DIRECTORY]
        .iter()
        .map(ToString::to_string)
        .collect();
    if !expected_entries.is_subset(&entries) {
        return Err(VendorError::UnmanagedDestination);
    }
    if entries != expected_entries {
        return Err(VendorError::DestinationInventoryMismatch);
    }
    let corpus = destination.join(CORPUS_DIRECTORY);
    if fs::symlink_metadata(&corpus).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(VendorError::DestinationSymlinkPath {
            path: CORPUS_DIRECTORY.to_owned(),
        });
    }
    let lock = load_lock(destination)?;
    // Verify corpus
    let verified = Corpus::open(&corpus).map_err(|e| VendorError::DestinationCorpusInvalid {
        path: truncate_display(&format!("{e}")),
    })?;
    let verified = verified
        .verify()
        .map_err(|e| VendorError::DestinationCorpusInvalid {
            path: truncate_display(&format!("{e}")),
        })?;
    let manifest_path = corpus.join("manifest.json");
    let manifest_value = load_json_strict_vendor(&manifest_path).map_err(|_| {
        VendorError::DestinationCorpusInvalid {
            path: "corpus/manifest.json".to_owned(),
        }
    })?;
    let manifest: Manifest = serde_json::from_value(manifest_value).map_err(|_| {
        VendorError::DestinationCorpusInvalid {
            path: "corpus/manifest.json".to_owned(),
        }
    })?;
    if lock.contract_major != manifest.contract_major
        || lock.contract_revision != manifest.contract_revision
        || lock.aggregate_digest != verified_aggregate(&corpus, &manifest)
    {
        // Python checks lock vs report.aggregate_digest and manifest fields
        // We compute via manifest aggregate, but also need to ensure lock.aggregate_digest matches verified digest
        // Use manifest's digest for comparison if verification succeeded; abi-contracts ensures manifest digest equals aggregate
        let report_digest = manifest.aggregate_digest.clone();
        if lock.aggregate_digest != report_digest {
            return Err(VendorError::DestinationLockMismatch {
                path: LOCK_NAME.to_owned(),
            });
        }
        // Also check major/revision mismatch already triggers same error
        return Err(VendorError::DestinationLockMismatch {
            path: LOCK_NAME.to_owned(),
        });
    }
    // Check lock digest vs report
    let total: u64 = manifest.artifacts.iter().map(|a| a.bytes).sum();
    let report = VerificationReport {
        aggregate_digest: lock.aggregate_digest.clone(),
        artifact_count: verified.artifact_count(),
        total_bytes: total,
    };
    // Need to ensure we compare lock fields already; if mismatch above, error.
    // Additional check: lock vs manifest major/revision already done.
    Ok((lock, report))
}

fn verified_aggregate(_corpus: &Path, manifest: &Manifest) -> String {
    manifest.aggregate_digest.clone()
}

fn compare_bytes(
    source: &Path,
    destination: &Path,
    manifest: &Manifest,
) -> Result<(), VendorError> {
    let mut paths = vec!["manifest.json".to_owned()];
    paths.extend(artifact_paths(manifest));
    for relative in paths {
        let source_bytes = fs::read(source.join(&relative)).map_err(|_| {
            VendorError::DestinationByteUnreadable {
                path: truncate_display(&relative),
            }
        })?;
        let dest_bytes =
            fs::read(destination.join(CORPUS_DIRECTORY).join(&relative)).map_err(|_| {
                VendorError::DestinationByteUnreadable {
                    path: truncate_display(&relative),
                }
            })?;
        if source_bytes != dest_bytes {
            return Err(VendorError::DestinationByteMismatch {
                path: truncate_display(&relative),
            });
        }
    }
    Ok(())
}

fn validate_expected_destination(
    source: &Path,
    destination: &Path,
    manifest: &Manifest,
    expected_lock: &Lock,
) -> Result<VerificationReport, VendorError> {
    let (lock, report) = validate_managed_destination(destination)?;
    if lock.source_repository != expected_lock.source_repository
        || lock.source_revision != expected_lock.source_revision
        || lock.contract_major != expected_lock.contract_major
        || lock.contract_revision != expected_lock.contract_revision
        || lock.aggregate_digest != expected_lock.aggregate_digest
    {
        return Err(VendorError::DestinationLockMismatch {
            path: LOCK_NAME.to_owned(),
        });
    }
    compare_bytes(source, destination, manifest)?;
    Ok(report)
}

fn copy_corpus(source: &Path, staged: &Path, manifest: &Manifest) -> Result<(), VendorError> {
    let corpus = staged.join(CORPUS_DIRECTORY);
    fs::create_dir(&corpus).map_err(|_| VendorError::CopyFailed {
        path: CORPUS_DIRECTORY.to_owned(),
    })?;
    // Set 0o700
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&corpus, fs::Permissions::from_mode(0o700));
    }
    let mut paths = vec!["manifest.json".to_owned()];
    paths.extend(artifact_paths(manifest));
    for relative in paths {
        let source_path = source.join(&relative);
        let dest_path = corpus.join(&relative);
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent).map_err(|_| VendorError::CopyFailed {
                path: truncate_display(&relative),
            })?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                // Ensure each created dir is 0o700 - best effort walk up
                let mut cur = parent;
                while cur != staged && cur != corpus {
                    let _ = fs::set_permissions(cur, fs::Permissions::from_mode(0o700));
                    if let Some(p) = cur.parent() {
                        cur = p;
                    } else {
                        break;
                    }
                }
                let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o700));
            }
        }
        let bytes = fs::read(&source_path).map_err(|_| VendorError::CopyFailed {
            path: truncate_display(&relative),
        })?;
        fs::write(&dest_path, bytes).map_err(|_| VendorError::CopyFailed {
            path: truncate_display(&relative),
        })?;
    }
    Ok(())
}

fn publish(staged: &Path, destination: &Path) -> Result<(), VendorError> {
    if !destination.exists() {
        fs::rename(staged, destination).map_err(|_| VendorError::PublicationFailed)?;
        return Ok(());
    }
    let token = uuid::Uuid::new_v4().simple().to_string()[..16].to_owned();
    let backup = destination
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!(
            ".{}.backup-{}",
            destination
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            token
        ));
    fs::rename(destination, &backup).map_err(|_| VendorError::PublicationFailed)?;
    if fs::rename(staged, destination).is_ok() {
        let _ = fs::remove_dir_all(&backup);
        Ok(())
    } else {
        let _ = fs::rename(&backup, destination);
        Err(VendorError::PublicationFailed)
    }
}

/// Verify one corpus directory, printing success or returning `ContractError`.
pub fn verify(corpus_path: &Path) -> Result<VerificationReport, ContractError> {
    let corpus = Corpus::open(corpus_path)?;
    let verified = corpus.verify()?;
    let manifest_path = corpus_path.join("manifest.json");
    let raw = fs::read(&manifest_path).map_err(|_| ContractError::ArtifactUnreadable {
        path: "manifest.json".to_owned(),
    })?;
    let value = parse_strict(&raw, "manifest.json")?;
    let manifest: Manifest =
        serde_json::from_value(value).map_err(|_| ContractError::ManifestInvalid {
            path: "manifest.json".to_owned(),
        })?;
    let total: u64 = manifest.artifacts.iter().map(|a| a.bytes).sum();
    Ok(VerificationReport {
        aggregate_digest: manifest.aggregate_digest,
        artifact_count: verified.artifact_count(),
        total_bytes: total,
    })
}

/// Vendor exact-byte corpus with lock, atomic publish, check/write modes.
/// Mirrors `tools/vendor_abbey_contracts.py::vendor`.
#[allow(clippy::too_many_lines)]
pub fn vendor(
    source: &Path,
    destination: &Path,
    source_revision: &str,
    check: bool,
) -> Result<VendorReport, VendorError> {
    validate_revision(source_revision)?;
    if fs::symlink_metadata(source).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(VendorError::SourceSymlink);
    }
    let source_resolved = fs::canonicalize(source).map_err(|_| VendorError::SourceUnreadable)?;
    if !source_resolved.is_dir() {
        return Err(VendorError::SourceNotDirectory);
    }
    let destination_abs = absolute_path(destination);
    if destination_abs == source_resolved || destination_abs.starts_with(&source_resolved) {
        // Python checks source in destination.parents - need to see if destination is inside source
        // Use starts_with after absolute
        if destination_abs != source_resolved {
            return Err(VendorError::DestinationInsideSource);
        }
        return Err(VendorError::DestinationInsideSource);
    }
    let parent = destination_abs
        .parent()
        .ok_or(VendorError::DestinationParentInvalid)?;
    if !parent.exists() || !parent.is_dir() {
        return Err(VendorError::DestinationParentInvalid);
    }
    let (manifest, source_report) = closed_manifest(&source_resolved)?;
    let expected_lock = expected_lock(&manifest, source_revision);
    if check {
        let report = validate_expected_destination(
            &source_resolved,
            &destination_abs,
            &manifest,
            &expected_lock,
        )?;
        return Ok(VendorReport {
            aggregate_digest: report.aggregate_digest,
            artifact_count: report.artifact_count,
            total_bytes: report.total_bytes,
            wrote: false,
        });
    }
    if fs::symlink_metadata(&destination_abs).is_ok_and(|m| m.file_type().is_symlink()) {
        return Err(VendorError::DestinationSymlink);
    }
    if destination_abs.exists() {
        validate_managed_destination(&destination_abs)?;
    }
    // staged temp dir
    let token = uuid::Uuid::new_v4().simple().to_string()[..16].to_owned();
    let staged_name = format!(
        ".{}.vendor-{}",
        destination_abs
            .file_name()
            .unwrap_or_default()
            .to_string_lossy(),
        token
    );
    let staged = parent.join(staged_name);
    fs::create_dir(&staged).map_err(|_| VendorError::PublicationFailed)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&staged, fs::Permissions::from_mode(0o700));
    }
    let result: Result<(), VendorError> = (|| {
        copy_corpus(&source_resolved, &staged, &manifest)?;
        let lock_bytes = lock_json_bytes(&expected_lock)?;
        fs::write(staged.join(LOCK_NAME), lock_bytes).map_err(|_| VendorError::CopyFailed {
            path: LOCK_NAME.to_owned(),
        })?;
        validate_expected_destination(&source_resolved, &staged, &manifest, &expected_lock)?;
        let (refreshed_manifest, refreshed_report) = closed_manifest(&source_resolved)?;
        if refreshed_manifest.contract_major != manifest.contract_major
            || refreshed_manifest.contract_revision != manifest.contract_revision
            || refreshed_manifest.aggregate_digest != manifest.aggregate_digest
            || refreshed_manifest.artifacts.len() != manifest.artifacts.len()
            || refreshed_report.aggregate_digest != source_report.aggregate_digest
            || refreshed_report.artifact_count != source_report.artifact_count
            || refreshed_report.total_bytes != source_report.total_bytes
        {
            return Err(VendorError::SourceChangedDuringCopy);
        }
        // Need deeper equality for artifacts list
        if refreshed_manifest
            .artifacts
            .iter()
            .zip(manifest.artifacts.iter())
            .any(|(a, b)| {
                a.path != b.path
                    || a.bytes != b.bytes
                    || a.sha256 != b.sha256
                    || a.media_type != b.media_type
            })
        {
            return Err(VendorError::SourceChangedDuringCopy);
        }
        publish(&staged, &destination_abs)?;
        Ok(())
    })();
    if let Err(e) = result {
        let _ = fs::remove_dir_all(&staged);
        return Err(e);
    }
    Ok(VendorReport {
        aggregate_digest: source_report.aggregate_digest,
        artifact_count: source_report.artifact_count,
        total_bytes: source_report.total_bytes,
        wrote: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn revision_validation() {
        assert!(validate_revision("8ceca077e1d888c2955a0aa52bcbb278c01967a5").is_ok());
        assert!(validate_revision("abc").is_err());
        assert!(validate_revision("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA").is_err());
    }
}
