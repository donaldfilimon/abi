//! Validation of a bundled plugin's on-disk structure and `abi-plugin.json`.
//!
//! Ported from `src/foundation/plugin_validator.zig`.
//!
//! ## What changed, and why
//!
//! The Zig validator required `mod.zig` + `stub.zig` beside the manifest, and an
//! `entry_point` ending in `.zig`. Those extensions are the only substantive
//! change in this port: the same files are now `mod.rs` and `stub.rs`, and the
//! entry point must end in `.rs`. The mod/stub *pair* requirement is retained —
//! it encodes the parity invariant that a feature and its no-op stub expose the
//! same API, which is the thing `tools/check_parity.zig` enforced.
//!
//! The path-safety rules are unchanged, because they are the security-relevant
//! part: an `entry_point` is used to open a file inside the plugin directory, so
//! it must not escape it. Absolute paths, `.`/`..` components, empty components,
//! and Windows drive-letter colons are all rejected.

use serde::Deserialize;
use std::path::Path;

/// Why a plugin directory failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    /// The plugin directory does not exist or is not a directory.
    MissingDirectory,
    /// `mod.rs` or `stub.rs` is absent.
    MissingModuleFile {
        /// The file that was expected.
        expected: &'static str,
    },
    /// `abi-plugin.json` is absent or unreadable.
    MissingManifest,
    /// `abi-plugin.json` is not valid JSON, or not a JSON object.
    MalformedManifest,
    /// A required manifest field is absent or not a string.
    MissingField {
        /// The field that was expected.
        field: &'static str,
    },
    /// A required manifest field is present but empty.
    EmptyField {
        /// The field that was empty.
        field: &'static str,
    },
    /// The `entry_point` is not a safe relative path.
    UnsafeEntryPoint {
        /// The rejected value.
        value: String,
    },
    /// The `entry_point` does not name an existing file.
    MissingEntryPoint {
        /// The path that was not found.
        value: String,
    },
    /// A `commands` or `context_providers` entry is not an object, or its
    /// `name` is absent/empty/not a string, or an `aliases` element is not a
    /// string.
    ///
    /// Zig raised its single `InvalidJson` for all of these; both map to
    /// `InvalidManifest` at the plugin-manager boundary, so splitting the
    /// variant out only makes the message more specific.
    MalformedDeclaration {
        /// `"commands"` or `"context_providers"`.
        section: &'static str,
    },
}

impl std::fmt::Display for ManifestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingDirectory => f.write_str("plugin directory does not exist"),
            Self::MissingModuleFile { expected } => write!(f, "plugin is missing {expected}"),
            Self::MissingManifest => f.write_str("plugin is missing abi-plugin.json"),
            Self::MalformedManifest => f.write_str("abi-plugin.json is not a JSON object"),
            Self::MissingField { field } => {
                write!(f, "abi-plugin.json is missing string field {field:?}")
            }
            Self::EmptyField { field } => write!(f, "abi-plugin.json field {field:?} is empty"),
            Self::UnsafeEntryPoint { value } => {
                write!(f, "unsafe entry_point {value:?}")
            }
            Self::MissingEntryPoint { value } => {
                write!(f, "entry_point {value:?} does not exist")
            }
            Self::MalformedDeclaration { section } => {
                write!(f, "abi-plugin.json {section} entry is malformed")
            }
        }
    }
}

impl std::error::Error for ManifestError {}

/// Files that must sit beside `abi-plugin.json`.
///
/// The pair is the parity invariant: `stub.rs` is the no-op implementation that
/// must expose the same API as `mod.rs`.
const REQUIRED_FILES: [&str; 2] = ["mod.rs", "stub.rs"];

/// The manifest file name.
pub const MANIFEST_FILE: &str = "abi-plugin.json";

/// Largest manifest that will be read, matching the Zig 64 KiB cap.
const MANIFEST_SIZE_LIMIT: u64 = 64 * 1024;

/// A slash-command a plugin declares in its manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestCommand {
    /// Command name. Required, and never empty.
    pub name: String,
    /// One-line summary. Empty when absent or not a string.
    pub summary: String,
    /// Alternate names for the command.
    pub aliases: Vec<String>,
}

/// A context provider a plugin declares in its manifest.
///
/// The plugin manager calls each provider through `run("__context__:<name>")`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestContextProvider {
    /// Provider name. Required, and never empty.
    pub name: String,
    /// One-line summary. Empty when absent or not a string.
    pub summary: String,
}

/// A validated `abi-plugin.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginManifest {
    /// Plugin name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Path to the plugin entry point, relative to the plugin directory.
    pub entry_point: String,
    /// The feature this plugin targets.
    pub target_feature: String,
    /// Slash-commands this plugin declares. Empty when absent.
    pub commands: Vec<ManifestCommand>,
    /// Context providers this plugin declares. Empty when absent.
    pub context_providers: Vec<ManifestContextProvider>,
}

/// The raw manifest shape, accepting both `snake_case` and `camelCase`.
///
/// The Zig version looked up `entry_point` then fell back to `entryPoint` (and
/// likewise for `target_feature`); serde aliases express the same thing.
#[derive(Deserialize)]
struct RawManifest {
    name: Option<String>,
    version: Option<String>,
    description: Option<String>,
    #[serde(alias = "entryPoint")]
    entry_point: Option<String>,
    #[serde(alias = "targetFeature")]
    target_feature: Option<String>,
    // Zig read these two with `obj.get("commands")` / `obj.get("context_providers")`
    // and no camelCase fallback, so there is deliberately no alias here. They stay
    // untyped because the Zig parser mixed strict and lenient rules per field —
    // see `parse_commands`.
    commands: Option<serde_json::Value>,
    context_providers: Option<serde_json::Value>,
}

/// Read a required, non-empty `name` from a declaration object.
///
/// Strict, matching Zig: absent, non-string, and empty all fail.
fn declaration_name(
    entry: &serde_json::Value,
    section: &'static str,
) -> Result<String, ManifestError> {
    let object = entry
        .as_object()
        .ok_or(ManifestError::MalformedDeclaration { section })?;
    let name = object
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or(ManifestError::MalformedDeclaration { section })?;
    if name.is_empty() {
        return Err(ManifestError::MalformedDeclaration { section });
    }
    Ok(name.to_string())
}

/// Read an optional `summary`, coercing a non-string to `""`.
///
/// Lenient, matching Zig: a `summary` of `42` is not an error, it is empty.
fn declaration_summary(entry: &serde_json::Value) -> String {
    entry
        .get("summary")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Parse the `commands` array.
///
/// A non-array value yields no commands rather than an error, matching Zig's
/// `if (cmds_val != .array) break :blk &.{}`. Within an array, however, every
/// entry must be an object with a non-empty string `name`, and every `aliases`
/// element must be a string. An `aliases` value that is present but not an array
/// is ignored, again matching Zig.
fn parse_commands(
    value: Option<&serde_json::Value>,
) -> Result<Vec<ManifestCommand>, ManifestError> {
    const SECTION: &str = "commands";
    let Some(items) = value.and_then(serde_json::Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut commands = Vec::with_capacity(items.len());
    for entry in items {
        let name = declaration_name(entry, SECTION)?;
        let summary = declaration_summary(entry);

        let mut aliases = Vec::new();
        if let Some(alias_items) = entry.get("aliases").and_then(serde_json::Value::as_array) {
            for alias in alias_items {
                let alias = alias
                    .as_str()
                    .ok_or(ManifestError::MalformedDeclaration { section: SECTION })?;
                aliases.push(alias.to_string());
            }
        }

        commands.push(ManifestCommand {
            name,
            summary,
            aliases,
        });
    }
    Ok(commands)
}

/// Parse the `context_providers` array, with the same array/entry rules as
/// [`parse_commands`].
fn parse_context_providers(
    value: Option<&serde_json::Value>,
) -> Result<Vec<ManifestContextProvider>, ManifestError> {
    const SECTION: &str = "context_providers";
    let Some(items) = value.and_then(serde_json::Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut providers = Vec::with_capacity(items.len());
    for entry in items {
        providers.push(ManifestContextProvider {
            name: declaration_name(entry, SECTION)?,
            summary: declaration_summary(entry),
        });
    }
    Ok(providers)
}

/// Validate the plugin directory at `plugin_path`.
pub fn validate(plugin_path: impl AsRef<Path>) -> Result<PluginManifest, ManifestError> {
    let dir = plugin_path.as_ref();
    if !dir.is_dir() {
        return Err(ManifestError::MissingDirectory);
    }

    for expected in &REQUIRED_FILES {
        if !dir.join(expected).is_file() {
            return Err(ManifestError::MissingModuleFile { expected });
        }
    }

    let manifest_path = dir.join(MANIFEST_FILE);
    let metadata = std::fs::metadata(&manifest_path).map_err(|_| ManifestError::MissingManifest)?;
    if metadata.len() > MANIFEST_SIZE_LIMIT {
        return Err(ManifestError::MalformedManifest);
    }
    let text =
        std::fs::read_to_string(&manifest_path).map_err(|_| ManifestError::MissingManifest)?;

    let raw: RawManifest =
        serde_json::from_str(&text).map_err(|_| ManifestError::MalformedManifest)?;

    let name = require(raw.name, "name")?;
    let version = require(raw.version, "version")?;
    let description = require(raw.description, "description")?;
    let entry_point = require(raw.entry_point, "entry_point")?;
    let target_feature = require(raw.target_feature, "target_feature")?;

    if !is_safe_entry_point(&entry_point) {
        return Err(ManifestError::UnsafeEntryPoint { value: entry_point });
    }
    if !dir.join(&entry_point).is_file() {
        return Err(ManifestError::MissingEntryPoint { value: entry_point });
    }

    Ok(PluginManifest {
        name,
        version,
        description,
        entry_point,
        target_feature,
        commands: parse_commands(raw.commands.as_ref())?,
        context_providers: parse_context_providers(raw.context_providers.as_ref())?,
    })
}

/// Whether the plugin directory at `plugin_path` is valid.
///
/// The boolean form Zig's `validatePluginStructure` returned. Prefer [`validate`],
/// which says *why* validation failed.
#[must_use]
pub fn is_valid(plugin_path: impl AsRef<Path>) -> bool {
    validate(plugin_path).is_ok()
}

fn require(value: Option<String>, field: &'static str) -> Result<String, ManifestError> {
    let value = value.ok_or(ManifestError::MissingField { field })?;
    if value.is_empty() {
        return Err(ManifestError::EmptyField { field });
    }
    Ok(value)
}

/// Whether `entry_point` is a safe path relative to the plugin directory.
///
/// Rejects: anything not ending in `.rs`, absolute paths, any `.` or `..`
/// component, empty components (so `a//b` and a trailing slash are out), and any
/// colon (which would allow a Windows drive-relative path like `C:evil.rs`).
#[must_use]
pub fn is_safe_entry_point(entry_point: &str) -> bool {
    // Case-insensitive extension check: `mod.RS` is the same file on a
    // case-insensitive filesystem, so rejecting it would be arbitrary.
    let has_rs_extension = std::path::Path::new(entry_point)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"));
    if !has_rs_extension {
        return false;
    }
    if entry_point.starts_with('/') || entry_point.starts_with('\\') {
        return false;
    }
    if entry_point.contains(':') {
        return false;
    }
    entry_point
        .split(['/', '\\'])
        .all(|part| !part.is_empty() && part != "." && part != "..")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temp_path;
    use std::path::PathBuf;

    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = temp_path::temp_file_path(name, "plugin");
            std::fs::create_dir_all(&dir).expect("create fixture dir");
            let fixture = Self { dir };
            fixture.write("mod.rs", "pub fn run() -> &'static str { \"fixture\" }\n");
            fixture.write("stub.rs", "pub fn run() -> &'static str { \"stub\" }\n");
            fixture
        }

        fn write(&self, relative: &str, contents: &str) {
            let path = self.dir.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create fixture parent");
            }
            std::fs::write(path, contents).expect("write fixture file");
        }

        fn manifest(&self, json: &str) {
            self.write(MANIFEST_FILE, json);
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    #[test]
    fn missing_directory_is_rejected() {
        let missing = temp_path::temp_file_path("abi_nonexistent_plugin", "plugin");
        assert_eq!(validate(&missing), Err(ManifestError::MissingDirectory));
        assert!(!is_valid(&missing));
    }

    #[test]
    fn a_well_formed_plugin_validates() {
        let fixture = Fixture::new("abi_plugin_ok");
        fixture.manifest(
            r#"{"name":"good","version":"0.1.0","description":"ok","target_feature":"plugins","entry_point":"mod.rs"}"#,
        );
        let manifest = validate(&fixture.dir).expect("valid plugin");
        assert_eq!(manifest.name, "good");
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.entry_point, "mod.rs");
        assert_eq!(manifest.target_feature, "plugins");
    }

    #[test]
    fn camel_case_aliases_and_a_nested_entry_point_are_accepted() {
        let fixture = Fixture::new("abi_plugin_nested");
        fixture.write("nested/entry.rs", "pub fn run() {}\n");
        fixture.manifest(
            r#"{"name":"nested","version":"0.1.0","description":"ok","targetFeature":"plugins","entryPoint":"nested/entry.rs"}"#,
        );
        let manifest = validate(&fixture.dir).expect("valid plugin");
        assert_eq!(manifest.entry_point, "nested/entry.rs");
    }

    #[test]
    fn a_missing_stub_is_rejected() {
        let fixture = Fixture::new("abi_plugin_no_stub");
        std::fs::remove_file(fixture.dir.join("stub.rs")).expect("remove stub");
        fixture.manifest(
            r#"{"name":"x","version":"0.1.0","description":"d","target_feature":"plugins","entry_point":"mod.rs"}"#,
        );
        assert_eq!(
            validate(&fixture.dir),
            Err(ManifestError::MissingModuleFile {
                expected: "stub.rs"
            })
        );
    }

    #[test]
    fn a_missing_manifest_is_rejected() {
        let fixture = Fixture::new("abi_plugin_no_manifest");
        assert_eq!(validate(&fixture.dir), Err(ManifestError::MissingManifest));
    }

    #[test]
    fn a_non_object_manifest_is_rejected() {
        let fixture = Fixture::new("abi_plugin_array_manifest");
        fixture.manifest("[]");
        assert_eq!(
            validate(&fixture.dir),
            Err(ManifestError::MalformedManifest)
        );
    }

    #[test]
    fn an_escaping_entry_point_is_rejected() {
        let fixture = Fixture::new("abi_plugin_escape");
        fixture.manifest(
            r#"{"name":"bad","version":"0.1.0","description":"bad","target_feature":"plugins","entry_point":"../mod.rs"}"#,
        );
        assert_eq!(
            validate(&fixture.dir),
            Err(ManifestError::UnsafeEntryPoint {
                value: "../mod.rs".to_string()
            })
        );
    }

    #[test]
    fn an_absent_entry_point_file_is_rejected() {
        let fixture = Fixture::new("abi_plugin_missing_entry");
        fixture.manifest(
            r#"{"name":"bad","version":"0.1.0","description":"bad","target_feature":"plugins","entry_point":"missing.rs"}"#,
        );
        assert_eq!(
            validate(&fixture.dir),
            Err(ManifestError::MissingEntryPoint {
                value: "missing.rs".to_string()
            })
        );
    }

    #[test]
    fn an_empty_required_field_is_rejected() {
        let fixture = Fixture::new("abi_plugin_empty_feature");
        fixture.manifest(
            r#"{"name":"bad","version":"0.1.0","description":"bad","target_feature":"","entry_point":"mod.rs"}"#,
        );
        assert_eq!(
            validate(&fixture.dir),
            Err(ManifestError::EmptyField {
                field: "target_feature"
            })
        );
    }

    #[test]
    fn an_absent_required_field_is_reported_by_name() {
        let fixture = Fixture::new("abi_plugin_no_version");
        fixture.manifest(
            r#"{"name":"bad","description":"bad","target_feature":"plugins","entry_point":"mod.rs"}"#,
        );
        assert_eq!(
            validate(&fixture.dir),
            Err(ManifestError::MissingField { field: "version" })
        );
    }

    #[test]
    fn safe_entry_points_are_accepted() {
        for value in ["mod.rs", "nested/entry.rs", "a/b/c/deep.rs"] {
            assert!(is_safe_entry_point(value), "{value} should be safe");
        }
    }

    #[test]
    fn unsafe_entry_points_are_rejected() {
        for value in [
            "",               // empty
            "mod.zig",        // wrong language
            "mod",            // no extension
            "/etc/passwd.rs", // absolute
            r"\windows\x.rs", // absolute, Windows form
            "../escape.rs",   // parent traversal
            "./same.rs",      // current-dir component
            "a/../../b.rs",   // traversal mid-path
            "a//b.rs",        // empty component
            "C:evil.rs",      // drive-relative
            "dir/",           // trailing slash, no file
        ] {
            assert!(!is_safe_entry_point(value), "{value:?} should be rejected");
        }
    }
}
