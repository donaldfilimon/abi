//! File-based credential persistence — the default backend.
//!
//! Ported from `src/foundation/credentials_file.zig`. Protection is POSIX
//! owner-only permissions (`0700` on the directory, `0600` on the file) plus a
//! Windows owner-only DACL. That is *not* encryption, an OS keychain, or a
//! Secure Enclave, and the Zig docs said so explicitly — the same boundary is
//! restated here so the claim does not quietly inflate during the port.

use super::{CREDENTIAL_FIELDS, Credentials, Secret};
use crate::errors::AbiError;
use crate::{env, text};
use std::path::{Path, PathBuf};
use zeroize::Zeroize as _;

/// Environment variable that overrides the credential file location.
pub const CREDENTIALS_PATH_ENV: &str = "ABI_CREDENTIALS_PATH";

/// Resolve the credential file path.
///
/// Precedence, matching the Zig implementation:
/// 1. `ABI_CREDENTIALS_PATH`;
/// 2. `$XDG_CONFIG_HOME/abi/credentials.json` (non-Windows only);
/// 3. `$HOME/.abi/credentials.json`, or `%USERPROFILE%` on Windows.
pub fn credentials_path() -> Result<PathBuf, AbiError> {
    if let Some(explicit) = env::get(CREDENTIALS_PATH_ENV) {
        return Ok(PathBuf::from(explicit));
    }

    if !cfg!(windows)
        && let Some(xdg) = env::get("XDG_CONFIG_HOME")
    {
        return Ok(PathBuf::from(text::path_join(&xdg, "abi/credentials.json")));
    }

    let home_var = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    let home = env::get(home_var).ok_or(AbiError::NotFound)?;
    Ok(PathBuf::from(text::path_join(
        &home,
        ".abi/credentials.json",
    )))
}

/// Load credentials from `path`.
///
/// A missing file yields empty credentials rather than an error — that is what
/// "no credentials configured yet" looks like. A file that parses as JSON but is
/// not an object is an error, because it signals corruption rather than absence.
pub fn load_from_path(path: impl AsRef<Path>) -> Result<Credentials, AbiError> {
    let path = path.as_ref();
    if !path.is_file() {
        return Ok(Credentials::default());
    }

    let mut content = std::fs::read_to_string(path)?;
    let parsed = serde_json::from_str::<serde_json::Value>(&content);
    // The raw file bytes hold plaintext secrets; wipe the buffer before it is
    // freed, matching the Zig `secureZero` on the read buffer.
    content.zeroize();

    let value = parsed.map_err(|_| AbiError::ParseError)?;
    let object = value.as_object().ok_or(AbiError::ParseError)?;

    let mut creds = Credentials::default();
    for field in CREDENTIAL_FIELDS {
        if let Some(text) = object.get(field.key).and_then(serde_json::Value::as_str) {
            creds.set(*field, Some(Secret::new(text)));
        }
    }
    Ok(creds)
}

/// Write `creds` to `path` with owner-only permissions.
///
/// Absent fields are omitted from the JSON rather than written as `null`, so a
/// cleared credential leaves no trace of having existed.
pub fn save_to_path(path: impl AsRef<Path>, creds: &Credentials) -> Result<(), AbiError> {
    let path = path.as_ref();

    let mut object = serde_json::Map::new();
    for field in CREDENTIAL_FIELDS {
        if let Some(secret) = creds.get(*field) {
            object.insert(
                field.key.to_string(),
                serde_json::Value::String(secret.expose().to_string()),
            );
        }
    }
    let mut payload = serde_json::to_string(&serde_json::Value::Object(object))?;

    let result = write_restricted(path, payload.as_bytes());
    payload.zeroize();
    result
}

/// Create the credential directory and force owner-only (`0700`) permissions.
///
/// The tightening is unconditional, which is intended for the *credential*
/// directory: an existing `~/.abi` left at `0755` by an older version should be
/// repaired. Do not call this on a directory the framework does not own — see
/// [`ensure_parent_dir`].
pub fn ensure_dir(path: impl AsRef<Path>) -> Result<(), AbiError> {
    let path = path.as_ref();
    std::fs::create_dir_all(path)?;
    set_mode(path, 0o700)?;
    apply_windows_owner_only_acl(path)?;
    Ok(())
}

/// Create the parent directory of a credential file, if missing.
///
/// Permissions are tightened **only when this call creates the directory**. That
/// distinction matters: `ABI_CREDENTIALS_PATH` can point anywhere, and an earlier
/// version of this port chmod'd the parent unconditionally — so pointing it at
/// `/tmp/creds.json` would attempt to `chmod 0700 /tmp`, either failing the save
/// outright or, on a system that permitted it, locking a shared directory nobody
/// asked it to touch. macOS refuses the chmod on the per-user temp directory
/// (`EPERM`), which is how this surfaced.
fn ensure_parent_dir(path: &Path) -> Result<(), AbiError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    if parent.as_os_str().is_empty() || parent.is_dir() {
        // Pre-existing directory: not ours to re-permission.
        return Ok(());
    }
    ensure_dir(parent)
}

/// Write `data` to `path` with `0600` permissions.
///
/// Permissions are set *before* the secret bytes are written, and re-asserted
/// afterwards. This ordering is load-bearing: `create`/truncate preserves the
/// mode of an existing file, so a previously world-readable credential file
/// would otherwise be world-readable for the window in which it holds the new
/// secret.
fn write_restricted(path: &Path, data: &[u8]) -> Result<(), AbiError> {
    use std::io::Write as _;

    ensure_parent_dir(path)?;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }

    let mut file = options.open(path)?;
    // Repair the mode of a pre-existing permissive file before writing secrets.
    set_mode(path, 0o600)?;
    file.write_all(data)?;
    file.sync_all()?;
    drop(file);
    set_mode(path, 0o600)?;
    apply_windows_owner_only_acl(path)?;
    Ok(())
}

/// Create a new owner-only secret file without replacing an existing path.
///
/// Permissions are established before bytes are written. On Windows the same
/// owner-only DACL used by credential persistence is applied; a DACL failure
/// removes the newly created file rather than leaving secret material behind.
pub fn write_owner_only_file_new(path: impl AsRef<Path>, data: &[u8]) -> Result<(), AbiError> {
    use std::io::Write as _;

    let path = path.as_ref();
    ensure_parent_dir(path)?;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    set_mode(path, 0o600)?;
    if let Err(error) = file.write_all(data).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error.into());
    }
    drop(file);
    if let Err(error) = set_mode(path, 0o600).and_then(|()| apply_windows_owner_only_acl(path)) {
        let _ = std::fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) -> Result<(), AbiError> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    Ok(())
}

#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn set_mode(_path: &Path, _mode: u32) -> Result<(), AbiError> {
    // No POSIX mode bits; Windows relies on the DACL below instead.
    Ok(())
}

/// Read the POSIX mode bits of `path`, for tests and `abi auth status`.
#[cfg(unix)]
pub fn mode_of(path: impl AsRef<Path>) -> Result<u32, AbiError> {
    use std::os::unix::fs::PermissionsExt as _;
    Ok(std::fs::metadata(path)?.permissions().mode() & 0o777)
}

/// Always `None` on Windows: there are no POSIX mode bits to report.
///
/// Present so `abi auth status` compiles and runs on Windows without a
/// `cfg` at its own call site. Protection there comes from the DACL in
/// [`super::windows_acl`], which is not expressible as a mode.
#[cfg(not(unix))]
pub fn mode_of(_path: impl AsRef<Path>) -> Result<u32, AbiError> {
    Err(AbiError::UnsupportedOperation)
}

/// Grant full access only to the owner via a Windows DACL.
///
/// Delegates to [`super::windows_acl`], whose real-file apply and read-back
/// path runs on Windows in CI. See that module for the exact DACL contract and
/// runtime evidence.
#[cfg(windows)]
fn apply_windows_owner_only_acl(path: &Path) -> Result<(), AbiError> {
    super::windows_acl::apply(path)
}

/// No-op on non-Windows targets; POSIX mode bits cover this.
///
/// Returns `Result` purely so the two `cfg` arms share one signature — clippy's
/// `unnecessary_wraps` sees only this arm, hence the allow.
#[cfg(not(windows))]
#[allow(clippy::unnecessary_wraps)]
fn apply_windows_owner_only_acl(_path: &Path) -> Result<(), AbiError> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::CredentialField;
    use crate::temp_path;

    fn scratch(name: &str) -> PathBuf {
        temp_path::temp_file_path(name, "json")
    }

    #[test]
    fn missing_file_loads_as_empty_credentials() {
        let path = scratch("abi_credentials_missing");
        let creds = load_from_path(&path).expect("missing file is not an error");
        assert_eq!(creds.len(), 0);
        for field in CREDENTIAL_FIELDS {
            assert!(creds.get(*field).is_none(), "{} should be unset", field.key);
        }
    }

    #[test]
    fn save_then_load_roundtrips_every_field() {
        let path = scratch("abi_credentials_roundtrip");

        let mut creds = Credentials::default();
        creds.set(
            CredentialField::OPENAI_API_KEY,
            Some(Secret::new("sk-test-key")),
        );
        creds.set(
            CredentialField::DISCORD_TOKEN,
            Some(Secret::new("token-123")),
        );
        creds.set(
            CredentialField::TWILIO_ACCOUNT_SID,
            Some(Secret::new("AC123")),
        );
        creds.set(
            CredentialField::TWILIO_AUTH_TOKEN,
            Some(Secret::new("twilio-secret")),
        );

        save_to_path(&path, &creds).expect("save");
        let loaded = load_from_path(&path).expect("load");

        assert_eq!(
            loaded
                .get(CredentialField::OPENAI_API_KEY)
                .map(Secret::expose),
            Some("sk-test-key")
        );
        assert_eq!(
            loaded
                .get(CredentialField::DISCORD_TOKEN)
                .map(Secret::expose),
            Some("token-123")
        );
        assert_eq!(
            loaded
                .get(CredentialField::TWILIO_ACCOUNT_SID)
                .map(Secret::expose),
            Some("AC123")
        );
        assert_eq!(
            loaded
                .get(CredentialField::TWILIO_AUTH_TOKEN)
                .map(Secret::expose),
            Some("twilio-secret")
        );
        // Unset fields stay unset.
        assert!(loaded.get(CredentialField::ANTHROPIC_API_KEY).is_none());
        assert!(loaded.get(CredentialField::GROK_API_KEY).is_none());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn absent_fields_are_omitted_not_nulled() {
        let path = scratch("abi_credentials_omit");
        let mut creds = Credentials::default();
        creds.set(CredentialField::GROK_API_KEY, Some(Secret::new("xai-key")));
        save_to_path(&path, &creds).expect("save");

        let raw = std::fs::read_to_string(&path).expect("read raw");
        assert!(raw.contains("grok_api_key"));
        assert!(!raw.contains("openai_api_key"), "raw was {raw}");
        assert!(!raw.contains("null"), "raw was {raw}");

        std::fs::remove_file(&path).ok();
    }

    #[cfg(unix)]
    #[test]
    fn save_tightens_an_existing_permissive_file() {
        use std::os::unix::fs::PermissionsExt as _;

        let path = scratch("abi_credentials_perms");
        std::fs::write(&path, "{}").expect("seed file");
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666))
            .expect("make permissive");

        let mut creds = Credentials::default();
        creds.set(
            CredentialField::ANTHROPIC_API_KEY,
            Some(Secret::new("anthropic-secret")),
        );
        save_to_path(&path, &creds).expect("save");

        assert_eq!(mode_of(&path).expect("mode"), 0o600);
        std::fs::remove_file(&path).ok();
    }

    #[cfg(unix)]
    #[test]
    fn credential_directory_is_owner_only() {
        let dir = temp_path::temp_file_path("abi_credentials_dir_perms", "d");
        ensure_dir(&dir).expect("ensure_dir");
        assert_eq!(mode_of(&dir).expect("mode"), 0o700);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(unix)]
    #[test]
    fn save_creates_a_locked_down_parent_directory() {
        let root = temp_path::temp_file_path("abi_credentials_parent", "d");
        let path = root.join("credentials.json");
        let mut creds = Credentials::default();
        creds.set(CredentialField::OPENAI_API_KEY, Some(Secret::new("sk-x")));
        save_to_path(&path, &creds).expect("save into new dir");

        assert_eq!(mode_of(&root).expect("dir mode"), 0o700);
        assert_eq!(mode_of(&path).expect("file mode"), 0o600);
        std::fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn save_does_not_repermission_a_pre_existing_parent() {
        // Regression: an earlier version of this port chmod'd the parent
        // unconditionally, so a credential path inside a shared directory tried
        // to `chmod 0700` that directory. macOS denies it for the per-user temp
        // dir, which turned every save into PermissionDenied.
        let shared = temp_path::temp_file_path("abi_credentials_shared_parent", "d");
        std::fs::create_dir_all(&shared).expect("create shared dir");
        set_mode(&shared, 0o755).expect("make group-readable");

        let path = shared.join("credentials.json");
        let mut creds = Credentials::default();
        creds.set(CredentialField::OPENAI_API_KEY, Some(Secret::new("sk-y")));
        save_to_path(&path, &creds).expect("save into a pre-existing dir");

        assert_eq!(
            mode_of(&shared).expect("dir mode"),
            0o755,
            "a pre-existing parent must keep its permissions"
        );
        // The file itself is still locked down.
        assert_eq!(mode_of(&path).expect("file mode"), 0o600);

        std::fs::remove_dir_all(&shared).ok();
    }

    #[test]
    fn non_object_json_is_a_parse_error() {
        let path = scratch("abi_credentials_invalid_shape");
        std::fs::write(&path, "[]").expect("write");
        assert_eq!(load_from_path(&path).unwrap_err(), AbiError::ParseError);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn malformed_json_is_a_parse_error() {
        let path = scratch("abi_credentials_malformed");
        std::fs::write(&path, "{not json").expect("write");
        assert_eq!(load_from_path(&path).unwrap_err(), AbiError::ParseError);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn non_string_field_values_are_ignored() {
        let path = scratch("abi_credentials_wrong_type");
        std::fs::write(&path, r#"{"openai_api_key": 42, "grok_api_key": "ok"}"#).expect("write");
        let loaded = load_from_path(&path).expect("load");
        assert!(loaded.get(CredentialField::OPENAI_API_KEY).is_none());
        assert_eq!(
            loaded
                .get(CredentialField::GROK_API_KEY)
                .map(Secret::expose),
            Some("ok")
        );
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn explicit_path_env_wins() {
        let _guard = env::lock_for_test();
        env::set_override(CREDENTIALS_PATH_ENV, "/explicit/creds.json");
        assert_eq!(
            credentials_path().expect("path"),
            PathBuf::from("/explicit/creds.json")
        );
        env::clear_override(CREDENTIALS_PATH_ENV);
    }

    #[cfg(not(windows))]
    #[test]
    fn xdg_config_home_is_used_when_no_explicit_path() {
        let _guard = env::lock_for_test();
        env::set_override(CREDENTIALS_PATH_ENV, "");
        env::set_override("XDG_CONFIG_HOME", "/xdg/config");
        assert_eq!(
            credentials_path().expect("path"),
            PathBuf::from("/xdg/config/abi/credentials.json")
        );
        env::clear_override("XDG_CONFIG_HOME");
        env::clear_override(CREDENTIALS_PATH_ENV);
    }

    #[cfg(not(windows))]
    #[test]
    fn home_is_the_final_fallback() {
        let _guard = env::lock_for_test();
        env::set_override(CREDENTIALS_PATH_ENV, "");
        env::set_override("XDG_CONFIG_HOME", "");
        env::set_override("HOME", "/home/tester");
        assert_eq!(
            credentials_path().expect("path"),
            PathBuf::from("/home/tester/.abi/credentials.json")
        );
        env::clear_override("HOME");
        env::clear_override("XDG_CONFIG_HOME");
        env::clear_override(CREDENTIALS_PATH_ENV);
    }
}
