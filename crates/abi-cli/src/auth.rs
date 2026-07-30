//! Local credential status, logout, and signin handlers.

use std::io::{self, BufRead, Write};

use abi_foundation::credentials::{
    self, Backend, CredentialField, Secret, backend_is_keychain, credentials_path,
};

use crate::app::Outcome;

const AUTH_USAGE: &str = "usage: abi auth <signin|logout|status>";
const STATUS_USAGE: &str = "usage: abi auth status";
const LOGOUT_USAGE: &str = "usage: abi auth logout";
const SIGNIN_USAGE: &str = "usage: abi auth signin <openai|anthropic|discord|grok|twilio>\n       (non-interactive: ABI_AUTH_TOKEN=<secret> abi auth signin <provider>)";

fn usage(text: &str) -> Outcome {
    Outcome::stderr(format!("error: {text}\n"), 2)
}

fn failure(operation: &str, error: impl std::fmt::Display) -> Outcome {
    Outcome::stderr(format!("error: auth {operation} failed: {error}\n"), 1)
}

fn backend_label() -> &'static str {
    if backend_is_keychain() {
        if cfg!(target_os = "macos") {
            "keychain (macOS login keychain, opt-in)"
        } else {
            "keychain requested (unsupported on this OS; using file — Windows/Linux Proposed)"
        }
    } else {
        "file (~/.abi/credentials.json)"
    }
}

fn status() -> Outcome {
    let prefix = format!("Authentication Status:\n  Backend:   {}\n", backend_label());
    let credentials = match credentials::load() {
        Ok(credentials) => credentials,
        Err(error) => {
            return Outcome::stderr(format!("{prefix}error: auth status failed: {error}\n"), 1);
        }
    };
    let configured = |field| {
        if credentials.get(field).is_some() {
            "configured"
        } else {
            "not configured"
        }
    };
    let twilio = if credentials
        .get(CredentialField::TWILIO_ACCOUNT_SID)
        .is_some()
        && credentials
            .get(CredentialField::TWILIO_AUTH_TOKEN)
            .is_some()
    {
        "configured"
    } else {
        "not configured"
    };

    Outcome::stderr(
        format!(
            "{prefix}  OpenAI:    {}\n  Anthropic: {}\n  Discord:   {}\n  Grok:      {}\n  Twilio:    {twilio}\n",
            configured(CredentialField::OPENAI_API_KEY),
            configured(CredentialField::ANTHROPIC_API_KEY),
            configured(CredentialField::DISCORD_TOKEN),
            configured(CredentialField::GROK_API_KEY),
        ),
        0,
    )
}

fn logout() -> Outcome {
    let path = match credentials_path() {
        Ok(path) => path,
        Err(error) => return failure("logout", error),
    };
    let mut cleared_something = false;
    match std::fs::remove_file(&path) {
        Ok(()) => cleared_something = true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return failure("logout", error),
    }

    if credentials::active_backend() == Backend::Keychain {
        if let Err(error) = credentials::clear_keychain() {
            return failure("logout", error);
        }
        cleared_something = true;
    }

    Outcome::stderr(
        if cleared_something {
            "Logged out. Credentials cleared.\n"
        } else {
            "No credentials found.\n"
        }
        .to_owned(),
        0,
    )
}

fn field_for_provider(provider: &str) -> Option<CredentialField> {
    match provider {
        "openai" => Some(CredentialField::OPENAI_API_KEY),
        "anthropic" => Some(CredentialField::ANTHROPIC_API_KEY),
        "discord" => Some(CredentialField::DISCORD_TOKEN),
        "grok" => Some(CredentialField::GROK_API_KEY),
        _ => None,
    }
}

/// Read a secret line. Prefer `ABI_AUTH_TOKEN` for non-interactive use.
///
/// On a TTY (Unix), echo is disabled for the duration of the read. Non-TTY
/// stdin reads one line (for pipes). Empty input is rejected.
fn read_secret(prompt: &str) -> Result<String, String> {
    // Prefer foundation env overlay so tests can isolate with set_override.
    if let Some(token) = abi_foundation::env::get("ABI_AUTH_TOKEN") {
        let trimmed = token.trim().to_string();
        if trimmed.is_empty() {
            return Err("ABI_AUTH_TOKEN is empty".into());
        }
        return Ok(trimmed);
    }

    let mut stderr = io::stderr();
    let _ = write!(stderr, "{prompt}");
    let _ = stderr.flush();

    #[cfg(unix)]
    {
        use std::io::IsTerminal;
        if io::stdin().is_terminal() {
            return crate::terminal::read_secret_line();
        }
    }

    let mut line = String::new();
    io::stdin()
        .lock()
        .read_line(&mut line)
        .map_err(|e| format!("failed to read secret: {e}"))?;
    let trimmed = line.trim().to_string();
    if trimmed.is_empty() {
        return Err("empty secret".into());
    }
    Ok(trimmed)
}

fn signin(provider: &str) -> Outcome {
    if provider == "twilio" {
        return signin_twilio();
    }
    let Some(field) = field_for_provider(provider) else {
        return usage(SIGNIN_USAGE);
    };
    let secret = match read_secret(&format!("Enter {provider} credential: ")) {
        Ok(s) => s,
        Err(err) => {
            return Outcome::stderr(format!("error: auth signin failed: {err}\n"), 1);
        }
    };
    let mut creds = match credentials::load() {
        Ok(c) => c,
        Err(err) => return failure("signin", err),
    };
    creds.set(field, Some(Secret::new(secret)));
    if let Err(err) = credentials::save(&creds) {
        return failure("signin", err);
    }
    Outcome::stderr(
        format!(
            "Signed in to {provider}. Credential stored ({})\n",
            backend_label()
        ),
        0,
    )
}

fn signin_twilio() -> Outcome {
    // Twilio needs SID + auth token. Non-interactive: ABI_AUTH_TOKEN="sid:token"
    // or ABI_TWILIO_ACCOUNT_SID + ABI_AUTH_TOKEN.
    let (sid, token) = if let Some(blob) = abi_foundation::env::get("ABI_AUTH_TOKEN") {
        if let Some((sid, token)) = blob.split_once(':') {
            let sid = sid.trim().to_string();
            let token = token.trim().to_string();
            if sid.is_empty() || token.is_empty() {
                return Outcome::stderr(
                    "error: auth signin failed: ABI_AUTH_TOKEN for twilio must be 'sid:token'\n"
                        .into(),
                    1,
                );
            }
            (sid, token)
        } else if let Some(sid) = abi_foundation::env::get("ABI_TWILIO_ACCOUNT_SID") {
            (sid.trim().to_string(), blob.trim().to_string())
        } else {
            return Outcome::stderr(
                "error: auth signin failed: twilio needs ABI_AUTH_TOKEN as 'sid:token' or ABI_TWILIO_ACCOUNT_SID + ABI_AUTH_TOKEN\n".into(),
                1,
            );
        }
    } else {
        let sid = match read_secret("Enter Twilio account SID: ") {
            Ok(s) => s,
            Err(err) => {
                return Outcome::stderr(format!("error: auth signin failed: {err}\n"), 1);
            }
        };
        let token = match read_secret("Enter Twilio auth token: ") {
            Ok(s) => s,
            Err(err) => {
                return Outcome::stderr(format!("error: auth signin failed: {err}\n"), 1);
            }
        };
        (sid, token)
    };

    let mut creds = match credentials::load() {
        Ok(c) => c,
        Err(err) => return failure("signin", err),
    };
    creds.set(CredentialField::TWILIO_ACCOUNT_SID, Some(Secret::new(sid)));
    creds.set(CredentialField::TWILIO_AUTH_TOKEN, Some(Secret::new(token)));
    if let Err(err) = credentials::save(&creds) {
        return failure("signin", err);
    }
    Outcome::stderr(
        format!(
            "Signed in to twilio. Credentials stored ({})\n",
            backend_label()
        ),
        0,
    )
}

/// Dispatch `abi auth`, excluding the top-level command token.
pub(crate) fn run(args: &[String]) -> Outcome {
    match args {
        [command] if command == "status" => status(),
        [command] if command == "logout" => logout(),
        [command, ..] if command == "status" => usage(STATUS_USAGE),
        [command, ..] if command == "logout" => usage(LOGOUT_USAGE),
        [command, service]
            if command == "signin"
                && matches!(
                    service.as_str(),
                    "openai" | "anthropic" | "discord" | "grok" | "twilio"
                ) =>
        {
            signin(service)
        }
        [command, ..] if command == "signin" => usage(SIGNIN_USAGE),
        _ => usage(AUTH_USAGE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi_foundation::{
        credentials::{BACKEND_ENV, CREDENTIALS_PATH_ENV, Credentials, Secret},
        env,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    struct CredentialEnvironment {
        path: std::path::PathBuf,
        /// Shared with any other test that mutates credential env overrides.
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl CredentialEnvironment {
        fn new() -> Self {
            // `env::lock_for_test` serializes against complete --live and other
            // credential-path override users across modules.
            let lock = env::lock_for_test();
            let suffix = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("abi-cli-auth-{}-{suffix}.json", std::process::id()));
            let _ = std::fs::remove_file(&path);
            env::set_override(
                CREDENTIALS_PATH_ENV,
                path.to_str().expect("UTF-8 test path"),
            );
            env::set_override(BACKEND_ENV, "file");
            // Never inherit a caller's ABI_AUTH_TOKEN into auth unit tests.
            env::set_override("ABI_AUTH_TOKEN", "");
            Self { path, _lock: lock }
        }
    }

    impl Drop for CredentialEnvironment {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            env::clear_override(CREDENTIALS_PATH_ENV);
            env::clear_override(BACKEND_ENV);
            env::clear_override("ABI_AUTH_TOKEN");
        }
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn status_matches_the_zig_field_order_and_twilio_pair_rule() {
        let _environment = CredentialEnvironment::new();
        let mut credentials = Credentials::new();
        credentials.set(CredentialField::OPENAI_API_KEY, Some(Secret::new("openai")));
        credentials.set(
            CredentialField::TWILIO_ACCOUNT_SID,
            Some(Secret::new("sid-only")),
        );
        credentials::save(&credentials).expect("save credentials");

        assert_eq!(
            run(&args(&["status"])),
            Outcome::stderr(
                "Authentication Status:\n  Backend:   file (~/.abi/credentials.json)\n  OpenAI:    configured\n  Anthropic: not configured\n  Discord:   not configured\n  Grok:      not configured\n  Twilio:    not configured\n"
                    .to_owned(),
                0,
            )
        );
    }

    #[test]
    fn logout_reports_absence_then_removes_the_file() {
        let environment = CredentialEnvironment::new();
        assert_eq!(run(&args(&["logout"])).stderr, "No credentials found.\n");
        std::fs::write(&environment.path, "{}").expect("credential fixture");
        assert_eq!(
            run(&args(&["logout"])).stderr,
            "Logged out. Credentials cleared.\n"
        );
        assert!(!environment.path.exists());
    }

    #[test]
    fn status_prints_backend_context_before_a_load_failure() {
        let environment = CredentialEnvironment::new();
        std::fs::write(&environment.path, "{").expect("malformed credential fixture");

        let outcome = run(&args(&["status"]));
        assert_eq!(outcome.exit_code, 1);
        assert!(outcome.stdout.is_empty());
        assert!(outcome.stderr.starts_with(
            "Authentication Status:\n  Backend:   file (~/.abi/credentials.json)\nerror:"
        ));
    }

    #[test]
    fn grammar_errors_are_usage_errors() {
        assert_eq!(run(&[]).exit_code, 2);
        assert_eq!(run(&args(&["status", "extra"])).exit_code, 2);
        assert_eq!(run(&args(&["signin", "unknown"])).exit_code, 2);
    }

    #[test]
    fn signin_via_env_token_stores_credential() {
        let environment = CredentialEnvironment::new();
        env::set_override("ABI_AUTH_TOKEN", "sk-test-anthropic-key");
        let outcome = run(&args(&["signin", "anthropic"]));
        env::clear_override("ABI_AUTH_TOKEN");
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stderr.contains("Signed in to anthropic"));
        let loaded = credentials::load().expect("load");
        assert!(loaded.get(CredentialField::ANTHROPIC_API_KEY).is_some());
        assert!(environment.path.is_file() || credentials_path().is_ok());
    }

    #[test]
    fn signin_twilio_via_sid_token_blob() {
        let _environment = CredentialEnvironment::new();
        env::set_override("ABI_AUTH_TOKEN", "ACsid123:authtoken456");
        let outcome = run(&args(&["signin", "twilio"]));
        env::clear_override("ABI_AUTH_TOKEN");
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        let loaded = credentials::load().expect("load");
        assert_eq!(
            loaded
                .get(CredentialField::TWILIO_ACCOUNT_SID)
                .map(Secret::expose),
            Some("ACsid123")
        );
        assert_eq!(
            loaded
                .get(CredentialField::TWILIO_AUTH_TOKEN)
                .map(Secret::expose),
            Some("authtoken456")
        );
    }
}
