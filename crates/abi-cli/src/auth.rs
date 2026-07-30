//! Local credential-status and logout command handlers.

use abi_foundation::credentials::{
    self, Backend, CredentialField, backend_is_keychain, credentials_path,
};

use crate::app::Outcome;

const AUTH_USAGE: &str = "usage: abi auth <signin|logout|status>";
const STATUS_USAGE: &str = "usage: abi auth status";
const LOGOUT_USAGE: &str = "usage: abi auth logout";
const SIGNIN_USAGE: &str = "usage: abi auth signin <openai|anthropic|discord|grok|twilio>";

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
            Outcome::stderr(
                "error: Rust interactive `auth signin` is not yet ported\n".to_owned(),
                1,
            )
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
    use std::sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    };

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

    struct CredentialEnvironment {
        path: std::path::PathBuf,
    }

    impl CredentialEnvironment {
        fn new() -> Self {
            let suffix = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("abi-cli-auth-{}-{suffix}.json", std::process::id()));
            let _ = std::fs::remove_file(&path);
            env::set_override(
                CREDENTIALS_PATH_ENV,
                path.to_str().expect("UTF-8 test path"),
            );
            env::set_override(BACKEND_ENV, "file");
            Self { path }
        }
    }

    impl Drop for CredentialEnvironment {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
            env::clear_override(CREDENTIALS_PATH_ENV);
            env::clear_override(BACKEND_ENV);
        }
    }

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn status_matches_the_zig_field_order_and_twilio_pair_rule() {
        let _guard = TEST_LOCK.lock().expect("test lock");
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
        let _guard = TEST_LOCK.lock().expect("test lock");
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
        let _guard = TEST_LOCK.lock().expect("test lock");
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
    fn grammar_errors_are_usage_errors_and_signin_is_explicitly_deferred() {
        assert_eq!(run(&[]).exit_code, 2);
        assert_eq!(run(&args(&["status", "extra"])).exit_code, 2);
        assert_eq!(run(&args(&["signin", "unknown"])).exit_code, 2);
        let signin = run(&args(&["signin", "openai"]));
        assert_eq!(signin.exit_code, 1);
        assert!(signin.stderr.contains("not yet ported"));
    }
}
