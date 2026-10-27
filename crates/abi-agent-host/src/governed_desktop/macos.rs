//! macOS desktop backend over the pinned system tools `lsappinfo` and `open`.
//!
//! Each tool runs from its absolute path with an empty environment, no shell,
//! null stdin and a bounded wait; a process that outlives the bound is killed.
use super::{DesktopBackend, DesktopError, valid_bundle_id};
use std::{
    io::{ErrorKind, Read},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const LSAPPINFO: &str = "/usr/bin/lsappinfo";
const OPEN: &str = "/usr/bin/open";
const TIMEOUT: Duration = Duration::from_secs(5);
const MAX_OUTPUT_BYTES: u64 = 16 * 1024;

/// Desktop backend for the local macOS session.
#[derive(Debug, Default)]
pub struct MacDesktop;

impl MacDesktop {
    /// A backend for the current user's macOS desktop session.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl DesktopBackend for MacDesktop {
    fn frontmost_bundle_id(&mut self) -> Result<String, DesktopError> {
        let front = run(LSAPPINFO, &["front"])?;
        let asn = front.trim();
        if !valid_asn(asn) {
            return Err(DesktopError::Failed);
        }
        parse_bundle_id(&run(LSAPPINFO, &["info", "-only", "bundleid", asn])?)
            .ok_or(DesktopError::Failed)
    }
    fn activate(&mut self, bundle_id: &str) -> Result<(), DesktopError> {
        if !valid_bundle_id(bundle_id) {
            return Err(DesktopError::Failed);
        }
        run(OPEN, &["-b", bundle_id]).map(drop)
    }
}

/// Run one pinned executable to completion within [`TIMEOUT`]; stdout on exit 0.
fn run(program: &str, args: &[&str]) -> Result<String, DesktopError> {
    let mut child = Command::new(program)
        .args(args)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|err| match err.kind() {
            ErrorKind::PermissionDenied => DesktopError::PermissionDenied,
            ErrorKind::NotFound => DesktopError::Unsupported,
            _ => DesktopError::Failed,
        })?;
    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(5)),
            _ => {
                // Timed out or unobservable: kill and reap, then fail closed.
                let _ = child.kill();
                let _ = child.wait();
                return Err(DesktopError::Failed);
            }
        }
    };
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .ok_or(DesktopError::Failed)?
        .take(MAX_OUTPUT_BYTES)
        .read_to_string(&mut stdout)
        .map_err(|_| DesktopError::Failed)?;
    if status.success() {
        Ok(stdout)
    } else {
        Err(DesktopError::Failed)
    }
}

/// `ASN:0x<hex>-0x<hex>:` as printed by `lsappinfo front`.
fn valid_asn(value: &str) -> bool {
    let Some(body) = value
        .strip_prefix("ASN:")
        .and_then(|rest| rest.strip_suffix(':'))
    else {
        return false;
    };
    let mut parts = body.split('-');
    let hex = |part: Option<&str>| {
        part.and_then(|p| p.strip_prefix("0x"))
            .is_some_and(|digits| {
                (1..=16).contains(&digits.len()) && digits.bytes().all(|b| b.is_ascii_hexdigit())
            })
    };
    hex(parts.next()) && hex(parts.next()) && parts.next().is_none()
}

/// Exactly one `bundleID="…"` (or older `"CFBundleIdentifier"="…"`) line.
fn parse_bundle_id(info: &str) -> Option<String> {
    let mut found = info.lines().filter_map(|line| {
        let line = line.trim();
        line.strip_prefix("bundleID=\"")
            .or_else(|| line.strip_prefix("\"CFBundleIdentifier\"=\""))
            .and_then(|rest| rest.strip_suffix('"'))
    });
    let bundle = found.next()?;
    (found.next().is_none() && valid_bundle_id(bundle)).then(|| bundle.to_owned())
}

#[cfg(test)]
mod tests {
    use super::{parse_bundle_id, valid_asn};

    #[test]
    fn lsappinfo_output_parsing_is_closed() {
        assert!(valid_asn("ASN:0x0-0x6d30d2a:"));
        for asn in ["", "ASN:0x0-0x6d30d2a", "ASN:0x0:", "ASN:0x0-0xzz:", "-x"] {
            assert!(!valid_asn(asn), "{asn}");
        }
        let current = "[ NULL ]  ASN:0x0-0x1: (in front)\n    bundleID=\"com.apple.finder\"\n";
        assert_eq!(
            parse_bundle_id(current).as_deref(),
            Some("com.apple.finder")
        );
        let older = "\"CFBundleIdentifier\"=\"com.apple.Terminal\"\n";
        assert_eq!(
            parse_bundle_id(older).as_deref(),
            Some("com.apple.Terminal")
        );
        for info in [
            "",
            "bundleID=[ NULL ]",
            "bundleID=\"bad id\"",
            "bundleID=\"a.b\"\nbundleID=\"c.d\"",
        ] {
            assert_eq!(parse_bundle_id(info), None, "{info}");
        }
    }
}
