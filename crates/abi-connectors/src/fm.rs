//! Apple `FoundationModels` on-device connector.
//!
//! Ported from `src/connectors/fm.zig`. On arm64 macOS with the
//! `foundationmodels` feature, links `libabi_fm_shim.dylib` (built from
//! `native/fm_shim.swift`) and drives real on-device inference. Everywhere
//! else, [`fm_available`] is `false` and [`complete_live`] returns
//! [`ConnectorError::FmUnavailable`] — never a fabricated reply.

#![cfg_attr(
    all(
        feature = "foundationmodels",
        target_os = "macos",
        target_arch = "aarch64"
    ),
    allow(unsafe_code)
)]

use crate::connector::{ConnectorError, Response, Result};

/// Configuration for the on-device `FoundationModels` client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FmConfig {
    /// Wall-clock budget hint (milliseconds). The Swift shim currently
    /// blocks until the session returns; reserved for future cancel wiring.
    pub timeout_ms: u32,
    /// Maximum tokens budget hint (~4 UTF-8 bytes/token for the out buffer).
    pub max_tokens: u32,
}

impl Default for FmConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 15_000,
            max_tokens: 512,
        }
    }
}

/// Default on-device config.
#[must_use]
pub fn fm_config() -> FmConfig {
    FmConfig::default()
}

/// Whether this binary was built with the `FoundationModels` Swift bridge
/// (feature + arm64 macOS). Independent of runtime model readiness.
#[must_use]
pub const fn fm_bridge_linked() -> bool {
    cfg!(all(
        feature = "foundationmodels",
        target_os = "macos",
        target_arch = "aarch64"
    ))
}

#[cfg(all(
    feature = "foundationmodels",
    target_os = "macos",
    target_arch = "aarch64"
))]
mod shim {
    use std::os::raw::c_char;

    unsafe extern "C" {
        pub fn abi_fm_available() -> i32;
        pub fn abi_fm_complete(prompt: *const c_char, out: *mut c_char, out_len: usize) -> i32;
    }
}

/// Probe whether the Apple `FoundationModels` runtime is ready.
///
/// Returns `true` only when the Swift bridge is linked **and**
/// `SystemLanguageModel.default.availability == .available`.
#[must_use]
pub fn fm_available() -> bool {
    #[cfg(all(
        feature = "foundationmodels",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: C ABI from libabi_fm_shim; no pointer args.
        unsafe { shim::abi_fm_available() == 1 }
    }
    #[cfg(not(all(
        feature = "foundationmodels",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    {
        false
    }
}

/// On-device completion against Apple `FoundationModels`.
///
/// Returns [`ConnectorError::FmUnavailable`] when the bridge is not linked or
/// the model is not ready; [`ConnectorError::FmSessionFailed`] when generation
/// throws. Never fabricates text.
pub fn complete_live(prompt: &str, config: FmConfig) -> Result<Response> {
    #[cfg(all(
        feature = "foundationmodels",
        target_os = "macos",
        target_arch = "aarch64"
    ))]
    {
        if !fm_available() {
            return Err(ConnectorError::FmUnavailable);
        }
        let cap = usize::try_from(config.max_tokens)
            .unwrap_or(512)
            .saturating_mul(4)
            .max(8 * 1024);
        let mut out = vec![0_u8; cap];
        let prompt_z =
            std::ffi::CString::new(prompt).map_err(|_| ConnectorError::InvalidResponse)?;

        // SAFETY: prompt is NUL-terminated CString; out has length `cap`.
        let rc = unsafe {
            shim::abi_fm_complete(
                prompt_z.as_ptr(),
                out.as_mut_ptr().cast::<std::os::raw::c_char>(),
                out.len(),
            )
        };
        if rc < 0 {
            return Err(match rc {
                -4 => ConnectorError::FmSessionFailed,
                _ => ConnectorError::FmUnavailable,
            });
        }
        let n = usize::try_from(rc).unwrap_or(0).min(out.len());
        out.truncate(n);
        // Drop trailing NUL if the shim wrote one inside the counted range.
        if out.last() == Some(&0) {
            out.pop();
        }
        Ok(Response {
            status: 200,
            body: String::from_utf8_lossy(&out).into_owned(),
        })
    }
    #[cfg(not(all(
        feature = "foundationmodels",
        target_os = "macos",
        target_arch = "aarch64"
    )))]
    {
        let _ = (prompt, config);
        Err(ConnectorError::FmUnavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults() {
        let cfg = fm_config();
        assert_eq!(cfg.timeout_ms, 15_000);
        assert_eq!(cfg.max_tokens, 512);
    }

    #[test]
    fn available_is_bool_and_complete_never_fabricates() {
        let reachable = fm_available();
        let _ = reachable;
        if fm_bridge_linked() && reachable {
            // Live model: accept success or session failure only.
            match complete_live("Reply with the single word: ok", fm_config()) {
                Ok(resp) => {
                    assert_eq!(resp.status, 200);
                    assert!(!resp.body.is_empty());
                }
                Err(ConnectorError::FmSessionFailed) => {}
                Err(other) => panic!("unexpected error with ready model: {other}"),
            }
        } else {
            assert!(matches!(
                complete_live("hello on-device", fm_config()),
                Err(ConnectorError::FmUnavailable)
            ));
        }
    }
}
