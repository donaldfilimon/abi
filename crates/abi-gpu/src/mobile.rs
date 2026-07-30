//! Mobile platform profile (simulated on desktop; native dispatch never claimed).
//!
//! Ported from the claim-honest subset of `src/features/mobile/mod.zig`.

use crate::detect_backend;

/// Declared mobile platforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// iOS.
    Ios,
    /// Android.
    Android,
    /// Desktop / unknown (simulated profile).
    Unknown,
}

impl Platform {
    /// Stable platform name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ios => "ios",
            Self::Android => "android",
            Self::Unknown => "unknown",
        }
    }
}

/// Runtime mode for the mobile feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    /// Native mobile OS (not claimed on desktop builds).
    NativePlatform,
    /// Simulated profile on desktop hosts.
    SimulatedProfile,
    /// Explicitly disabled.
    Disabled,
}

impl RuntimeMode {
    /// Stable mode name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::NativePlatform => "native_platform",
            Self::SimulatedProfile => "simulated_profile",
            Self::Disabled => "disabled",
        }
    }
}

/// Platform detection status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformStatus {
    /// Detected platform.
    pub platform: Platform,
    /// Feature available for reporting.
    pub available: bool,
    /// Whether accelerated native mobile dispatch is active (always false).
    pub accelerated: bool,
    /// Operator message.
    pub message: &'static str,
}

/// Device profile used by diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceProfile {
    /// Detected platform.
    pub platform: Platform,
    /// Runtime mode.
    pub mode: RuntimeMode,
    /// Native mobile dispatch (always false in this port).
    pub native_dispatch: bool,
    /// Whether this is a simulated desktop profile.
    pub simulated: bool,
    /// GPU accelerated flag from host GPU detection (does not imply mobile native).
    pub accelerated: bool,
    /// Simulated screen width.
    pub width: u32,
    /// Simulated screen height.
    pub height: u32,
    /// Simulated screen density.
    pub density: f32,
}

/// Detect host platform for the mobile feature.
#[must_use]
pub fn detect_platform() -> PlatformStatus {
    // Rust desktop/macOS/linux/windows targets use the simulated profile.
    // Real iOS/Android targets would flip here if cross-compiled.
    if cfg!(target_os = "ios") {
        return PlatformStatus {
            platform: Platform::Ios,
            available: true,
            accelerated: false,
            message: "iOS platform detected; mobile feature active, native dispatch pending",
        };
    }
    if cfg!(target_os = "android") {
        return PlatformStatus {
            platform: Platform::Android,
            available: true,
            accelerated: false,
            message: "Android platform detected; mobile feature active, native dispatch pending",
        };
    }
    // Desktop mobile profile is always simulated — do not inherit GPU kernel
    // acceleration into the mobile surface (native_dispatch remains false).
    let _ = detect_backend();
    PlatformStatus {
        platform: Platform::Unknown,
        available: true,
        accelerated: false,
        message: "Desktop platform; mobile feature active with simulated mobile profile",
    }
}

/// Simulated / host device profile for diagnostics.
#[must_use]
pub fn device_profile() -> DeviceProfile {
    let status = detect_platform();
    let (mode, simulated, width, height, density) = match status.platform {
        Platform::Ios | Platform::Android => (RuntimeMode::NativePlatform, false, 390, 844, 3.0),
        Platform::Unknown => (RuntimeMode::SimulatedProfile, true, 390, 844, 2.0),
    };
    DeviceProfile {
        platform: status.platform,
        mode,
        native_dispatch: false,
        simulated,
        accelerated: status.accelerated,
        width,
        height,
        density,
    }
}

/// One-line status for `abi backends`.
#[must_use]
pub fn status_line() -> String {
    let profile = device_profile();
    format!(
        "platform={} mode={} native_dispatch=false simulated={} accelerated={} screen={}x{}@{:.1}",
        profile.platform.name(),
        profile.mode.name(),
        profile.simulated,
        profile.accelerated,
        profile.width,
        profile.height,
        profile.density,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desktop_is_simulated_without_native_dispatch() {
        let profile = device_profile();
        assert!(!profile.native_dispatch);
        assert!(!profile.accelerated);
        if !cfg!(any(target_os = "ios", target_os = "android")) {
            assert_eq!(profile.platform, Platform::Unknown);
            assert!(profile.simulated);
            assert_eq!(profile.mode, RuntimeMode::SimulatedProfile);
        }
        let line = status_line();
        assert!(line.contains("native_dispatch=false"));
    }
}
