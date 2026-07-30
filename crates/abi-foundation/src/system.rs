//! Host and process introspection.
//!
//! Ported from `src/foundation/os.zig`. The Zig `OSController` bundled host facts
//! (pid, platform, arch, hostname, CPU count) together with generic file
//! operations, because both needed the same `allocator` + `std.Io` pair threaded
//! through them. Rust needs neither, so the file operations stay in
//! [`crate::io`] and this module is just the host facts — as free functions,
//! since there is no state to carry.

use crate::env;

/// The host operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Platform {
    /// macOS.
    MacOs,
    /// Linux.
    Linux,
    /// Windows.
    Windows,
    /// FreeBSD.
    FreeBsd,
    /// Something else.
    Unknown,
}

impl Platform {
    /// The lowercase name used in `abi backends` and `gpu_status` output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MacOs => "macos",
            Self::Linux => "linux",
            Self::Windows => "windows",
            Self::FreeBsd => "freebsd",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for Platform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The host CPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    /// 64-bit x86.
    X86_64,
    /// 64-bit ARM.
    Aarch64,
    /// 32-bit ARM.
    Arm,
    /// 64-bit RISC-V.
    Riscv64,
    /// 32-bit WebAssembly.
    Wasm32,
    /// Something else.
    Unknown,
}

impl Arch {
    /// The lowercase name used in diagnostics output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::X86_64 => "x86_64",
            Self::Aarch64 => "aarch64",
            Self::Arm => "arm",
            Self::Riscv64 => "riscv64",
            Self::Wasm32 => "wasm32",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A snapshot of host facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemInfo {
    /// The host operating system.
    pub platform: Platform,
    /// The host CPU architecture.
    pub arch: Arch,
    /// The host name, or `"unknown"` if it could not be determined.
    pub hostname: String,
    /// Available parallelism, at least 1.
    pub cpu_count: u32,
}

/// Resource ceilings a subsystem may declare.
///
/// Carried over from Zig's `Limits`. These are *declarations* consumed by the
/// scheduler and the OS-control policy; nothing in this module enforces them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Maximum resident memory in mebibytes.
    pub max_memory_mb: u32,
    /// Maximum CPU share as a percentage.
    pub max_cpu_percent: u8,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_memory_mb: 1024,
            max_cpu_percent: 50,
        }
    }
}

/// The host operating system, resolved at compile time.
#[must_use]
pub const fn platform() -> Platform {
    if cfg!(target_os = "macos") {
        Platform::MacOs
    } else if cfg!(target_os = "linux") {
        Platform::Linux
    } else if cfg!(windows) {
        Platform::Windows
    } else if cfg!(target_os = "freebsd") {
        Platform::FreeBsd
    } else {
        Platform::Unknown
    }
}

/// The host CPU architecture, resolved at compile time.
#[must_use]
pub const fn arch() -> Arch {
    if cfg!(target_arch = "x86_64") {
        Arch::X86_64
    } else if cfg!(target_arch = "aarch64") {
        Arch::Aarch64
    } else if cfg!(target_arch = "arm") {
        Arch::Arm
    } else if cfg!(target_arch = "riscv64") {
        Arch::Riscv64
    } else if cfg!(target_arch = "wasm32") {
        Arch::Wasm32
    } else {
        Arch::Unknown
    }
}

/// This process's ID.
#[must_use]
pub fn pid() -> u32 {
    std::process::id()
}

/// The host name.
///
/// Read from `HOSTNAME`, then `COMPUTERNAME` (Windows), then `hostname(1)`,
/// falling back to `"unknown"`.
///
/// The Zig version called `gethostname(2)` directly on POSIX. `std` exposes no
/// equivalent, and adding a `libc` FFI call to a crate that denies `unsafe_code`
/// is not worth it for a diagnostics string, so this consults the environment
/// first and only then shells out. Same `"unknown"` fallback either way.
#[must_use]
pub fn hostname() -> String {
    if let Some(name) = env::get("HOSTNAME") {
        return name;
    }
    if cfg!(windows)
        && let Some(name) = env::get("COMPUTERNAME")
    {
        return name;
    }
    if !cfg!(windows)
        && let Ok(output) = std::process::Command::new("hostname").output()
        && output.status.success()
        && let Ok(text) = String::from_utf8(output.stdout)
    {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "unknown".to_string()
}

/// Available parallelism, at least 1.
#[must_use]
pub fn cpu_count() -> u32 {
    std::thread::available_parallelism()
        .map_or(1, std::num::NonZero::get)
        .try_into()
        .unwrap_or(u32::MAX)
}

/// The process's current working directory.
pub fn current_dir() -> std::io::Result<std::path::PathBuf> {
    std::env::current_dir()
}

/// A snapshot of every host fact above.
#[must_use]
pub fn system_info() -> SystemInfo {
    SystemInfo {
        platform: platform(),
        arch: arch(),
        hostname: hostname(),
        cpu_count: cpu_count(),
    }
}

/// The names of entries directly inside `path`, unsorted.
pub fn list_directory(path: impl AsRef<std::path::Path>) -> std::io::Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(path)? {
        names.push(entry?.file_name().to_string_lossy().into_owned());
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platform_is_recognised_on_supported_hosts() {
        // The port targets macOS, Linux and Windows; Unknown on one of those
        // would mean the cfg chain is broken.
        if cfg!(any(target_os = "macos", target_os = "linux", windows)) {
            assert_ne!(platform(), Platform::Unknown);
        }
    }

    #[test]
    fn arch_is_recognised_on_this_host() {
        if cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
            assert_ne!(arch(), Arch::Unknown);
        }
    }

    #[test]
    fn platform_and_arch_render_lowercase_names() {
        assert_eq!(Platform::MacOs.to_string(), "macos");
        assert_eq!(Platform::Windows.name(), "windows");
        assert_eq!(Arch::Aarch64.to_string(), "aarch64");
        assert_eq!(Arch::X86_64.name(), "x86_64");
    }

    #[test]
    fn pid_is_nonzero() {
        assert!(pid() > 0);
    }

    #[test]
    fn cpu_count_is_at_least_one() {
        assert!(cpu_count() >= 1);
    }

    #[test]
    fn hostname_is_never_empty() {
        let name = hostname();
        assert!(!name.is_empty());
    }

    #[test]
    fn hostname_prefers_the_environment_override() {
        let _guard = env::lock_for_test();
        env::set_override("HOSTNAME", "test-host");
        assert_eq!(hostname(), "test-host");
        env::clear_override("HOSTNAME");
    }

    #[test]
    fn system_info_is_internally_consistent() {
        let _guard = env::lock_for_test();
        let info = system_info();
        assert_eq!(info.platform, platform());
        assert_eq!(info.arch, arch());
        assert!(info.cpu_count >= 1);
        assert!(!info.hostname.is_empty());
    }

    #[test]
    fn default_limits_match_the_zig_values() {
        let limits = Limits::default();
        assert_eq!(limits.max_memory_mb, 1024);
        assert_eq!(limits.max_cpu_percent, 50);
    }

    #[test]
    fn current_dir_resolves() {
        let dir = current_dir().expect("cwd");
        assert!(dir.is_absolute());
    }

    #[test]
    fn list_directory_sees_a_created_file() {
        let root = crate::temp_path::temp_file_path("abi_system_listdir", "d");
        std::fs::create_dir_all(&root).expect("mkdir");
        std::fs::write(root.join("marker.txt"), b"x").expect("write");

        let names = list_directory(&root).expect("list");
        assert!(names.iter().any(|n| n == "marker.txt"), "got {names:?}");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn list_directory_errors_on_a_missing_path() {
        let missing = crate::temp_path::temp_file_path("abi_system_missing_dir", "d");
        assert!(list_directory(&missing).is_err());
    }
}
