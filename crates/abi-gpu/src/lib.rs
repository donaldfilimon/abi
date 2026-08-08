//! Honest GPU / accelerator backend detection.
//!
//! Preferred backend is Metal on macOS. When the optional `metal-kernels`
//! feature links and initializes a Metal DOT pipeline, `accelerated=true`
//! for that path; otherwise CPU SIMD fallback with `accelerated=false`.
//!
//! Also hosts claim-honest **shaders**, **mlir**, and **mobile** report
//! surfaces (validation / textual lowering / simulated profile).

pub mod metal_kernels;
pub mod mlir;
pub mod mobile;
pub mod shaders;

pub use mlir::{
    Dialect, ModuleSpec, lower as lower_mlir, toolchain_status as mlir_toolchain_status,
};
pub use mobile::{device_profile as mobile_device_profile, status_line as mobile_status_line};
pub use shaders::{
    Language as ShaderLanguage, ShaderSource, compile as compile_shader,
    compiler_status as shader_compiler_status,
};

use abi_wdbx::best_cpu_backend;

/// Declared GPU backends, matching Zig's `Backend` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Deterministic vectorized CPU fallback.
    Simulated,
    /// Apple Metal (preferred on macOS; optional DOT kernels via `metal-kernels`).
    Metal,
    /// Vulkan (declared; not linked).
    Vulkan,
    /// CUDA (declared; not linked).
    Cuda,
    /// WebGPU (declared; not linked).
    Webgpu,
    /// OpenGL (declared; not linked).
    Opengl,
    /// WebGL2 (declared; not linked).
    Webgl2,
}

impl Backend {
    /// Stable lowercase name used in `gpu_status` and `abi backends`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Simulated => "simulated",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Cuda => "cuda",
            Self::Webgpu => "webgpu",
            Self::Opengl => "opengl",
            Self::Webgl2 => "webgl2",
        }
    }

    /// All seven declared backends in Zig's list order.
    pub const ALL: [Self; 7] = [
        Self::Simulated,
        Self::Metal,
        Self::Vulkan,
        Self::Cuda,
        Self::Webgpu,
        Self::Opengl,
        Self::Webgl2,
    ];
}

/// Capability report for one declared backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendCapabilities {
    /// Declared backend.
    pub backend: Backend,
    /// Whether this backend is *declared* available on the host (not "native
    /// kernels run").
    pub available: bool,
    /// Whether native accelerated kernels are linked and initialized.
    pub accelerated: bool,
    /// Whether native kernels are compiled into this build.
    pub native_kernels: bool,
    /// Operator-facing explanation.
    pub message: &'static str,
}

/// Detection result for the preferred backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendStatus {
    /// Preferred backend for this host.
    pub backend: Backend,
    /// Whether a safe backend is available (always true — CPU fallback).
    pub available: bool,
    /// Whether native acceleration is active for at least one kernel.
    pub accelerated: bool,
    /// Operator-facing explanation.
    pub message: &'static str,
}

/// Preferred backend for the compile target — Metal on macOS, simulated
/// elsewhere — matching Zig's `preferredBackendForTarget`.
#[must_use]
pub const fn preferred_backend() -> Backend {
    if cfg!(target_os = "macos") {
        Backend::Metal
    } else {
        Backend::Simulated
    }
}

/// Capability table for one backend.
///
/// Metal may report `accelerated=true` when `metal-kernels` init succeeds.
#[must_use]
pub fn backend_capabilities(backend: Backend) -> BackendCapabilities {
    match backend {
        Backend::Metal => {
            let platform_available = cfg!(target_os = "macos");
            let linked = metal_kernels::kernels_linked();
            let active = metal_kernels::kernels_active();
            BackendCapabilities {
                backend: Backend::Metal,
                available: platform_available,
                accelerated: active,
                native_kernels: linked,
                message: if !platform_available {
                    "Metal is only available on macOS; vectorized CPU fallback active"
                } else if active {
                    "Metal DOT kernel linked and initialized; vector ops may dispatch on GPU"
                } else if linked {
                    "Metal kernels linked but device/pipeline init failed; vectorized CPU fallback active"
                } else {
                    "Metal is the preferred backend on macOS but native kernels are not linked; vectorized CPU fallback active"
                },
            }
        }
        Backend::Vulkan => BackendCapabilities {
            backend: Backend::Vulkan,
            available: false,
            accelerated: false,
            native_kernels: false,
            message: "Vulkan backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        Backend::Cuda => BackendCapabilities {
            backend: Backend::Cuda,
            available: false,
            accelerated: false,
            native_kernels: false,
            message: "CUDA backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        Backend::Webgpu => BackendCapabilities {
            backend: Backend::Webgpu,
            available: false,
            accelerated: false,
            native_kernels: false,
            message: "WebGPU backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        Backend::Opengl => BackendCapabilities {
            backend: Backend::Opengl,
            available: false,
            accelerated: false,
            native_kernels: false,
            message: "OpenGL backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        Backend::Webgl2 => BackendCapabilities {
            backend: Backend::Webgl2,
            available: false,
            accelerated: false,
            native_kernels: false,
            message: "WebGL2 backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        Backend::Simulated => BackendCapabilities {
            backend: Backend::Simulated,
            available: true,
            accelerated: false,
            native_kernels: false,
            message: "Deterministic vectorized CPU fallback backend active",
        },
    }
}

/// All seven capability rows.
#[must_use]
pub fn backend_capabilities_list() -> [BackendCapabilities; 7] {
    [
        backend_capabilities(Backend::Simulated),
        backend_capabilities(Backend::Metal),
        backend_capabilities(Backend::Vulkan),
        backend_capabilities(Backend::Cuda),
        backend_capabilities(Backend::Webgpu),
        backend_capabilities(Backend::Opengl),
        backend_capabilities(Backend::Webgl2),
    ]
}

/// Detect the preferred backend's status.
#[must_use]
pub fn detect_backend() -> BackendStatus {
    let caps = backend_capabilities(preferred_backend());
    BackendStatus {
        backend: caps.backend,
        available: caps.available,
        accelerated: caps.accelerated,
        message: caps.message,
    }
}

/// MCP `gpu_status` report line.
///
/// Format: `backend=… available=… accelerated=… capabilities=N message=…`
/// where `N` is the length of the declared capability table (always 7).
#[must_use]
pub fn status_text() -> String {
    let status = detect_backend();
    let caps = backend_capabilities_list();
    format!(
        "backend={} available={} accelerated={} capabilities={} message={}",
        status.backend.name(),
        if status.available { "true" } else { "false" },
        if status.accelerated { "true" } else { "false" },
        caps.len(),
        status.message,
    )
}

/// Effective CPU backend name (portable SIMD path used for real work).
#[must_use]
pub fn effective_cpu_backend_name() -> &'static str {
    best_cpu_backend().name()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_table_has_exactly_seven_entries() {
        assert_eq!(backend_capabilities_list().len(), 7);
        assert_eq!(Backend::ALL.len(), 7);
    }

    #[test]
    fn acceleration_matches_metal_kernel_init() {
        let metal = backend_capabilities(Backend::Metal);
        if metal_kernels::kernels_active() {
            assert!(metal.accelerated);
            assert!(metal.native_kernels);
            assert!(metal.message.contains("initialized") || metal.message.contains("GPU"));
            if preferred_backend() == Backend::Metal {
                assert!(detect_backend().accelerated);
            }
        } else {
            assert!(!metal.accelerated);
            for caps in backend_capabilities_list() {
                if caps.backend != Backend::Metal {
                    assert!(!caps.accelerated, "{:?}", caps.backend);
                }
            }
            if !metal_kernels::kernels_linked() {
                assert!(
                    detect_backend().message.contains("fallback")
                        || detect_backend().message.contains("CPU")
                        || detect_backend().message.contains("not linked")
                );
            }
        }
    }

    #[test]
    fn preferred_backend_is_metal_on_macos_else_simulated() {
        if cfg!(target_os = "macos") {
            assert_eq!(preferred_backend(), Backend::Metal);
            assert!(detect_backend().available);
            assert_eq!(detect_backend().backend, Backend::Metal);
        } else {
            assert_eq!(preferred_backend(), Backend::Simulated);
            assert!(detect_backend().available);
        }
    }

    #[test]
    fn status_text_matches_the_mcp_wire_shape() {
        let text = status_text();
        assert!(text.starts_with("backend="));
        assert!(text.contains(" available="));
        assert!(
            text.contains(" accelerated=false") || text.contains(" accelerated=true"),
            "{text}"
        );
        assert!(text.contains(" capabilities=7 "));
        assert!(text.contains(" message="));
        if metal_kernels::kernels_active() && preferred_backend() == Backend::Metal {
            assert!(text.contains("accelerated=true"), "{text}");
        } else {
            assert!(text.contains("accelerated=false"), "{text}");
        }
    }

    #[test]
    fn backend_names_match_zig() {
        assert_eq!(Backend::Simulated.name(), "simulated");
        assert_eq!(Backend::Metal.name(), "metal");
        assert_eq!(Backend::Vulkan.name(), "vulkan");
        assert_eq!(Backend::Cuda.name(), "cuda");
        assert_eq!(Backend::Webgpu.name(), "webgpu");
        assert_eq!(Backend::Opengl.name(), "opengl");
        assert_eq!(Backend::Webgl2.name(), "webgl2");
    }

    #[test]
    fn effective_cpu_backend_is_nonempty() {
        assert_ne!(effective_cpu_backend_name(), "");
    }
}
