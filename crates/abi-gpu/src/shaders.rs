//! Shader source validation (no external compiler toolchain linked).
//!
//! Ported from `src/features/shaders/mod.zig`. Produces a local validation
//! artifact only — `compiler_status().available` is always `false`.

use abi_foundation::wyhash;

/// Supported shader language labels (report surface only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    /// Zig kernel DSL (historical label from the Zig port).
    ZigKernel,
    /// WebGPU Shading Language.
    Wgsl,
    /// Metal Shading Language.
    Msl,
    /// SPIR-V assembly text.
    SpirvText,
}

impl Language {
    /// Stable name used in status/artifacts.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ZigKernel => "zig-kernel",
            Self::Wgsl => "wgsl",
            Self::Msl => "msl",
            Self::SpirvText => "spirv-text",
        }
    }
}

/// Input shader source.
#[derive(Debug, Clone, Copy)]
pub struct ShaderSource<'a> {
    /// Artifact name.
    pub name: &'a str,
    /// Language dialect.
    pub language: Language,
    /// Source text.
    pub source: &'a str,
}

/// Validation report (no native compile).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Artifact name.
    pub name: String,
    /// Language dialect.
    pub language: Language,
    /// Detected entry point label.
    pub entry_point: String,
    /// Source byte length.
    pub source_bytes: usize,
    /// Wyhash checksum of the source (Zig-compatible).
    pub checksum: u64,
}

/// Local validation artifact string (not a binary shader).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderArtifact {
    /// Artifact name.
    pub name: String,
    /// Language dialect.
    pub language: Language,
    /// Entry point label.
    pub entry_point: String,
    /// Backend label (`validated-local`).
    pub backend: &'static str,
    /// Serialized validation summary.
    pub bytes: String,
}

/// External toolchain status — always unavailable in this port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerStatus {
    /// Whether an external compiler is linked (always false).
    pub available: bool,
    /// Backend label.
    pub backend: &'static str,
    /// Operator message.
    pub message: &'static str,
}

/// Shader validation / compile errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderError {
    /// Empty or null-containing name.
    InvalidName,
    /// Empty or null-containing source.
    InvalidSource,
    /// Missing language entry point.
    MissingEntryPoint,
    /// Unbalanced `()[]{}` delimiters.
    UnbalancedDelimiters,
}

impl std::fmt::Display for ShaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidName => write!(f, "invalid shader name"),
            Self::InvalidSource => write!(f, "invalid shader source"),
            Self::MissingEntryPoint => write!(f, "missing shader entry point"),
            Self::UnbalancedDelimiters => write!(f, "unbalanced shader delimiters"),
        }
    }
}

impl std::error::Error for ShaderError {}

/// Compiler status (external toolchains not linked).
#[must_use]
pub const fn compiler_status() -> CompilerStatus {
    CompilerStatus {
        available: false,
        backend: "validated-local",
        message: "external shader compiler toolchains are not linked; source validation artifact is active",
    }
}

/// Validate source and return a detailed report.
pub fn validate_detailed(source: ShaderSource<'_>) -> Result<ValidationReport, ShaderError> {
    if source.name.is_empty() || source.name.bytes().any(|b| b == 0) {
        return Err(ShaderError::InvalidName);
    }
    if source.source.is_empty() || source.source.bytes().any(|b| b == 0) {
        return Err(ShaderError::InvalidSource);
    }
    validate_balanced_delimiters(source.source)?;
    let entry_point = detect_entry_point(source.language, source.source)?;
    Ok(ValidationReport {
        name: source.name.to_owned(),
        language: source.language,
        entry_point: entry_point.to_owned(),
        source_bytes: source.source.len(),
        checksum: wyhash::hash(0, source.source.as_bytes()),
    })
}

/// Validate source (drop report).
pub fn validate(source: ShaderSource<'_>) -> Result<(), ShaderError> {
    validate_detailed(source).map(|_| ())
}

/// Produce a local validation artifact string (not a compiled binary).
pub fn compile(source: ShaderSource<'_>) -> Result<ShaderArtifact, ShaderError> {
    let report = validate_detailed(source)?;
    let status = compiler_status();
    let bytes = format!(
        "shader={};language={};backend={};entry={};source_bytes={};checksum={:x}",
        report.name,
        report.language.name(),
        status.backend,
        report.entry_point,
        report.source_bytes,
        report.checksum
    );
    Ok(ShaderArtifact {
        name: report.name,
        language: report.language,
        entry_point: report.entry_point,
        backend: status.backend,
        bytes,
    })
}

fn detect_entry_point(language: Language, source: &str) -> Result<&'static str, ShaderError> {
    match language {
        Language::ZigKernel | Language::Wgsl => {
            if source.contains("fn main") {
                Ok("main")
            } else {
                Err(ShaderError::MissingEntryPoint)
            }
        }
        Language::Msl => {
            if source.contains("kernel") {
                Ok("kernel")
            } else if source.contains("main") {
                Ok("main")
            } else {
                Err(ShaderError::MissingEntryPoint)
            }
        }
        Language::SpirvText => {
            if source.contains("OpEntryPoint") {
                Ok("OpEntryPoint")
            } else {
                Err(ShaderError::MissingEntryPoint)
            }
        }
    }
}

fn validate_balanced_delimiters(source: &str) -> Result<(), ShaderError> {
    let mut braces: i32 = 0;
    let mut parens: i32 = 0;
    let mut brackets: i32 = 0;
    for byte in source.bytes() {
        match byte {
            b'{' => braces += 1,
            b'}' => {
                braces -= 1;
                if braces < 0 {
                    return Err(ShaderError::UnbalancedDelimiters);
                }
            }
            b'(' => parens += 1,
            b')' => {
                parens -= 1;
                if parens < 0 {
                    return Err(ShaderError::UnbalancedDelimiters);
                }
            }
            b'[' => brackets += 1,
            b']' => {
                brackets -= 1;
                if brackets < 0 {
                    return Err(ShaderError::UnbalancedDelimiters);
                }
            }
            _ => {}
        }
    }
    if braces != 0 || parens != 0 || brackets != 0 {
        return Err(ShaderError::UnbalancedDelimiters);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_compiles_sample() {
        let artifact = compile(ShaderSource {
            name: "status",
            language: Language::ZigKernel,
            source: "fn main() void {}",
        })
        .expect("compile");
        assert_eq!(artifact.entry_point, "main");
        assert!(artifact.bytes.contains("shader=status"));
        assert!(artifact.bytes.contains("validated-local"));
        assert!(!compiler_status().available);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert_eq!(
            validate(ShaderSource {
                name: "",
                language: Language::ZigKernel,
                source: "fn main() void {}",
            }),
            Err(ShaderError::InvalidName)
        );
        assert_eq!(
            validate(ShaderSource {
                name: "copy",
                language: Language::ZigKernel,
                source: "fn helper() void {}",
            }),
            Err(ShaderError::MissingEntryPoint)
        );
        assert_eq!(
            validate(ShaderSource {
                name: "copy",
                language: Language::ZigKernel,
                source: "fn main() void {",
            }),
            Err(ShaderError::UnbalancedDelimiters)
        );
    }

    #[test]
    fn language_entry_points() {
        assert_eq!(
            validate_detailed(ShaderSource {
                name: "wgsl",
                language: Language::Wgsl,
                source: "@compute fn main() {}",
            })
            .unwrap()
            .entry_point,
            "main"
        );
        assert_eq!(
            validate_detailed(ShaderSource {
                name: "msl",
                language: Language::Msl,
                source: "kernel void add() {}",
            })
            .unwrap()
            .entry_point,
            "kernel"
        );
        assert_eq!(
            validate_detailed(ShaderSource {
                name: "spv",
                language: Language::SpirvText,
                source: "OpEntryPoint GLCompute %main \"main\"",
            })
            .unwrap()
            .entry_point,
            "OpEntryPoint"
        );
    }
}
