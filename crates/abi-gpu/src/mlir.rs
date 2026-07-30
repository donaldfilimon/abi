//! Textual MLIR-style IR lowering (no external MLIR/LLVM toolchain linked).
//!
//! Ported from `src/features/mlir/mod.zig`.

use std::fmt::Write as _;

use abi_foundation::wyhash;

/// Supported dialect labels for the textual IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Dialect {
    /// Affine dialect.
    Affine,
    /// Linalg (default).
    #[default]
    Linalg,
    /// Tensor dialect.
    Tensor,
    /// GPU dialect.
    Gpu,
}

impl Dialect {
    /// Stable dialect name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Affine => "affine",
            Self::Linalg => "linalg",
            Self::Tensor => "tensor",
            Self::Gpu => "gpu",
        }
    }
}

/// Module lowering request.
#[derive(Debug, Clone)]
pub struct ModuleSpec<'a> {
    /// Module symbol name.
    pub name: &'a str,
    /// Dialect label.
    pub dialect: Dialect,
    /// Operation names to emit.
    pub operations: &'a [&'a str],
}

/// Analysis of a module spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleAnalysis {
    /// Module name.
    pub module_name: String,
    /// Dialect.
    pub dialect: Dialect,
    /// Operation count.
    pub operation_count: usize,
    /// Wyhash checksum of name/dialect/ops.
    pub checksum: u64,
}

/// Lowering result with textual IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoweringResult {
    /// Module name.
    pub module_name: String,
    /// Dialect.
    pub dialect: Dialect,
    /// Target backend label.
    pub target_backend: &'static str,
    /// Textual IR body.
    pub ir: String,
}

/// External MLIR toolchain status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ToolchainStatus {
    /// Whether external tools are linked (always false).
    pub available: bool,
    /// Backend label.
    pub backend: &'static str,
    /// Operator message.
    pub message: &'static str,
}

/// MLIR analysis / lowering errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlirError {
    /// Invalid module symbol.
    InvalidModuleName,
    /// Empty/invalid operation name.
    InvalidOperation,
}

impl std::fmt::Display for MlirError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidModuleName => write!(f, "invalid MLIR module name"),
            Self::InvalidOperation => write!(f, "invalid MLIR operation"),
        }
    }
}

impl std::error::Error for MlirError {}

/// Toolchain status (external MLIR tools not linked).
#[must_use]
pub const fn toolchain_status() -> ToolchainStatus {
    ToolchainStatus {
        available: false,
        backend: "textual-local",
        message: "external MLIR tools are not linked; textual ABI IR lowering is active",
    }
}

/// Analyze a module spec.
pub fn analyze(spec: &ModuleSpec<'_>) -> Result<ModuleAnalysis, MlirError> {
    if spec.name.is_empty() || !is_symbol_name(spec.name) {
        return Err(MlirError::InvalidModuleName);
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(spec.name.as_bytes());
    buf.extend_from_slice(spec.dialect.name().as_bytes());
    for op in spec.operations {
        if op.is_empty() || op.bytes().any(|b| b == 0) {
            return Err(MlirError::InvalidOperation);
        }
        buf.extend_from_slice(op.as_bytes());
        buf.push(0xff);
    }
    Ok(ModuleAnalysis {
        module_name: spec.name.to_owned(),
        dialect: spec.dialect,
        operation_count: spec.operations.len(),
        checksum: wyhash::hash(0, &buf),
    })
}

/// Lower a module to structured textual IR.
pub fn lower(spec: &ModuleSpec<'_>) -> Result<LoweringResult, MlirError> {
    let analysis = analyze(spec)?;
    let mut ir = format!(
        "module @{} attributes {{abi.dialect = \"{}\", abi.ops = {}, abi.checksum = \"{:x}\"}} {{\n",
        analysis.module_name,
        analysis.dialect.name(),
        analysis.operation_count,
        analysis.checksum
    );
    if spec.operations.is_empty() {
        ir.push_str("  // no operations requested\n");
    } else {
        for (i, op) in spec.operations.iter().enumerate() {
            let _ = write!(
                ir,
                "  abi.op @{}_{i} attributes {{abi.name = ",
                analysis.dialect.name()
            );
            append_mlir_string(&mut ir, op);
            ir.push_str("}\n");
        }
    }
    ir.push_str("}\n");
    Ok(LoweringResult {
        module_name: analysis.module_name,
        dialect: analysis.dialect,
        target_backend: toolchain_status().backend,
        ir,
    })
}

fn is_symbol_name(value: &str) -> bool {
    !value.is_empty()
        && value.bytes().all(|b| {
            matches!(b,
                b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b'.'
            )
        })
}

fn append_mlir_string(out: &mut String, value: &str) {
    out.push('"');
    for byte in value.bytes() {
        match byte {
            b'"' => out.push_str("\\22"),
            b'\\' => out.push_str("\\5C"),
            b'\n' => out.push_str("\\0A"),
            b'\r' => out.push_str("\\0D"),
            b'\t' => out.push_str("\\09"),
            0x00..=0x08 | 0x0b..=0x0c | 0x0e..=0x1f => {
                let _ = write!(out, "\\{byte:02X}");
            }
            _ => out.push(byte as char),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_textual_ir() {
        let result = lower(&ModuleSpec {
            name: "status",
            dialect: Dialect::Linalg,
            operations: &["matmul"],
        })
        .expect("lower");
        assert!(result.ir.contains("module @status"));
        assert!(result.ir.contains("abi.name = \"matmul\""));
        assert_eq!(result.target_backend, "textual-local");
        assert!(!toolchain_status().available);
    }

    #[test]
    fn rejects_bad_symbols_and_escapes() {
        assert_eq!(
            analyze(&ModuleSpec {
                name: "bad name",
                dialect: Dialect::Linalg,
                operations: &[],
            }),
            Err(MlirError::InvalidModuleName)
        );
        let result = lower(&ModuleSpec {
            name: "train",
            dialect: Dialect::Linalg,
            operations: &["quote \" op"],
        })
        .expect("lower");
        assert!(result.ir.contains("quote \\22 op"));
    }
}
