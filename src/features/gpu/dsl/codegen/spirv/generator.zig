//! SPIR-V Generator Facade
//!
//! Re-exports the SpirvGenerator and SPIR-V specification constants.

const generator_mod = @import("generator/generator.zig");
const spec_mod = @import("generator/spec.zig");

pub const SpirvGenerator = generator_mod.SpirvGenerator;

pub const SPIRV_MAGIC = spec_mod.SPIRV_MAGIC;
pub const SPIRV_VERSION = spec_mod.SPIRV_VERSION;
pub const GENERATOR_ID = spec_mod.GENERATOR_ID;

pub const OpCode = spec_mod.OpCode;
pub const Capability = spec_mod.Capability;
pub const ExecutionModel = spec_mod.ExecutionModel;
pub const AddressingModel = spec_mod.AddressingModel;
pub const MemoryModel = spec_mod.MemoryModel;
pub const ExecutionMode = spec_mod.ExecutionMode;
pub const StorageClass = spec_mod.StorageClass;
pub const Decoration = spec_mod.Decoration;
pub const BuiltIn = spec_mod.BuiltIn;
pub const MemorySemantics = spec_mod.MemorySemantics;
pub const Scope = spec_mod.Scope;
