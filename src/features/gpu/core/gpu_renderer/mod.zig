const config_mod = @import("config.zig");
const pipelines_mod = @import("pipelines.zig");
const buffers_mod = @import("buffers.zig");

pub const config = config_mod;
pub const pipelines = pipelines_mod;
pub const buffers = buffers_mod;
pub const buffer_operations = @import("buffer_operations.zig");

// Re-export commonly used configuration types for backwards compatibility.
pub const has_webgpu_support = config_mod.has_webgpu_support;
pub const GpuError = config_mod.GpuError;
pub const SPIRVOptimizationLevel = config_mod.SPIRVOptimizationLevel;
pub const SPIRVCompilerOptions = config_mod.SPIRVCompilerOptions;
pub const MSLOptimizationLevel = config_mod.MSLOptimizationLevel;
pub const MetalVersion = config_mod.MetalVersion;
pub const MSLCompilerOptions = config_mod.MSLCompilerOptions;
pub const PTXOptimizationLevel = config_mod.PTXOptimizationLevel;
pub const CudaComputeCapability = config_mod.CudaComputeCapability;
pub const PTXCompilerOptions = config_mod.PTXCompilerOptions;
pub const SPIRVCompiler = config_mod.SPIRVCompiler;
pub const MSLCompiler = config_mod.MSLCompiler;
pub const PTXCompiler = config_mod.PTXCompiler;
pub const GPUConfig = config_mod.GPUConfig;
pub const PowerPreference = config_mod.PowerPreference;
pub const Backend = config_mod.Backend;
pub const BufferUsage = config_mod.BufferUsage;
pub const TextureFormat = config_mod.TextureFormat;
pub const ShaderStage = config_mod.ShaderStage;
pub const Color = config_mod.Color;
pub const GPUHandle = config_mod.GPUHandle;
pub const MathUtils = config_mod.MathUtils;

// Pipeline layer exports
pub const Shader = pipelines_mod.Shader;
pub const BindGroup = pipelines_mod.BindGroup;
pub const BindGroupDesc = pipelines_mod.BindGroupDesc;
pub const RendererStats = pipelines_mod.RendererStats;
pub const ComputeDispatchInfo = pipelines_mod.ComputeDispatchInfo;
pub const CpuFallbackFn = pipelines_mod.CpuFallbackFn;
pub const ComputePipelineDesc = pipelines_mod.ComputePipelineDesc;
pub const ComputePipeline = pipelines_mod.ComputePipeline;
pub const ComputeDispatch = pipelines_mod.ComputeDispatch;

// Buffer layer exports
pub const MockGPU = buffers_mod.MockGPU;
pub const HardwareContext = buffers_mod.HardwareContext;
pub const GPUContext = buffers_mod.GPUContext;
pub const BufferManager = buffers_mod.BufferManager;
pub const Buffer = buffers_mod.Buffer;
