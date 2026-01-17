//! Stub for GPU feature when disabled
//!
//! Mirrors the public API of mod.zig while delegating to the real module.
const compute_gpu = @import("mod.zig");

// Acceleration API
pub const acceleration = compute_gpu.acceleration;
pub const Accelerator = compute_gpu.Accelerator;
pub const AcceleratorConfig = compute_gpu.AcceleratorConfig;
pub const AcceleratorError = compute_gpu.AcceleratorError;

// Access and memory modes
pub const AccessHint = compute_gpu.AccessHint;
pub const AccessMode = compute_gpu.AccessMode;
pub const AddressSpace = compute_gpu.AddressSpace;
pub const AsyncTransfer = compute_gpu.AsyncTransfer;

// Backend detection and info
pub const availableBackends = compute_gpu.availableBackends;
pub const Backend = compute_gpu.Backend;
pub const BackendAvailability = compute_gpu.BackendAvailability;
pub const backendAvailability = compute_gpu.backendAvailability;
pub const backendDescription = compute_gpu.backendDescription;
pub const backendDisplayName = compute_gpu.backendDisplayName;
pub const backendFlag = compute_gpu.backendFlag;
pub const backendFromString = compute_gpu.backendFromString;
pub const BackendInfo = compute_gpu.BackendInfo;
pub const backendName = compute_gpu.backendName;
pub const backendSupportsKernels = compute_gpu.backendSupportsKernels;

// Backend factory exports
pub const backend_factory = compute_gpu.backend_factory;
pub const BackendFactory = compute_gpu.BackendFactory;
pub const BackendInstance = compute_gpu.BackendInstance;
pub const BackendFeature = compute_gpu.BackendFeature;
pub const createBackend = compute_gpu.createBackend;
pub const createBestBackend = compute_gpu.createBestBackend;
pub const destroyBackend = compute_gpu.destroyBackend;

// DSL operations
pub const BinaryOp = compute_gpu.BinaryOp;
pub const UnaryOp = compute_gpu.UnaryOp;
pub const BuiltinFn = compute_gpu.BuiltinFn;
pub const BuiltinVar = compute_gpu.BuiltinVar;
pub const Expr = compute_gpu.Expr;
pub const Stmt = compute_gpu.Stmt;
pub const DslType = compute_gpu.DslType;
pub const ScalarType = compute_gpu.ScalarType;
pub const VectorType = compute_gpu.VectorType;
pub const MatrixType = compute_gpu.MatrixType;

// Buffer types
pub const Buffer = compute_gpu.Buffer;
pub const BufferFlags = compute_gpu.BufferFlags;
pub const BufferOptions = compute_gpu.BufferOptions;
pub const BufferStats = compute_gpu.BufferStats;
pub const BufferView = compute_gpu.BufferView;
pub const GpuBuffer = compute_gpu.GpuBuffer;
pub const MappedBuffer = compute_gpu.MappedBuffer;
pub const UnifiedBuffer = compute_gpu.UnifiedBuffer;

// Kernel cache
pub const CacheStats = compute_gpu.CacheStats;
pub const KernelCache = compute_gpu.KernelCache;
pub const KernelCacheConfig = compute_gpu.KernelCacheConfig;

// Compilation
pub const CodegenError = compute_gpu.CodegenError;
pub const compile = compute_gpu.compile;
pub const compileAll = compute_gpu.compileAll;
pub const CompiledKernel = compute_gpu.CompiledKernel;
pub const CompileError = compute_gpu.CompileError;
pub const compileKernel = compute_gpu.compileKernel;
pub const CompileOptions = compute_gpu.CompileOptions;
pub const compileToKernelSource = compute_gpu.compileToKernelSource;

// Compute
pub const ComputeTask = compute_gpu.ComputeTask;
pub const createDefaultKernels = compute_gpu.createDefaultKernels;
pub const cuda_loader = compute_gpu.cuda_loader;

// Device management
pub const defaultDevice = compute_gpu.defaultDevice;
pub const defaultDeviceLabel = compute_gpu.defaultDeviceLabel;
pub const deinit = compute_gpu.deinit;
pub const DetectionLevel = compute_gpu.DetectionLevel;
pub const Device = compute_gpu.Device;
pub const DeviceCapability = compute_gpu.DeviceCapability;
pub const DeviceFeature = compute_gpu.DeviceFeature;
pub const DeviceInfo = compute_gpu.DeviceInfo;
pub const DeviceManager = compute_gpu.DeviceManager;
pub const DeviceSelector = compute_gpu.DeviceSelector;
pub const DeviceType = compute_gpu.DeviceType;
pub const discoverDevices = compute_gpu.discoverDevices;

// Diagnostics and error handling
pub const diagnostics = compute_gpu.diagnostics;
pub const DiagnosticsInfo = compute_gpu.DiagnosticsInfo;
pub const error_handling = compute_gpu.error_handling;
pub const ErrorContext = compute_gpu.ErrorContext;
pub const GpuErrorCode = compute_gpu.GpuErrorCode;
pub const GpuErrorType = compute_gpu.GpuErrorType;

// Dispatcher exports
pub const dispatcher = compute_gpu.dispatcher;
pub const KernelDispatcher = compute_gpu.KernelDispatcher;
pub const DispatchError = compute_gpu.DispatchError;
pub const CompiledKernelHandle = compute_gpu.CompiledKernelHandle;
pub const KernelArgs = compute_gpu.KernelArgs;

// DSL module
pub const dsl = compute_gpu.dsl;
pub const ElementType = compute_gpu.ElementType;
pub const ensureInitialized = compute_gpu.ensureInitialized;

// Events
pub const Event = compute_gpu.Event;
pub const EventFlags = compute_gpu.EventFlags;
pub const EventOptions = compute_gpu.EventOptions;
pub const EventState = compute_gpu.EventState;

// Execution
pub const ExecutionResult = compute_gpu.ExecutionResult;
pub const ExecutionStats = compute_gpu.ExecutionStats;

// Recovery and failover
pub const failover = compute_gpu.failover;
pub const FailoverManager = compute_gpu.FailoverManager;
pub const recovery = compute_gpu.recovery;
pub const RecoveryManager = compute_gpu.RecoveryManager;

// Generated code
pub const GeneratedSource = compute_gpu.GeneratedSource;

// Backend queries
pub const getAvailableBackends = compute_gpu.getAvailableBackends;
pub const getBestBackend = compute_gpu.getBestBackend;
pub const getBestKernelBackend = compute_gpu.getBestKernelBackend;

// Main GPU type
pub const Gpu = compute_gpu.Gpu;
pub const GpuConfig = compute_gpu.GpuConfig;
pub const GpuError = compute_gpu.GpuError;
pub const GpuMemoryPool = compute_gpu.GpuMemoryPool;
pub const GpuStats = compute_gpu.GpuStats;
pub const GpuStream = compute_gpu.GpuStream;

// Health and status
pub const HealthStatus = compute_gpu.HealthStatus;

// Initialization
pub const init = compute_gpu.init;
pub const interface = compute_gpu.interface;
pub const isEnabled = compute_gpu.isEnabled;
pub const isGpuAvailable = compute_gpu.isGpuAvailable;
pub const isInitialized = compute_gpu.isInitialized;

// Kernel types
pub const KernelBuilder = compute_gpu.KernelBuilder;
pub const KernelConfig = compute_gpu.KernelConfig;
pub const KernelError = compute_gpu.KernelError;
pub const KernelIR = compute_gpu.KernelIR;
pub const KernelSource = compute_gpu.KernelSource;

// Launch configuration
pub const LaunchConfig = compute_gpu.LaunchConfig;

// Device listing
pub const listBackendInfo = compute_gpu.listBackendInfo;
pub const listDevices = compute_gpu.listDevices;
pub const LoadBalanceStrategy = compute_gpu.LoadBalanceStrategy;

// Matrix operations
pub const MatrixDims = compute_gpu.MatrixDims;
pub const matrixMultiply = compute_gpu.matrixMultiply;

// Memory management
pub const MemoryBandwidth = compute_gpu.MemoryBandwidth;
pub const MemoryError = compute_gpu.MemoryError;
pub const MemoryInfo = compute_gpu.MemoryInfo;
pub const MemoryLocation = compute_gpu.MemoryLocation;
pub const MemoryMode = compute_gpu.MemoryMode;
pub const MemoryPool = compute_gpu.MemoryPool;
pub const MemoryStats = compute_gpu.MemoryStats;

// Module status
pub const moduleEnabled = compute_gpu.moduleEnabled;
pub const MultiGpuConfig = compute_gpu.MultiGpuConfig;
pub const OccupancyResult = compute_gpu.OccupancyResult;

// Portable kernels
pub const PortableKernelSource = compute_gpu.PortableKernelSource;

// Profiling
pub const Profiler = compute_gpu.Profiler;
pub const profiling = compute_gpu.profiling;
pub const TimingResult = compute_gpu.TimingResult;

// Operations
pub const reduceSum = compute_gpu.reduceSum;
pub const vectorAdd = compute_gpu.vectorAdd;

// Stream management
pub const stream = compute_gpu.stream;
pub const StreamFlags = compute_gpu.StreamFlags;
pub const StreamManager = compute_gpu.StreamManager;
pub const StreamOptions = compute_gpu.StreamOptions;
pub const StreamPriority = compute_gpu.StreamPriority;
pub const StreamState = compute_gpu.StreamState;

// Summary and unified API
pub const summary = compute_gpu.summary;
pub const unified = compute_gpu.unified;
pub const unified_buffer = compute_gpu.unified_buffer;

// Builtin kernels module
pub const builtin_kernels = compute_gpu.builtin_kernels;
