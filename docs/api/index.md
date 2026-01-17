# ABI API Reference

Auto-generated API documentation.

## Modules

- [abi](#abi)
- [gpu](#gpu)
- [ai](#ai)
- [database](#database)
- [network](#network)
- [runtime](#runtime)

---

<a name="abi"></a>
## abi Module

**Source:** `src/abi.zig`

### Overview

ABI Framework - Main Library Interface

High level entrypoints and re-exports for the modernized runtime.


### Public API

#### `core`

Core utilities and fundamental types

#### `features`

Feature modules grouped for discoverability

#### `ai`

Individual feature namespaces re-exported at the root for ergonomic imports.

#### `gpu`

#### `database`

#### `web`

#### `monitoring`

#### `connectors`

#### `network`

#### `compute`

#### `framework`

Framework orchestration layer that coordinates features and plugins.

#### `Feature`

#### `Framework`

#### `FrameworkOptions`

#### `FrameworkConfiguration`

#### `RuntimeConfig`

#### `runtimeConfigFromOptions`

#### `logging`

#### `plugins`

#### `observability`

#### `platform`

#### `simd`

#### `utils`

#### `config`

#### `vectorAdd`

#### `vectorDot`

#### `vectorL2Norm`

#### `cosineSimilarity`

#### `hasSimdSupport`

#### `GpuBackend`

#### `GpuBackendInfo`

#### `GpuBackendAvailability`

#### `GpuBackendDetectionLevel`

#### `GpuDeviceInfo`

#### `GpuDeviceCapability`

#### `GpuBuffer`

#### `GpuBufferFlags`

#### `GpuMemoryPool`

#### `GpuMemoryStats`

#### `GpuMemoryError`

#### `Gpu`

#### `GpuConfig`

#### `GpuUnifiedBuffer`

#### `GpuDevice`

#### `GpuDeviceType`

#### `GpuDeviceFeature`

#### `GpuDeviceSelector`

#### `GpuStream`

#### `GpuEvent`

#### `GpuExecutionResult`

#### `GpuMemoryMode`

#### `GpuHealthStatus`

#### `KernelBuilder`

#### `KernelIR`

#### `PortableKernelSource`

#### `NetworkConfig`

#### `NetworkState`

#### `TransformerConfig`

#### `TransformerModel`

#### `StreamingGenerator`

#### `StreamToken`

#### `StreamState`

#### `GenerationConfig`

#### `discord`

#### `DiscordClient`

#### `DiscordConfig`

#### `DiscordTools`

#### `wdbx`

Compatibility namespace for the WDBX tooling.

#### `database`

#### `helpers`

#### `cli`

#### `http`

#### `createDatabase`

#### `connectDatabase`

#### `closeDatabase`

#### `insertVector`

#### `searchVectors`

#### `deleteVector`

#### `updateVector`

#### `getVector`

#### `listVectors`

#### `getStats`

#### `optimize`

#### `backup`

#### `restore`

#### `init`

Initialise the ABI framework and return the orchestration handle.

#### `shutdown`

Convenience wrapper around `Framework.deinit`.

#### `version`

Get framework version information.

#### `createDefaultFramework`

Create a framework with default configuration.

#### `createFramework`

Create a framework with custom configuration.


---

<a name="gpu"></a>
## gpu Module

**Source:** `src/compute/gpu/unified.zig`

### Overview

Unified GPU API

Main entry point for the unified GPU API.
Provides a single interface for all GPU backends with:
- High-level operations (vectorAdd, matrixMultiply, etc.)
- Custom kernel compilation and execution
- Smart buffer management
- Device discovery and selection
- Stream/event synchronization

## Quick Start

```zig
var gpu = try Gpu.init(allocator, .{});
defer gpu.deinit();

// Create buffers
var a = try gpu.createBufferFromSlice(f32, &[_]f32{ 1, 2, 3, 4 }, .{});
var b = try gpu.createBufferFromSlice(f32, &[_]f32{ 5, 6, 7, 8 }, .{});
var result = try gpu.createBuffer(4 * @sizeOf(f32), .{});
defer { gpu.destroyBuffer(&a); gpu.destroyBuffer(&b); gpu.destroyBuffer(&result); }

// Run operation
_ = try gpu.vectorAdd(&a, &b, &result);

// Read results
var output: [4]f32 = undefined;
try result.read(f32, &output);
```


### Public API

#### `Backend`

#### `Device`

#### `DeviceSelector`

#### `DeviceType`

#### `DeviceFeature`

#### `Stream`

#### `StreamOptions`

#### `StreamPriority`

#### `Event`

#### `EventOptions`

#### `Buffer`

#### `BufferOptions`

#### `MemoryMode`

#### `MemoryLocation`

#### `AccessHint`

#### `ElementType`

#### `KernelBuilder`

#### `KernelIR`

#### `PortableKernelSource`

#### `DeviceGroup`

#### `WorkDistribution`

#### `DeviceBarrier`

#### `PeerTransfer`

#### `MetricsCollector`

#### `MetricsSummary`

#### `KernelMetrics`

#### `LoadBalanceStrategy`

Load balance strategy for multi-GPU.

#### `GpuConfig`

GPU configuration.

#### `ExecutionResult`

Execution result with timing and statistics.

#### `throughputGBps`

Get throughput in GB/s.

#### `elementsPerSecond`

Get elements per second.

#### `MatrixDims`

Matrix dimensions for matrix operations.

#### `LaunchConfig`

Kernel launch configuration.

#### `CompiledKernel`

Compiled kernel handle.

#### `deinit`

#### `MemoryInfo`

GPU memory information.

#### `GpuStats`

GPU statistics.

#### `HealthStatus`

Health status.

#### `MultiGpuConfig`

Multi-GPU configuration.

#### `Gpu`

Main unified GPU API.

#### `init`

Initialize the unified GPU API.

#### `deinit`

Deinitialize and cleanup.

#### `selectDevice`

Select a device based on criteria.

#### `getActiveDevice`

Get the currently active device.

#### `listDevices`

List all available devices.

#### `enableMultiGpu`

Enable multi-GPU mode.

#### `getDeviceGroup`

Get multi-GPU device group (if enabled).

#### `distributeWork`

Distribute work across multiple GPUs.

#### `createBuffer`

Create a new buffer.

#### `createBufferFromSlice`

Create a buffer from a typed slice.

#### `destroyBuffer`

Destroy a buffer.

#### `vectorAdd`

Vector addition: result = a + b

#### `matrixMultiply`

Matrix multiplication: result = a * b

#### `reduceSum`

Reduce sum: returns sum of all elements.

#### `dotProduct`

Dot product: returns a · b

#### `softmax`

Softmax: output = softmax(input)

#### `compileKernel`

Compile a kernel from portable source.

#### `launchKernel`

Launch a compiled kernel.

#### `synchronize`

Synchronize all pending operations.

#### `createStream`

Create a new stream.

#### `createEvent`

Create a new event.

#### `getStats`

Get GPU statistics.

#### `getMemoryInfo`

Get memory information.

#### `checkHealth`

Check GPU health.

#### `isAvailable`

Check if GPU is available.

#### `getBackend`

Get the active backend.

#### `isProfilingEnabled`

Check if profiling is enabled.

#### `enableProfiling`

Enable profiling (creates metrics collector if not exists).

#### `disableProfiling`

Disable profiling.

#### `getMetricsSummary`

Get metrics summary (if profiling enabled).

#### `getKernelMetrics`

Get kernel-specific metrics (if profiling enabled).

#### `getMetricsCollector`

Get the metrics collector directly (for advanced usage).

#### `resetMetrics`

Reset all profiling metrics.

#### `isMultiGpuEnabled`

Check if multi-GPU is enabled.

#### `getMultiGpuStats`

Get multi-GPU statistics (if enabled).

#### `activeDeviceCount`

Get the number of active devices.


---

<a name="ai"></a>
## ai Module

**Source:** `src/features/ai/mod.zig`

### Overview

AI feature module with agents, transformers, training, and federated learning.

Provides high-level interfaces for AI functionality including agent creation,
transformer models, training pipelines, and federated learning coordination.


### Public API

#### `agent`

#### `model_registry`

#### `training`

#### `federated`

#### `transformer`

#### `streaming`

#### `tools`

#### `explore`

#### `llm`

#### `memory`

#### `prompts`

#### `abbey`

#### `embeddings`

#### `eval`

#### `rag`

#### `templates`

#### `Agent`

#### `ModelRegistry`

#### `ModelInfo`

#### `TrainingConfig`

#### `TrainingReport`

#### `TrainingResult`

#### `TrainError`

#### `OptimizerType`

#### `LearningRateSchedule`

#### `CheckpointStore`

#### `Checkpoint`

#### `loadCheckpoint`

#### `saveCheckpoint`

#### `GradientAccumulator`

#### `LlmTrainingConfig`

#### `LlamaTrainer`

#### `TrainableModel`

#### `trainLlm`

#### `trainable_model`

#### `Tool`

#### `ToolResult`

#### `ToolRegistry`

#### `TaskTool`

#### `Subagent`

#### `DiscordTools`

#### `registerDiscordTools`

#### `ExploreAgent`

#### `ExploreConfig`

#### `ExploreLevel`

#### `ExploreResult`

#### `Match`

#### `ExplorationStats`

#### `QueryIntent`

#### `ParsedQuery`

#### `QueryUnderstanding`

#### `PromptBuilder`

#### `Persona`

#### `PersonaType`

#### `PromptFormat`

#### `getPersona`

#### `listPersonas`

#### `Abbey`

#### `AbbeyConfig`

#### `AbbeyResponse`

#### `AbbeyStats`

#### `ReasoningChain`

#### `ReasoningStep`

#### `Confidence`

#### `ConfidenceLevel`

#### `EmotionalState`

#### `EmotionType`

#### `ConversationContext`

#### `TopicTracker`

#### `LlmEngine`

#### `LlmModel`

#### `LlmConfig`

#### `GgufFile`

#### `BpeTokenizer`

#### `AiError`

#### `init`

#### `deinit`

#### `isEnabled`

#### `isInitialized`

#### `createRegistry`

#### `train`

#### `trainWithResult`

#### `createAgent`

#### `createAgentWithConfig`

#### `processMessage`

#### `createTransformer`

#### `inferText`

#### `embedText`

#### `encodeTokens`

#### `decodeTokens`


---

<a name="database"></a>
## database Module

**Source:** `src/features/database/mod.zig`

### Overview

Database feature facade and convenience helpers.

### Public API

#### `database`

#### `db_helpers`

#### `storage`

#### `wdbx`

#### `cli`

#### `http`

#### `fulltext`

#### `hybrid`

#### `filter`

#### `batch`

#### `clustering`

#### `formats`

#### `quantization`

#### `Database`

#### `DatabaseHandle`

#### `SearchResult`

#### `VectorView`

#### `Stats`

#### `InvertedIndex`

#### `Bm25Config`

#### `TokenizerConfig`

#### `TextSearchResult`

#### `QueryParser`

#### `HybridSearchEngine`

#### `HybridConfig`

#### `HybridResult`

#### `FusionMethod`

#### `FilterBuilder`

#### `FilterExpression`

#### `FilterCondition`

#### `FilterOperator`

#### `MetadataValue`

#### `MetadataStore`

#### `FilteredSearch`

#### `FilteredResult`

#### `BatchProcessor`

#### `BatchConfig`

#### `BatchRecord`

#### `BatchResult`

#### `BatchWriter`

#### `BatchOperationBuilder`

#### `BatchImporter`

#### `KMeans`

#### `ClusterStats`

#### `FitOptions`

#### `FitResult`

#### `euclideanDistance`

#### `cosineSimilarity`

#### `silhouetteScore`

#### `elbowMethod`

#### `ScalarQuantizer`

#### `ProductQuantizer`

#### `QuantizationError`

#### `UnifiedFormat`

#### `UnifiedFormatBuilder`

#### `FormatHeader`

#### `FormatFlags`

#### `TensorDescriptor`

#### `DataType`

#### `Converter`

#### `ConversionOptions`

#### `TargetFormat`

#### `CompressionType`

#### `StreamingWriter`

#### `StreamingReader`

#### `MappedFile`

#### `MemoryCursor`

#### `FormatVectorDatabase`

#### `FormatVectorRecord`

#### `FormatSearchResult`

#### `fromGguf`

#### `toGguf`

#### `GgufTensorType`

#### `DatabaseFeatureError`

#### `init`

#### `deinit`

#### `isEnabled`

#### `isInitialized`

#### `open`

#### `connect`

#### `close`

#### `insert`

#### `search`

#### `remove`

#### `update`

#### `get`

#### `list`

#### `stats`

#### `optimize`

#### `backup`

#### `restore`

#### `openFromFile`

#### `openOrCreate`


---

<a name="network"></a>
## network Module

**Source:** `src/features/network/mod.zig`

### Overview

Network feature module for distributed compute coordination.

Provides node registry, task/result serialization protocols, and cluster state
management for distributed computing scenarios.


### Public API

#### `retry`

#### `rate_limiter`

#### `connection_pool`

#### `raft`

#### `transport`

#### `raft_transport`

#### `circuit_breaker`

#### `NodeRegistry`

#### `NodeInfo`

#### `NodeStatus`

#### `TaskEnvelope`

#### `ResultEnvelope`

#### `ResultStatus`

#### `encodeTask`

#### `decodeTask`

#### `encodeResult`

#### `decodeResult`

#### `TaskScheduler`

#### `SchedulerConfig`

#### `SchedulerError`

#### `TaskPriority`

#### `TaskState`

#### `ComputeNode`

#### `LoadBalancingStrategy`

#### `SchedulerStats`

#### `HealthCheck`

#### `ClusterConfig`

#### `HaError`

#### `NodeHealth`

#### `ClusterState`

#### `HealthCheckResult`

#### `FailoverPolicy`

#### `ServiceDiscovery`

#### `DiscoveryConfig`

#### `DiscoveryBackend`

#### `ServiceInstance`

#### `ServiceStatus`

#### `DiscoveryError`

#### `LoadBalancer`

#### `LoadBalancerConfig`

#### `LoadBalancerStrategy`

#### `LoadBalancerError`

#### `NodeState`

#### `NodeStats`

#### `RetryConfig`

#### `RetryResult`

#### `RetryError`

#### `RetryStrategy`

#### `RetryExecutor`

#### `RetryableErrors`

#### `BackoffCalculator`

#### `retryOperation`

#### `retryWithStrategy`

#### `RateLimiter`

#### `RateLimiterConfig`

#### `RateLimitAlgorithm`

#### `AcquireResult`

#### `TokenBucketLimiter`

#### `SlidingWindowLimiter`

#### `FixedWindowLimiter`

#### `LimiterStats`

#### `ConnectionPool`

#### `ConnectionPoolConfig`

#### `PooledConnection`

#### `ConnectionState`

#### `ConnectionStats`

#### `HostKey`

#### `PoolStats`

#### `PoolBuilder`

#### `RaftNode`

#### `RaftState`

#### `RaftConfig`

#### `RaftError`

#### `RaftStats`

#### `LogEntry`

#### `RequestVoteRequest`

#### `RequestVoteResponse`

#### `AppendEntriesRequest`

#### `AppendEntriesResponse`

#### `PeerState`

#### `createRaftCluster`

#### `RaftPersistence`

#### `PersistentState`

#### `RaftSnapshotManager`

#### `SnapshotConfig`

#### `SnapshotMetadata`

#### `SnapshotInfo`

#### `InstallSnapshotRequest`

#### `InstallSnapshotResponse`

#### `ConfigChangeType`

#### `ConfigChangeRequest`

#### `applyConfigChange`

#### `TcpTransport`

#### `TransportConfig`

#### `TransportError`

#### `TransportStats`

#### `MessageType`

#### `MessageHeader`

#### `PeerConnection`

#### `RpcSerializer`

#### `parseAddress`

#### `RaftTransport`

#### `RaftTransportConfig`

#### `RaftTransportStats`

#### `PeerAddress`

#### `CircuitBreaker`

#### `CircuitConfig`

#### `CircuitState`

#### `CircuitRegistry`

#### `CircuitStats`

#### `CircuitMetrics`

#### `CircuitMetricEntry`

#### `NetworkOperationError`

#### `AggregateStats`

#### `NetworkError`

#### `NetworkConfig`

#### `NetworkState`

#### `init`

#### `deinit`

#### `isEnabled`

#### `isInitialized`

#### `init`

#### `initWithConfig`

#### `deinit`

#### `defaultRegistry`

#### `defaultConfig`


---

<a name="runtime"></a>
## runtime Module

**Source:** `src/compute/runtime/mod.zig`

### Public API

#### `engine`

#### `async`

#### `benchmark`

#### `workload`

#### `future`

#### `cancellation`

#### `task_group`

#### `DistributedComputeEngine`

#### `EngineConfig`

#### `EngineError`

#### `TaskId`

#### `BenchmarkResult`

#### `runBenchmarks`

#### `ExecutionContext`

#### `WorkloadHints`

#### `WorkloadVTable`

#### `ResultHandle`

#### `ResultVTable`

#### `WorkItem`

#### `GPUWorkloadVTable`

#### `runWorkItem`

#### `matMul`

#### `dense`

#### `relu`

#### `MatrixMultiplyTask`

#### `MlpTask`

#### `AsyncRuntime`

#### `AsyncRuntimeOptions`

#### `TaskHandle`

#### `AsyncTaskGroup`

#### `AsyncError`

#### `Future`

#### `FutureState`

#### `FutureResult`

#### `Promise`

#### `CancellationToken`

#### `all`

#### `race`

#### `delay`

#### `CancellationSource`

#### `CancellationState`

#### `CancellationReason`

#### `LinkedCancellation`

#### `ScopedCancellation`

#### `TaskGroup`

#### `TaskGroupConfig`

#### `TaskGroupBuilder`

#### `ScopedTaskGroup`

#### `TaskContext`

#### `TaskFn`

#### `TaskState`

#### `TaskResult`

#### `TaskInfo`

#### `GroupStats`

#### `parallelForEach`

#### `createEngine`

#### `submitTask`

#### `waitForResult`

#### `runTask`

#### `runWorkload`

Alias for runTask() - runs a workload and waits for the result @param engine_instance The compute engine instance @param ResultType The expected result type @param work The workload/task to execute @param timeout_ms Timeout in milliseconds (0=immediate check, null=wait indefinitely) @return The workload result


---


---

*Generated for 6/6 modules*
