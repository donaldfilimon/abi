# Code API Index (Scanned)

Scanned 205 Zig files under `src/`. This index lists public declarations discovered along with leading doc comments.

## src/examples/advanced_zig_gpu.zig

- type `SpirvShader`

SPIR-V shader module for GPU compute operations
Using Zig's native SPIR-V support without vendor-specific toolchains


```zig
pub const SpirvShader = struct {
```

- fn `createComputeShader`

Create a simple compute shader in SPIR-V format


```zig
pub fn createComputeShader(allocator: std.mem.Allocator, workgroup_size: [3]u32) !SpirvShader {
```

- fn `deinit`

```zig
pub fn deinit(self: SpirvShader) void {
```

- type `GpuBuffer`

Advanced GPU buffer with memory mapping and synchronization


```zig
pub const GpuBuffer = struct {
```

- fn `create`

Create a GPU buffer with specified properties


```zig
pub fn create(allocator: std.mem.Allocator, size: usize, host_visible: bool) !GpuBuffer {
```

- fn `map`

Map buffer to CPU address space (if host-visible)


```zig
pub fn map(self: *GpuBuffer) ![]u8 {
```

- fn `unmap`

Unmap buffer from CPU address space


```zig
pub fn unmap(self: *GpuBuffer) void {
```

- fn `copyFromHost`

Copy data to buffer with bounds checking


```zig
pub fn copyFromHost(self: *GpuBuffer, source: []const u8, offset: usize) !void {
```

- fn `copyToHost`

Copy data from buffer to host memory


```zig
pub fn copyToHost(self: *GpuBuffer, destination: []u8, offset: usize) !void {
```

- fn `deinit`

```zig
pub fn deinit(self: *GpuBuffer) void {
```

- type `VulkanComputePipeline`

Vulkan compute pipeline using Zig's Vulkan bindings
This demonstrates how Zig can directly interface with Vulkan without C bindings


```zig
pub const VulkanComputePipeline = struct {
```

- fn `init`

Initialize Vulkan compute pipeline


```zig
pub fn init(allocator: std.mem.Allocator, shader: SpirvShader) !VulkanComputePipeline {
```

- fn `dispatch`

Dispatch compute work to GPU


```zig
pub fn dispatch(self: *VulkanComputePipeline, workgroup_count: [3]u32) !void {
```

- fn `deinit`

```zig
pub fn deinit(self: *VulkanComputePipeline) void {
```

- type `WebGpuComputePipeline`

WebGPU compute pipeline for browser and cross-platform compatibility
Demonstrates Zig's WebGPU support for web deployment


```zig
pub const WebGpuComputePipeline = struct {
```

- fn `init`

Initialize WebGPU compute pipeline


```zig
pub fn init(allocator: std.mem.Allocator, shader: SpirvShader) !WebGpuComputePipeline {
```

- fn `dispatch`

Dispatch compute work using WebGPU


```zig
pub fn dispatch(self: *WebGpuComputePipeline, workgroup_count: [3]u32) !void {
```

- fn `deinit`

```zig
pub fn deinit(self: *WebGpuComputePipeline) void {
```

- type `GpuMemoryPool`

Advanced GPU memory pool with defragmentation


```zig
pub const GpuMemoryPool = struct {
```

- type `MemoryBlock`

```zig
pub const MemoryBlock = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, total_size: usize) !GpuMemoryPool {
```

- fn `alloc`

Allocate memory from pool using first-fit algorithm


```zig
pub fn alloc(self: *GpuMemoryPool, size: usize, alignment: usize) !usize {
```

- fn `free`

Free memory back to pool


```zig
pub fn free(self: *GpuMemoryPool, allocation_id: usize) !void {
```

- fn `getStats`

Get memory usage statistics


```zig
pub fn getStats(self: *GpuMemoryPool) MemoryStats {
```

- type `MemoryStats`

```zig
pub const MemoryStats = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *GpuMemoryPool) void {
```

- fn `main`

Main demonstration function


```zig
pub fn main() !void {
```

## src/examples/ai.zig

- const `ai`

```zig
pub const ai = @import("../features/ai/mod.zig");
```

## src/examples/ai_demo.zig

- type `DenseLayer`

Simple neural network layer


```zig
pub const DenseLayer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*DenseLayer {
```

- fn `deinit`

```zig
pub fn deinit(self: *DenseLayer) void {
```

- fn `forward`

```zig
pub fn forward(self: *DenseLayer, input: []const f32, output: []f32) void {
```

- fn `relu`

Simple ReLU activation function


```zig
pub fn relu(x: f32) f32 {
```

- fn `softmax`

Simple softmax activation for classification


```zig
pub fn softmax(input: []f32, output: []f32) void {
```

- type `SimpleNN`

Simple feed-forward neural network


```zig
pub const SimpleNN = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize) !*SimpleNN {
```

- fn `deinit`

```zig
pub fn deinit(self: *SimpleNN) void {
```

- fn `forward`

```zig
pub fn forward(self: *SimpleNN, input: []const f32, output: []f32) void {
```

- fn `main`

```zig
pub fn main() !void {
```

## src/examples/demo_http_client.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/examples/enterprise_features_demo.zig

- type `EnterpriseMLPlatform`

Enterprise ML Platform


```zig
pub const EnterpriseMLPlatform = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*EnterpriseMLPlatform {
```

- fn `deinit`

```zig
pub fn deinit(self: *EnterpriseMLPlatform) void {
```

- fn `registerProductionModel`

Register a production model with comprehensive tracking


```zig
pub fn registerProductionModel(
```

- fn `evaluateModel`

Perform comprehensive model evaluation


```zig
pub fn evaluateModel(
```

- fn `deployToProduction`

Deploy model to production with monitoring


```zig
pub fn deployToProduction(self: *EnterpriseMLPlatform, model_id: []const u8) !void {
```

- fn `runSystemMonitoring`

Comprehensive system monitoring


```zig
pub fn runSystemMonitoring(self: *EnterpriseMLPlatform) !void {
```

- type `SecurityAuditor`

Security auditor for compliance and audit logging


```zig
pub const SecurityAuditor = struct {
```

- type `AuditEvent`

```zig
pub const AuditEvent = struct {
```

- type `EventType`

```zig
pub const EventType = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) SecurityAuditor {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityAuditor) void {
```

- fn `logEvent`

```zig
pub fn logEvent(self: *SecurityAuditor, event: AuditEvent) !void {
```

- fn `getAuditLog`

```zig
pub fn getAuditLog(self: *SecurityAuditor) []AuditEvent {
```

- fn `generateSecurityReport`

```zig
pub fn generateSecurityReport(self: *SecurityAuditor) !void {
```

- fn `main`

Main demonstration function


```zig
pub fn main() !void {
```

## src/examples/gpu.zig

- const `gpu`

```zig
pub const gpu = @import("../features/gpu/mod.zig");
```

## src/examples/gpu_acceleration_demo.zig

- type `GPUTrainer`

GPU-accelerated neural network trainer


```zig
pub const GPUTrainer = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *GPUTrainer) void {
```

- fn `trainDistributed`

Train with GPU acceleration and distributed processing


```zig
pub fn trainDistributed(
```

## src/examples/gpu_ai_acceleration_demo.zig

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
```

- fn `initWithData`

```zig
pub fn initWithData(allocator: std.mem.Allocator, shape: []const usize, values: []const f32) !*Tensor {
```

- fn `size`

```zig
pub fn size(self: Tensor) usize {
```

- fn `deinit`

```zig
pub fn deinit(self: *Tensor) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) MatrixOps {
```

- fn `matmul`

```zig
pub fn matmul(self: *const MatrixOps, a: *Tensor, b: *Tensor, c: *Tensor) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_size: usize, hidden_size: usize, output_size: usize) !*SimpleNeuralNetwork {
```

- fn `deinit`

```zig
pub fn deinit(self: *SimpleNeuralNetwork) void {
```

- fn `forward`

Forward pass through the network


```zig
pub fn forward(self: *SimpleNeuralNetwork, input: []const f32, output: []f32) !void {
```

- fn `main`

Main demonstration function


```zig
pub fn main() !void {
```

## src/examples/gpu_neural_network_integration.zig

- type `DenseLayer`

Simple neural network layer


```zig
pub const DenseLayer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize) !*DenseLayer {
```

- fn `deinit`

```zig
pub fn deinit(self: *DenseLayer) void {
```

- fn `forward`

```zig
pub fn forward(self: *DenseLayer, input: []const f32, output: []f32) void {
```

- type `NeuralNetwork`

Neural network that could be GPU-accelerated


```zig
pub const NeuralNetwork = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, use_gpu: bool) !*NeuralNetwork {
```

- fn `deinit`

```zig
pub fn deinit(self: *NeuralNetwork) void {
```

- fn `addLayer`

```zig
pub fn addLayer(self: *NeuralNetwork, input_size: usize, output_size: usize) !void {
```

- fn `forward`

```zig
pub fn forward(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
```

- fn `main`

```zig
pub fn main() !void {
```

## src/examples/monitoring.zig

- type `Monitoring`

```zig
pub const Monitoring = struct {
```

- fn `printHello`

```zig
pub fn printHello() void {
```

## src/examples/plugins/example_plugin.zig

- fn `isCompatible`

```zig
pub fn isCompatible(self: PluginInfo, framework_abi: PluginVersion) bool {
```

- fn `init`

```zig
pub fn init(major: u32, minor: u32, patch: u32) PluginVersion {
```

- fn `isCompatible`

```zig
pub fn isCompatible(self: PluginVersion, required: PluginVersion) bool {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) PluginConfig {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginConfig) void {
```

- fn `log`

```zig
pub fn log(self: *PluginContext, level: u8, message: []const u8) void {
```

## src/examples/rl_complete_example.zig

- type `SimpleEnvironment`

Simple CartPole-like environment for demonstration


```zig
pub const SimpleEnvironment = struct {
```

- fn `init`

```zig
pub fn init() SimpleEnvironment {
```

- fn `reset`

```zig
pub fn reset(self: *SimpleEnvironment) []f32 {
```

- fn `step`

```zig
pub fn step(self: *SimpleEnvironment, action: usize) struct { state: []f32, reward: f32, done: bool } {
```

- fn `main`

Complete RL training demonstration


```zig
pub fn main() !void {
```

## src/examples/showcase_optimizations.zig

- pub `inline`

Fast inline square root approximation


```zig
pub inline fn fastSqrt(x: f32) f32 {
```

- pub `inline`

Inline vector dot product with manual unrolling


```zig
pub inline fn dotProduct(a: []const f32, b: []const f32) f32 {
```

- pub `inline`

Inline matrix-vector multiplication


```zig
pub inline fn matrixVectorMul(matrix: []const f32, vector: []const f32, result: []f32, rows: usize, cols: usize) void {
```

- pub `inline`

Inline lookup table sine approximation


```zig
pub inline fn fastSin(angle_degrees: f32) f32 {
```

- fn `stackVectorAdd`

Stack-based vector operations for small arrays


```zig
pub fn stackVectorAdd(comptime size: usize, a_data: [size]f32, b_data: [size]f32) [size]f32 {
```

- fn `adaptiveVectorOperation`

Adaptive allocation strategy based on size


```zig
pub fn adaptiveVectorOperation(allocator: std.mem.Allocator, size: usize) !void {
```

- fn `runOptimizationShowcase`

```zig
pub fn runOptimizationShowcase(allocator: std.mem.Allocator) !void {
```

- fn `main`

Main function to run the optimization showcase


```zig
pub fn main() !void {
```

## src/examples/transformer_complete_example.zig

- type `CompleteTransformer`

Complete transformer model for sequence processing


```zig
pub const CompleteTransformer = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *CompleteTransformer) void {
```

- fn `forward`

Forward pass through complete transformer


```zig
pub fn forward(self: *CompleteTransformer, input_tokens: []const u32, output: []f32) !void {
```

- fn `generate`

Generate text using the transformer


```zig
pub fn generate(self: *CompleteTransformer, prompt: []const u32, max_length: usize) ![]u32 {
```

- fn `main`

Demonstration of complete transformer capabilities


```zig
pub fn main() !void {
```

## src/examples/transformer_example.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/examples/utilities_demo.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/examples/zig_gpu_build_demo.zig

- type `GpuBackend`

GPU Backend enumeration with priorities


```zig
pub const GpuBackend = enum {
```

- fn `priority`

Get priority value for backend selection


```zig
pub fn priority(self: GpuBackend) u32 {
```

- fn `isAvailable`

Check if backend is available on this platform


```zig
pub fn isAvailable(self: GpuBackend) bool {
```

- fn `name`

Get human-readable name


```zig
pub fn name(self: GpuBackend) []const u8 {
```

- type `GpuCapabilities`

GPU Device capabilities structure


```zig
pub const GpuCapabilities = struct {
```

- fn `detect`

Detect capabilities based on platform


```zig
pub fn detect() GpuCapabilities {
```

- type `GpuBuildConfig`

Build configuration for GPU features


```zig
pub const GpuBuildConfig = struct {
```

- fn `validate`

Validate build configuration


```zig
pub fn validate(self: GpuBuildConfig) !void {
```

- fn `getRecommendedBackend`

Get recommended backend for current platform


```zig
pub fn getRecommendedBackend(self: GpuBuildConfig) GpuBackend {
```

- fn `linkGpuLibraries`

Link libraries based on GPU backend requirements


```zig
pub fn linkGpuLibraries(config: GpuBuildConfig) void {
```

- fn `main`

Main demonstration function


```zig
pub fn main() !void {
```

## src/features/ai/activation.zig

- type `ActivationType`

Available activation function types with detailed mathematical definitions


```zig
pub const ActivationType = enum {
```

- type `ActivationConfig`

Activation function configuration


```zig
pub const ActivationConfig = struct {
```

- type `ActivationProcessor`

High-performance activation function processor


```zig
pub const ActivationProcessor = struct {
```

- fn `init`

```zig
pub fn init(config: ActivationConfig) ActivationProcessor {
```

- fn `activate`

Apply activation function to a single value


```zig
pub fn activate(self: *const ActivationProcessor, x: f32) f32 {
```

- fn `activateBatch`

Apply activation function to an array (with SIMD optimization)


```zig
pub fn activateBatch(self: *const ActivationProcessor, output: []f32, input: []const f32) void {
```

- fn `derivative`

Compute derivative of activation function


```zig
pub fn derivative(self: *const ActivationProcessor, x: f32, y: f32) f32 {
```

- fn `derivativeBatch`

Batch derivative computation


```zig
pub fn derivativeBatch(self: *const ActivationProcessor, output: []f32, input: []const f32, forward_output: []const f32) void {
```

- type `ActivationRegistry`

Activation function registry for dynamic dispatch


```zig
pub const ActivationRegistry = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) ActivationRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *ActivationRegistry) void {
```

- fn `register`

```zig
pub fn register(self: *ActivationRegistry, name: []const u8, func: ActivationFn) !void {
```

- fn `get`

```zig
pub fn get(self: *ActivationRegistry, name: []const u8) ?ActivationFn {
```

## src/features/ai/agent.zig

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- const `AgentError`

Errors that an agent operation can produce.


```zig
pub const AgentError = error{
```

- type `PersonaType`

Simple set of personas that are safe to use across the codebase.


```zig
pub const PersonaType = enum {
```

- type `AgentCapabilities`

Capabilities flag set – kept small for now, can expand later without
breaking the ABI.


```zig
pub const AgentCapabilities = packed struct(u8) {
```

- type `AgentConfig`

Configuration supplied when constructing an agent.


```zig
pub const AgentConfig = struct {
```

- fn `validate`

```zig
pub fn validate(self: AgentConfig) AgentError!void {
```

- type `Agent`

Minimal Agent implementation – tracks persona and a simple message history.


```zig
pub const Agent = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: AgentConfig) AgentError!*Agent {
```

- fn `deinit`

```zig
pub fn deinit(self: *Agent) void {
```

- fn `process`

Returns a copy of the response so callers can manage lifetime.


```zig
pub fn process(self: *Agent, input: []const u8, allocator: Allocator) AgentError![]const u8 {
```

- fn `clearHistory`

```zig
pub fn clearHistory(self: *Agent) void {
```

- fn `historyCount`

```zig
pub fn historyCount(self: *const Agent) usize {
```

- fn `getPersona`

```zig
pub fn getPersona(self: *const Agent) PersonaType {
```

- fn `setPersona`

```zig
pub fn setPersona(self: *Agent, persona: PersonaType) void {
```

- fn `name`

```zig
pub fn name(self: *const Agent) []const u8 {
```

## src/features/ai/ai_core.zig

- type `ActivationUtils`

High-performance activation function utilities


```zig
pub const ActivationUtils = struct {
```

- pub `inline`

Inline fast approximation functions for better performance


```zig
pub inline fn fastSigmoid(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastTanh(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastExp(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastGelu(x: f32) f32 {
```

- pub `inline`

```zig
pub inline fn fastSqrt(x: f32) f32 {
```

- pub `inline`

Vectorized ReLU activation with SIMD optimization


```zig
pub inline fn vectorizedRelu(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedSigmoid(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedTanh(data: []f32) void {
```

- pub `inline`

Vectorized Leaky ReLU activation with SIMD optimization


```zig
pub inline fn vectorizedLeakyRelu(data: []f32) void {
```

- pub `inline`

```zig
pub inline fn vectorizedGelu(data: []f32) void {
```

- pub `inline`

Optimized softmax with numerical stability


```zig
pub inline fn stableSoftmax(data: []f32) void {
```

- pub `inline`

Optimized log softmax with numerical stability


```zig
pub inline fn stableLogSoftmax(data: []f32) void {
```

- type `LayerType`

Neural network layer types with enhanced coverage


```zig
pub const LayerType = enum {
```

- type `Activation`

Enhanced activation functions with optimized implementations


```zig
pub const Activation = enum {
```

- pub `inline`

Inline function for quick activation checks


```zig
pub inline fn isNonlinear(self: Activation) bool {
```

- pub `inline`

Inline function for gradient requirements


```zig
pub inline fn requiresGradient(self: Activation) bool {
```

- type `WeightInit`

Comprehensive weight initialization strategies


```zig
pub const WeightInit = enum {
```

- type `Regularization`

Advanced regularization configuration


```zig
pub const Regularization = struct {
```

- type `MemoryStrategy`

Memory allocation strategy


```zig
pub const MemoryStrategy = enum {
```

- type `ComputeBackend`

Computation backend


```zig
pub const ComputeBackend = enum {
```

- type `Layer`

Enhanced neural network layer with comprehensive functionality


```zig
pub const Layer = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, layer_type: LayerType, input_shape: []const usize, output_shape: []const usize) !*Layer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Layer, allocator: Allocator) void {
```

- fn `saveToFile`

Save layer to file


```zig
pub fn saveToFile(self: *Layer, writer: anytype) !void {
```

- fn `loadFromFile`

Load layer from file


```zig
pub fn loadFromFile(allocator: Allocator, reader: *std.fs.File.Reader) !*Layer {
```

- fn `initializeWeights`

```zig
pub fn initializeWeights(self: *Layer, allocator: Allocator, rng: *Random) !void {
```

- fn `forward`

```zig
pub fn forward(self: *Layer, input: []const f32, output: []f32) !void {
```

- type `LossFunction`

Loss functions with comprehensive coverage


```zig
pub const LossFunction = enum {
```

- type `Optimizer`

Optimizers with state-of-the-art algorithms


```zig
pub const Optimizer = enum {
```

- type `LRScheduler`

Learning rate scheduling strategies


```zig
pub const LRScheduler = enum {
```

- type `DataAugmentation`

Data augmentation techniques


```zig
pub const DataAugmentation = struct {
```

- type `TrainingConfig`

Model training configuration with advanced options


```zig
pub const TrainingConfig = struct {
```

- type `TrainingMetrics`

Comprehensive training metrics


```zig
pub const TrainingMetrics = struct {
```

- type `NeuralNetwork`

Neural network model with enhanced capabilities


```zig
pub const NeuralNetwork = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_shape: []const usize, output_shape: []const usize) !*NeuralNetwork {
```

- fn `deinit`

```zig
pub fn deinit(self: *NeuralNetwork) void {
```

- fn `saveToFile`

Save neural network to file in binary format


```zig
pub fn saveToFile(self: *NeuralNetwork, file_path: []const u8) !void {
```

- fn `trainStep`

Train network on a single input-target pair


```zig
pub fn trainStep(self: *NeuralNetwork, input: []const f32, target: []const f32) !f32 {
```

- fn `loadFromFile`

Load neural network from file


```zig
pub fn loadFromFile(allocator: std.mem.Allocator, file_path: []const u8) !*NeuralNetwork {
```

- fn `setTraining`

```zig
pub fn setTraining(self: *NeuralNetwork, is_training: bool) void {
```

- fn `addLayer`

```zig
pub fn addLayer(self: *NeuralNetwork, layer: *Layer) !void {
```

- fn `addDenseLayer`

```zig
pub fn addDenseLayer(self: *NeuralNetwork, units: usize, activation: ?Activation) !void {
```

- fn `addConv2DLayer`

```zig
pub fn addConv2DLayer(self: *NeuralNetwork, filters: usize, kernel_size: [2]usize, activation: ?Activation) !void {
```

- fn `addDropoutLayer`

```zig
pub fn addDropoutLayer(self: *NeuralNetwork, rate: f32) !void {
```

- fn `addBatchNormLayer`

```zig
pub fn addBatchNormLayer(self: *NeuralNetwork) !void {
```

- fn `addLSTMLayer`

```zig
pub fn addLSTMLayer(self: *NeuralNetwork, units: usize, return_sequences: bool) !void {
```

- fn `addAttentionLayer`

```zig
pub fn addAttentionLayer(self: *NeuralNetwork, num_heads: usize, head_dim: usize) !void {
```

- fn `compile`

```zig
pub fn compile(self: *NeuralNetwork) !void {
```

- fn `forward`

```zig
pub fn forward(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
```

- fn `predict`

```zig
pub fn predict(self: *NeuralNetwork, input: []const f32, output: []f32) !void {
```

- fn `predictBatch`

```zig
pub fn predictBatch(self: *NeuralNetwork, inputs: []const []const f32, outputs: [][]f32) !void {
```

- fn `getParameterCount`

```zig
pub fn getParameterCount(self: *const NeuralNetwork) usize {
```

- fn `getMemoryUsage`

```zig
pub fn getMemoryUsage(self: *const NeuralNetwork) usize {
```

- type `EmbeddingGenerator`

Advanced embedding generator with multiple architectures


```zig
pub const EmbeddingGenerator = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize) !*EmbeddingGenerator {
```

- fn `initTransformer`

```zig
pub fn initTransformer(allocator: std.mem.Allocator, input_size: usize, embedding_size: usize, num_heads: usize) !*EmbeddingGenerator {
```

- fn `deinit`

```zig
pub fn deinit(self: *EmbeddingGenerator) void {
```

- fn `generateEmbedding`

```zig
pub fn generateEmbedding(self: *EmbeddingGenerator, input: []const f32, embedding: []f32) !void {
```

- fn `generateEmbeddingsBatch`

```zig
pub fn generateEmbeddingsBatch(self: *EmbeddingGenerator, inputs: []const []const f32, embeddings: [][]f32) !void {
```

- fn `computeSimilarity`

```zig
pub fn computeSimilarity(self: *EmbeddingGenerator, embedding1: []const f32, embedding2: []const f32) f32 {
```

- fn `findNearestNeighbors`

```zig
pub fn findNearestNeighbors(
```

- type `ModelTrainer`

Enhanced model trainer with comprehensive optimization support


```zig
pub const ModelTrainer = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *ModelTrainer) void {
```

- fn `train`

```zig
pub fn train(
```

- const `Network`

```zig
pub const Network = NeuralNetwork;
```

- const `Embedding`

```zig
pub const Embedding = EmbeddingGenerator;
```

- const `transformer`

```zig
pub const transformer = @import("transformer.zig");
```

- const `Neural`

```zig
pub const Neural = @import("neural.zig");
```

- const `LocalML`

```zig
pub const LocalML = @import("localml.zig");
```

- const `DynamicRouter`

```zig
pub const DynamicRouter = @import("dynamic.zig");
```

- const `DataStructures`

```zig
pub const DataStructures = @import("data_structures/mod.zig");
```

- const `Trainer`

```zig
pub const Trainer = ModelTrainer;
```

- const `Config`

```zig
pub const Config = TrainingConfig;
```

- const `Metrics`

```zig
pub const Metrics = TrainingMetrics;
```

- const `Loss`

```zig
pub const Loss = LossFunction;
```

- const `Opt`

```zig
pub const Opt = Optimizer;
```

- const `agent`

```zig
pub const agent = @import("agent.zig");
```

- const `enhanced_agent`

```zig
pub const enhanced_agent = @import("enhanced_agent.zig");
```

- const `reinforcement_learning`

```zig
pub const reinforcement_learning = @import("reinforcement_learning.zig");
```

- const `distributed_training`

```zig
pub const distributed_training = @import("distributed_training.zig");
```

- const `model_serialization`

```zig
pub const model_serialization = @import("model_serialization.zig");
```

- fn `createMLP`

```zig
pub fn createMLP(allocator: std.mem.Allocator, layer_sizes: []const usize, activations: []const Activation) !*NeuralNetwork {
```

- fn `createCNN`

```zig
pub fn createCNN(allocator: std.mem.Allocator, input_shape: []const usize, num_classes: usize) !*NeuralNetwork {
```

## src/features/ai/data_structures/batch_queue.zig

- type `BatchQueue`

High-performance batch queue for processing data in batches


```zig
pub const BatchQueue = struct {
```

- fn `init`

Initialize a new batch queue


```zig
pub fn init(allocator: std.mem.Allocator, batch_size: usize) !*Self {
```

- fn `deinit`

Deinitialize the queue


```zig
pub fn deinit(self: *Self) void {
```

- fn `enqueue`

Add data to the queue


```zig
pub fn enqueue(self: *Self, data: []const u8) !void {
```

- fn `dequeueBatch`

Get the next batch if available


```zig
pub fn dequeueBatch(self: *Self) ?[]u8 {
```

## src/features/ai/data_structures/bloom_filter.zig

- type `BloomFilter`

Bloom filter for efficient set membership testing


```zig
pub const BloomFilter = struct {
```

- fn `init`

Initialize a new bloom filter


```zig
pub fn init(allocator: std.mem.Allocator, size: usize, hash_count: u32) !*Self {
```

- fn `deinit`

Deinitialize the filter


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add an item to the filter


```zig
pub fn add(self: *Self, data: []const u8) void {
```

- fn `contains`

Check if an item might be in the filter


```zig
pub fn contains(self: *Self, data: []const u8) bool {
```

## src/features/ai/data_structures/cache.zig

- fn `ThreadSafeCache`

Thread-safe LRU cache implementation


```zig
pub fn ThreadSafeCache(comptime K: type, comptime V: type) type {
```

- fn `init`

Initialize a new thread-safe cache


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the cache


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get a value from the cache


```zig
pub fn get(self: *Self, key: K) ?V {
```

- fn `put`

Put a value in the cache


```zig
pub fn put(self: *Self, key: K, value: V) !void {
```

- fn `LRUCache`

LRU Cache implementation


```zig
pub fn LRUCache(comptime K: type, comptime V: type) type {
```

- fn `init`

Initialize a new LRU cache


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the cache


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get a value from the cache


```zig
pub fn get(self: *Self, key: K) ?V {
```

- fn `put`

Put a value in the cache


```zig
pub fn put(self: *Self, key: K, value: V) !void {
```

## src/features/ai/data_structures/circular_buffer.zig

- fn `CircularBuffer`

High-performance circular buffer for time series data


```zig
pub fn CircularBuffer(comptime T: type) type {
```

- fn `init`

Initialize a new circular buffer


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the buffer


```zig
pub fn deinit(self: *Self) void {
```

- fn `push`

Add an element to the buffer


```zig
pub fn push(self: *Self, value: T) void {
```

- fn `pop`

Remove and return the oldest element


```zig
pub fn pop(self: *Self) ?T {
```

- fn `RingBuffer`

Alias for CircularBuffer - RingBuffer is the same implementation


```zig
pub fn RingBuffer(comptime T: type) type {
```

## src/features/ai/data_structures/compressed_vector.zig

- type `CompressedVector`

Compressed vector implementation using sparse storage


```zig
pub const CompressedVector = struct {
```

- fn `init`

Initialize a new compressed vector


```zig
pub fn init(allocator: std.mem.Allocator, dimension: usize) !*Self {
```

- fn `deinit`

Deinitialize the vector


```zig
pub fn deinit(self: *Self) void {
```

- fn `set`

Set a value at the specified index


```zig
pub fn set(self: *Self, index: usize, value: f32) !void {
```

- fn `get`

Get a value at the specified index


```zig
pub fn get(self: *Self, index: usize) f32 {
```

- fn `nnz`

Get the number of non-zero elements


```zig
pub fn nnz(self: *Self) usize {
```

- fn `dot`

Calculate dot product with another compressed vector


```zig
pub fn dot(self: *Self, other: *const Self) f32 {
```

- fn `add`

Add another compressed vector to this one


```zig
pub fn add(self: *Self, other: *const Self) !void {
```

## src/features/ai/data_structures/dense_matrix.zig

- type `DenseMatrix`

Dense matrix implementation with contiguous memory layout


```zig
pub const DenseMatrix = struct {
```

- fn `init`

Initialize a new dense matrix


```zig
pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*Self {
```

- fn `deinit`

Deinitialize the matrix


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get element at position (i, j)


```zig
pub fn get(self: *Self, i: usize, j: usize) f32 {
```

- fn `set`

Set element at position (i, j)


```zig
pub fn set(self: *Self, i: usize, j: usize, value: f32) void {
```

- fn `getRow`

Get a row as a slice


```zig
pub fn getRow(self: *Self, i: usize) ?[]f32 {
```

- fn `getCol`

Get a column as a slice (creates a copy)


```zig
pub fn getCol(self: *Self, j: usize) ![]f32 {
```

- fn `mul`

Matrix multiplication (self * other)


```zig
pub fn mul(self: *Self, other: *const Self) !*Self {
```

- fn `add`

Element-wise addition


```zig
pub fn add(self: *Self, other: *const Self) !void {
```

- fn `mulScalar`

Scalar multiplication


```zig
pub fn mulScalar(self: *Self, scalar: f32) void {
```

## src/features/ai/data_structures/graph.zig

- type `Graph`

Generic graph implementation


```zig
pub const Graph = struct {
```

- fn `init`

Initialize a new graph


```zig
pub fn init(allocator: std.mem.Allocator, vertices: usize, directed: bool) !*Self {
```

- fn `deinit`

Deinitialize the graph


```zig
pub fn deinit(self: *Self) void {
```

- fn `addEdge`

Add an edge between two vertices


```zig
pub fn addEdge(self: *Self, from: usize, to: usize) !void {
```

- fn `removeEdge`

Remove an edge between two vertices


```zig
pub fn removeEdge(self: *Self, from: usize, to: usize) !void {
```

- fn `getNeighbors`

Get neighbors of a vertex


```zig
pub fn getNeighbors(self: *Self, vertex: usize) ?[]usize {
```

- fn `bfs`

Perform breadth-first search


```zig
pub fn bfs(self: *Self, start: usize, visitor: anytype) !void {
```

- fn `dfs`

Perform depth-first search


```zig
pub fn dfs(self: *Self, start: usize, visitor: anytype) !void {
```

- type `DirectedGraph`

Directed graph (alias for Graph with directed=true)


```zig
pub const DirectedGraph = struct {
```

- fn `init`

Initialize a new directed graph


```zig
pub fn init(allocator: std.mem.Allocator, vertices: usize) !*Self {
```

- fn `deinit`

Deinitialize the graph


```zig
pub fn deinit(self: *Self) void {
```

- fn `addEdge`

Add a directed edge


```zig
pub fn addEdge(self: *Self, from: usize, to: usize) !void {
```

- fn `getNeighbors`

Get neighbors (outgoing edges)


```zig
pub fn getNeighbors(self: *Self, vertex: usize) ?[]usize {
```

- fn `getReverseNeighbors`

Get reverse neighbors (incoming edges)


```zig
pub fn getReverseNeighbors(self: *Self, vertex: usize) !std.ArrayList(usize) {
```

- type `BipartiteGraph`

Bipartite graph implementation


```zig
pub const BipartiteGraph = struct {
```

- fn `init`

Initialize a new bipartite graph


```zig
pub fn init(allocator: std.mem.Allocator, size_a: usize, size_b: usize) !*Self {
```

- fn `deinit`

Deinitialize the graph


```zig
pub fn deinit(self: *Self) void {
```

- fn `addEdge`

Add an edge between sets (only allowed between different sets)


```zig
pub fn addEdge(self: *Self, a_vertex: usize, b_vertex: usize) !void {
```

- fn `validateBipartite`

Check if the graph is bipartite (validate no edges within same set)


```zig
pub fn validateBipartite(self: *Self) bool {
```

- fn `getSetA`

Get vertices in set A


```zig
pub fn getSetA(self: *Self) std.ArrayList(usize) {
```

- fn `getSetB`

Get vertices in set B


```zig
pub fn getSetB(self: *Self) std.ArrayList(usize) {
```

## src/features/ai/data_structures/lockfree.zig

- const `LockFreeError`

Lock-free data structure errors


```zig
pub const LockFreeError = error{
```

- type `LockFreeStats`

Performance statistics for lock-free operations


```zig
pub const LockFreeStats = struct {
```

- fn `recordOperation`

```zig
pub fn recordOperation(self: *LockFreeStats, success: bool, latency_ns: u64) void {
```

- fn `successRate`

```zig
pub fn successRate(self: *const LockFreeStats) f32 {
```

- fn `lockFreeQueue`

Lock-free queue using Michael & Scott algorithm


```zig
pub fn lockFreeQueue(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `enqueue`

```zig
pub fn enqueue(self: *Self, data: T) !void {
```

- fn `dequeue`

```zig
pub fn dequeue(self: *Self) ?T {
```

- fn `lockFreeStack`

Lock-free stack using Treiber algorithm


```zig
pub fn lockFreeStack(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `push`

```zig
pub fn push(self: *Self, data: T) !void {
```

- fn `pop`

```zig
pub fn pop(self: *Self) ?T {
```

- fn `lockFreeHashMap`

Lock-free hash map using hopscotch hashing


```zig
pub fn lockFreeHashMap(comptime K: type, comptime V: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `put`

```zig
pub fn put(self: *Self, key: K, value: V) !bool {
```

- fn `get`

```zig
pub fn get(self: *Self, key: K) ?V {
```

- fn `workStealingDeque`

Lock-free work-stealing deque


```zig
pub fn workStealingDeque(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `push`

```zig
pub fn push(self: *Self, item: T) bool {
```

- fn `pop`

```zig
pub fn pop(self: *Self) ?T {
```

- fn `steal`

```zig
pub fn steal(self: *Self) ?T {
```

- fn `mpmcQueue`

Multi-producer, multi-consumer queue with batching


```zig
pub fn mpmcQueue(comptime T: type, comptime capacity: usize) type {
```

- fn `init`

```zig
pub fn init() Self {
```

- fn `enqueue`

```zig
pub fn enqueue(self: *Self, item: T) bool {
```

- fn `dequeue`

```zig
pub fn dequeue(self: *Self) ?T {
```

## src/features/ai/data_structures/memory_pool.zig

- fn `MemoryPool`

Generic memory pool for object reuse


```zig
pub fn MemoryPool(comptime T: type) type {
```

- fn `init`

Initialize a new memory pool


```zig
pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the pool


```zig
pub fn deinit(self: *Self) void {
```

- fn `get`

Get an object from the pool


```zig
pub fn get(self: *Self) ?*T {
```

- fn `put`

Return an object to the pool


```zig
pub fn put(self: *Self, object: *T) void {
```

## src/features/ai/data_structures/mod.zig

- const `lockFreeQueue`

```zig
pub const lockFreeQueue = @import("lockfree.zig").lockFreeQueue;
```

- const `LockFreeStack`

```zig
pub const LockFreeStack = @import("lockfree.zig").lockFreeStack;
```

- const `lockFreeHashMap`

```zig
pub const lockFreeHashMap = @import("lockfree.zig").lockFreeHashMap;
```

- const `workStealingDeque`

```zig
pub const workStealingDeque = @import("lockfree.zig").workStealingDeque;
```

- const `mpmcQueue`

```zig
pub const mpmcQueue = @import("lockfree.zig").mpmcQueue;
```

- const `CircularBuffer`

```zig
pub const CircularBuffer = @import("circular_buffer.zig").CircularBuffer;
```

- const `RingBuffer`

```zig
pub const RingBuffer = @import("circular_buffer.zig").RingBuffer;
```

- const `BatchQueue`

```zig
pub const BatchQueue = @import("batch_queue.zig").BatchQueue;
```

- const `MemoryPool`

```zig
pub const MemoryPool = @import("memory_pool.zig").MemoryPool;
```

- const `ObjectPool`

```zig
pub const ObjectPool = @import("object_pool.zig").ObjectPool;
```

- const `ThreadSafeCache`

```zig
pub const ThreadSafeCache = @import("cache.zig").ThreadSafeCache;
```

- const `LRUCache`

```zig
pub const LRUCache = @import("cache.zig").LRUCache;
```

- const `BloomFilter`

```zig
pub const BloomFilter = @import("bloom_filter.zig").BloomFilter;
```

- const `CountMinSketch`

```zig
pub const CountMinSketch = @import("probabilistic.zig").CountMinSketch;
```

- const `HyperLogLog`

```zig
pub const HyperLogLog = @import("probabilistic.zig").HyperLogLog;
```

- const `VectorStore`

```zig
pub const VectorStore = @import("vector_store.zig").VectorStore;
```

- const `SparseMatrix`

```zig
pub const SparseMatrix = @import("sparse_matrix.zig").SparseMatrix;
```

- const `DenseMatrix`

```zig
pub const DenseMatrix = @import("dense_matrix.zig").DenseMatrix;
```

- const `CompressedVector`

```zig
pub const CompressedVector = @import("compressed_vector.zig").CompressedVector;
```

- const `KDTree`

```zig
pub const KDTree = @import("spatial.zig").KDTree;
```

- const `QuadTree`

```zig
pub const QuadTree = @import("spatial.zig").QuadTree;
```

- const `BallTree`

```zig
pub const BallTree = @import("spatial.zig").BallTree;
```

- const `LSHForest`

```zig
pub const LSHForest = @import("spatial.zig").LSHForest;
```

- const `Graph`

```zig
pub const Graph = @import("graph.zig").Graph;
```

- const `DirectedGraph`

```zig
pub const DirectedGraph = @import("graph.zig").DirectedGraph;
```

- const `BipartiteGraph`

```zig
pub const BipartiteGraph = @import("graph.zig").BipartiteGraph;
```

- const `TimeSeries`

```zig
pub const TimeSeries = @import("time_series.zig").TimeSeries;
```

- const `TimeSeriesBuffer`

```zig
pub const TimeSeriesBuffer = @import("time_series.zig").TimeSeriesBuffer;
```

- const `SlidingWindow`

```zig
pub const SlidingWindow = @import("sliding_window.zig").SlidingWindow;
```

- const `ExponentialMovingAverage`

```zig
pub const ExponentialMovingAverage = @import("statistics.zig").ExponentialMovingAverage;
```

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- type `DataStructureConfig`

Configuration for data structure initialization


```zig
pub const DataStructureConfig = struct {
```

- type `DataStructureStats`

Performance statistics for data structures


```zig
pub const DataStructureStats = struct {
```

- fn `reset`

Reset all statistics


```zig
pub fn reset(self: *DataStructureStats) void {
```

- fn `recordOperation`

Update operation statistics


```zig
pub fn recordOperation(self: *DataStructureStats, success: bool, latency_ns: u64) void {
```

- fn `createLockFreeQueue`

Initialize a lock-free queue with the specified capacity


```zig
pub fn createLockFreeQueue(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*lockFreeQueue(T) {
```

- fn `createLockFreeStack`

Initialize a lock-free stack with the specified capacity


```zig
pub fn createLockFreeStack(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*LockFreeStack(T) {
```

- fn `createLockFreeHashMap`

Initialize a concurrent hash map with the specified capacity


```zig
pub fn createLockFreeHashMap(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*lockFreeHashMap(K, V) {
```

- fn `createCircularBuffer`

Initialize a circular buffer for time series data


```zig
pub fn createCircularBuffer(comptime T: type, allocator: std.mem.Allocator, capacity: usize) !*CircularBuffer(T) {
```

- fn `createMemoryPool`

Initialize a memory pool for object reuse


```zig
pub fn createMemoryPool(comptime T: type, allocator: std.mem.Allocator, pool_size: usize) !*MemoryPool(T) {
```

- fn `createLRUCache`

Initialize a thread-safe LRU cache


```zig
pub fn createLRUCache(comptime K: type, comptime V: type, allocator: std.mem.Allocator, capacity: usize) !*LRUCache(K, V) {
```

- fn `createVectorStore`

Initialize a vector store for embedding storage and similarity search


```zig
pub fn createVectorStore(comptime T: type, allocator: std.mem.Allocator, dimensions: usize, capacity: usize) !*VectorStore(T) {
```

- fn `createKDTree`

Initialize a KD-tree for spatial indexing


```zig
pub fn createKDTree(comptime T: type, allocator: std.mem.Allocator, dimensions: usize) !*KDTree(T) {
```

- fn `createSparseMatrix`

Initialize a sparse matrix for efficient storage of sparse data


```zig
pub fn createSparseMatrix(comptime T: type, allocator: std.mem.Allocator, rows: usize, cols: usize) !*SparseMatrix(T) {
```

- fn `createTimeSeriesBuffer`

Initialize a time series buffer with automatic windowing


```zig
pub fn createTimeSeriesBuffer(comptime T: type, allocator: std.mem.Allocator, window_size: usize) !*TimeSeriesBuffer(T) {
```

- type `DataStructureFactory`

Data structure factory for creating optimized instances based on use case


```zig
pub const DataStructureFactory = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: DataStructureConfig) DataStructureFactory {
```

- fn `createOptimizedQueue`

Create an optimized queue for the specified use case


```zig
pub fn createOptimizedQueue(self: *DataStructureFactory, comptime T: type, use_case: enum { high_throughput, low_latency, memory_efficient }) !*lockFreeQueue(T) {
```

- fn `createOptimizedCache`

Create an optimized cache for the specified access pattern


```zig
pub fn createOptimizedCache(self: *DataStructureFactory, comptime K: type, comptime V: type, access_pattern: enum { temporal, random, sequential }) !*LRUCache(K, V) {
```

- fn `getStats`

Get current statistics


```zig
pub fn getStats(self: DataStructureFactory) DataStructureStats {
```

## src/features/ai/data_structures/object_pool.zig

- fn `ObjectPool`

Generic object pool for type-safe object reuse


```zig
pub fn ObjectPool(comptime T: type) type {
```

- fn `init`

Initialize a new object pool


```zig
pub fn init(allocator: std.mem.Allocator, initial_capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the pool


```zig
pub fn deinit(self: *Self) void {
```

- fn `acquire`

Get an object from the pool


```zig
pub fn acquire(self: *Self) ?*T {
```

- fn `release`

Return an object to the pool


```zig
pub fn release(self: *Self, object: *T) void {
```

## src/features/ai/data_structures/probabilistic.zig

- type `CountMinSketch`

Count-Min Sketch for frequency estimation


```zig
pub const CountMinSketch = struct {
```

- fn `init`

Initialize a new Count-Min Sketch


```zig
pub fn init(allocator: std.mem.Allocator, depth: usize, width: usize) !*Self {
```

- fn `deinit`

Deinitialize the sketch


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add an item to the sketch


```zig
pub fn add(self: *Self, data: []const u8) void {
```

- fn `estimate`

Estimate the frequency of an item


```zig
pub fn estimate(self: *Self, data: []const u8) u32 {
```

- type `HyperLogLog`

HyperLogLog for cardinality estimation


```zig
pub const HyperLogLog = struct {
```

- fn `init`

Initialize a new HyperLogLog


```zig
pub fn init(allocator: std.mem.Allocator, b: u32) !*Self {
```

- fn `deinit`

Deinitialize the HyperLogLog


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add an item to the HyperLogLog


```zig
pub fn add(self: *Self, data: []const u8) void {
```

- fn `estimate`

Estimate the cardinality


```zig
pub fn estimate(self: *Self) usize {
```

## src/features/ai/data_structures/sliding_window.zig

- type `SlidingWindow`

Sliding window data structure


```zig
pub const SlidingWindow = struct {
```

- fn `init`

Initialize a new sliding window


```zig
pub fn init(allocator: std.mem.Allocator, max_size: usize) !*Self {
```

- fn `deinit`

Deinitialize the window


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add a value to the window


```zig
pub fn add(self: *Self, value: f32) void {
```

- fn `size`

Get current window size


```zig
pub fn size(self: *Self) usize {
```

- fn `isFull`

Check if window is full


```zig
pub fn isFull(self: *Self) bool {
```

- fn `average`

Get average of current window


```zig
pub fn average(self: *Self) f32 {
```

- fn `min`

Get minimum value in window


```zig
pub fn min(self: *Self) f32 {
```

- fn `max`

Get maximum value in window


```zig
pub fn max(self: *Self) f32 {
```

- fn `stdDev`

Get standard deviation of current window


```zig
pub fn stdDev(self: *Self) f32 {
```

- fn `get`

Get value at specific index (0 = newest)


```zig
pub fn get(self: *Self, index: usize) ?f32 {
```

- fn `clear`

Clear all data from the window


```zig
pub fn clear(self: *Self) void {
```

- fn `getAll`

Get all values as a slice


```zig
pub fn getAll(self: *Self) []f32 {
```

## src/features/ai/data_structures/sparse_matrix.zig

- fn `SparseMatrix`

Sparse matrix implementation using COO (Coordinate) format


```zig
pub fn SparseMatrix(comptime T: type) type {
```

- fn `init`

Initialize a new sparse matrix


```zig
pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !*Self {
```

- fn `deinit`

Deinitialize the matrix


```zig
pub fn deinit(self: *Self) void {
```

- fn `set`

Set a value at the specified position


```zig
pub fn set(self: *Self, row: usize, col: usize, value: T) !void {
```

- fn `get`

Get a value at the specified position


```zig
pub fn get(self: *Self, row: usize, col: usize) T {
```

- fn `nnz`

Get the number of non-zero elements


```zig
pub fn nnz(self: *Self) usize {
```

## src/features/ai/data_structures/spatial.zig

- fn `KDTree`

KD-tree for efficient nearest neighbor search in k-dimensional space


```zig
pub fn KDTree(comptime T: type) type {
```

- fn `init`

Initialize a new KD-tree


```zig
pub fn init(allocator: std.mem.Allocator, dimensions: usize) !*Self {
```

- fn `deinit`

Deinitialize the tree


```zig
pub fn deinit(self: *Self) void {
```

- fn `insert`

Insert a point into the tree


```zig
pub fn insert(self: *Self, point: []const T) !void {
```

- fn `nearestNeighbor`

Find nearest neighbor to a query point


```zig
pub fn nearestNeighbor(self: *Self, query: []const T) !?[]T {
```

- type `QuadTree`

Quad-tree for 2D spatial indexing


```zig
pub const QuadTree = struct {
```

- fn `init`

Initialize a new quad tree


```zig
pub fn init(allocator: std.mem.Allocator, x: f32, y: f32, width: f32, height: f32, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the tree


```zig
pub fn deinit(self: *Self) void {
```

- fn `insert`

Insert a point into the tree


```zig
pub fn insert(self: *Self, x: f32, y: f32) !void {
```

- type `BallTree`

Ball-tree for hierarchical clustering


```zig
pub const BallTree = struct {
```

- fn `init`

Initialize a new ball tree


```zig
pub fn init(allocator: std.mem.Allocator, points: []const []const f32) !*Self {
```

- fn `deinit`

Deinitialize the tree


```zig
pub fn deinit(self: *Self) void {
```

- type `LSHForest`

LSH Forest for approximate nearest neighbor search


```zig
pub const LSHForest = struct {
```

- fn `init`

Initialize a new LSH forest


```zig
pub fn init(allocator: std.mem.Allocator, num_tables: usize, num_hashes: usize) !*Self {
```

- fn `deinit`

Deinitialize the forest


```zig
pub fn deinit(self: *Self) void {
```

- fn `add`

Add a point to the forest


```zig
pub fn add(self: *Self, point: []const f32) !void {
```

- fn `query`

Query approximate nearest neighbors


```zig
pub fn query(self: *Self, query_point: []const f32, k: usize) !std.ArrayList([]f32) {
```

## src/features/ai/data_structures/statistics.zig

- type `ExponentialMovingAverage`

Exponential moving average calculator


```zig
pub const ExponentialMovingAverage = struct {
```

- fn `init`

Initialize a new EMA calculator


```zig
pub fn init(allocator: std.mem.Allocator, alpha: f32) !*Self {
```

- fn `deinit`

Deinitialize the EMA calculator


```zig
pub fn deinit(self: *Self) void {
```

- fn `update`

Add a new value and update the EMA


```zig
pub fn update(self: *Self, new_value: f32) void {
```

- fn `get`

Get current EMA value


```zig
pub fn get(self: *Self) f32 {
```

- fn `reset`

Reset the EMA calculator


```zig
pub fn reset(self: *Self) void {
```

- fn `getAlpha`

Get the smoothing factor


```zig
pub fn getAlpha(self: *Self) f32 {
```

- fn `setAlpha`

Set a new smoothing factor


```zig
pub fn setAlpha(self: *Self, alpha: f32) !void {
```

- type `RunningStats`

Running statistics calculator


```zig
pub const RunningStats = struct {
```

- fn `init`

Initialize a new running statistics calculator


```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

Deinitialize the calculator


```zig
pub fn deinit(self: *Self) void {
```

- fn `update`

Add a new value


```zig
pub fn update(self: *Self, value: f32) void {
```

- fn `mean`

Get the mean


```zig
pub fn mean(self: *Self) f32 {
```

- fn `variance`

Get the variance


```zig
pub fn variance(self: *Self) f32 {
```

- fn `stdDev`

Get the standard deviation


```zig
pub fn stdDev(self: *Self) f32 {
```

- fn `min`

Get the minimum value


```zig
pub fn min(self: *Self) f32 {
```

- fn `max`

Get the maximum value


```zig
pub fn max(self: *Self) f32 {
```

- fn `range`

Get the range (max - min)


```zig
pub fn range(self: *Self) f32 {
```

- fn `reset`

Reset all statistics


```zig
pub fn reset(self: *Self) void {
```

- fn `summary`

Get a summary of all statistics


```zig
pub fn summary(self: *Self) struct {
```

- type `OnlineVariance`

Online variance calculator using Welford's method


```zig
pub const OnlineVariance = struct {
```

- fn `init`

Initialize a new online variance calculator


```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

Deinitialize the calculator


```zig
pub fn deinit(self: *Self) void {
```

- fn `update`

Add a new value


```zig
pub fn update(self: *Self, value: f32) void {
```

- fn `getMean`

Get the current mean


```zig
pub fn getMean(self: *Self) f32 {
```

- fn `getVariance`

Get the current variance


```zig
pub fn getVariance(self: *Self) f32 {
```

- fn `getStdDev`

Get the current standard deviation


```zig
pub fn getStdDev(self: *Self) f32 {
```

- fn `reset`

Reset the calculator


```zig
pub fn reset(self: *Self) void {
```

## src/features/ai/data_structures/time_series.zig

- type `TimeSeriesPoint`

Time series data point


```zig
pub const TimeSeriesPoint = struct {
```

- type `TimeSeriesBuffer`

Time series buffer for storing time-stamped data


```zig
pub const TimeSeriesBuffer = struct {
```

- fn `init`

Initialize a new time series buffer


```zig
pub fn init(allocator: std.mem.Allocator, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the buffer


```zig
pub fn deinit(self: *Self) void {
```

- fn `addPoint`

Add a data point


```zig
pub fn addPoint(self: *Self, timestamp: i64, value: f32) !void {
```

- fn `getValueAt`

Get value at specific timestamp (exact match)


```zig
pub fn getValueAt(self: *Self, timestamp: i64) ?f32 {
```

- fn `getInterpolatedValueAt`

Get interpolated value at timestamp


```zig
pub fn getInterpolatedValueAt(self: *Self, timestamp: i64) ?f32 {
```

- fn `getValuesInRange`

Get values in time range


```zig
pub fn getValuesInRange(self: *Self, start_time: i64, end_time: i64) !std.ArrayList(TimeSeriesPoint) {
```

- fn `simpleMovingAverage`

Calculate simple moving average


```zig
pub fn simpleMovingAverage(self: *Self, window_size: usize) !std.ArrayList(f32) {
```

- fn `getStats`

Get statistics for the time series


```zig
pub fn getStats(self: *Self) struct {
```

- const `TimeSeries`

Time series data structure (alias for TimeSeriesBuffer for backward compatibility)


```zig
pub const TimeSeries = TimeSeriesBuffer;
```

## src/features/ai/data_structures/vector_store.zig

- fn `VectorStore`

Vector store for embedding storage and similarity search


```zig
pub fn VectorStore(comptime T: type) type {
```

- fn `init`

Initialize a new vector store


```zig
pub fn init(allocator: std.mem.Allocator, dimensions: usize, capacity: usize) !*Self {
```

- fn `deinit`

Deinitialize the vector store


```zig
pub fn deinit(self: *Self) void {
```

- fn `addVector`

Add a vector to the store


```zig
pub fn addVector(self: *Self, vector: []const T) !void {
```

- fn `getVector`

Get a vector by index


```zig
pub fn getVector(self: *Self, index: usize) ?[]T {
```

- fn `cosineSimilarity`

Calculate cosine similarity between two vectors


```zig
pub fn cosineSimilarity(self: *Self, a: []const T, b: []const T) f32 {
```

## src/features/ai/distributed_training.zig

- type `DistributedConfig`

Distributed training configuration


```zig
pub const DistributedConfig = struct {
```

- type `AllReduceBackend`

```zig
pub const AllReduceBackend = enum {
```

- type `ParameterServer`

Parameter server for distributed training


```zig
pub const ParameterServer = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, worker_count: usize) !ParameterServer {
```

- fn `deinit`

```zig
pub fn deinit(self: *ParameterServer) void {
```

- fn `registerParameters`

Register model parameters


```zig
pub fn registerParameters(self: *ParameterServer, params: []const []const f32) !void {
```

- fn `pushGradients`

Push gradients from worker


```zig
pub fn pushGradients(self: *ParameterServer, worker_id: usize, gradients: []const []const f32) !void {
```

- fn `pullParameters`

Pull updated parameters for worker


```zig
pub fn pullParameters(self: *ParameterServer, worker_id: usize) ![][]f32 {
```

- fn `applyGradients`

Apply accumulated gradients and reset


```zig
pub fn applyGradients(self: *ParameterServer, learning_rate: f32) !void {
```

- fn `getStats`

Get current parameter statistics


```zig
pub fn getStats(self: *ParameterServer) struct {
```

- type `DistributedTrainer`

Distributed trainer coordinator


```zig
pub const DistributedTrainer = struct {
```

- type `Worker`

```zig
pub const Worker = struct {
```

- type `WorkerStats`

```zig
pub const WorkerStats = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: DistributedConfig, model_factory: *const fn (Allocator) anyerror!*anyopaque) !DistributedTrainer {
```

- fn `deinit`

```zig
pub fn deinit(self: *DistributedTrainer) void {
```

- fn `registerModel`

Register model parameters with parameter server


```zig
pub fn registerModel(self: *DistributedTrainer, model_params: []const []const f32) !void {
```

- fn `train`

Start distributed training


```zig
pub fn train(self: *DistributedTrainer, dataset: []const []const f32, targets: []const []const f32, epochs: usize) !void {
```

- fn `getStats`

Get training statistics


```zig
pub fn getStats(self: *DistributedTrainer) struct {
```

- fn `saveCheckpoint`

Save checkpoint


```zig
pub fn saveCheckpoint(self: *DistributedTrainer, path: []const u8) !void {
```

- type `GradientAllReduce`

Gradient accumulation and all-reduce operations


```zig
pub const GradientAllReduce = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, gradient_size: usize, num_workers: usize) !GradientAllReduce {
```

- fn `deinit`

```zig
pub fn deinit(self: *GradientAllReduce) void {
```

- fn `accumulate`

Accumulate gradients from worker


```zig
pub fn accumulate(self: *GradientAllReduce, worker_gradients: []const f32) !void {
```

- fn `allReduce`

Perform all-reduce operation (average across workers)


```zig
pub fn allReduce(self: *GradientAllReduce) void {
```

- fn `getReducedGradients`

Get final reduced gradients


```zig
pub fn getReducedGradients(self: *GradientAllReduce) []f32 {
```

- fn `reset`

Reset for next accumulation


```zig
pub fn reset(self: *GradientAllReduce) void {
```

- type `MixedPrecision`

Mixed precision training utilities


```zig
pub const MixedPrecision = struct {
```

- type `Precision`

```zig
pub const Precision = enum {
```

- fn `fp32ToFp16`

Convert FP32 to FP16


```zig
pub fn fp32ToFp16(value: f32) u16 {
```

- fn `fp16ToFp32`

Convert FP16 to FP32


```zig
pub fn fp16ToFp32(value: u16) f32 {
```

- type `LossScaler`

Loss scaling for gradient stability in mixed precision


```zig
pub const LossScaler = struct {
```

- fn `init`

```zig
pub fn init(initial_scale: f32, growth_factor: f32, backoff_factor: f32, max_scale: f32) LossScaler {
```

- fn `scaleLoss`

```zig
pub fn scaleLoss(self: LossScaler, loss: f32) f32 {
```

- fn `unscaleGradients`

```zig
pub fn unscaleGradients(self: LossScaler, gradients: []f32) void {
```

- fn `update`

```zig
pub fn update(self: *LossScaler, has_overflow: bool) void {
```

## src/features/ai/dynamic.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `RouterError`

Router-specific error types


```zig
pub const RouterError = error{
```

- type `Persona`

Represents a single conversational persona with basic metrics.


```zig
pub const Persona = struct {
```

- fn `validate`

```zig
pub fn validate(self: Persona) bool {
```

- type `Query`

Represents a user query with context information.


```zig
pub const Query = struct {
```

- fn `validate`

```zig
pub fn validate(self: Query) RouterError!void {
```

- type `TransformerModel`

Placeholder transformer model used to evaluate personas.


```zig
pub const TransformerModel = struct {
```

- fn `scorePersona`

Score a persona for the given query.


```zig
pub fn scorePersona(self: TransformerModel, persona: Persona, query: Query) RouterError!f32 {
```

- type `DynamicPersonaRouter`

Router selects the best persona for a given query.


```zig
pub const DynamicPersonaRouter = struct {
```

- fn `init`

```zig
pub fn init(personas: []const Persona) RouterError!DynamicPersonaRouter {
```

- fn `select`

Select a persona based on query context and user needs.


```zig
pub fn select(self: DynamicPersonaRouter, query: Query) RouterError!Persona {
```

- fn `example`

Example usage of the router.


```zig
pub fn example() !void {
```

## src/features/ai/enhanced_agent.zig

- type `AgentState`

Agent state management with compile-time validation


```zig
pub const AgentState = enum(u8) {
```

- fn `canTransitionTo`

Compile-time state transition validation


```zig
pub fn canTransitionTo(comptime from: AgentState, comptime to: AgentState) bool {
```

- type `AgentCapabilities`

Agent capabilities with packed struct for memory efficiency


```zig
pub const AgentCapabilities = packed struct(u32) {
```

- fn `validateCapabilities`

Compile-time capability validation


```zig
pub fn validateCapabilities(comptime caps: AgentCapabilities) bool {
```

- type `AgentConfig`

Enhanced agent configuration with compile-time optimizations


```zig
pub const AgentConfig = struct {
```

- fn `validate`

Compile-time validation of configuration


```zig
pub fn validate(comptime config: AgentConfig) !void {
```

- type `MemoryEntry`

Advanced memory entry with vectorized operations support


```zig
pub const MemoryEntry = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, content: []const u8, importance: f32) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self, allocator: Allocator) void {
```

- fn `addTag`

```zig
pub fn addTag(self: *Self, allocator: Allocator, key: []const u8, value: []const u8) !void {
```

- fn `updateAccess`

Update access statistics with SIMD-optimized importance calculation


```zig
pub fn updateAccess(self: *Self, enable_simd: bool) void {
```

- fn `init`

```zig
pub fn init(base_allocator: Allocator, pool_size: usize) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `allocator`

```zig
pub fn allocator(self: *Self) Allocator {
```

- fn `getStats`

Get current memory usage statistics


```zig
pub fn getStats(self: *Self) struct { total_allocated: usize, peak_allocated: usize, free_blocks: usize } {
```

- fn `reset`

Reset the allocator, freeing all allocations


```zig
pub fn reset(self: *Self) void {
```

- fn `defragment`

Defragment the free list by merging adjacent blocks


```zig
pub fn defragment(self: *Self) void {
```

- type `EnhancedAgent`

Enhanced AI Agent with advanced performance optimizations


```zig
pub const EnhancedAgent = struct {
```

- type `PerformanceStats`

Enhanced performance tracking with SIMD support


```zig
pub const PerformanceStats = struct {
```

- fn `updateResponseTime`

```zig
pub fn updateResponseTime(self: *PerformanceStats, response_time_ms: f64) void {
```

- fn `recordSuccess`

```zig
pub fn recordSuccess(self: *PerformanceStats) void {
```

- fn `recordFailure`

```zig
pub fn recordFailure(self: *PerformanceStats) void {
```

- fn `init`

Initialize enhanced agent with compile-time validation


```zig
pub fn init(allocator: Allocator, comptime config: AgentConfig) !*Self {
```

- fn `deinit`

Deinitialize agent with proper cleanup


```zig
pub fn deinit(self: *Self) void {
```

- fn `processInput`

Process user input with enhanced error handling and concurrency


```zig
pub fn processInput(self: *Self, input: []const u8) ![]const u8 {
```

- fn `storeMemory`

Enhanced memory storage with SIMD optimization


```zig
pub fn storeMemory(self: *Self, content: []const u8, importance: f32) !void {
```

- fn `getStats`

Enhanced statistics with detailed metrics


```zig
pub fn getStats(self: *const Self) PerformanceStats {
```

- fn `searchMemory`

Enhanced semantic memory search with vector similarity


```zig
pub fn searchMemory(self: *const Self, query: []const u8) ![]MemoryEntry {
```

- fn `learn`

Enhanced learning with reinforcement-based importance adjustment


```zig
pub fn learn(self: *Self, input: []const u8, feedback: f32) !void {
```

- fn `getState`

Get current agent state safely


```zig
pub fn getState(self: *const Self) AgentState {
```

- fn `healthCheck`

Health check for agent status


```zig
pub fn healthCheck(self: *const Self) struct { healthy: bool, issues: []const []const u8 } {
```

## src/features/ai/layer.zig

- type `LayerType`

Neural network layer types with enhanced coverage


```zig
pub const LayerType = enum {
```

- type `WeightInit`

Weight initialization strategies with enhanced coverage


```zig
pub const WeightInit = enum {
```

- type `PaddingMode`

Padding modes for convolution layers


```zig
pub const PaddingMode = enum {
```

- type `PoolingMode`

Pooling modes for pooling layers


```zig
pub const PoolingMode = enum {
```

- type `AttentionType`

Attention mechanisms


```zig
pub const AttentionType = enum {
```

- type `RNNCellType`

RNN cell types


```zig
pub const RNNCellType = enum {
```

- type `Regularization`

Advanced regularization configuration with comprehensive techniques


```zig
pub const Regularization = struct {
```

- type `LayerConfig`

Enhanced layer configuration structure with comprehensive parameters


```zig
pub const LayerConfig = struct {
```

- type `Layer`

Enhanced neural network layer with comprehensive functionality


```zig
pub const Layer = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, config: LayerConfig) anyerror!*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initializeWeights`

Initialize weights and biases for the layer


```zig
pub fn initializeWeights(self: *Self, rng: std.Random) anyerror!void {
```

- fn `forward`

Forward pass through the layer


```zig
pub fn forward(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
```

- fn `backward`

Backward pass through the layer


```zig
pub fn backward(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
```

- fn `getInputSize`

Get input size for the layer


```zig
pub fn getInputSize(self: *const Self) usize {
```

- fn `getOutputSize`

Get output size for the layer


```zig
pub fn getOutputSize(self: *const Self) usize {
```

- fn `setTraining`

Set training mode


```zig
pub fn setTraining(self: *Self, is_training: bool) void {
```

- fn `setInference`

Set inference mode


```zig
pub fn setInference(self: *Self) void {
```

- fn `freeze`

Freeze layer parameters


```zig
pub fn freeze(self: *Self) void {
```

- fn `unfreeze`

Unfreeze layer parameters


```zig
pub fn unfreeze(self: *Self) void {
```

- fn `resetState`

Reset layer state (useful for RNNs)


```zig
pub fn resetState(self: *Self) void {
```

- fn `getMemoryUsage`

Get memory usage of the layer


```zig
pub fn getMemoryUsage(self: *const Self) usize {
```

- fn `getParameterCount`

Get parameter count


```zig
pub fn getParameterCount(self: *const Self) usize {
```

## src/features/ai/localml.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `MLError`

LocalML-specific error types


```zig
pub const MLError = error{
```

- type `DataRow`

Represents a single data point with two features and a label


```zig
pub const DataRow = struct {
```

- fn `validate`

Validates that all values in the data row are finite numbers


```zig
pub fn validate(self: DataRow) MLError!void {
```

- fn `fromArray`

Creates a DataRow from an array of values
Expects exactly 3 values: [x1, x2, label]


```zig
pub fn fromArray(values: []const f64) MLError!DataRow {
```

- fn `toArray`

Converts the DataRow to an array representation


```zig
pub fn toArray(self: DataRow) [3]f64 {
```

- fn `normalize`

Creates a copy of the DataRow with normalized features


```zig
pub fn normalize(self: DataRow, x1_min: f64, x1_max: f64, x2_min: f64, x2_max: f64) DataRow {
```

- fn `distance`

Calculates the Euclidean distance between two data points


```zig
pub fn distance(self: DataRow, other: DataRow) f64 {
```

- type `Model`

A simple linear/logistic regression model


```zig
pub const Model = struct {
```

- fn `init`

Creates a new untrained model with zero-initialized parameters


```zig
pub fn init() Model {
```

- fn `initWithParams`

Creates a model with pre-initialized parameters


```zig
pub fn initWithParams(w1: f64, w2: f64, bias: f64) Model {
```

- fn `predict`

Makes a prediction for a given input
Returns the raw linear combination for regression


```zig
pub fn predict(self: Model, row: DataRow) MLError!f64 {
```

- fn `predictProba`

Makes a classification prediction using logistic function
Returns a probability between 0 and 1


```zig
pub fn predictProba(self: Model, row: DataRow) MLError!f64 {
```

- fn `train`

Trains the model using gradient descent


```zig
pub fn train(self: *Model, data: []const DataRow, learning_rate: f64, epochs: usize) MLError!void {
```

- fn `evaluate`

Evaluates the model on test data and returns mean squared error


```zig
pub fn evaluate(self: Model, test_data: []const DataRow) MLError!f64 {
```

- fn `accuracy`

Calculates classification accuracy on binary classification data


```zig
pub fn accuracy(self: Model, test_data: []const DataRow, threshold: f64) MLError!f64 {
```

- fn `reset`

Resets the model to untrained state


```zig
pub fn reset(self: *Model) void {
```

- fn `toJson`

Serializes the model to JSON format for persistence


```zig
pub fn toJson(self: Model, allocator: Allocator) ![]u8 {
```

- fn `fromJson`

Deserializes a model from JSON format


```zig
pub fn fromJson(allocator: Allocator, json_data: []const u8) !Model {
```

- type `DataProcessor`

Data preprocessing utilities


```zig
pub const DataProcessor = struct {
```

- fn `normalizeDataset`

Normalizes a dataset by scaling features to [0, 1] range


```zig
pub fn normalizeDataset(allocator: Allocator, data: []const DataRow) ![]DataRow {
```

- fn `trainTestSplit`

Splits dataset into training and testing sets


```zig
pub fn trainTestSplit(allocator: Allocator, data: []const DataRow, train_ratio: f64, random_seed: u64) !struct { train: []DataRow, @"test": []DataRow } {
```

- fn `standardizeDataset`

Standardizes a dataset using z-score normalization (mean=0, std=1)


```zig
pub fn standardizeDataset(allocator: Allocator, data: []const DataRow) ![]DataRow {
```

- type `CrossValidator`

Cross-validation utilities


```zig
pub const CrossValidator = struct {
```

- fn `kFoldValidation`

Performs k-fold cross-validation on a model


```zig
pub fn kFoldValidation(allocator: Allocator, data: []const DataRow, k: usize, learning_rate: f64, epochs: usize) !struct { mean_accuracy: f64, std_accuracy: f64 } {
```

- type `KNNClassifier`

K-Nearest Neighbors classifier for non-parametric classification


```zig
pub const KNNClassifier = struct {
```

- fn `init`

```zig
pub fn init(training_data: []const DataRow, k: usize) MLError!KNNClassifier {
```

- fn `predict`

```zig
pub fn predict(self: KNNClassifier, allocator: Allocator, query_point: DataRow) !f64 {
```

- fn `readDataset`

Reads a dataset from a CSV file
Expected format: x1,x2,label (one row per line)


```zig
pub fn readDataset(allocator: std.mem.Allocator, path: []const u8) ![]DataRow {
```

- fn `saveDataset`

Saves a dataset to a CSV file


```zig
pub fn saveDataset(path: []const u8, data: []const DataRow) !void {
```

- fn `saveModel`

Saves a trained model to file in a simple text format


```zig
pub fn saveModel(path: []const u8, model: Model) !void {
```

- fn `loadModel`

Loads a trained model from file


```zig
pub fn loadModel(path: []const u8) !Model {
```

## src/features/ai/mod.zig

- const `neural`

```zig
pub const neural = @import("neural.zig");
```

- const `layer`

```zig
pub const layer = @import("layer.zig");
```

- const `activation`

```zig
pub const activation = @import("activation.zig");
```

- const `localml`

```zig
pub const localml = @import("localml.zig");
```

- const `transformer`

```zig
pub const transformer = @import("transformer.zig");
```

- const `reinforcement_learning`

```zig
pub const reinforcement_learning = @import("reinforcement_learning.zig");
```

- const `enhanced_agent`

```zig
pub const enhanced_agent = @import("enhanced_agent.zig");
```

- const `agent`

```zig
pub const agent = @import("agent.zig");
```

- const `model_serialization`

```zig
pub const model_serialization = @import("model_serialization.zig");
```

- const `model_registry`

```zig
pub const model_registry = @import("model_registry.zig");
```

- const `distributed_training`

```zig
pub const distributed_training = @import("distributed_training.zig");
```

- const `dynamic`

```zig
pub const dynamic = @import("dynamic.zig");
```

- const `data_structures`

```zig
pub const data_structures = @import("data_structures/mod.zig");
```

- const `ai_core`

```zig
pub const ai_core = @import("ai_core.zig");
```

## src/features/ai/model_registry.zig

- type `ModelEntry`

Model registry entry


```zig
pub const ModelEntry = struct {
```

- type `TrainingConfig`

```zig
pub const TrainingConfig = struct {
```

- type `TrainingMetrics`

```zig
pub const TrainingMetrics = struct {
```

- type `DeploymentStatus`

```zig
pub const DeploymentStatus = enum {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, id: []const u8, name: []const u8, version: []const u8) !ModelEntry {
```

- fn `deinit`

```zig
pub fn deinit(self: *ModelEntry, allocator: Allocator) void {
```

- type `ModelRegistry`

Advanced model registry


```zig
pub const ModelRegistry = struct {
```

- type `PerformanceMetrics`

```zig
pub const PerformanceMetrics = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator) ModelRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *ModelRegistry) void {
```

- fn `registerModel`

Register a new model


```zig
pub fn registerModel(self: *ModelRegistry, entry: *ModelEntry) !void {
```

- fn `getModel`

Get model by ID


```zig
pub fn getModel(self: *ModelRegistry, id: []const u8) ?*ModelEntry {
```

- fn `getModelVersions`

Get all versions of a model


```zig
pub fn getModelVersions(self: *ModelRegistry, name: []const u8) ?[]*ModelEntry {
```

- fn `getLatestVersion`

Get latest version of a model


```zig
pub fn getLatestVersion(self: *ModelRegistry, name: []const u8) ?*ModelEntry {
```

- fn `compareModels`

Compare two models


```zig
pub fn compareModels(self: *ModelRegistry, id1: []const u8, id2: []const u8) !ModelComparison {
```

- fn `recordMetrics`

Record performance metrics


```zig
pub fn recordMetrics(self: *ModelRegistry, model_id: []const u8, metrics: PerformanceMetrics) !void {
```

- fn `getPerformanceHistory`

Get performance history


```zig
pub fn getPerformanceHistory(self: *ModelRegistry, model_id: []const u8) ?[]PerformanceMetrics {
```

- fn `promoteToProduction`

Promote model to production


```zig
pub fn promoteToProduction(self: *ModelRegistry, model_id: []const u8) !void {
```

- fn `archiveOldVersions`

Archive old model versions


```zig
pub fn archiveOldVersions(self: *ModelRegistry, model_name: []const u8, keep_versions: usize) !void {
```

- fn `searchByTags`

Search models by tags


```zig
pub fn searchByTags(self: *ModelRegistry, tags: []const []const u8) ![]*ModelEntry {
```

- type `ModelComparison`

Model comparison result


```zig
pub const ModelComparison = struct {
```

- fn `format`

```zig
pub fn format(
```

- type `RegistryCLI`

Model registry CLI interface


```zig
pub const RegistryCLI = struct {
```

- fn `init`

```zig
pub fn init(registry: *ModelRegistry) RegistryCLI {
```

- fn `listModels`

```zig
pub fn listModels(self: RegistryCLI) !void {
```

- fn `showModelDetails`

```zig
pub fn showModelDetails(self: RegistryCLI, model_id: []const u8) !void {
```

- fn `compareModelsCLI`

```zig
pub fn compareModelsCLI(self: RegistryCLI, id1: []const u8, id2: []const u8) !void {
```

## src/features/ai/model_serialization.zig

- type `FormatVersion`

Model serialization format versions


```zig
pub const FormatVersion = enum(u32) {
```

- fn `current`

```zig
pub fn current() FormatVersion {
```

- type `ModelMetadata`

Model metadata structure


```zig
pub const ModelMetadata = struct {
```

- type `TrainingConfig`

```zig
pub const TrainingConfig = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, architecture: []const u8, input_shape: []const usize, output_shape: []const usize) ModelMetadata {
```

- fn `deinit`

```zig
pub fn deinit(self: *ModelMetadata, allocator: Allocator) void {
```

- fn `addCustomField`

```zig
pub fn addCustomField(self: *ModelMetadata, allocator: Allocator, key: []const u8, value: []const u8) !void {
```

- type `ModelSerializer`

Advanced model serializer


```zig
pub const ModelSerializer = struct {
```

- type `SerializationOptions`

```zig
pub const SerializationOptions = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, options: SerializationOptions) ModelSerializer {
```

- fn `serializeModel`

Serialize complete model with metadata


```zig
pub fn serializeModel(
```

- fn `deserializeModel`

Deserialize complete model


```zig
pub fn deserializeModel(
```

- fn `serializeWeights`

Serialize model weights only (for fine-tuning)


```zig
pub fn serializeWeights(_: *ModelSerializer, model: *anyopaque, writer: anytype) !void {
```

- fn `loadWeights`

Load weights into existing model


```zig
pub fn loadWeights(_: *ModelSerializer, model: *anyopaque, reader: anytype) !void {
```

- fn `exportModel`

Export model to different formats (ONNX, TensorFlow, etc.)


```zig
pub fn exportModel(self: *ModelSerializer, model: *anyopaque, format: ExportFormat, writer: anytype) !void {
```

- type `ExportFormat`

```zig
pub const ExportFormat = enum {
```

- type `ModelValidator`

Model validation and integrity checking


```zig
pub const ModelValidator = struct {
```

- fn `validateModel`

```zig
pub fn validateModel(model: *anyopaque, metadata: ModelMetadata) !void {
```

- fn `calculateChecksum`

```zig
pub fn calculateChecksum(data: []const u8) u64 {
```

- fn `validateChecksum`

```zig
pub fn validateChecksum(reader: anytype, expected_checksum: u64) !void {
```

- type `ModelCompression`

Model compression utilities


```zig
pub const ModelCompression = struct {
```

- type `CompressionType`

```zig
pub const CompressionType = enum {
```

- fn `compress`

Compress model data


```zig
pub fn compress(allocator: Allocator, data: []const u8, compression_type: CompressionType) ![]u8 {
```

- fn `decompress`

Decompress model data


```zig
pub fn decompress(allocator: Allocator, compressed_data: []const u8, compression_type: CompressionType) ![]u8 {
```

- type `ModelRegistry`

Model registry for versioning and management


```zig
pub const ModelRegistry = struct {
```

- type `ModelEntry`

```zig
pub const ModelEntry = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator) ModelRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *ModelRegistry) void {
```

- fn `registerModel`

```zig
pub fn registerModel(self: *ModelRegistry, name: []const u8, entry: ModelEntry) !void {
```

- fn `getModel`

```zig
pub fn getModel(self: *ModelRegistry, name: []const u8) ?ModelEntry {
```

- fn `listModels`

```zig
pub fn listModels(self: *ModelRegistry, allocator: Allocator) ![]ModelEntry {
```

- fn `saveToFile`

Save registry to disk


```zig
pub fn saveToFile(self: *ModelRegistry, path: []const u8) !void {
```

- fn `loadFromFile`

Load registry from disk


```zig
pub fn loadFromFile(self: *ModelRegistry, path: []const u8) !void {
```

## src/features/ai/neural.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- type `LayerType`

Neural network layer types


```zig
pub const LayerType = enum {
```

- type `Activation`

Activation functions with mixed precision support


```zig
pub const Activation = enum {
```

- fn `apply`

Apply activation function to a f32 value


```zig
pub fn apply(self: Activation, x: f32) f32 {
```

- fn `applyF16`

Apply activation function to a f16 value


```zig
pub fn applyF16(self: Activation, x: f16) f16 {
```

- fn `derivative`

Derivative of activation function (f32)


```zig
pub fn derivative(self: Activation, x: f32) f32 {
```

- fn `derivativeF16`

Derivative of activation function (f16)


```zig
pub fn derivativeF16(self: Activation, x: f16) f16 {
```

- type `Precision`

Precision mode for computations


```zig
pub const Precision = enum {
```

- type `TrainingConfig`

Neural network training configuration with enhanced memory options


```zig
pub const TrainingConfig = struct {
```

- type `LayerConfig`

Layer configuration


```zig
pub const LayerConfig = struct {
```

- type `NetworkConfig`

Complete neural network configuration


```zig
pub const NetworkConfig = struct {
```

- type `MemoryPool`

Memory pool for efficient buffer reuse


```zig
pub const MemoryPool = struct {
```

- type `PoolConfig`

Memory pool configuration


```zig
pub const PoolConfig = struct {
```

- type `PooledBuffer`

Pooled buffer


```zig
pub const PooledBuffer = struct {
```

- fn `release`

Return buffer to pool


```zig
pub fn release(self: *PooledBuffer) void {
```

- fn `slice`

Get buffer as slice of requested size


```zig
pub fn slice(self: *PooledBuffer, size: usize) []f32 {
```

- type `TrackedBuffer`

Enhanced buffer with liveness tracking


```zig
pub const TrackedBuffer = struct {
```

- fn `isStale`

Check if buffer is stale (not accessed recently)


```zig
pub fn isStale(self: TrackedBuffer, current_time: u64, stale_threshold_ns: u64) bool {
```

- fn `markAccessed`

Update access time


```zig
pub fn markAccessed(self: *TrackedBuffer, current_time: u64) void {
```

- type `LivenessConfig`

Liveness analysis configuration


```zig
pub const LivenessConfig = struct {
```

- fn `init`

Initialize memory pool


```zig
pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !*MemoryPool {
```

- fn `deinit`

Deinitialize memory pool


```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate buffer from pool or create new one


```zig
pub fn allocBuffer(self: *MemoryPool, size: usize) !*PooledBuffer {
```

- fn `returnBuffer`

Return buffer to pool for reuse


```zig
pub fn returnBuffer(self: *MemoryPool, buffer: *PooledBuffer) void {
```

- fn `getStats`

Get pool statistics


```zig
pub fn getStats(self: *MemoryPool) struct {
```

- fn `initLivenessAnalysis`

Initialize liveness analysis


```zig
pub fn initLivenessAnalysis(self: *MemoryPool, config: LivenessConfig) void {
```

- fn `recordBufferAccess`

Record buffer access for liveness analysis


```zig
pub fn recordBufferAccess(self: *MemoryPool, buffer: *PooledBuffer) void {
```

- fn `performLivenessCleanup`

Perform liveness-based cleanup


```zig
pub fn performLivenessCleanup(self: *MemoryPool, current_time: u64) void {
```

- fn `getLivenessStats`

Get liveness statistics


```zig
pub fn getLivenessStats(self: *MemoryPool) struct {
```

- type `Layer`

Neural network layer with enhanced memory safety and mixed precision support


```zig
pub const Layer = struct {
```

- fn `init`

Initialize a new layer with memory pool support


```zig
pub fn init(allocator: std.mem.Allocator, config: LayerConfig, memory_pool: ?*MemoryPool) !*Layer {
```

- fn `initF16`

Initialize f16 versions for mixed precision training


```zig
pub fn initF16(self: *Layer) !void {
```

- fn `syncToF32`

Synchronize f16 weights/biases back to f32 after training


```zig
pub fn syncToF32(self: *Layer) void {
```

- fn `forwardMixed`

Forward pass with mixed precision support


```zig
pub fn forwardMixed(self: *Layer, input: []const f32, use_f16: bool) ![]f32 {
```

- fn `backwardMixed`

Backward pass with mixed precision support


```zig
pub fn backwardMixed(
```

- fn `allocBuffer`

Allocate buffer using memory pool if available, fallback to allocator


```zig
pub fn allocBuffer(self: *Layer, size: usize) ![]f32 {
```

- fn `freeBuffer`

Free buffer using memory pool if available, fallback to allocator


```zig
pub fn freeBuffer(self: *Layer, buffer: []f32) void {
```

- fn `deinit`

Free layer resources with proper cleanup


```zig
pub fn deinit(self: *Layer) void {
```

- fn `forward`

Forward pass through the layer with memory pool support


```zig
pub fn forward(self: *Layer, input: []const f32) ![]f32 {
```

- fn `backward`

Backward pass through the layer with memory pool support


```zig
pub fn backward(
```

- type `CheckpointState`

Gradient checkpointing state


```zig
pub const CheckpointState = struct {
```

- type `NeuralNetwork`

Neural network for learning embeddings with enhanced memory safety


```zig
pub const NeuralNetwork = struct {
```

- fn `init`

Initialize a new neural network with optional memory pool


```zig
pub fn init(allocator: std.mem.Allocator, config: TrainingConfig) !*NeuralNetwork {
```

- fn `initDefault`

Initialize a new neural network with default configuration (backward compatibility)


```zig
pub fn initDefault(allocator: std.mem.Allocator) !*NeuralNetwork {
```

- fn `deinit`

Free network resources with proper cleanup


```zig
pub fn deinit(self: *NeuralNetwork) void {
```

- fn `deinitEnhanced`

Deinitialize with enhanced cleanup (for MemoryPool with liveness analysis)


```zig
pub fn deinitEnhanced(self: *MemoryPool) void {
```

- fn `addLayer`

Add a layer to the network with memory pool support


```zig
pub fn addLayer(self: *NeuralNetwork, config: LayerConfig) !void {
```

- fn `saveToFile`

Save network to file (basic implementation)


```zig
pub fn saveToFile(self: *NeuralNetwork, path: []const u8) !void {
```

- fn `loadFromFile`

Load network from file (basic implementation)


```zig
pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !*NeuralNetwork {
```

- fn `forward`

Forward pass through the network with memory optimization


```zig
pub fn forward(self: *NeuralNetwork, input: []const f32) ![]f32 {
```

- fn `forwardMixed`

Forward pass with mixed precision support


```zig
pub fn forwardMixed(self: *NeuralNetwork, input: []const f32) ![]f32 {
```

- fn `trainStep`

Train the network on a single sample with memory optimization


```zig
pub fn trainStep(
```

- fn `trainStepMixed`

Train the network on a single sample with mixed precision support


```zig
pub fn trainStepMixed(
```

## src/features/ai/reinforcement_learning.zig

- type `Environment`

Environment interface for RL interactions


```zig
pub const Environment = struct {
```

- fn `reset`

```zig
pub fn reset(self: Environment) ![]f32 {
```

- fn `step`

```zig
pub fn step(self: Environment, action: usize) !struct { state: []f32, reward: f32, done: bool, info: ?[]const u8 } {
```

- fn `getObservationSpace`

```zig
pub fn getObservationSpace(self: Environment) []const usize {
```

- fn `getActionSpace`

```zig
pub fn getActionSpace(self: Environment) []const usize {
```

- fn `render`

```zig
pub fn render(self: Environment) !void {
```

- type `ExperienceReplay`

Experience replay buffer for DQN training


```zig
pub const ExperienceReplay = struct {
```

- type `Experience`

```zig
pub const Experience = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, capacity: usize) !ExperienceReplay {
```

- fn `deinit`

```zig
pub fn deinit(self: *ExperienceReplay) void {
```

- fn `add`

```zig
pub fn add(self: *ExperienceReplay, state: []const f32, action: usize, reward: f32, next_state: []const f32, done: bool) !void {
```

- fn `sample`

```zig
pub fn sample(self: *ExperienceReplay, batch_size: usize) ![]Experience {
```

- fn `size`

```zig
pub fn size(self: ExperienceReplay) usize {
```

- type `DQN`

Deep Q-Network implementation


```zig
pub const DQN = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *DQN) void {
```

- fn `selectAction`

Select action using ε-greedy policy


```zig
pub fn selectAction(self: *DQN, state: []const f32, num_actions: usize) !usize {
```

- fn `getBestAction`

Get best action from Q-network


```zig
pub fn getBestAction(self: *DQN, state: []const f32) !usize {
```

- fn `trainOnBatch`

Train on batch of experiences


```zig
pub fn trainOnBatch(self: *DQN, network: anytype) !void {
```

- fn `updateTargetNetwork`

Update target network with current network weights


```zig
pub fn updateTargetNetwork(self: *DQN, network: anytype) !void {
```

- fn `save`

Save DQN model


```zig
pub fn save(_: *DQN, path: []const u8) !void {
```

- fn `load`

Load DQN model


```zig
pub fn load(allocator: Allocator, path: []const u8) !DQN {
```

- type `PolicyGradient`

Policy Gradient Agent (REINFORCE)


```zig
pub const PolicyGradient = struct {
```

- type `Trajectory`

```zig
pub const Trajectory = struct {
```

- fn `init`

```zig
pub fn init(allocator: Allocator) Trajectory {
```

- fn `deinit`

```zig
pub fn deinit(self: *Trajectory) void {
```

- fn `init`

```zig
pub fn init(allocator: Allocator, policy_network: *anyopaque, learning_rate: f32, gamma: f32) !PolicyGradient {
```

- fn `deinit`

```zig
pub fn deinit(self: *PolicyGradient) void {
```

- fn `collectTrajectory`

Collect trajectory by interacting with environment


```zig
pub fn collectTrajectory(self: *PolicyGradient, env: Environment, max_steps: usize) !void {
```

- fn `train`

Train policy on collected trajectories


```zig
pub fn train(self: *PolicyGradient) !void {
```

- type `ActorCritic`

Actor-Critic implementation


```zig
pub const ActorCritic = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *ActorCritic) void {
```

- fn `train`

Train on single transition


```zig
pub fn train(self: *ActorCritic, state: []const f32, action: usize, reward: f32, next_state: []const f32, done: bool) !void {
```

- const `ExplorationStrategy`

Exploration strategies


```zig
pub const ExplorationStrategy = union(enum) {
```

- fn `selectAction`

```zig
pub fn selectAction(self: ExplorationStrategy, q_values: []const f32, rng: Random) usize {
```

## src/features/ai/transformer.zig

- type `MultiHeadAttention`

Multi-Head Attention implementation


```zig
pub const MultiHeadAttention = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, embed_dim: usize, num_heads: usize) !*MultiHeadAttention {
```

- fn `deinit`

```zig
pub fn deinit(self: *MultiHeadAttention, allocator: std.mem.Allocator) void {
```

- fn `forward`

Forward pass for multi-head attention


```zig
pub fn forward(self: *MultiHeadAttention, query: []const f32, key: []const f32, value: []const f32, output: []f32) !void {
```

- type `PositionalEncoding`

Positional Encoding for transformer architectures


```zig
pub const PositionalEncoding = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, max_seq_len: usize, embed_dim: usize) !*PositionalEncoding {
```

- fn `deinit`

```zig
pub fn deinit(self: *PositionalEncoding, allocator: std.mem.Allocator) void {
```

- fn `encode`

```zig
pub fn encode(self: *PositionalEncoding, input: []f32, seq_len: usize) void {
```

- type `TransformerBlock`

Transformer Block with self-attention and feed-forward network


```zig
pub const TransformerBlock = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, embed_dim: usize, num_heads: usize, ff_dim: usize, dropout_rate: f32) !*TransformerBlock {
```

- fn `deinit`

```zig
pub fn deinit(self: *TransformerBlock, allocator: std.mem.Allocator) void {
```

- fn `forward`

```zig
pub fn forward(self: *TransformerBlock, input: []f32, seq_len: usize) !void {
```

- type `FeedForwardNetwork`

Feed-Forward Network for transformer blocks


```zig
pub const FeedForwardNetwork = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, input_dim: usize, hidden_dim: usize) !*FeedForwardNetwork {
```

- fn `deinit`

```zig
pub fn deinit(self: *FeedForwardNetwork, allocator: std.mem.Allocator) void {
```

- fn `forward`

```zig
pub fn forward(self: *FeedForwardNetwork, input: []f32, output: []f32) !void {
```

- type `LayerNorm`

Layer Normalization for transformer blocks


```zig
pub const LayerNorm = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, size: usize) !*LayerNorm {
```

- fn `deinit`

```zig
pub fn deinit(self: *LayerNorm, allocator: std.mem.Allocator) void {
```

- fn `forward`

```zig
pub fn forward(self: *LayerNorm, input: []f32, seq_len: usize) void {
```

- type `Transformer`

Complete Transformer model


```zig
pub const Transformer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embed_dim: usize, num_layers: usize, num_heads: usize, max_seq_len: usize) !*Transformer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Transformer, allocator: std.mem.Allocator) void {
```

- fn `forward`

```zig
pub fn forward(self: *Transformer, input_tokens: []const u32, output_logits: []f32) !void {
```

- type `Embedding`

Embedding layer for transformers


```zig
pub const Embedding = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, vocab_size: usize, embed_dim: usize) !*Embedding {
```

- fn `deinit`

```zig
pub fn deinit(self: *Embedding, allocator: std.mem.Allocator) void {
```

- fn `forward`

```zig
pub fn forward(self: *Embedding, tokens: []const u32, output: []f32) !void {
```

## src/features/connectors/mod.zig

- const `openai`

```zig
pub const openai = @import("openai.zig");
```

- const `ollama`

```zig
pub const ollama = @import("ollama.zig");
```

- const `plugin`

```zig
pub const plugin = @import("plugin.zig");
```

- const `mod`

```zig
pub const mod = @import("mod.zig");
```

## src/features/connectors/ollama.zig

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- fn `embedText`

```zig
pub fn embedText(allocator: Allocator, host: []const u8, model: []const u8, text: []const u8) ![]f32 {
```

## src/features/connectors/openai.zig

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- const `Error`

Errors that can occur during OpenAI API operations


```zig
pub const Error = error{
```

- fn `embedText`

Embeds the given text using the OpenAI embeddings API

Args:
- allocator: Memory allocator for dynamic allocations
- base_url: Base URL for the OpenAI API (e.g., "https://api.openai.com/v1")
- api_key: OpenAI API key for authentication
- model: The model to use for embeddings (e.g., "text-embedding-ada-002")
- text: The text to embed

Returns:
- A slice of f32 values representing the embedding vector
- The caller owns the returned memory and must free it

Errors:
- MissingApiKey: If the api_key is empty
- NetworkError: If there's a network communication error or non-200 response
- InvalidResponse: If the API response format is unexpected
- OutOfMemory: If memory allocation fails


```zig
pub fn embedText(allocator: Allocator, base_url: []const u8, api_key: []const u8, model: []const u8, text: []const u8) Error![]f32 {
```

## src/features/connectors/plugin.zig

- type `EmbeddingApi`

```zig
pub const EmbeddingApi = extern struct {
```

- var `PLUGIN_INTERFACE`

```zig
pub var PLUGIN_INTERFACE: iface.PluginInterface = .{
```

- fn `abi_plugin_create`

```zig
pub fn abi_plugin_create() callconv(.c) ?*const iface.PluginInterface {
```

- fn `getInterface`

```zig
pub fn getInterface() *const iface.PluginInterface {
```

## src/features/database/cli.zig

- const `Db`

Re-export database types for convenience


```zig
pub const Db = database.Db;
```

- const `DbError`

```zig
pub const DbError = database.Db.DbError;
```

- const `Result`

```zig
pub const Result = database.Db.Result;
```

- const `WdbxHeader`

```zig
pub const WdbxHeader = database.WdbxHeader;
```

- type `Command`

WDBX CLI command types


```zig
pub const Command = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?Command {
```

- fn `getDescription`

```zig
pub fn getDescription(self: Command) []const u8 {
```

- type `OutputFormat`

Output format options


```zig
pub const OutputFormat = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?OutputFormat {
```

- fn `getExtension`

```zig
pub fn getExtension(self: OutputFormat) []const u8 {
```

- type `LogLevel`

Log level enumeration


```zig
pub const LogLevel = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?LogLevel {
```

- fn `toInt`

```zig
pub fn toInt(self: LogLevel) u8 {
```

- type `Options`

CLI options structure


```zig
pub const Options = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *Options, allocator: std.mem.Allocator) void {
```

- type `WdbxCLI`

WDBX CLI implementation


```zig
pub const WdbxCLI = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: Options) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `run`

```zig
pub fn run(self: *Self) !void {
```

- fn `showHelp`

```zig
pub fn showHelp(self: *Self) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, level: LogLevel) Logger {
```

- fn `deinit`

```zig
pub fn deinit(self: *Logger) void {
```

- fn `trace`

```zig
pub fn trace(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `debug`

```zig
pub fn debug(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `info`

```zig
pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `warn`

```zig
pub fn warn(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `err`

```zig
pub fn err(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `fatal`

```zig
pub fn fatal(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `main`

Main entry point


```zig
pub fn main() !void {
```

## src/features/database/config.zig

- const `ConfigValidationError`

Configuration validation error codes


```zig
pub const ConfigValidationError = error{
```

- type `ConfigValidator`

Configuration schema validator


```zig
pub const ConfigValidator = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `validateConfig`

Validate entire configuration with comprehensive checks


```zig
pub fn validateConfig(self: *Self, config: *const WdbxConfig) ConfigValidationError!void {
```

- fn `generateValidationReport`

Generate validation report


```zig
pub fn generateValidationReport(self: *Self, config: *const WdbxConfig) ![]const u8 {
```

- type `WdbxConfig`

Configuration file format (JSON-based)


```zig
pub const WdbxConfig = struct {
```

- type `DatabaseConfig`

```zig
pub const DatabaseConfig = struct {
```

- type `ServerConfig`

```zig
pub const ServerConfig = struct {
```

- type `PerformanceConfig`

```zig
pub const PerformanceConfig = struct {
```

- type `MonitoringConfig`

```zig
pub const MonitoringConfig = struct {
```

- type `SecurityConfig`

```zig
pub const SecurityConfig = struct {
```

- type `LoggingConfig`

```zig
pub const LoggingConfig = struct {
```

- type `ConfigManager`

Configuration manager


```zig
pub const ConfigManager = struct {
```

- const `DEFAULT_CONFIG_FILE`

Default configuration file name


```zig
pub const DEFAULT_CONFIG_FILE = ".wdbx-config";
```

- fn `init`

Initialize configuration manager


```zig
pub fn init(allocator: std.mem.Allocator, config_path: ?[]const u8) !*Self {
```

- fn `deinit`

Deinitialize configuration manager


```zig
pub fn deinit(self: *Self) void {
```

- fn `getConfig`

Get current configuration


```zig
pub fn getConfig(self: *Self) *const WdbxConfig {
```

- fn `loadFromFile`

Load configuration from file


```zig
pub fn loadFromFile(self: *Self) !void {
```

- fn `createDefaultConfigFile`

Create default configuration file (Zig 0.16 compatible)


```zig
pub fn createDefaultConfigFile(self: *Self) !void {
```

- fn `applyEnvironmentOverrides`

Apply environment variable overrides


```zig
pub fn applyEnvironmentOverrides(self: *Self) !void {
```

- fn `checkAndReload`

Check if configuration file has been modified and reload if necessary


```zig
pub fn checkAndReload(self: *Self) !bool {
```

- fn `validate`

Validate configuration values using comprehensive schema validation


```zig
pub fn validate(self: *Self) !void {
```

- fn `getValue`

Get configuration value by key path (e.g., "database.dimensions")


```zig
pub fn getValue(self: *Self, key: []const u8) !?[]const u8 {
```

- fn `setValue`

Set configuration value by key path (e.g., "database.dimensions=128")


```zig
pub fn setValue(self: *Self, key: []const u8, value: []const u8) !void {
```

- fn `listAll`

List all configuration values


```zig
pub fn listAll(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `save`

Save current configuration to file


```zig
pub fn save(self: *Self) !void {
```

- type `ConfigUtils`

Configuration utilities


```zig
pub const ConfigUtils = struct {
```

- fn `getValue`

Get configuration value by path (e.g., "database.hnsw_m")


```zig
pub fn getValue(config: *const WdbxConfig, path: []const u8, allocator: std.mem.Allocator) !?[]const u8 {
```

- fn `printSummary`

Print configuration summary


```zig
pub fn printSummary(config: *const WdbxConfig) void {
```

## src/features/database/core.zig

- const `WdbxError`

WDBX standardized error codes with numeric IDs for consistent error handling


```zig
pub const WdbxError = error{
```

- type `ErrorCodes`

Error code mapping for consistent error handling across modules


```zig
pub const ErrorCodes = struct {
```

- fn `getErrorCode`

```zig
pub fn getErrorCode(err: WdbxError) u32 {
```

- fn `getErrorDescription`

```zig
pub fn getErrorDescription(err: WdbxError) []const u8 {
```

- fn `getErrorCategory`

```zig
pub fn getErrorCategory(err: WdbxError) []const u8 {
```

- fn `formatError`

Format error for logging and display


```zig
pub fn formatError(allocator: std.mem.Allocator, err: WdbxError, context: ?[]const u8) ![]const u8 {
```

- type `VERSION`

WDBX version information


```zig
pub const VERSION = struct {
```

- const `MAJOR`

```zig
pub const MAJOR = 1;
```

- const `MINOR`

```zig
pub const MINOR = 0;
```

- const `PATCH`

```zig
pub const PATCH = 0;
```

- const `PRE_RELEASE`

```zig
pub const PRE_RELEASE = "alpha";
```

- fn `string`

```zig
pub fn string() []const u8 {
```

- fn `isCompatible`

```zig
pub fn isCompatible(major: u32, minor: u32) bool {
```

- type `OutputFormat`

Output format options


```zig
pub const OutputFormat = enum {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?OutputFormat {
```

- fn `toString`

```zig
pub fn toString(self: OutputFormat) []const u8 {
```

- type `LogLevel`

Log level enumeration


```zig
pub const LogLevel = enum(u8) {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?LogLevel {
```

- fn `toString`

```zig
pub fn toString(self: LogLevel) []const u8 {
```

- fn `toInt`

```zig
pub fn toInt(self: LogLevel) u8 {
```

- type `Config`

Common WDBX configuration


```zig
pub const Config = struct {
```

- fn `init`

```zig
pub fn init() Config {
```

- fn `validate`

```zig
pub fn validate(self: *const Config) WdbxError!void {
```

- type `Timer`

Performance timer for benchmarking


```zig
pub const Timer = struct {
```

- fn `init`

```zig
pub fn init() Timer {
```

- fn `elapsed`

```zig
pub fn elapsed(self: *const Timer) u64 {
```

- fn `elapsedMs`

```zig
pub fn elapsedMs(self: *const Timer) f64 {
```

- fn `elapsedUs`

```zig
pub fn elapsedUs(self: *const Timer) f64 {
```

- fn `restart`

```zig
pub fn restart(self: *Timer) void {
```

- type `Logger`

Simple logging utility


```zig
pub const Logger = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, level: LogLevel) Logger {
```

- fn `deinit`

```zig
pub fn deinit(self: *Logger) void {
```

- fn `log`

```zig
pub fn log(self: *Logger, level: LogLevel, comptime fmt: []const u8, args: anytype) !void {
```

- fn `debug`

```zig
pub fn debug(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `info`

```zig
pub fn info(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `warn`

```zig
pub fn warn(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `err`

```zig
pub fn err(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- fn `fatal`

```zig
pub fn fatal(self: *Logger, comptime fmt: []const u8, args: anytype) !void {
```

- type `MemoryStats`

Memory usage statistics


```zig
pub const MemoryStats = struct {
```

- fn `init`

```zig
pub fn init() MemoryStats {
```

- fn `allocate`

```zig
pub fn allocate(self: *MemoryStats, size: usize) void {
```

- fn `deallocate`

```zig
pub fn deallocate(self: *MemoryStats, size: usize) void {
```

- fn `reset`

```zig
pub fn reset(self: *MemoryStats) void {
```

## src/features/database/database.zig

- const `DatabaseError`

Database-specific error types


```zig
pub const DatabaseError = error{
```

- const `MAGIC`

Magic identifier for ABI files (7 bytes + NUL)


```zig
pub const MAGIC: [8]u8 = "WDBXAI\x00\x00".*;
```

- const `FORMAT_VERSION`

Current file format version


```zig
pub const FORMAT_VERSION: u16 = 1;
```

- const `DEFAULT_PAGE_SIZE`

Default page size for file operations (4 KiB)


```zig
pub const DEFAULT_PAGE_SIZE: u32 = 4096;
```

- type `WdbxHeader`

File-header fixed at 4 KiB (4096 bytes)


```zig
pub const WdbxHeader = extern struct {
```

- fn `validateMagic`

```zig
pub fn validateMagic(self: *const WdbxHeader) bool {
```

- fn `createDefault`

```zig
pub fn createDefault() WdbxHeader {
```

- type `Db`

```zig
pub const Db = struct {
```

- const `DbError`

```zig
pub const DbError = error{
```

- fn `isInitialized`

```zig
pub fn isInitialized(self: *const Db) bool {
```

- fn `init`

```zig
pub fn init(self: *Db, dim: u16) DbError!void {
```

- fn `addEmbedding`

```zig
pub fn addEmbedding(self: *Db, embedding: []const f32) DbError!u64 {
```

- fn `addEmbeddingsBatch`

```zig
pub fn addEmbeddingsBatch(self: *Db, embeddings: []const []const f32) DbError![]u64 {
```

- fn `search`

```zig
pub fn search(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) DbError![]Result {
```

- type `Result`

```zig
pub const Result = struct {
```

- fn `lessThanAsc`

```zig
pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
```

- type `DbStats`

```zig
pub const DbStats = struct {
```

- fn `getAverageSearchTime`

```zig
pub fn getAverageSearchTime(self: *const DbStats) u64 {
```

- fn `open`

```zig
pub fn open(path: []const u8, create_if_missing: bool) DbError!*Db {
```

- fn `close`

```zig
pub fn close(self: *Db) void {
```

- fn `getStats`

```zig
pub fn getStats(self: *const Db) DbStats {
```

- fn `getRowCount`

```zig
pub fn getRowCount(self: *const Db) u64 {
```

- fn `getDimension`

```zig
pub fn getDimension(self: *const Db) u16 {
```

- type `HNSWIndex`

HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search


```zig
pub const HNSWIndex = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, id: u64, vector: []const f32, layer: u32) !*Node {
```

- fn `deinit`

```zig
pub fn deinit(self: *Node, allocator: std.mem.Allocator) void {
```

- fn `addConnection`

```zig
pub fn addConnection(self: *Node, allocator: std.mem.Allocator, node_id: u64) !void {
```

- fn `compare`

```zig
pub fn compare(_: void, a: SearchResult, b: SearchResult) std.math.Order {
```

- fn `compare`

```zig
pub fn compare(_: void, a: QueueEntry, b: QueueEntry) std.math.Order {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, dimension: u16) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addVector`

Add a vector to the HNSW index


```zig
pub fn addVector(self: *Self, id: u64, vector: []const f32) !void {
```

- fn `search`

Search for approximate nearest neighbors


```zig
pub fn search(self: *Self, query: []const f32, top_k: usize) ![]SearchResult {
```

- fn `initHNSW`

Initialize HNSW index for faster search


```zig
pub fn initHNSW(self: *Db) !void {
```

- fn `addToHNSW`

Add vector to HNSW index


```zig
pub fn addToHNSW(self: *Db, id: u64, vector: []const f32) !void {
```

- fn `setHNSWParams`

Adjust HNSW parameters (useful for tests/benchmarks)


```zig
pub fn setHNSWParams(self: *Db, params: struct { max_connections: u32 = 16, ef_construction: u32 = 200, ef_search: u32 = 100 }) void {
```

- fn `searchHNSW`

Search using HNSW index (fallback to brute force if not available)


```zig
pub fn searchHNSW(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator) ![]Result {
```

- fn `searchParallel`

Parallel search using multiple threads for brute force search


```zig
pub fn searchParallel(self: *Db, query: []const f32, top_k: usize, allocator: std.mem.Allocator, num_threads: u32) ![]Result {
```

## src/features/database/database_sharding.zig

- type `GlobalResult`

```zig
pub const GlobalResult = struct {
```

- type `ShardedDb`

```zig
pub const ShardedDb = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, paths: []const []const u8, dim: u16) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `getDimension`

```zig
pub fn getDimension(self: *const Self) u16 {
```

- fn `getTotalRowCount`

```zig
pub fn getTotalRowCount(self: *const Self) u64 {
```

- fn `addEmbedding`

```zig
pub fn addEmbedding(self: *Self, embedding: []const f32) !struct { shard: usize, local_index: u64 } {
```

- fn `search`

```zig
pub fn search(self: *Self, query: []const f32, k: usize, allocator: std.mem.Allocator) ![]GlobalResult {
```

## src/features/database/db_helpers.zig

- const `Db`

```zig
pub const Db = engine.Db;
```

- const `DatabaseError`

```zig
pub const DatabaseError = engine.DatabaseError;
```

- const `WdbxHeader`

```zig
pub const WdbxHeader = engine.WdbxHeader;
```

- const `Result`

```zig
pub const Result = engine.Db.Result;
```

- const `DbStats`

```zig
pub const DbStats = engine.Db.DbStats;
```

- type `helpers`

Shared helpers for manipulating vectors and database-oriented payloads.


```zig
pub const helpers = struct {
```

- fn `parseVector`

```zig
pub fn parseVector(allocator: std.mem.Allocator, input: []const u8) ![]f32 {
```

- fn `parseJsonVector`

```zig
pub fn parseJsonVector(allocator: std.mem.Allocator, node: json.Value) ![]f32 {
```

- fn `formatNearestNeighborResponse`

```zig
pub fn formatNearestNeighborResponse(allocator: std.mem.Allocator, result: Result) ![]u8 {
```

- fn `formatKnnResponse`

```zig
pub fn formatKnnResponse(allocator: std.mem.Allocator, k: usize, results: []const Result) ![]u8 {
```

## src/features/database/http.zig

- const `WdbxHttpServer`

```zig
pub const WdbxHttpServer = wdbx_http.WdbxHttpServer;
```

- const `ServerConfig`

```zig
pub const ServerConfig = wdbx_http.ServerConfig;
```

- const `createServer`

```zig
pub const createServer = wdbx_http.WdbxHttpServer.init;
```

- const `HttpError`

```zig
pub const HttpError = wdbx_http.HttpError;
```

- const `Response`

```zig
pub const Response = wdbx_http.Response;
```

## src/features/database/mod.zig

- const `database`

```zig
pub const database = @import("database.zig");
```

- const `core`

```zig
pub const core = @import("core.zig");
```

- const `config`

```zig
pub const config = @import("config.zig");
```

- const `utils`

```zig
pub const utils = @import("utils.zig");
```

- const `db_helpers`

```zig
pub const db_helpers = @import("db_helpers.zig");
```

- const `unified`

```zig
pub const unified = @import("unified.zig");
```

- const `database_sharding`

```zig
pub const database_sharding = @import("database_sharding.zig");
```

- const `http`

```zig
pub const http = @import("http.zig");
```

- const `cli`

```zig
pub const cli = @import("cli.zig");
```

- const `mod`

```zig
pub const mod = @import("mod.zig");
```

## src/features/database/unified.zig

- const `WdbxCLI`

```zig
pub const WdbxCLI = cli.WdbxCLI;
```

- const `WdbxHttpServer`

```zig
pub const WdbxHttpServer = http.WdbxHttpServer;
```

- const `Command`

```zig
pub const Command = cli.Command;
```

- const `Options`

```zig
pub const Options = cli.Options;
```

- const `ServerConfig`

```zig
pub const ServerConfig = http.ServerConfig;
```

- const `WdbxError`

```zig
pub const WdbxError = wdbx_core.WdbxError;
```

- const `VERSION`

```zig
pub const VERSION = wdbx_core.VERSION;
```

- const `OutputFormat`

```zig
pub const OutputFormat = wdbx_core.OutputFormat;
```

- const `LogLevel`

```zig
pub const LogLevel = wdbx_core.LogLevel;
```

- const `Config`

```zig
pub const Config = wdbx_core.Config;
```

- const `Timer`

```zig
pub const Timer = wdbx_core.Timer;
```

- const `Logger`

```zig
pub const Logger = wdbx_core.Logger;
```

- const `MemoryStats`

```zig
pub const MemoryStats = wdbx_core.MemoryStats;
```

- const `main`

```zig
pub const main = cli.main;
```

- const `createServer`

```zig
pub const createServer = http.createServer;
```

- type `wdbx`

```zig
pub const wdbx = struct {
```

- const `cli_module`

```zig
pub const cli_module = cli;
```

- const `core_module`

```zig
pub const core_module = wdbx_core;
```

- const `http_module`

```zig
pub const http_module = http;
```

- fn `createCLI`

```zig
pub fn createCLI(allocator: std.mem.Allocator, options: Options) !*WdbxCLI {
```

- fn `createHttpServer`

```zig
pub fn createHttpServer(allocator: std.mem.Allocator, config: ServerConfig) !*WdbxHttpServer {
```

- const `version`

```zig
pub const version = VERSION.string();
```

- const `version_major`

```zig
pub const version_major = VERSION.MAJOR;
```

- const `version_minor`

```zig
pub const version_minor = VERSION.MINOR;
```

- const `version_patch`

```zig
pub const version_patch = VERSION.PATCH;
```

- fn `quickStart`

```zig
pub fn quickStart(allocator: std.mem.Allocator) !void {
```

- fn `startHttpServer`

```zig
pub fn startHttpServer(allocator: std.mem.Allocator, port: u16) !*WdbxHttpServer {
```

## src/features/database/utils.zig

- type `utils`

Utility helpers for the WDBX module.

The primary purpose of this file is to provide small, reusable
functions that are used across the WDBX codebase.  Currently we
expose a single helper for freeing optional string slices.

This module deliberately stays tiny; if more shared helpers are
required in the future, they can be added here while keeping the
implementation consistent and testable.


```zig
pub const utils = struct {
```

- pub `inline`

Frees an optional slice if it is allocated.

Zig's optional types can be `null` or a value.  In the
WDBX code many structs own optional `[]const u8` strings that
need to be freed when the struct is dropped.  This helper
abstracts the pattern

```zig
if (opt) |ptr| allocator.free(ptr);
```

Usage:

```zig
utils.freeOptional(allocator, self.db_path);
```

# Parameters

- `allocator`: the allocator that owns the memory.
- `opt`: the optional slice to free, or `null`.

# Panics

Does not panic. If `opt` is not `null`, `allocator.free`
is called with the slice; the slice must have been allocated
from that allocator.


```zig
pub inline fn freeOptional(allocator: std.mem.Allocator, opt: ?[]const u8) void {
```

## src/features/gpu/backends/backends.zig

- type `Backend`

Supported GPU backends


```zig
pub const Backend = enum {
```

- fn `toString`

```zig
pub fn toString(self: Backend) []const u8 {
```

- fn `getPriority`

```zig
pub fn getPriority(self: Backend) u8 {
```

- type `Capabilities`

Backend capabilities and features


```zig
pub const Capabilities = struct {
```

- fn `format`

```zig
pub fn format(
```

- const `BackendConfig`

Backend-specific configuration


```zig
pub const BackendConfig = union(Backend) {
```

- type `VulkanConfig`

Vulkan-specific configuration


```zig
pub const VulkanConfig = struct {
```

- type `MetalConfig`

Metal-specific configuration


```zig
pub const MetalConfig = struct {
```

- type `DX12Config`

DirectX 12-specific configuration


```zig
pub const DX12Config = struct {
```

- type `OpenGLConfig`

OpenGL-specific configuration


```zig
pub const OpenGLConfig = struct {
```

- type `CUDAConfig`

CUDA-specific configuration


```zig
pub const CUDAConfig = struct {
```

- type `OpenCLConfig`

OpenCL-specific configuration


```zig
pub const OpenCLConfig = struct {
```

- type `WebGPUConfig`

WebGPU-specific configuration


```zig
pub const WebGPUConfig = struct {
```

- type `CPUConfig`

CPU fallback configuration


```zig
pub const CPUConfig = struct {
```

- type `BackendManager`

Multi-Backend Manager


```zig
pub const BackendManager = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*BackendManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *BackendManager) void {
```

- fn `detectAvailableBackends`

Detect which backends are available on this system


```zig
pub fn detectAvailableBackends(self: *BackendManager) !void {
```

- fn `setDefaultConfigs`

Set default configurations for all backends


```zig
pub fn setDefaultConfigs(self: *BackendManager) !void {
```

- fn `selectBestBackend`

Select the best available backend


```zig
pub fn selectBestBackend(self: *BackendManager) ?Backend {
```

- fn `selectBackend`

Force a specific backend


```zig
pub fn selectBackend(self: *BackendManager, backend: Backend) !void {
```

- fn `getCapabilities`

Get capabilities for a backend


```zig
pub fn getCapabilities(self: *BackendManager, backend: Backend) !Capabilities {
```

- fn `createRenderer`

Create a renderer for the current backend


```zig
pub fn createRenderer(self: *BackendManager, config: gpu_renderer.GPUConfig) !*gpu_renderer.GPURenderer {
```

- type `ShaderCompiler`

Backend-specific shader compiler


```zig
pub const ShaderCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, backend: Backend) !*ShaderCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *ShaderCompiler) void {
```

- fn `compileShader`

Compile shader source to backend-specific format


```zig
pub fn compileShader(self: *ShaderCompiler, source: []const u8, shader_type: enum { vertex, fragment, compute }) ![]const u8 {
```

## src/features/gpu/backends/mod.zig

- const `backends`

```zig
pub const backends = @import("backends.zig");
```

## src/features/gpu/benchmark/benchmarks.zig

- type `BenchmarkConfig`

```zig
pub const BenchmarkConfig = struct {
```

- fn `validate`

```zig
pub fn validate(self: BenchmarkConfig) !void {
```

- type `WorkloadType`

```zig
pub const WorkloadType = enum {
```

- fn `displayName`

```zig
pub fn displayName(self: WorkloadType) []const u8 {
```

- fn `complexityClass`

```zig
pub fn complexityClass(self: WorkloadType) []const u8 {
```

- type `PerformanceGrade`

```zig
pub const PerformanceGrade = enum {
```

- fn `displayName`

```zig
pub fn displayName(self: PerformanceGrade) []const u8 {
```

- fn `colorCode`

```zig
pub fn colorCode(self: PerformanceGrade) []const u8 {
```

- fn `toString`

```zig
pub fn toString(self: PerformanceGrade) []const u8 {
```

- type `ExecutionContext`

```zig
pub const ExecutionContext = struct {
```

- type `BenchmarkResult`

```zig
pub const BenchmarkResult = struct {
```

## src/features/gpu/benchmark/mod.zig

- const `benchmarks`

```zig
pub const benchmarks = @import("benchmarks.zig");
```

## src/features/gpu/compute/gpu_ai_acceleration.zig

- type `Tensor`

Tensor data type for GPU operations


```zig
pub const Tensor = struct {
```

- fn `create`

Create a new tensor


```zig
pub fn create(allocator: std.mem.Allocator, shape: []const usize) !*Tensor {
```

- fn `initWithData`

Initialize tensor with values


```zig
pub fn initWithData(allocator: std.mem.Allocator, shape: []const usize, values: []const f32) !*Tensor {
```

- fn `uploadToGpu`

Upload tensor to GPU


```zig
pub fn uploadToGpu(self: *Tensor, renderer: *gpu_renderer.GPURenderer) !void {
```

- fn `downloadFromGpu`

Download tensor from GPU


```zig
pub fn downloadFromGpu(self: *Tensor, renderer: *gpu_renderer.GPURenderer) !void {
```

- fn `size`

Get tensor element count


```zig
pub fn size(self: Tensor) usize {
```

- fn `deinit`

Cleanup tensor resources


```zig
pub fn deinit(self: *Tensor) void {
```

- type `MatrixOps`

GPU-accelerated matrix operations


```zig
pub const MatrixOps = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) MatrixOps {
```

- fn `matmul`

Matrix multiplication: C = A * B


```zig
pub fn matmul(self: *MatrixOps, a: *Tensor, b: *Tensor, c: *Tensor) !void {
```

- fn `transpose`

Matrix transpose


```zig
pub fn transpose(self: *MatrixOps, input: *Tensor, output: *Tensor) !void {
```

- fn `elementWiseAdd`

Element-wise operations


```zig
pub fn elementWiseAdd(self: *MatrixOps, a: *Tensor, b: *Tensor, result: *Tensor) !void {
```

- fn `elementWiseMultiply`

```zig
pub fn elementWiseMultiply(self: *MatrixOps, a: *Tensor, b: *Tensor, result: *Tensor) !void {
```

- type `NeuralNetworkOps`

GPU-accelerated neural network operations


```zig
pub const NeuralNetworkOps = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) NeuralNetworkOps {
```

- fn `denseForward`

Dense layer forward pass: output = activation(input * weights + biases)


```zig
pub fn denseForward(self: *NeuralNetworkOps, input: *Tensor, weights: *Tensor, biases: *Tensor, output: *Tensor, activation: kernels.ActivationType) !void {
```

- fn `conv2dForward`

Convolution operation (simplified 2D convolution)


```zig
pub fn conv2dForward(self: *NeuralNetworkOps, input: *Tensor, kernels_tensor: *Tensor, biases: *Tensor, output: *Tensor, stride: usize, padding: usize) !void {
```

- fn `maxPool2d`

Max pooling operation


```zig
pub fn maxPool2d(self: *NeuralNetworkOps, input: *Tensor, output: *Tensor, kernel_size: usize, stride: usize) !void {
```

- type `TrainingAcceleration`

Training acceleration for neural networks


```zig
pub const TrainingAcceleration = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) TrainingAcceleration {
```

- fn `denseBackward`

Backpropagation for dense layer


```zig
pub fn denseBackward(self: *TrainingAcceleration, input: *Tensor, weights: *Tensor, output_grad: *Tensor, input_grad: *Tensor, weights_grad: *Tensor, biases_grad: *Tensor, activation: kernels.ActivationType) !void {
```

- fn `sgdStep`

SGD optimizer step


```zig
pub fn sgdStep(self: *TrainingAcceleration, weights: *Tensor, biases: *Tensor, weights_grad: *Tensor, biases_grad: *Tensor, learning_rate: f32) void {
```

- type `AIMLAcceleration`

Main AI/ML Acceleration Manager


```zig
pub const AIMLAcceleration = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *gpu_renderer.GPURenderer) !*AIMLAcceleration {
```

- fn `deinit`

```zig
pub fn deinit(self: *AIMLAcceleration) void {
```

- fn `verifyBackendCapabilities`

Verify backend capabilities before initialization


```zig
pub fn verifyBackendCapabilities(renderer: *gpu_renderer.GPURenderer) !void {
```

- fn `createTensor`

Create and track a tensor


```zig
pub fn createTensor(self: *AIMLAcceleration, shape: []const usize) !*Tensor {
```

- fn `createTensorWithData`

Create tensor with data and track it


```zig
pub fn createTensorWithData(self: *AIMLAcceleration, shape: []const usize, data: []const f32) !*Tensor {
```

- fn `getStats`

Get performance statistics


```zig
pub fn getStats(self: *AIMLAcceleration) struct {
```

- fn `demo`

Example usage and demonstration


```zig
pub fn demo() !void {
```

## src/features/gpu/compute/gpu_backend_manager.zig

- type `HardwareCapabilities`

Hardware capabilities structure for GPU backend management


```zig
pub const HardwareCapabilities = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !HardwareCapabilities {
```

- fn `deinit`

```zig
pub fn deinit(self: *HardwareCapabilities) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*CUDADriver {
```

- fn `deinit`

```zig
pub fn deinit(self: *CUDADriver) void {
```

- fn `getDeviceProperties`

```zig
pub fn getDeviceProperties(self: *CUDADriver, device_id: u32) !HardwareCapabilities {
```

- fn `getDeviceCount`

```zig
pub fn getDeviceCount(self: *CUDADriver) !u32 {
```

- type `MemoryBandwidthBenchmark`

Memory Bandwidth Benchmark stub


```zig
pub const MemoryBandwidthBenchmark = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: anytype) !*MemoryBandwidthBenchmark {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryBandwidthBenchmark) void {
```

- fn `measureBandwidth`

```zig
pub fn measureBandwidth(self: *MemoryBandwidthBenchmark, buffer_size: usize, iterations: u32) !f64 {
```

- type `ComputeThroughputBenchmark`

Compute Throughput Benchmark stub


```zig
pub const ComputeThroughputBenchmark = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: anytype) !*ComputeThroughputBenchmark {
```

- fn `deinit`

```zig
pub fn deinit(self: *ComputeThroughputBenchmark) void {
```

- fn `measureComputeThroughput`

```zig
pub fn measureComputeThroughput(_self: *ComputeThroughputBenchmark, workgroup_size: u32, iterations: u32) !f64 {
```

- type `PerformanceMeasurement`

Performance measurement structure


```zig
pub const PerformanceMeasurement = struct {
```

- type `BenchmarkResult`

Benchmark result structure


```zig
pub const BenchmarkResult = struct {
```

- type `PerformanceProfiler`

Performance Profiler stub


```zig
pub const PerformanceProfiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: anytype) !*PerformanceProfiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceProfiler) void {
```

- fn `startTiming`

```zig
pub fn startTiming(self: *PerformanceProfiler, operation_name: []const u8) !void {
```

- fn `endTiming`

```zig
pub fn endTiming(self: *PerformanceProfiler) !u64 {
```

- fn `stopTiming`

```zig
pub fn stopTiming(self: *PerformanceProfiler) !u64 {
```

- fn `runWorkloadBenchmark`

```zig
pub fn runWorkloadBenchmark(self: *PerformanceProfiler, workload: anytype, size: usize, config: anytype) !f64 {
```

- type `GPUBackendManager`

Enhanced GPU Backend Manager with comprehensive error handling and resource management


```zig
pub const GPUBackendManager = struct {
```

- type `BackendStatistics`

Backend usage statistics


```zig
pub const BackendStatistics = struct {
```

- fn `init`

Initialize GPU Backend Manager with comprehensive setup


```zig
pub fn init(allocator: std.mem.Allocator) !*GPUBackendManager {
```

- fn `deinit`

Safely deinitialize GPU Backend Manager with comprehensive cleanup


```zig
pub fn deinit(self: *GPUBackendManager) void {
```

- fn `getStatistics`

Get backend statistics


```zig
pub fn getStatistics(self: *GPUBackendManager) BackendStatistics {
```

- fn `isReady`

Check if manager is properly initialized


```zig
pub fn isReady(self: *GPUBackendManager) bool {
```

- fn `selectBackend`

Force selection of a specific backend with validation and statistics


```zig
pub fn selectBackend(self: *GPUBackendManager, backend: BackendType) GPUBackendError!void {
```

- fn `hasBackend`

Check if a specific backend is available


```zig
pub fn hasBackend(self: *GPUBackendManager, backend: BackendType) bool {
```

- fn `getBackendCapabilities`

Get capabilities for a specific backend


```zig
pub fn getBackendCapabilities(self: *GPUBackendManager, backend: BackendType) !HardwareCapabilities {
```

- fn `compileShader`

Compile shader for current backend with comprehensive error handling


```zig
pub fn compileShader(
```

- fn `printSystemInfo`

Print comprehensive system information with performance metrics


```zig
pub fn printSystemInfo(self: *GPUBackendManager) void {
```

- fn `getSystemInfoString`

Get system information as a formatted string


```zig
pub fn getSystemInfoString(self: *GPUBackendManager, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `validateConfiguration`

Validate current backend configuration


```zig
pub fn validateConfiguration(self: *GPUBackendManager) GPUBackendError!void {
```

- fn `getRecommendedBackendForWorkload`

Get recommended backend based on workload characteristics


```zig
pub fn getRecommendedBackendForWorkload(
```

- type `WorkloadCharacteristics`

Workload characteristics for backend recommendation


```zig
pub const WorkloadCharacteristics = struct {
```

## src/features/gpu/compute/kernels.zig

- type `KernelConfig`

Configuration for GPU kernel operations


```zig
pub const KernelConfig = struct {
```

- type `LayerType`

Neural network layer types


```zig
pub const LayerType = enum {
```

- type `ActivationType`

Activation functions


```zig
pub const ActivationType = enum {
```

- type `OptimizerType`

Optimization algorithms


```zig
pub const OptimizerType = enum {
```

- type `KernelManager`

GPU Kernel Manager - Manages specialized compute kernels


```zig
pub const KernelManager = struct {
```

- type `Kernel`

```zig
pub const Kernel = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *GPURenderer) !*KernelManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *KernelManager) void {
```

- fn `createDenseLayer`

Create a dense neural network layer kernel


```zig
pub fn createDenseLayer(
```

- fn `forwardDense`

Forward pass through a dense layer


```zig
pub fn forwardDense(self: *KernelManager, kernel_idx: usize, input_handle: u32, output_handle: u32) !void {
```

- fn `backwardDense`

Backward pass through a dense layer


```zig
pub fn backwardDense(self: *KernelManager, kernel_idx: usize, input_handle: u32, grad_output_handle: u32, grad_input_handle: u32) !void {
```

- fn `createConvLayer`

Create a convolutional layer kernel


```zig
pub fn createConvLayer(
```

- fn `createAttentionLayer`

Create an attention layer kernel for transformer models


```zig
pub fn createAttentionLayer(
```

- fn `matrixMultiplyGPU`

Perform matrix multiplication optimized for GPU


```zig
pub fn matrixMultiplyGPU(
```

- fn `softmaxGPU`

Compute softmax activation function on GPU


```zig
pub fn softmaxGPU(
```

- fn `layerNormGPU`

Compute layer normalization on GPU


```zig
pub fn layerNormGPU(
```

- fn `adamUpdateGPU`

Update weights using Adam optimizer on GPU


```zig
pub fn adamUpdateGPU(
```

- type `MemoryPool`

GPU Memory Pool for efficient memory management


```zig
pub const MemoryPool = struct {
```

- type `BufferInfo`

```zig
pub const BufferInfo = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, renderer: *GPURenderer) !*MemoryPool {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate or reuse a buffer from the pool


```zig
pub fn allocBuffer(self: *MemoryPool, size: usize, usage: gpu_renderer.BufferUsage) !u32 {
```

- fn `freeBuffer`

Return a buffer to the pool for reuse


```zig
pub fn freeBuffer(self: *MemoryPool, handle: u32) !void {
```

- fn `cleanup`

Clean up old unused buffers to free memory


```zig
pub fn cleanup(self: *MemoryPool, max_age_ms: i64) !void {
```

- fn `getStats`

Get memory pool statistics


```zig
pub fn getStats(self: *MemoryPool) MemoryStats {
```

- type `MemoryStats`

```zig
pub const MemoryStats = struct {
```

- type `BackendSupport`

GPU Backend Support for multiple APIs


```zig
pub const BackendSupport = struct {
```

- type `Backend`

Supported GPU backends


```zig
pub const Backend = enum {
```

- type `Capabilities`

Backend capabilities


```zig
pub const Capabilities = struct {
```

- fn `init`

Initialize backend support detection


```zig
pub fn init(allocator: std.mem.Allocator) !*BackendSupport {
```

- fn `deinit`

```zig
pub fn deinit(self: *BackendSupport) void {
```

- fn `selectBestBackend`

Select the best available backend


```zig
pub fn selectBestBackend(self: *BackendSupport) ?Backend {
```

- fn `selectBackend`

Force selection of a specific backend


```zig
pub fn selectBackend(self: *BackendSupport, backend: Backend) !void {
```

- fn `detectAvailableBackends`

Detect available GPU backends


```zig
pub fn detectAvailableBackends(self: *BackendSupport) ![]Backend {
```

- fn `getCapabilities`

Get capabilities for a specific backend


```zig
pub fn getCapabilities(self: *BackendSupport, backend: Backend) !Capabilities {
```

- const `GPURenderer`

```zig
pub const GPURenderer = gpu_renderer.GPURenderer;
```

- const `GpuError`

```zig
pub const GpuError = gpu_renderer.GpuError;
```

## src/features/gpu/compute/mod.zig

- const `kernels`

```zig
pub const kernels = @import("kernels.zig");
```

- const `gpu_backend_manager`

```zig
pub const gpu_backend_manager = @import("gpu_backend_manager.zig");
```

- const `gpu_ai_acceleration`

```zig
pub const gpu_ai_acceleration = @import("gpu_ai_acceleration.zig");
```

## src/features/gpu/core/backend.zig

- type `Db`

```zig
pub const Db = struct {
```

- type `Result`

```zig
pub const Result = struct {
```

- fn `lessThanAsc`

```zig
pub fn lessThanAsc(_: void, a: Result, b: Result) bool {
```

- type `Header`

```zig
pub const Header = struct {
```

- fn `open`

```zig
pub fn open(_: []const u8, _: bool) !*Db {
```

- fn `close`

```zig
pub fn close(_: *Db) void {
```

- fn `init`

```zig
pub fn init(_: *Db, _: u32) !void {
```

- fn `addEmbedding`

```zig
pub fn addEmbedding(_: *Db, _: []const f32) !u32 {
```

- type `GpuBackendConfig`

Configuration for the GPU backend.


```zig
pub const GpuBackendConfig = struct {
```

- type `GpuBackend`

Main context for GPU-accelerated operations.


```zig
pub const GpuBackend = struct {
```

- const `Error`

Error set for all GPU backend operations.


```zig
pub const Error = error{
```

- fn `init`

Initialize the GPU backend with the given configuration.


```zig
pub fn init(allocator: std.mem.Allocator, config: GpuBackendConfig) Error!*GpuBackend {
```

- fn `deinit`

Clean up and release all resources held by the backend.


```zig
pub fn deinit(self: *GpuBackend) void {
```

- fn `isGpuAvailable`

Check if a suitable GPU is available for compute.


```zig
pub fn isGpuAvailable(self: *const GpuBackend) bool {
```

- fn `searchSimilar`

Perform a vector similarity search for the given query vector against the database.
Returns the top_k closest results. Falls back to CPU if GPU is unavailable.


```zig
pub fn searchSimilar(self: *GpuBackend, db: *const Db, query: []const f32, top_k: usize) Error![]Db.Result {
```

- fn `hasMemoryFor`

Returns true if there is enough GPU memory for an operation.


```zig
pub fn hasMemoryFor(self: *const GpuBackend, bytes: u64) bool {
```

- fn `batchSearch`

Batch version of searchSimilar, processes multiple queries and returns results for each.


```zig
pub fn batchSearch(self: *GpuBackend, db: *const Db, queries: []const []const f32, top_k: usize) Error![][]Db.Result {
```

- type `BatchConfig`

Configuration for batch processing.


```zig
pub const BatchConfig = struct {
```

- type `BatchProcessor`

Utility for batch processing of vector search queries.


```zig
pub const BatchProcessor = struct {
```

- fn `init`

```zig
pub fn init(backend: *GpuBackend, config: BatchConfig) BatchProcessor {
```

- fn `processBatch`

Process a batch of queries with optional progress reporting.


```zig
pub fn processBatch(self: *BatchProcessor, db: *const Db, queries: []const []const f32, top_k: usize) ![][]Db.Result {
```

- fn `processBatchWithCallback`

Process queries with a callback for each result.


```zig
pub fn processBatchWithCallback(
```

- type `GpuStats`

Tracks performance statistics for GPU operations.


```zig
pub const GpuStats = struct {
```

- fn `recordOperation`

```zig
pub fn recordOperation(self: *GpuStats, duration_us: u64, memory_used: u64, used_cpu: bool) void {
```

- fn `getAverageOperationTime`

```zig
pub fn getAverageOperationTime(self: *const GpuStats) u64 {
```

- fn `print`

```zig
pub fn print(self: *const GpuStats) void {
```

## src/features/gpu/core/gpu_renderer.zig

- const `has_webgpu_support`

```zig
pub const has_webgpu_support = @hasDecl(std, "gpu") and @hasDecl(std.gpu, "Instance");
```

- const `GpuError`

GPU renderer errors


```zig
pub const GpuError = error{
```

- type `SPIRVOptimizationLevel`

SPIR-V compiler optimization levels


```zig
pub const SPIRVOptimizationLevel = enum {
```

- pub `inline`

```zig
pub inline fn toInt(self: SPIRVOptimizationLevel) u32 {
```

- type `SPIRVCompilerOptions`

SPIR-V compiler configuration structure


```zig
pub const SPIRVCompilerOptions = struct {
```

- fn `validate`

```zig
pub fn validate(self: SPIRVCompilerOptions) !void {
```

- type `MSLOptimizationLevel`

Metal Shading Language optimization levels


```zig
pub const MSLOptimizationLevel = enum {
```

- pub `inline`

```zig
pub inline fn toInt(self: MSLOptimizationLevel) u32 {
```

- type `MetalVersion`

Metal target versions


```zig
pub const MetalVersion = enum {
```

- pub `inline`

```zig
pub inline fn toVersionString(self: MetalVersion) []const u8 {
```

- type `MSLCompilerOptions`

Metal Shading Language compiler configuration structure


```zig
pub const MSLCompilerOptions = struct {
```

- type `Platform`

```zig
pub const Platform = enum {
```

- pub `inline`

```zig
pub inline fn isApplePlatform(self: Platform) bool {
```

- fn `validate`

```zig
pub fn validate(self: MSLCompilerOptions) !void {
```

- type `PTXOptimizationLevel`

PTX optimization levels


```zig
pub const PTXOptimizationLevel = enum {
```

- pub `inline`

```zig
pub inline fn toString(self: PTXOptimizationLevel) []const u8 {
```

- type `CudaComputeCapability`

CUDA compute capabilities


```zig
pub const CudaComputeCapability = enum {
```

- pub `inline`

```zig
pub inline fn toString(self: CudaComputeCapability) []const u8 {
```

- pub `inline`

```zig
pub inline fn getMajorVersion(self: CudaComputeCapability) u32 {
```

- type `PTXCompilerOptions`

PTX (Parallel Thread Execution) compiler configuration structure


```zig
pub const PTXCompilerOptions = struct {
```

- fn `validate`

```zig
pub fn validate(self: PTXCompilerOptions) !void {
```

- pub `inline`

```zig
pub inline fn getComputeCapabilityString(self: PTXCompilerOptions) []const u8 {
```

- type `SPIRVCompiler`

SPIR-V compiler for Vulkan, OpenGL, and OpenCL backends


```zig
pub const SPIRVCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: SPIRVCompilerOptions) !*SPIRVCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *SPIRVCompiler) void {
```

- fn `compileShader`

```zig
pub fn compileShader(self: *SPIRVCompiler, source: []const u8, stage: ShaderStage) ![]u8 {
```

- fn `validateSPIRV`

```zig
pub fn validateSPIRV(_self: *SPIRVCompiler, spirv_code: []const u8) !bool {
```

- fn `disassembleSPIRV`

```zig
pub fn disassembleSPIRV(self: *SPIRVCompiler, spirv_code: []const u8) ![]u8 {
```

- type `MSLCompiler`

Metal Shading Language compiler for Apple platforms


```zig
pub const MSLCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: MSLCompilerOptions) !*MSLCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *MSLCompiler) void {
```

- fn `compileShader`

```zig
pub fn compileShader(self: *MSLCompiler, source: []const u8, stage: ShaderStage) ![]u8 {
```

- type `PTXCompiler`

PTX compiler for NVIDIA CUDA platforms


```zig
pub const PTXCompiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, options: PTXCompilerOptions) !*PTXCompiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *PTXCompiler) void {
```

- fn `compileKernel`

```zig
pub fn compileKernel(self: *PTXCompiler, source: []const u8) ![]u8 {
```

- type `GPUConfig`

GPU renderer configuration with compile-time optimization


```zig
pub const GPUConfig = struct {
```

- fn `validate`

Compile-time validation of configuration


```zig
pub fn validate(comptime config: GPUConfig) void {
```

- type `PowerPreference`

Power preference for GPU selection


```zig
pub const PowerPreference = enum {
```

- pub `inline`

Inline function for quick preference checks


```zig
pub inline fn isHighPerformance(self: PowerPreference) bool {
```

- type `Backend`

GPU backend types with platform detection and compile-time optimization


```zig
pub const Backend = enum {
```

- pub `inline`

```zig
pub inline fn isAvailable(self: Backend) bool {
```

- fn `getBest`

```zig
pub fn getBest() Backend {
```

- pub `inline`

Inline function for performance checks


```zig
pub inline fn requiresGPU(self: Backend) bool {
```

- fn `getPriority`

Get priority score for backend selection (higher is better)


```zig
pub fn getPriority(self: Backend) u8 {
```

- fn `toString`

Convert backend to human-readable string


```zig
pub fn toString(self: Backend) []const u8 {
```

- type `BufferUsage`

GPU buffer usage flags


```zig
pub const BufferUsage = packed struct {
```

- pub `inline`

Inline function for quick usage checks


```zig
pub inline fn isReadable(self: BufferUsage) bool {
```

- pub `inline`

Inline function for quick usage checks


```zig
pub inline fn isWritable(self: BufferUsage) bool {
```

- type `TextureFormat`

GPU texture format


```zig
pub const TextureFormat = enum {
```

- pub `inline`

Inline function for format properties


```zig
pub inline fn getBytesPerPixel(self: TextureFormat) u32 {
```

- pub `inline`

Inline function for format checks


```zig
pub inline fn isFloatFormat(self: TextureFormat) bool {
```

- type `ShaderStage`

Shader stage types


```zig
pub const ShaderStage = enum {
```

- pub `inline`

```zig
pub inline fn toWebGPU(self: ShaderStage) u32 {
```

- type `Color`

Color for clearing operations with inline utility functions


```zig
pub const Color = struct {
```

- const `BLACK`

Compile-time color constants


```zig
pub const BLACK = Color{ .r = 0.0, .g = 0.0, .b = 0.0, .a = 1.0 };
```

- const `WHITE`

```zig
pub const WHITE = Color{ .r = 1.0, .g = 1.0, .b = 1.0, .a = 1.0 };
```

- const `RED`

```zig
pub const RED = Color{ .r = 1.0, .g = 0.0, .b = 0.0, .a = 1.0 };
```

- const `GREEN`

```zig
pub const GREEN = Color{ .r = 0.0, .g = 1.0, .b = 0.0, .a = 1.0 };
```

- const `BLUE`

```zig
pub const BLUE = Color{ .r = 0.0, .g = 0.0, .b = 1.0, .a = 1.0 };
```

- pub `inline`

Inline utility functions


```zig
pub inline fn fromRGB(r: f32, g: f32, b: f32) Color {
```

- pub `inline`

```zig
pub inline fn lerp(a: Color, b: Color, t: f32) Color {
```

- pub `inline`

Inline function to convert to packed format


```zig
pub inline fn toPackedRGBA(self: Color) u32 {
```

- type `GPUHandle`

GPU resource handle with generation for safety and inline utilities


```zig
pub const GPUHandle = struct {
```

- pub `inline`

```zig
pub inline fn invalid() GPUHandle {
```

- pub `inline`

```zig
pub inline fn isValid(self: GPUHandle) bool {
```

- pub `inline`

Inline function for handle comparison


```zig
pub inline fn equals(self: GPUHandle, other: GPUHandle) bool {
```

- type `MathUtils`

High-performance math utilities with SIMD operations


```zig
pub const MathUtils = struct {
```

- pub `inline`

Inline vector operations


```zig
pub inline fn vectorAdd(comptime T: type, a: []const T, b: []const T, result: []T) void {
```

- pub `inline`

Inline matrix multiplication with cache-friendly access patterns


```zig
pub inline fn matrixMultiply(comptime T: type, a: []const T, b: []const T, result: []T, size: usize) void {
```

- pub `inline`

Inline approximation functions for faster math


```zig
pub inline fn fastSqrt(x: f32) f32 {
```

- pub `inline`

Inline function for fast approximate equality


```zig
pub inline fn approxEqual(a: f32, b: f32) bool {
```

- type `Instance`

```zig
pub const Instance = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Instance {
```

- fn `deinit`

```zig
pub fn deinit(self: *Instance) void {
```

- fn `requestAdapter`

```zig
pub fn requestAdapter(self: *Instance) !*Adapter {
```

- type `Adapter`

```zig
pub const Adapter = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Adapter {
```

- fn `deinit`

```zig
pub fn deinit(self: *Adapter) void {
```

- fn `getName`

```zig
pub fn getName(self: *Adapter) []const u8 {
```

- fn `requestDevice`

```zig
pub fn requestDevice(self: *Adapter) !*Device {
```

- type `Device`

```zig
pub const Device = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Device {
```

- fn `deinit`

```zig
pub fn deinit(self: *Device) void {
```

- fn `createBuffer`

```zig
pub fn createBuffer(self: *Device, size: usize, usage: BufferUsage) !*MockGPU.Buffer {
```

- type `Queue`

```zig
pub const Queue = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator) !*Queue {
```

- fn `deinit`

```zig
pub fn deinit(self: *Queue) void {
```

- fn `writeBuffer`

```zig
pub fn writeBuffer(self: *Queue, buffer: *MockGPU.Buffer, data: []const u8) void {
```

- fn `submit`

```zig
pub fn submit(self: *Queue) void {
```

- fn `onSubmittedWorkDone`

```zig
pub fn onSubmittedWorkDone(self: *Queue) void {
```

- type `Buffer`

```zig
pub const Buffer = struct {
```

- fn `create`

```zig
pub fn create(allocator: std.mem.Allocator, size: usize) !*MockGPU.Buffer {
```

- fn `deinit`

```zig
pub fn deinit(self: *MockGPU.Buffer) void {
```

- fn `getMappedRange`

```zig
pub fn getMappedRange(self: *MockGPU.Buffer, comptime T: type, offset: usize, length: usize) ?[]T {
```

- fn `unmap`

```zig
pub fn unmap(self: *MockGPU.Buffer) void {
```

- type `GPUContext`

Extended GPU context with compiler support


```zig
pub const GPUContext = struct {
```

- fn `init`

Initialize GPU context with error handling


```zig
pub fn init(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initVulkan`

Initialize Vulkan-specific context


```zig
pub fn initVulkan(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initMetal`

Initialize Metal-specific context


```zig
pub fn initMetal(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initDX12`

Initialize DirectX 12-specific context


```zig
pub fn initDX12(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initOpenGL`

Initialize OpenGL-specific context


```zig
pub fn initOpenGL(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initOpenCL`

Initialize OpenCL-specific context


```zig
pub fn initOpenCL(allocator: std.mem.Allocator) !GPUContext {
```

- fn `initCUDA`

Initialize CUDA-specific context


```zig
pub fn initCUDA(allocator: std.mem.Allocator) !GPUContext {
```

- fn `deinit`

Clean up GPU resources


```zig
pub fn deinit(self: *GPUContext) void {
```

- fn `printDeviceInfo`

Get device info for debugging


```zig
pub fn printDeviceInfo(self: *GPUContext) void {
```

- type `BufferManager`

Buffer manager for simplified GPU buffer operations


```zig
pub const BufferManager = struct {
```

- fn `createBuffer`

Create a GPU buffer with specified type and usage


```zig
pub fn createBuffer(self: BufferManager, comptime T: type, size: u64, usage: BufferUsage) !*MockGPU.Buffer {
```

- fn `writeBuffer`

Write data to GPU buffer


```zig
pub fn writeBuffer(self: BufferManager, buffer: *MockGPU.Buffer, data: anytype) void {
```

- fn `readBuffer`

Read data from GPU buffer


```zig
pub fn readBuffer(self: BufferManager, comptime T: type, buffer: *MockGPU.Buffer, size: u64, allocator: std.mem.Allocator) ![]T {
```

- fn `createBufferWithData`

Create a buffer with initial data


```zig
pub fn createBufferWithData(self: BufferManager, comptime T: type, data: []const T, usage: BufferUsage) !*MockGPU.Buffer {
```

- type `Buffer`

GPU buffer resource with platform abstraction


```zig
pub const Buffer = struct {
```

- fn `init`

```zig
pub fn init(gpu_buffer: *MockGPU.Buffer, size: usize, usage: BufferUsage, id: u64) Buffer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Buffer) void {
```

- fn `map`

```zig
pub fn map(self: *Buffer, allocator: std.mem.Allocator) ![]u8 {
```

- fn `unmap`

```zig
pub fn unmap(self: *Buffer) void {
```

- type `Shader`

Shader resource


```zig
pub const Shader = struct {
```

- fn `compile`

```zig
pub fn compile(allocator: std.mem.Allocator, stage: ShaderStage, source: []const u8) !Shader {
```

- fn `deinit`

```zig
pub fn deinit(self: *Shader) void {
```

- type `BindGroup`

Bind group for resource binding


```zig
pub const BindGroup = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, id: u64) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addBuffer`

```zig
pub fn addBuffer(self: *Self, buffer: Buffer) !void {
```

- type `RendererStats`

Lightweight renderer statistics


```zig
pub const RendererStats = struct {
```

- type `GPURenderer`

Main GPU renderer with cross-platform support and CPU fallbacks


```zig
pub const GPURenderer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: GPUConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `createBuffer`

Create a GPU buffer with specified usage


```zig
pub fn createBuffer(self: *Self, size: usize, usage: BufferUsage) !u32 {
```

- fn `createBufferWithData`

Convenience: create a buffer initialized with data


```zig
pub fn createBufferWithData(self: *Self, comptime T: type, data: []const T, usage: BufferUsage) !u32 {
```

- fn `destroyBuffer`

Destroy a buffer by handle


```zig
pub fn destroyBuffer(self: *Self, handle: u32) !void {
```

- fn `writeBuffer`

Write raw bytes into a buffer


```zig
pub fn writeBuffer(self: *Self, handle: u32, data: []const u8) !void {
```

- fn `readBuffer`

Read raw bytes from a buffer (copies into a new slice)


```zig
pub fn readBuffer(self: *Self, handle: u32, allocator: std.mem.Allocator) ![]u8 {
```

- fn `copyBuffer`

Copy contents from src to dst (copies min(src.size, dst.size) bytes)


```zig
pub fn copyBuffer(self: *Self, src_handle: u32, dst_handle: u32) !usize {
```

- fn `computeVectorDotBuffers`

Compute vector dot product directly on buffers (length in f32 elements)


```zig
pub fn computeVectorDotBuffers(self: *Self, a_handle: u32, b_handle: u32, length: usize) !f32 {
```

- fn `beginFrame`

Begin frame rendering


```zig
pub fn beginFrame(self: *Self) !void {
```

- fn `endFrame`

End frame rendering


```zig
pub fn endFrame(self: *Self) !void {
```

- fn `clear`

Clear the render target with specified color


```zig
pub fn clear(self: *Self, color: Color) !void {
```

- fn `vectorAdd`

High-performance vector addition with optimized memory patterns


```zig
pub fn vectorAdd(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `matrixMultiply`

High-performance matrix multiplication with cache-optimized implementation


```zig
pub fn matrixMultiply(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `imageProcessing`

High-performance image processing with optimized blur algorithm


```zig
pub fn imageProcessing(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `computeMatrixMultiply`

High-performance matrix multiplication with optimized algorithms


```zig
pub fn computeMatrixMultiply(self: *Self, a: []const f32, b: []const f32, result: []f32, m: u32, n: u32, k: u32) !void {
```

- fn `computeNeuralInference`

Run neural network inference


```zig
pub fn computeNeuralInference(self: *Self, input: []const f32, weights: []const f32, output: []f32) !void {
```

- fn `renderNeuralNetwork`

Render neural network visualization


```zig
pub fn renderNeuralNetwork(self: *Self, neural_engine: anytype) !void {
```

- fn `runExamples`

High-performance example runner with benchmarking


```zig
pub fn runExamples(self: *Self, allocator: std.mem.Allocator) !void {
```

- fn `getFPS`

Get current FPS


```zig
pub fn getFPS(self: *Self) f32 {
```

- fn `getFrameCount`

Get frame count


```zig
pub fn getFrameCount(self: *Self) u64 {
```

- fn `getStats`

Get current renderer stats (copy)


```zig
pub fn getStats(self: *Self) RendererStats {
```

- fn `runExamples`

Standalone function for running optimized GPU examples


```zig
pub fn runExamples() !void {
```

- fn `main`

Main function for running the combined GPU examples


```zig
pub fn main() !void {
```

## src/features/gpu/core/mod.zig

- const `GPURenderer`

```zig
pub const GPURenderer = @import("gpu_renderer.zig").GPURenderer;
```

- const `GPUConfig`

```zig
pub const GPUConfig = @import("gpu_renderer.zig").GPUConfig;
```

- const `GpuError`

```zig
pub const GpuError = @import("gpu_renderer.zig").GpuError;
```

- const `Color`

```zig
pub const Color = @import("gpu_renderer.zig").Color;
```

- const `GPUHandle`

```zig
pub const GPUHandle = @import("gpu_renderer.zig").GPUHandle;
```

- const `Backend`

```zig
pub const Backend = @import("gpu_renderer.zig").Backend;
```

- const `PowerPreference`

```zig
pub const PowerPreference = @import("gpu_renderer.zig").PowerPreference;
```

- const `has_webgpu_support`

```zig
pub const has_webgpu_support = @import("gpu_renderer.zig").has_webgpu_support;
```

- const `GpuBackend`

```zig
pub const GpuBackend = @import("backend.zig").GpuBackend;
```

- const `GpuBackendConfig`

```zig
pub const GpuBackendConfig = @import("backend.zig").GpuBackendConfig;
```

- const `GpuBackendError`

```zig
pub const GpuBackendError = @import("backend.zig").GpuBackend.Error;
```

- const `BatchConfig`

```zig
pub const BatchConfig = @import("backend.zig").BatchConfig;
```

- const `BatchProcessor`

```zig
pub const BatchProcessor = @import("backend.zig").BatchProcessor;
```

- const `GpuStats`

```zig
pub const GpuStats = @import("backend.zig").GpuStats;
```

- const `Db`

```zig
pub const Db = @import("backend.zig").Db;
```

- const `KernelManager`

```zig
pub const KernelManager = @import("../compute/kernels.zig").KernelManager;
```

- const `GPUBackendManager`

```zig
pub const GPUBackendManager = @import("../compute/gpu_backend_manager.zig").GPUBackendManager;
```

- const `SPIRVCompiler`

```zig
pub const SPIRVCompiler = @import("../compute/gpu_backend_manager.zig").SPIRVCompiler;
```

- const `BackendType`

```zig
pub const BackendType = @import("../compute/gpu_backend_manager.zig").BackendType;
```

- const `HardwareCapabilities`

```zig
pub const HardwareCapabilities = @import("../compute/gpu_backend_manager.zig").HardwareCapabilities;
```

- const `MemoryBandwidthBenchmark`

```zig
pub const MemoryBandwidthBenchmark = @import("../compute/gpu_backend_manager.zig").MemoryBandwidthBenchmark;
```

- const `ComputeThroughputBenchmark`

```zig
pub const ComputeThroughputBenchmark = @import("../compute/gpu_backend_manager.zig").ComputeThroughputBenchmark;
```

- const `PerformanceProfiler`

```zig
pub const PerformanceProfiler = @import("../compute/gpu_backend_manager.zig").PerformanceProfiler;
```

- const `MemoryPool`

```zig
pub const MemoryPool = @import("../memory/memory_pool.zig").MemoryPool;
```

- const `BackendSupport`

```zig
pub const BackendSupport = @import("../compute/kernels.zig").BackendSupport;
```

- const `BenchmarkConfig`

```zig
pub const BenchmarkConfig = @import("../benchmark/benchmarks.zig").BenchmarkConfig;
```

- const `WorkloadType`

```zig
pub const WorkloadType = @import("../benchmark/benchmarks.zig").WorkloadType;
```

- const `PerformanceGrade`

```zig
pub const PerformanceGrade = @import("../benchmark/benchmarks.zig").PerformanceGrade;
```

- const `BenchmarkResult`

```zig
pub const BenchmarkResult = @import("../benchmark/benchmarks.zig").BenchmarkResult;
```

- const `Allocator`

```zig
pub const Allocator = std.mem.Allocator;
```

- fn `initDefault`

Initialize the GPU system with default configuration


```zig
pub fn initDefault(allocator: std.mem.Allocator) !*GPURenderer {
```

- fn `isGpuAvailable`

Check if GPU acceleration is available


```zig
pub fn isGpuAvailable() bool {
```

## src/features/gpu/cross_compilation.zig

- const `CrossCompilationError`

Cross-compilation specific errors


```zig
pub const CrossCompilationError = error{
```

- type `CrossCompilationTarget`

Cross-compilation target configuration with validation and error handling


```zig
pub const CrossCompilationTarget = struct {
```

- fn `init`

Create a new cross-compilation target with validation


```zig
pub fn init(
```

- fn `deinit`

Safely deinitialize the target and free resources


```zig
pub fn deinit(self: *CrossCompilationTarget) void {
```

- fn `validate`

Validate the target configuration


```zig
pub fn validate(self: *const CrossCompilationTarget) CrossCompilationError!void {
```

- fn `description`

Get a human-readable description of the target


```zig
pub fn description(self: *const CrossCompilationTarget) []const u8 {
```

- fn `supportsFeature`

Check if this target supports a specific feature


```zig
pub fn supportsFeature(self: *const CrossCompilationTarget, feature: TargetFeature) bool {
```

- type `TargetFeature`

Target feature enumeration


```zig
pub const TargetFeature = enum {
```

- type `GPUBackend`

GPU backend selection for cross-compilation


```zig
pub const GPUBackend = enum {
```

- type `OptimizationLevel`

Optimization level for cross-compilation


```zig
pub const OptimizationLevel = enum {
```

- type `TargetFeatures`

Target-specific features with enhanced capability detection


```zig
pub const TargetFeatures = struct {
```

- fn `init`

Initialize target features based on architecture and OS


```zig
pub fn init(
```

- fn `deinit`

Safely deinitialize target features


```zig
pub fn deinit(self: *TargetFeatures) void {
```

- fn `validate`

Validate feature configuration


```zig
pub fn validate(self: *const TargetFeatures) CrossCompilationError!void {
```

- fn `supportsFeature`

Check if a specific feature is supported


```zig
pub fn supportsFeature(self: *const TargetFeatures, feature: TargetFeature) bool {
```

- fn `getFeatureSummary`

Get feature summary as a human-readable string


```zig
pub fn getFeatureSummary(self: *const TargetFeatures, allocator: std.mem.Allocator) ![]const u8 {
```

- type `MemoryModel`

Memory model for target architecture


```zig
pub const MemoryModel = enum {
```

- type `ThreadingModel`

Threading model for target architecture


```zig
pub const ThreadingModel = enum {
```

- type `CrossCompilationManager`

Cross-compilation manager with enhanced error handling and resource management


```zig
pub const CrossCompilationManager = struct {
```

- fn `init`

Initialize the cross-compilation manager


```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

Deinitialize the manager and free all resources


```zig
pub fn deinit(self: *Self) void {
```

- fn `hash`

```zig
pub fn hash(self: TargetKey) u64 {
```

- fn `eql`

```zig
pub fn eql(self: TargetKey, other: TargetKey) bool {
```

- fn `format`

Create a human-readable string representation


```zig
pub fn format(self: TargetKey, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
```

- fn `deinit`

```zig
pub fn deinit(self: *BuildConfig) void {
```

- fn `registerTarget`

Register a cross-compilation target with validation


```zig
pub fn registerTarget(self: *Self, target: CrossCompilationTarget) CrossCompilationError!void {
```

- fn `getTarget`

Get cross-compilation target for architecture


```zig
pub fn getTarget(self: *Self, arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag, abi: std.Target.Abi) ?*CrossCompilationTarget {
```

- fn `getBuildConfig`

Get build configuration for target


```zig
pub fn getBuildConfig(self: *Self, arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag, abi: std.Target.Abi) ?*BuildConfig {
```

- type `PredefinedTargets`

Predefined cross-compilation targets


```zig
pub const PredefinedTargets = struct {
```

- fn `wasmTarget`

WebAssembly target for web deployment


```zig
pub fn wasmTarget(allocator: std.mem.Allocator) !CrossCompilationTarget {
```

- fn `arm64Target`

ARM64 target for mobile and embedded systems


```zig
pub fn arm64Target(allocator: std.mem.Allocator, os: std.Target.Os.Tag) !CrossCompilationTarget {
```

- fn `riscv64Target`

RISC-V target for embedded and HPC systems


```zig
pub fn riscv64Target(allocator: std.mem.Allocator, os: std.Target.Os.Tag) !CrossCompilationTarget {
```

- fn `x86_64Target`

x86_64 target for desktop systems


```zig
pub fn x86_64Target(allocator: std.mem.Allocator, os: std.Target.Os.Tag) !CrossCompilationTarget {
```

- type `CrossCompilationUtils`

Cross-compilation utility functions


```zig
pub const CrossCompilationUtils = struct {
```

- fn `supportsGPUAcceleration`

Check if target architecture supports GPU acceleration


```zig
pub fn supportsGPUAcceleration(arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag) bool {
```

- fn `getRecommendedGPUBackend`

Get recommended GPU backend for target


```zig
pub fn getRecommendedGPUBackend(arch: std.Target.Cpu.Arch, os: std.Target.Os.Tag) GPUBackend {
```

- fn `supportsSIMD`

Check if target supports SIMD operations


```zig
pub fn supportsSIMD(arch: std.Target.Cpu.Arch) bool {
```

- fn `getOptimalMemoryAlignment`

Get optimal memory alignment for target


```zig
pub fn getOptimalMemoryAlignment(arch: std.Target.Cpu.Arch) u32 {
```

- fn `getOptimalThreadCount`

Get optimal thread count for target


```zig
pub fn getOptimalThreadCount(arch: std.Target.Cpu.Arch, _: std.Target.Os.Tag) u32 {
```

- fn `logCrossCompilationTarget`

Log cross-compilation target information


```zig
pub fn logCrossCompilationTarget(target: *const CrossCompilationTarget) void {
```

## src/features/gpu/demo/advanced_gpu_demo.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/features/gpu/demo/enhanced_gpu_demo.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/features/gpu/demo/gpu_demo.zig

- fn `format`

```zig
pub fn format(self: PerformanceMetrics, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
```

- fn `fromTemperature`

```zig
pub fn fromTemperature(temp_c: f32) ThermalState {
```

- fn `detect`

```zig
pub fn detect() ArchitectureFeatures {
```

- fn `logFeatures`

```zig
pub fn logFeatures(self: ArchitectureFeatures) void {
```

- fn `main`

```zig
pub fn main() !void {
```

- fn `update`

```zig
pub fn update(self: *HardwareMonitor) void {
```

- fn `checkThrottling`

```zig
pub fn checkThrottling(self: *ThermalMonitor) bool {
```

- fn `checkPowerLimit`

```zig
pub fn checkPowerLimit(self: *PowerMonitor) bool {
```

- fn `deinit`

```zig
pub fn deinit(self: ComprehensiveReportData) void {
```

## src/features/gpu/demo/mod.zig

- const `gpu_demo`

```zig
pub const gpu_demo = @import("gpu_demo.zig");
```

- const `enhanced_gpu_demo`

```zig
pub const enhanced_gpu_demo = @import("enhanced_gpu_demo.zig");
```

- const `advanced_gpu_demo`

```zig
pub const advanced_gpu_demo = @import("advanced_gpu_demo.zig");
```

- const `test_shader`

```zig
pub const test_shader = @embedFile("test_shader.glsl");
```

## src/features/gpu/gpu_examples.zig

- fn `init`

Initialize GPU context with error handling


```zig
pub fn init() !GPUContext {
```

- fn `deinit`

Clean up GPU resources


```zig
pub fn deinit(self: *GPUContext) void {
```

- fn `printDeviceInfo`

Get device info for debugging


```zig
pub fn printDeviceInfo(self: *GPUContext) void {
```

- fn `init`

Initialize compute pipeline with shader


```zig
pub fn init(device: *gpu.Device, shader_source: []const u8) !ComputePipeline {
```

- fn `deinit`

Clean up pipeline resources


```zig
pub fn deinit(self: *ComputePipeline) void {
```

- fn `createBuffer`

Create a GPU buffer with specified type and usage


```zig
pub fn createBuffer(self: BufferManager, comptime T: type, size: u64, usage: gpu.Buffer.Usage) !*gpu.Buffer {
```

- fn `writeBuffer`

Write data to GPU buffer


```zig
pub fn writeBuffer(self: BufferManager, buffer: *gpu.Buffer, data: anytype) void {
```

- fn `readBuffer`

Read data from GPU buffer with staging buffer


```zig
pub fn readBuffer(self: BufferManager, comptime T: type, buffer: *gpu.Buffer, size: u64, allocator: std.mem.Allocator) ![]T {
```

- fn `createBufferWithData`

Create a buffer with initial data


```zig
pub fn createBufferWithData(self: BufferManager, comptime T: type, data: []const T, usage: gpu.Buffer.Usage) !*gpu.Buffer {
```

- fn `main`

Main function demonstrating GPU compute capabilities


```zig
pub fn main() !void {
```

## src/features/gpu/gpu_renderer.zig

- const `GpuError`

GPU renderer errors


```zig
pub const GpuError = error{
```

- type `GPUConfig`

GPU renderer configuration


```zig
pub const GPUConfig = struct {
```

- type `PowerPreference`

Power preference for GPU selection


```zig
pub const PowerPreference = enum {
```

- type `Backend`

GPU backend types with platform detection


```zig
pub const Backend = enum {
```

- fn `isAvailable`

```zig
pub fn isAvailable(self: Backend) bool {
```

- fn `getBest`

```zig
pub fn getBest() Backend {
```

- type `BufferUsage`

GPU buffer usage flags with WebGPU compatibility


```zig
pub const BufferUsage = packed struct {
```

- fn `toWebGPU`

```zig
pub fn toWebGPU(self: BufferUsage) u32 {
```

- type `TextureFormat`

GPU texture format with format translation


```zig
pub const TextureFormat = enum {
```

- fn `toWebGPU`

```zig
pub fn toWebGPU(self: TextureFormat) []const u8 {
```

- type `ShaderStage`

Shader stage types


```zig
pub const ShaderStage = enum {
```

- fn `toWebGPU`

```zig
pub fn toWebGPU(self: ShaderStage) u32 {
```

- type `Color`

Color for clearing operations


```zig
pub const Color = struct {
```

- type `GPUHandle`

GPU resource handle with generation for safety


```zig
pub const GPUHandle = struct {
```

- fn `invalid`

```zig
pub fn invalid() GPUHandle {
```

- fn `isValid`

```zig
pub fn isValid(self: GPUHandle) bool {
```

- type `Buffer`

GPU buffer resource with platform abstraction


```zig
pub const Buffer = struct {
```

- fn `map`

```zig
pub fn map(self: *Buffer) ![]u8 {
```

- fn `unmap`

```zig
pub fn unmap(self: *Buffer) void {
```

- fn `isValid`

```zig
pub fn isValid(self: *const Buffer) bool {
```

- type `Shader`

Shader resource with cross-platform compilation


```zig
pub const Shader = struct {
```

- fn `compile`

```zig
pub fn compile(allocator: std.mem.Allocator, stage: ShaderStage, source: []const u8) !Shader {
```

- type `ComputePipeline`

Compute pipeline for AI operations


```zig
pub const ComputePipeline = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, compute_shader: Shader) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addBindGroup`

```zig
pub fn addBindGroup(self: *Self, bind_group: BindGroup) !void {
```

- type `BindGroup`

Bind group for resource binding


```zig
pub const BindGroup = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addBuffer`

```zig
pub fn addBuffer(self: *Self, buffer: Buffer) !void {
```

- type `GPURenderer`

Main GPU renderer with cross-platform support


```zig
pub const GPURenderer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: GPUConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `createBuffer`

Create a GPU buffer with specified usage


```zig
pub fn createBuffer(self: *Self, size: usize, usage: BufferUsage) !Buffer {
```

- fn `beginFrame`

Begin frame rendering


```zig
pub fn beginFrame(self: *Self) !void {
```

- fn `endFrame`

End frame rendering


```zig
pub fn endFrame(self: *Self) !void {
```

- fn `clear`

Clear the render target with specified color


```zig
pub fn clear(self: *Self, color: Color) !void {
```

- fn `renderNeuralNetwork`

Render neural network visualization


```zig
pub fn renderNeuralNetwork(self: *Self, neural_engine: anytype) !void {
```

- fn `computeMatrixMultiply`

Run matrix multiplication compute shader


```zig
pub fn computeMatrixMultiply(self: *Self, a: []const f32, b: []const f32, result: []f32, m: u32, n: u32, k: u32) !void {
```

- fn `computeNeuralInference`

Run neural network inference on GPU


```zig
pub fn computeNeuralInference(self: *Self, input: []const f32, weights: []const f32, output: []f32) !void {
```

- fn `getFPS`

Get current FPS


```zig
pub fn getFPS(self: *Self) f32 {
```

- fn `getFrameCount`

Get frame count


```zig
pub fn getFrameCount(self: *Self) u64 {
```

## src/features/gpu/hardware_detection.zig

- type `BackendType`

GPU backend options supported by the framework.


```zig
pub const BackendType = enum {
```

- fn `priority`

Get the priority value for backend selection (higher = better)


```zig
pub fn priority(self: BackendType) u32 {
```

- fn `displayName`

Get a display name for the backend


```zig
pub fn displayName(self: BackendType) []const u8 {
```

- fn `supportsCompute`

Check if backend supports compute operations


```zig
pub fn supportsCompute(self: BackendType) bool {
```

- fn `supportsGraphics`

Check if backend supports graphics operations


```zig
pub fn supportsGraphics(self: BackendType) bool {
```

- fn `isCrossPlatform`

Check if backend is cross-platform


```zig
pub fn isCrossPlatform(self: BackendType) bool {
```

- fn `shaderLanguage`

Get the shader language used by this backend


```zig
pub fn shaderLanguage(self: BackendType) []const u8 {
```

- fn `supportedPlatforms`

Get supported platforms for this backend


```zig
pub fn supportedPlatforms(self: BackendType) []const []const u8 {
```

- fn `isAvailable`

Check if backend is available on current platform


```zig
pub fn isAvailable(self: BackendType) bool {
```

- type `PerformanceTier`

Coarse performance tier classification used by demos and heuristics.


```zig
pub const PerformanceTier = enum {
```

- type `GPUType`

Lightweight GPU type bucket for convenience helpers.


```zig
pub const GPUType = enum {
```

- type `SystemCapabilities`

Basic system level capabilities reported alongside detection results.


```zig
pub const SystemCapabilities = struct {
```

- type `RealGPUInfo`

Minimal real GPU information record retained for compatibility with
higher-level code. Fields map to historic structure layouts.


```zig
pub const RealGPUInfo = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *RealGPUInfo) void {
```

- type `GPUDetectionResult`

Aggregate detection result returned by the detector.


```zig
pub const GPUDetectionResult = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *GPUDetectionResult) void {
```

- type `GPUDetector`

Main detector type. The current implementation synthesizes a conservative
fallback profile so that higher layers can rely on deterministic data even
when platform specific detection hooks are unavailable.


```zig
pub const GPUDetector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) GPUDetector {
```

- fn `deinit`

```zig
pub fn deinit(self: *GPUDetector) void {
```

- fn `detectGPUs`

```zig
pub fn detectGPUs(self: *GPUDetector) !GPUDetectionResult {
```

- fn `isHardwareDetectionAvailable`

Runtime flag used by higher layers to decide whether to attempt real
hardware probing. The stub implementation always returns , which
encourages callers to use conservative defaults without failing builds on
unsupported targets.


```zig
pub fn isHardwareDetectionAvailable() bool {
```

- fn `determineRecommendedBackend`

Return the most desirable backend present in the supplied GPU list.


```zig
pub fn determineRecommendedBackend(gpus: []RealGPUInfo) BackendType {
```

- fn `logGPUDetectionResults`

Convenience helper used by demos to print a concise summary.


```zig
pub fn logGPUDetectionResults(result: *const GPUDetectionResult) void {
```

## src/features/gpu/libraries/cuda_integration.zig

- type `CUDACapabilities`

CUDA device capabilities


```zig
pub const CUDACapabilities = struct {
```

- type `CUDAFeatures`

```zig
pub const CUDAFeatures = packed struct {
```

- type `CUDARenderer`

CUDA renderer implementation


```zig
pub const CUDARenderer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initialize`

Initialize CUDA context and device


```zig
pub fn initialize(self: *Self) !void {
```

- fn `getCapabilities`

Get device capabilities


```zig
pub fn getCapabilities(self: *Self) !CUDACapabilities {
```

- fn `launchKernel`

Launch CUDA kernel


```zig
pub fn launchKernel(self: *Self, kernel: *anyopaque, grid_dim: [3]u32, block_dim: [3]u32, shared_mem_bytes: u32, args: []const *anyopaque) !void {
```

- fn `allocateDeviceMemory`

Allocate device memory


```zig
pub fn allocateDeviceMemory(self: *Self, size: u64) !*anyopaque {
```

- fn `freeDeviceMemory`

Free device memory


```zig
pub fn freeDeviceMemory(self: *Self, memory: *anyopaque) void {
```

- fn `copyMemory`

Copy memory between host and device


```zig
pub fn copyMemory(self: *Self, dst: *anyopaque, src: *anyopaque, size: u64, kind: MemoryCopyKind) !void {
```

- type `MemoryCopyKind`

```zig
pub const MemoryCopyKind = enum {
```

- fn `synchronize`

Synchronize CUDA stream


```zig
pub fn synchronize(self: *Self) !void {
```

- type `CUDAUtils`

CUDA utility functions


```zig
pub const CUDAUtils = struct {
```

- fn `isCUDAAvailable`

Check if CUDA is available


```zig
pub fn isCUDAAvailable() bool {
```

- fn `getDeviceCount`

Get number of CUDA devices


```zig
pub fn getDeviceCount() !u32 {
```

- fn `compileKernel`

Compile CUDA kernel from source


```zig
pub fn compileKernel(source: []const u8, kernel_name: []const u8) !*anyopaque {
```

- fn `createStream`

Create CUDA stream


```zig
pub fn createStream() !*anyopaque {
```

- fn `destroyStream`

Destroy CUDA stream


```zig
pub fn destroyStream(stream: *anyopaque) void {
```

## src/features/gpu/libraries/mach_gpu_integration.zig

- type `MachDeviceType`

Mach/GPU device types


```zig
pub const MachDeviceType = enum {
```

- type `MachCapabilities`

Mach/GPU capabilities


```zig
pub const MachCapabilities = struct {
```

- type `MachFeatures`

```zig
pub const MachFeatures = packed struct {
```

- type `MachLimits`

```zig
pub const MachLimits = struct {
```

- type `MachRenderer`

Mach/GPU renderer implementation


```zig
pub const MachRenderer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initialize`

Initialize Mach/GPU device


```zig
pub fn initialize(self: *Self, device_type: MachDeviceType) !void {
```

- fn `getCapabilities`

Get device capabilities


```zig
pub fn getCapabilities(self: *Self) !MachCapabilities {
```

- fn `createComputePipeline`

Create a compute pipeline


```zig
pub fn createComputePipeline(self: *Self, shader_module: *anyopaque) !*anyopaque {
```

- fn `createRenderPipeline`

Create a render pipeline


```zig
pub fn createRenderPipeline(self: *Self, pipeline_info: *anyopaque) !*anyopaque {
```

- fn `dispatchCompute`

Execute compute shader


```zig
pub fn dispatchCompute(self: *Self, command_encoder: *anyopaque, group_count_x: u32, group_count_y: u32, group_count_z: u32) !void {
```

- fn `createBuffer`

Create buffer


```zig
pub fn createBuffer(self: *Self, size: u64, usage: BufferUsage) !*anyopaque {
```

- fn `createTexture`

Create texture


```zig
pub fn createTexture(self: *Self, texture_info: *anyopaque) !*anyopaque {
```

- type `BufferUsage`

```zig
pub const BufferUsage = packed struct {
```

- type `MachUtils`

Mach/GPU utility functions


```zig
pub const MachUtils = struct {
```

- fn `isMachGPUAvailable`

Check if Mach/GPU is available


```zig
pub fn isMachGPUAvailable() bool {
```

- fn `getOptimalDeviceType`

Get optimal device type for current platform


```zig
pub fn getOptimalDeviceType() MachDeviceType {
```

- fn `createShaderModule`

Create shader module from WGSL source


```zig
pub fn createShaderModule(device: *anyopaque, wgsl_source: []const u8) !*anyopaque {
```

- fn `compileGLSLToWGSL`

Compile GLSL to WGSL


```zig
pub fn compileGLSLToWGSL(glsl_source: []const u8, shader_type: ShaderType) ![]const u8 {
```

- type `ShaderType`

```zig
pub const ShaderType = enum {
```

## src/features/gpu/libraries/mod.zig

- const `vulkan_bindings`

```zig
pub const vulkan_bindings = @import("vulkan_bindings.zig");
```

- const `mach_gpu_integration`

```zig
pub const mach_gpu_integration = @import("mach_gpu_integration.zig");
```

- const `cuda_integration`

```zig
pub const cuda_integration = @import("cuda_integration.zig");
```

- const `simd_optimizations`

```zig
pub const simd_optimizations = @import("simd_optimizations_minimal.zig");
```

- const `VulkanRenderer`

```zig
pub const VulkanRenderer = vulkan_bindings.VulkanRenderer;
```

- const `VulkanCapabilities`

```zig
pub const VulkanCapabilities = vulkan_bindings.VulkanCapabilities;
```

- const `VulkanUtils`

```zig
pub const VulkanUtils = vulkan_bindings.VulkanUtils;
```

- const `AdvancedVulkanFeatures`

```zig
pub const AdvancedVulkanFeatures = vulkan_bindings.AdvancedVulkanFeatures;
```

- const `MachRenderer`

```zig
pub const MachRenderer = mach_gpu_integration.MachRenderer;
```

- const `MachCapabilities`

```zig
pub const MachCapabilities = mach_gpu_integration.MachCapabilities;
```

- const `MachUtils`

```zig
pub const MachUtils = mach_gpu_integration.MachUtils;
```

- const `CUDARenderer`

```zig
pub const CUDARenderer = cuda_integration.CUDARenderer;
```

- const `CUDACapabilities`

```zig
pub const CUDACapabilities = cuda_integration.CUDACapabilities;
```

- const `CUDAUtils`

```zig
pub const CUDAUtils = cuda_integration.CUDAUtils;
```

- const `VectorTypes`

```zig
pub const VectorTypes = simd_optimizations.VectorTypes;
```

- const `SIMDMath`

```zig
pub const SIMDMath = simd_optimizations.SIMDMath;
```

- const `SIMDGraphics`

```zig
pub const SIMDGraphics = simd_optimizations.SIMDGraphics;
```

- const `SIMDCompute`

```zig
pub const SIMDCompute = simd_optimizations.SIMDCompute;
```

- const `SIMDBenchmarks`

```zig
pub const SIMDBenchmarks = simd_optimizations.SIMDBenchmarks;
```

- type `GPULibraryManager`

Unified GPU library manager


```zig
pub const GPULibraryManager = struct {
```

- type `AvailableLibraries`

```zig
pub const AvailableLibraries = packed struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initVulkan`

Initialize Vulkan renderer


```zig
pub fn initVulkan(self: *Self) !void {
```

- fn `initMachGPU`

Initialize Mach/GPU renderer


```zig
pub fn initMachGPU(self: *Self, device_type: mach_gpu_integration.MachDeviceType) !void {
```

- fn `initCUDA`

Initialize CUDA renderer


```zig
pub fn initCUDA(self: *Self) !void {
```

- fn `getAvailableLibraries`

Get available libraries information


```zig
pub fn getAvailableLibraries(self: *Self) AvailableLibraries {
```

- fn `runSIMDBenchmarks`

Run SIMD benchmarks


```zig
pub fn runSIMDBenchmarks(self: *Self) !void {
```

- fn `getLibraryStatus`

Get comprehensive library status


```zig
pub fn getLibraryStatus(self: *Self) LibraryStatus {
```

- type `LibraryStatus`

```zig
pub const LibraryStatus = struct {
```

- type `LibraryState`

```zig
pub const LibraryState = enum {
```

- const `GPULibraryError`

Error types for GPU library operations


```zig
pub const GPULibraryError = error{
```

## src/features/gpu/libraries/simd_optimizations.zig

- type `VectorTypes`

Vector types for common graphics operations


```zig
pub const VectorTypes = struct {
```

- const `Vec4f`

4-component float vector (RGBA, XYZW)


```zig
pub const Vec4f = @Vector(4, f32);
```

- const `Vec3f`

3-component float vector (RGB, XYZ)


```zig
pub const Vec3f = @Vector(3, f32);
```

- const `Vec2f`

2-component float vector (UV, XY)


```zig
pub const Vec2f = @Vector(2, f32);
```

- const `Vec4i`

4-component integer vector


```zig
pub const Vec4i = @Vector(4, i32);
```

- const `Vec3i`

3-component integer vector


```zig
pub const Vec3i = @Vector(3, i32);
```

- const `Vec2i`

2-component integer vector


```zig
pub const Vec2i = @Vector(2, i32);
```

- const `Vec4u`

4-component unsigned integer vector


```zig
pub const Vec4u = @Vector(4, u32);
```

- const `Vec3u`

3-component unsigned integer vector


```zig
pub const Vec3u = @Vector(3, u32);
```

- const `Vec2u`

2-component unsigned integer vector


```zig
pub const Vec2u = @Vector(2, u32);
```

- const `Mat4x4f`

4x4 matrix as 4 vectors


```zig
pub const Mat4x4f = [4]Vec4f;
```

- const `Mat3x3f`

3x3 matrix as 3 vectors


```zig
pub const Mat3x3f = [3]Vec3f;
```

- const `Mat2x2f`

2x2 matrix as 2 vectors


```zig
pub const Mat2x2f = [2]Vec2f;
```

- type `SIMDMath`

SIMD math operations


```zig
pub const SIMDMath = struct {
```

- fn `add`

Vector addition


```zig
pub fn add(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `sub`

Vector subtraction


```zig
pub fn sub(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `mul`

Vector multiplication


```zig
pub fn mul(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `div`

Vector division


```zig
pub fn div(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `dot`

Vector dot product


```zig
pub fn dot(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f) f32 {
```

- fn `cross`

Vector cross product (3D)


```zig
pub fn cross(a: VectorTypes.Vec3f, b: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `length`

Vector length


```zig
pub fn length(v: VectorTypes.Vec4f) f32 {
```

- fn `normalize`

Vector normalization


```zig
pub fn normalize(v: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `lerp`

Vector linear interpolation


```zig
pub fn lerp(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f, t: f32) VectorTypes.Vec4f {
```

- fn `mat4MulVec4`

Matrix-vector multiplication (4x4)


```zig
pub fn mat4MulVec4(m: VectorTypes.Mat4x4f, v: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `mat4MulMat4`

Matrix multiplication (4x4)


```zig
pub fn mat4MulMat4(a: VectorTypes.Mat4x4f, b: VectorTypes.Mat4x4f) VectorTypes.Mat4x4f {
```

- fn `fma`

Advanced SIMD operations for AI/ML workloads
SIMD fused multiply-add (FMA) operation


```zig
pub fn fma(a: anytype, b: anytype, c: anytype) @TypeOf(a) {
```

- fn `horizontalSum`

SIMD horizontal sum (sum all elements in vector)


```zig
pub fn horizontalSum(v: anytype) std.meta.Child(@TypeOf(v)) {
```

- fn `horizontalMax`

SIMD horizontal maximum (find max element in vector)


```zig
pub fn horizontalMax(v: anytype) std.meta.Child(@TypeOf(v)) {
```

- fn `horizontalMin`

SIMD horizontal minimum (find min element in vector)


```zig
pub fn horizontalMin(v: anytype) std.meta.Child(@TypeOf(v)) {
```

- fn `max`

SIMD element-wise maximum


```zig
pub fn max(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `min`

SIMD element-wise minimum


```zig
pub fn min(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `abs`

SIMD element-wise absolute value


```zig
pub fn abs(v: anytype) @TypeOf(v) {
```

- fn `sqrt`

SIMD element-wise square root


```zig
pub fn sqrt(v: anytype) @TypeOf(v) {
```

- fn `rsqrt`

SIMD element-wise reciprocal square root


```zig
pub fn rsqrt(v: anytype) @TypeOf(v) {
```

- fn `exp`

SIMD element-wise exponential


```zig
pub fn exp(v: anytype) @TypeOf(v) {
```

- fn `log`

SIMD element-wise logarithm (natural log)


```zig
pub fn log(v: anytype) @TypeOf(v) {
```

- fn `sigmoid`

SIMD element-wise sigmoid activation


```zig
pub fn sigmoid(v: anytype) @TypeOf(v) {
```

- fn `tanh`

SIMD element-wise tanh activation


```zig
pub fn tanh(v: anytype) @TypeOf(v) {
```

- fn `relu`

SIMD element-wise ReLU activation


```zig
pub fn relu(v: anytype) @TypeOf(v) {
```

- fn `leakyRelu`

SIMD element-wise Leaky ReLU activation


```zig
pub fn leakyRelu(v: anytype, alpha: std.meta.Child(@TypeOf(v))) @TypeOf(v) {
```

- fn `gelu`

SIMD element-wise GELU activation (approximation)


```zig
pub fn gelu(v: anytype) @TypeOf(v) {
```

- fn `matMulSIMD`

SIMD matrix multiplication (optimized for small matrices)


```zig
pub fn matMulSIMD(a: []const f32, b: []const f32, c: []f32, m: usize, n: usize, p: usize) void {
```

- fn `softmaxSIMD`

SIMD vectorized softmax (numerically stable)


```zig
pub fn softmaxSIMD(input: []f32, output: []f32) void {
```

- fn `batchNormSIMD`

SIMD batch normalization (vectorized)


```zig
pub fn batchNormSIMD(input: []f32, output: []f32, gamma: f32, beta: f32, epsilon: f32) void {
```

- type `SIMDGraphics`

SIMD graphics operations


```zig
pub const SIMDGraphics = struct {
```

- fn `blendColors`

Color blending (alpha blending)


```zig
pub fn blendColors(src: VectorTypes.Vec4f, dst: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `rgbToHsv`

Color space conversion (RGB to HSV)


```zig
pub fn rgbToHsv(rgb: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `hsvToRgb`

Color space conversion (HSV to RGB)


```zig
pub fn hsvToRgb(hsv: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `gammaCorrect`

Gamma correction


```zig
pub fn gammaCorrect(color: VectorTypes.Vec3f, gamma: f32) VectorTypes.Vec3f {
```

- fn `toneMapReinhard`

Tone mapping (Reinhard)


```zig
pub fn toneMapReinhard(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `toneMapACES`

Tone mapping (ACES)


```zig
pub fn toneMapACES(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- type `SIMDCompute`

SIMD compute operations


```zig
pub const SIMDCompute = struct {
```

- fn `addArrays`

Parallel array addition


```zig
pub fn addArrays(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `mulArrays`

Parallel array multiplication


```zig
pub fn mulArrays(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `scaleArray`

Parallel array scaling


```zig
pub fn scaleArray(a: []const f32, scale: f32, result: []f32) void {
```

- fn `sumArray`

Parallel array sum reduction


```zig
pub fn sumArray(a: []const f32) f32 {
```

- fn `dotProduct`

Parallel array dot product


```zig
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
```

- type `SIMDBenchmarks`

SIMD performance benchmarks


```zig
pub const SIMDBenchmarks = struct {
```

- fn `benchmarkSIMDvsScalar`

Benchmark SIMD vs scalar operations


```zig
pub fn benchmarkSIMDvsScalar(allocator: std.mem.Allocator, array_size: usize) !void {
```

- fn `benchmarkMatrixOperations`

Benchmark matrix operations


```zig
pub fn benchmarkMatrixOperations(allocator: std.mem.Allocator) !void {
```

## src/features/gpu/libraries/simd_optimizations_minimal.zig

- type `VectorTypes`

Vector types for common graphics operations


```zig
pub const VectorTypes = struct {
```

- const `Vec4f`

4-component float vector (RGBA, XYZW)


```zig
pub const Vec4f = @Vector(4, f32);
```

- const `Vec3f`

3-component float vector (RGB, XYZ)


```zig
pub const Vec3f = @Vector(3, f32);
```

- const `Vec2f`

2-component float vector (UV, XY)


```zig
pub const Vec2f = @Vector(2, f32);
```

- const `Mat4x4f`

4x4 matrix as 4 vectors


```zig
pub const Mat4x4f = [4]Vec4f;
```

- type `SIMDMath`

SIMD math operations


```zig
pub const SIMDMath = struct {
```

- fn `add`

Vector addition


```zig
pub fn add(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `sub`

Vector subtraction


```zig
pub fn sub(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `mul`

Vector multiplication


```zig
pub fn mul(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `div`

Vector division


```zig
pub fn div(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `dot`

Vector dot product


```zig
pub fn dot(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f) f32 {
```

- fn `length`

Vector length


```zig
pub fn length(v: VectorTypes.Vec4f) f32 {
```

- fn `normalize`

Vector normalization


```zig
pub fn normalize(v: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `lerp`

Vector linear interpolation


```zig
pub fn lerp(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f, t: f32) VectorTypes.Vec4f {
```

- fn `mat4MulVec4`

Matrix-vector multiplication (4x4)


```zig
pub fn mat4MulVec4(m: VectorTypes.Mat4x4f, v: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- type `SIMDGraphics`

SIMD graphics operations


```zig
pub const SIMDGraphics = struct {
```

- fn `blendColors`

Color blending (alpha blending)


```zig
pub fn blendColors(src: VectorTypes.Vec4f, dst: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `rgbToHsv`

Color space conversion (RGB to HSV)


```zig
pub fn rgbToHsv(rgb: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `hsvToRgb`

Color space conversion (HSV to RGB)


```zig
pub fn hsvToRgb(hsv: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `toneMapReinhard`

Tone mapping (Reinhard)


```zig
pub fn toneMapReinhard(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `toneMapACES`

Tone mapping (ACES)


```zig
pub fn toneMapACES(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- type `SIMDCompute`

SIMD compute operations


```zig
pub const SIMDCompute = struct {
```

- fn `addArrays`

Simple array addition (scalar version for now)


```zig
pub fn addArrays(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `mulArrays`

Simple array multiplication (scalar version for now)


```zig
pub fn mulArrays(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `scaleArray`

Simple array scaling (scalar version for now)


```zig
pub fn scaleArray(a: []const f32, scale: f32, result: []f32) void {
```

- fn `sumArray`

Simple array sum (scalar version for now)


```zig
pub fn sumArray(a: []const f32) f32 {
```

- fn `dotProduct`

Simple array dot product (scalar version for now)


```zig
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
```

- type `SIMDBenchmarks`

SIMD performance benchmarks


```zig
pub const SIMDBenchmarks = struct {
```

- fn `benchmarkSIMDvsScalar`

Benchmark operations


```zig
pub fn benchmarkSIMDvsScalar(allocator: std.mem.Allocator, array_size: usize) !void {
```

- fn `benchmarkMatrixOperations`

Benchmark matrix operations


```zig
pub fn benchmarkMatrixOperations(allocator: std.mem.Allocator) !void {
```

## src/features/gpu/libraries/simd_optimizations_simple.zig

- type `VectorTypes`

Vector types for common graphics operations


```zig
pub const VectorTypes = struct {
```

- const `Vec4f`

4-component float vector (RGBA, XYZW)


```zig
pub const Vec4f = @Vector(4, f32);
```

- const `Vec3f`

3-component float vector (RGB, XYZ)


```zig
pub const Vec3f = @Vector(3, f32);
```

- const `Vec2f`

2-component float vector (UV, XY)


```zig
pub const Vec2f = @Vector(2, f32);
```

- const `Vec4i`

4-component integer vector


```zig
pub const Vec4i = @Vector(4, i32);
```

- const `Vec3i`

3-component integer vector


```zig
pub const Vec3i = @Vector(3, i32);
```

- const `Vec2i`

2-component integer vector


```zig
pub const Vec2i = @Vector(2, i32);
```

- const `Vec4u`

4-component unsigned integer vector


```zig
pub const Vec4u = @Vector(4, u32);
```

- const `Vec3u`

3-component unsigned integer vector


```zig
pub const Vec3u = @Vector(3, u32);
```

- const `Vec2u`

2-component unsigned integer vector


```zig
pub const Vec2u = @Vector(2, u32);
```

- const `Mat4x4f`

4x4 matrix as 4 vectors


```zig
pub const Mat4x4f = [4]Vec4f;
```

- const `Mat3x3f`

3x3 matrix as 3 vectors


```zig
pub const Mat3x3f = [3]Vec3f;
```

- const `Mat2x2f`

2x2 matrix as 2 vectors


```zig
pub const Mat2x2f = [2]Vec2f;
```

- type `SIMDMath`

SIMD math operations


```zig
pub const SIMDMath = struct {
```

- fn `add`

Vector addition


```zig
pub fn add(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `sub`

Vector subtraction


```zig
pub fn sub(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `mul`

Vector multiplication


```zig
pub fn mul(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `div`

Vector division


```zig
pub fn div(a: anytype, b: anytype) @TypeOf(a) {
```

- fn `dot`

Vector dot product


```zig
pub fn dot(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f) f32 {
```

- fn `cross`

Vector cross product (3D)


```zig
pub fn cross(a: VectorTypes.Vec3f, b: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `length`

Vector length


```zig
pub fn length(v: VectorTypes.Vec4f) f32 {
```

- fn `normalize`

Vector normalization


```zig
pub fn normalize(v: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `lerp`

Vector linear interpolation


```zig
pub fn lerp(a: VectorTypes.Vec4f, b: VectorTypes.Vec4f, t: f32) VectorTypes.Vec4f {
```

- fn `mat4MulVec4`

Matrix-vector multiplication (4x4)


```zig
pub fn mat4MulVec4(m: VectorTypes.Mat4x4f, v: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `mat4MulMat4`

Matrix multiplication (4x4)


```zig
pub fn mat4MulMat4(a: VectorTypes.Mat4x4f, b: VectorTypes.Mat4x4f) VectorTypes.Mat4x4f {
```

- type `SIMDGraphics`

SIMD graphics operations


```zig
pub const SIMDGraphics = struct {
```

- fn `blendColors`

Color blending (alpha blending)


```zig
pub fn blendColors(src: VectorTypes.Vec4f, dst: VectorTypes.Vec4f) VectorTypes.Vec4f {
```

- fn `rgbToHsv`

Color space conversion (RGB to HSV)


```zig
pub fn rgbToHsv(rgb: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `hsvToRgb`

Color space conversion (HSV to RGB)


```zig
pub fn hsvToRgb(hsv: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `gammaCorrect`

Gamma correction


```zig
pub fn gammaCorrect(color: VectorTypes.Vec3f, gamma: f32) VectorTypes.Vec3f {
```

- fn `toneMapReinhard`

Tone mapping (Reinhard)


```zig
pub fn toneMapReinhard(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- fn `toneMapACES`

Tone mapping (ACES)


```zig
pub fn toneMapACES(color: VectorTypes.Vec3f) VectorTypes.Vec3f {
```

- type `SIMDCompute`

SIMD compute operations


```zig
pub const SIMDCompute = struct {
```

- fn `addArrays`

Parallel array addition


```zig
pub fn addArrays(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `mulArrays`

Parallel array multiplication


```zig
pub fn mulArrays(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `scaleArray`

Parallel array scaling


```zig
pub fn scaleArray(a: []const f32, scale: f32, result: []f32) void {
```

- fn `sumArray`

Parallel array sum reduction


```zig
pub fn sumArray(a: []const f32) f32 {
```

- fn `dotProduct`

Parallel array dot product


```zig
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
```

- type `SIMDBenchmarks`

SIMD performance benchmarks


```zig
pub const SIMDBenchmarks = struct {
```

- fn `benchmarkSIMDvsScalar`

Benchmark SIMD vs scalar operations


```zig
pub fn benchmarkSIMDvsScalar(allocator: std.mem.Allocator, array_size: usize) !void {
```

- fn `benchmarkMatrixOperations`

Benchmark matrix operations


```zig
pub fn benchmarkMatrixOperations(allocator: std.mem.Allocator) !void {
```

## src/features/gpu/libraries/vulkan_bindings.zig

- type `VulkanVersion`

Vulkan API version and capabilities


```zig
pub const VulkanVersion = enum(u32) {
```

- type `VulkanCapabilities`

Vulkan device capabilities and features


```zig
pub const VulkanCapabilities = struct {
```

- type `DeviceType`

```zig
pub const DeviceType = enum {
```

- type `MemoryHeap`

```zig
pub const MemoryHeap = struct {
```

- type `MemoryHeapFlags`

```zig
pub const MemoryHeapFlags = packed struct {
```

- type `MemoryType`

```zig
pub const MemoryType = struct {
```

- type `MemoryPropertyFlags`

```zig
pub const MemoryPropertyFlags = packed struct {
```

- type `QueueFamily`

```zig
pub const QueueFamily = struct {
```

- type `QueueFlags`

```zig
pub const QueueFlags = packed struct {
```

- type `Extension`

```zig
pub const Extension = struct {
```

- type `DeviceFeatures`

```zig
pub const DeviceFeatures = packed struct {
```

- type `DeviceLimits`

```zig
pub const DeviceLimits = struct {
```

- type `VulkanRenderer`

Vulkan renderer implementation


```zig
pub const VulkanRenderer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initialize`

Initialize Vulkan instance and device


```zig
pub fn initialize(self: *Self) !void {
```

- fn `getCapabilities`

Get device capabilities


```zig
pub fn getCapabilities(self: *Self) !VulkanCapabilities {
```

- fn `createComputePipeline`

Create a compute pipeline


```zig
pub fn createComputePipeline(self: *Self, shader_module: *anyopaque) !*anyopaque {
```

- fn `createGraphicsPipeline`

Create a graphics pipeline


```zig
pub fn createGraphicsPipeline(self: *Self, pipeline_info: *anyopaque) !*anyopaque {
```

- fn `dispatchCompute`

Execute compute shader


```zig
pub fn dispatchCompute(self: *Self, command_buffer: *anyopaque, group_count_x: u32, group_count_y: u32, group_count_z: u32) !void {
```

- fn `allocateMemory`

Memory management


```zig
pub fn allocateMemory(self: *Self, size: u64, memory_type: u32) !*anyopaque {
```

- fn `freeMemory`

```zig
pub fn freeMemory(self: *Self, memory: *anyopaque) void {
```

- type `VulkanUtils`

Vulkan utility functions


```zig
pub const VulkanUtils = struct {
```

- fn `isVulkanAvailable`

Check if Vulkan is available on the system


```zig
pub fn isVulkanAvailable() bool {
```

- fn `getAvailableExtensions`

Get available Vulkan extensions


```zig
pub fn getAvailableExtensions(allocator: std.mem.Allocator) ![]const []const u8 {
```

- fn `findMemoryType`

Get optimal memory type for given requirements


```zig
pub fn findMemoryType(physical_device: *anyopaque, type_filter: u32, properties: u32) !u32 {
```

- fn `createShaderModule`

Create shader module from SPIR-V bytecode


```zig
pub fn createShaderModule(device: *anyopaque, code: []const u8) !*anyopaque {
```

- fn `compileGLSLToSPIRV`

Compile GLSL to SPIR-V


```zig
pub fn compileGLSLToSPIRV(glsl_source: []const u8, shader_type: ShaderType) ![]const u8 {
```

- type `ShaderType`

```zig
pub const ShaderType = enum {
```

- type `AdvancedVulkanFeatures`

Advanced Vulkan features


```zig
pub const AdvancedVulkanFeatures = struct {
```

- type `RayTracing`

Ray tracing support


```zig
pub const RayTracing = struct {
```

- fn `isSupported`

```zig
pub fn isSupported(device: *VulkanRenderer) bool {
```

- fn `createRayTracingPipeline`

```zig
pub fn createRayTracingPipeline(device: *VulkanRenderer, pipeline_info: *anyopaque) !*anyopaque {
```

- type `MeshShaders`

Mesh shader support


```zig
pub const MeshShaders = struct {
```

- fn `isSupported`

```zig
pub fn isSupported(device: *VulkanRenderer) bool {
```

- fn `createMeshPipeline`

```zig
pub fn createMeshPipeline(device: *VulkanRenderer, pipeline_info: *anyopaque) !*anyopaque {
```

- type `VariableRateShading`

Variable rate shading support


```zig
pub const VariableRateShading = struct {
```

- fn `isSupported`

```zig
pub fn isSupported(device: *VulkanRenderer) bool {
```

- fn `setShadingRate`

```zig
pub fn setShadingRate(command_buffer: *anyopaque, shading_rate: ShadingRate) void {
```

- type `ShadingRate`

```zig
pub const ShadingRate = enum {
```

- type `MultiView`

Multi-view rendering


```zig
pub const MultiView = struct {
```

- fn `isSupported`

```zig
pub fn isSupported(device: *VulkanRenderer) bool {
```

- fn `createMultiViewRenderPass`

```zig
pub fn createMultiViewRenderPass(device: *VulkanRenderer, view_count: u32) !*anyopaque {
```

## src/features/gpu/memory/memory_pool.zig

- type `MemoryPoolConfig`

GPU Memory Pool Configuration


```zig
pub const MemoryPoolConfig = struct {
```

- type `MemoryStats`

Memory pool statistics


```zig
pub const MemoryStats = struct {
```

- type `BufferMetadata`

Buffer metadata for pool management


```zig
pub const BufferMetadata = struct {
```

- type `MemoryPool`

GPU Memory Pool Manager


```zig
pub const MemoryPool = struct {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate a buffer from the pool or create a new one


```zig
pub fn allocBuffer(
```

- fn `freeBuffer`

Return a buffer to the pool for reuse


```zig
pub fn freeBuffer(self: *MemoryPool, handle: u32) !void {
```

- fn `cleanup`

Periodic cleanup of old buffers


```zig
pub fn cleanup(self: *MemoryPool) !void {
```

- fn `getStats`

Get comprehensive memory statistics


```zig
pub fn getStats(self: *MemoryPool) MemoryStats {
```

- fn `defragment`

Defragment memory pool (advanced feature)


```zig
pub fn defragment(self: *MemoryPool) !void {
```

- fn `prefetchBuffers`

Prefetch buffers for anticipated usage patterns


```zig
pub fn prefetchBuffers(self: *MemoryPool, sizes: []const usize, usage: gpu_renderer.BufferUsage) !void {
```

- fn `resizeBuffer`

Resize a buffer while maintaining pool efficiency


```zig
pub fn resizeBuffer(self: *MemoryPool, old_handle: u32, new_size: usize) !u32 {
```

- fn `getMemoryReport`

Get memory usage report


```zig
pub fn getMemoryReport(self: *MemoryPool, allocator: std.mem.Allocator) ![]const u8 {
```

## src/features/gpu/memory/mod.zig

- const `memory_pool`

```zig
pub const memory_pool = @import("memory_pool.zig");
```

## src/features/gpu/mobile/mobile_platform_support.zig

- type `MobilePlatformManager`

Mobile platform support manager


```zig
pub const MobilePlatformManager = struct {
```

- type `MobilePlatform`

```zig
pub const MobilePlatform = enum {
```

- type `MobileGPUBackend`

```zig
pub const MobileGPUBackend = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initializeMobileBackend`

Initialize mobile GPU backend


```zig
pub fn initializeMobileBackend(self: *Self) !void {
```

- fn `getMobileCapabilities`

Get mobile-specific capabilities


```zig
pub fn getMobileCapabilities(self: *Self) MobileCapabilities {
```

- type `MobileCapabilities`

Mobile platform capabilities


```zig
pub const MobileCapabilities = struct {
```

- type `PowerManagement`

Power management for mobile devices


```zig
pub const PowerManagement = struct {
```

- type `PowerMode`

```zig
pub const PowerMode = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `setPowerMode`

Set power mode


```zig
pub fn setPowerMode(self: *Self, mode: PowerMode) void {
```

- fn `getOptimalGPUSettings`

Get optimal GPU settings for current power mode


```zig
pub fn getOptimalGPUSettings(self: *Self) GPUSettings {
```

- type `GPUSettings`

```zig
pub const GPUSettings = struct {
```

- type `ThermalManagement`

Thermal management for mobile devices


```zig
pub const ThermalManagement = struct {
```

- type `ThermalState`

```zig
pub const ThermalState = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `updateThermalState`

Update thermal state


```zig
pub fn updateThermalState(self: *Self, temperature: f32) void {
```

- fn `getThermalThrottlingFactor`

Get thermal throttling factor


```zig
pub fn getThermalThrottlingFactor(self: *Self) f32 {
```

- fn `getThermalGPUSettings`

Get recommended GPU settings for thermal state


```zig
pub fn getThermalGPUSettings(self: *Self) PowerManagement.GPUSettings {
```

## src/features/gpu/mobile/mod.zig

- const `mobile_platform_support`

```zig
pub const mobile_platform_support = @import("mobile_platform_support.zig");
```

- const `MobilePlatformManager`

```zig
pub const MobilePlatformManager = mobile_platform_support.MobilePlatformManager;
```

- const `MobileCapabilities`

```zig
pub const MobileCapabilities = mobile_platform_support.MobileCapabilities;
```

- const `PowerManagement`

```zig
pub const PowerManagement = mobile_platform_support.PowerManagement;
```

- const `ThermalManagement`

```zig
pub const ThermalManagement = mobile_platform_support.ThermalManagement;
```

## src/features/gpu/mod.zig

- const `gpu_renderer`

```zig
pub const gpu_renderer = @import("gpu_renderer.zig");
```

- const `unified_memory`

```zig
pub const unified_memory = @import("unified_memory.zig");
```

- const `hardware_detection`

```zig
pub const hardware_detection = @import("hardware_detection.zig");
```

- const `backends`

```zig
pub const backends = @import("backends/mod.zig");
```

- const `compute`

```zig
pub const compute = @import("compute/mod.zig");
```

- const `core`

```zig
pub const core = @import("core/mod.zig");
```

- const `wasm_support`

```zig
pub const wasm_support = @import("wasm_support.zig");
```

- const `cross_compilation`

```zig
pub const cross_compilation = @import("cross_compilation.zig");
```

- const `memory`

```zig
pub const memory = @import("memory/mod.zig");
```

- const `testing`

```zig
pub const testing = @import("testing/mod.zig");
```

- const `benchmark`

```zig
pub const benchmark = @import("benchmark/mod.zig");
```

- const `mobile`

```zig
pub const mobile = @import("mobile/mod.zig");
```

- const `optimizations`

```zig
pub const optimizations = @import("optimizations/mod.zig");
```

- const `libraries`

```zig
pub const libraries = @import("libraries/mod.zig");
```

- const `mod`

```zig
pub const mod = @import("mod.zig");
```

- const `gpu_examples`

```zig
pub const gpu_examples = @import("gpu_examples.zig");
```

- const `GPUContext`

```zig
pub const GPUContext = core.gpu_renderer.GPUContext;
```

## src/features/gpu/optimizations/backend_detection.zig

- type `BackendDetector`

Enhanced backend detection system


```zig
pub const BackendDetector = struct {
```

- type `BackendType`

```zig
pub const BackendType = enum {
```

- type `BackendInfo`

```zig
pub const BackendInfo = struct {
```

- type `BackendVersion`

```zig
pub const BackendVersion = struct {
```

- type `BackendCapabilities`

```zig
pub const BackendCapabilities = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `detectAllBackends`

Detect all available backends


```zig
pub fn detectAllBackends(self: *Self) !void {
```

- fn `getRecommendedBackend`

Get the recommended backend


```zig
pub fn getRecommendedBackend(self: *Self) ?BackendType {
```

- fn `getDetectedBackends`

Get all detected backends


```zig
pub fn getDetectedBackends(self: *Self) []const BackendInfo {
```

- fn `getBackendInfo`

Get backend info by type


```zig
pub fn getBackendInfo(self: *Self, backend_type: BackendType) ?*BackendInfo {
```

- fn `isBackendAvailable`

Check if a specific backend is available


```zig
pub fn isBackendAvailable(self: *Self, backend_type: BackendType) bool {
```

## src/features/gpu/optimizations/mod.zig

- const `platform_optimizations`

```zig
pub const platform_optimizations = @import("platform_optimizations.zig");
```

- const `backend_detection`

```zig
pub const backend_detection = @import("backend_detection.zig");
```

- const `PlatformOptimizations`

```zig
pub const PlatformOptimizations = platform_optimizations.PlatformOptimizations;
```

- const `PlatformConfig`

```zig
pub const PlatformConfig = platform_optimizations.PlatformConfig;
```

- const `PlatformMetrics`

```zig
pub const PlatformMetrics = platform_optimizations.PlatformMetrics;
```

- const `PlatformUtils`

```zig
pub const PlatformUtils = platform_optimizations.PlatformUtils;
```

- const `BackendDetector`

```zig
pub const BackendDetector = backend_detection.BackendDetector;
```

## src/features/gpu/optimizations/platform_optimizations.zig

- type `PlatformOptimizations`

Platform-specific optimization strategies


```zig
pub const PlatformOptimizations = struct {
```

- type `TargetPlatform`

```zig
pub const TargetPlatform = enum {
```

- type `OptimizationLevel`

```zig
pub const OptimizationLevel = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, platform: TargetPlatform, level: OptimizationLevel) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `getOptimizationConfig`

Get platform-specific optimization configuration


```zig
pub fn getOptimizationConfig(self: *Self) PlatformConfig {
```

- type `PlatformConfig`

Platform-specific configuration


```zig
pub const PlatformConfig = struct {
```

- type `MemoryManagementConfig`

```zig
pub const MemoryManagementConfig = struct {
```

- type `HeapTypeOptimization`

```zig
pub const HeapTypeOptimization = enum {
```

- type `CommandOptimizationConfig`

```zig
pub const CommandOptimizationConfig = struct {
```

- type `PipelineOptimizationConfig`

```zig
pub const PipelineOptimizationConfig = struct {
```

- type `SynchronizationConfig`

```zig
pub const SynchronizationConfig = struct {
```

- type `ShaderOptimizationConfig`

```zig
pub const ShaderOptimizationConfig = struct {
```

- type `PlatformMetrics`

Platform-specific performance metrics


```zig
pub const PlatformMetrics = struct {
```

- fn `benchmark`

```zig
pub fn benchmark(self: *PlatformOptimizations, config: PlatformConfig) !void {
```

- type `PlatformUtils`

Platform optimization utilities


```zig
pub const PlatformUtils = struct {
```

- fn `detectPlatform`

Detect the current platform


```zig
pub fn detectPlatform() PlatformOptimizations.TargetPlatform {
```

- fn `getOptimalOptimizationLevel`

Get optimal optimization level for platform


```zig
pub fn getOptimalOptimizationLevel(platform: PlatformOptimizations.TargetPlatform) PlatformOptimizations.OptimizationLevel {
```

- fn `supportsFeature`

Check if platform supports specific feature


```zig
pub fn supportsFeature(platform: PlatformOptimizations.TargetPlatform, feature: PlatformFeature) bool {
```

- type `PlatformFeature`

```zig
pub const PlatformFeature = enum {
```

## src/features/gpu/testing/cross_platform_tests.zig

- type `CrossPlatformTestSuite`

Cross-platform test suite


```zig
pub const CrossPlatformTestSuite = struct {
```

- type `TargetPlatform`

```zig
pub const TargetPlatform = struct {
```

- type `TestResult`

```zig
pub const TestResult = struct {
```

- type `TestStatus`

```zig
pub const TestStatus = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `addTargetPlatform`

Add target platform for testing


```zig
pub fn addTargetPlatform(self: *Self, os: std.Target.Os.Tag, arch: std.Target.Cpu.Arch, abi: std.Target.Abi, name: []const u8) !void {
```

- fn `runAllTests`

Run all tests across all platforms


```zig
pub fn runAllTests(self: *Self) !void {
```

## src/features/gpu/testing/mod.zig

- const `cross_platform_tests`

```zig
pub const cross_platform_tests = @import("cross_platform_tests.zig");
```

- const `CrossPlatformTestSuite`

```zig
pub const CrossPlatformTestSuite = cross_platform_tests.CrossPlatformTestSuite;
```

## src/features/gpu/unified_memory.zig

- const `UnifiedMemoryError`

Unified memory specific errors


```zig
pub const UnifiedMemoryError = error{
```

- type `UnifiedMemoryType`

Unified Memory Architecture types


```zig
pub const UnifiedMemoryType = enum {
```

- type `UnifiedMemoryConfig`

Unified Memory Configuration


```zig
pub const UnifiedMemoryConfig = struct {
```

- type `UnifiedMemoryManager`

Unified Memory Manager with enhanced error handling and resource management


```zig
pub const UnifiedMemoryManager = struct {
```

- type `MemoryStatistics`

Memory usage statistics


```zig
pub const MemoryStatistics = struct {
```

- fn `init`

Initialize the unified memory manager with comprehensive setup


```zig
pub fn init(allocator: std.mem.Allocator) UnifiedMemoryError!Self {
```

- fn `deinit`

Safely deinitialize the unified memory manager with cleanup verification


```zig
pub fn deinit(self: *Self) void {
```

- fn `getStatistics`

Get current memory statistics


```zig
pub fn getStatistics(self: *Self) MemoryStatistics {
```

- fn `resetStatistics`

Reset memory statistics (useful for benchmarking)


```zig
pub fn resetStatistics(self: *Self) void {
```

- fn `allocateUnified`

Allocate unified memory that can be accessed by both CPU and GPU


```zig
pub fn allocateUnified(self: *Self, size: usize, alignment: u29) UnifiedMemoryError![]u8 {
```

- fn `freeUnified`

Free unified memory with statistics tracking


```zig
pub fn freeUnified(self: *Self, memory: []u8) void {
```

- fn `getPerformanceInfo`

Get unified memory performance characteristics


```zig
pub fn getPerformanceInfo(self: *Self) struct {
```

- type `UnifiedBuffer`

Unified Memory Buffer for zero-copy operations


```zig
pub const UnifiedBuffer = struct {
```

- fn `create`

Create a new unified buffer


```zig
pub fn create(manager: *UnifiedMemoryManager, size: usize) !Self {
```

- fn `destroy`

Destroy the unified buffer


```zig
pub fn destroy(self: *const Self) void {
```

- fn `getData`

Get raw data pointer


```zig
pub fn getData(self: *Self) []u8 {
```

- fn `getSize`

Get buffer size


```zig
pub fn getSize(self: *Self) usize {
```

- fn `isGpuAccessible`

Check if buffer is GPU accessible


```zig
pub fn isGpuAccessible(self: *const Self) bool {
```

- fn `transferToGpu`

Zero-copy data transfer (if supported)


```zig
pub fn transferToGpu(self: *Self) !void {
```

- fn `transferFromGpu`

Zero-copy data transfer from GPU (if supported)


```zig
pub fn transferFromGpu(self: *Self) !void {
```

## src/features/gpu/wasm_support.zig

- const `WASMError`

WebAssembly specific errors


```zig
pub const WASMError = error{
```

- type `WASMConfig`

WebAssembly compilation configuration


```zig
pub const WASMConfig = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *WASMConfig) void {
```

- type `WASMArchitecture`

WebAssembly architecture variants


```zig
pub const WASMArchitecture = enum {
```

- type `WASMOptimizationLevel`

WebAssembly optimization levels


```zig
pub const WASMOptimizationLevel = enum {
```

- type `WASMMemoryModel`

WebAssembly memory model


```zig
pub const WASMMemoryModel = enum {
```

- type `WASMGPUBackend`

WebAssembly GPU backend


```zig
pub const WASMGPUBackend = enum {
```

- type `WASMCompiler`

WebAssembly compilation manager with enhanced error handling


```zig
pub const WASMCompiler = struct {
```

- type `CompilationStatistics`

Compilation statistics for performance monitoring


```zig
pub const CompilationStatistics = struct {
```

- fn `init`

Initialize WASM compiler with validation


```zig
pub fn init(allocator: std.mem.Allocator, config: WASMConfig) WASMError!Self {
```

- fn `deinit`

Safely deinitialize the WASM compiler


```zig
pub fn deinit(self: *Self) void {
```

- fn `getStatistics`

Get compilation statistics


```zig
pub fn getStatistics(self: *Self) CompilationStatistics {
```

- fn `compileToWASM`

Compile Zig code to WebAssembly with comprehensive error handling


```zig
pub fn compileToWASM(self: *Self, source_files: []const []const u8, output_path: []const u8) WASMError!void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WASMConfig) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `getGPUBackendFlags`

Get GPU backend compilation flags


```zig
pub fn getGPUBackendFlags(self: *Self) ![]const []const u8 {
```

- fn `getMemoryFlags`

Get memory model compilation flags


```zig
pub fn getMemoryFlags(self: *Self) ![]const []const u8 {
```

- type `WASMRuntime`

WebAssembly runtime environment with enhanced error handling


```zig
pub const WASMRuntime = struct {
```

- type `RuntimeStatistics`

Runtime statistics for performance monitoring


```zig
pub const RuntimeStatistics = struct {
```

- fn `init`

Initialize WASM runtime with validation


```zig
pub fn init(allocator: std.mem.Allocator, config: WASMConfig) WASMError!Self {
```

- fn `deinit`

Safely deinitialize the WASM runtime


```zig
pub fn deinit(self: *Self) void {
```

- fn `getStatistics`

Get runtime statistics


```zig
pub fn getStatistics(self: *Self) RuntimeStatistics {
```

- fn `initGPUContext`

Initialize GPU context for WebAssembly with comprehensive error handling


```zig
pub fn initGPUContext(self: *Self) WASMError!void {
```

- fn `execute`

Execute WebAssembly module with performance monitoring


```zig
pub fn execute(self: *Self, module_path: []const u8) WASMError!void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WASMConfig) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `allocate`

Allocate memory in WASM linear memory


```zig
pub fn allocate(self: *Self, size: usize) ![]u8 {
```

- fn `getMemoryUsage`

Get current memory usage


```zig
pub fn getMemoryUsage(self: *Self) MemoryUsage {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WASMConfig) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initialize`

Initialize GPU context


```zig
pub fn initialize(self: *Self) !void {
```

- type `PredefinedWASMConfigs`

Predefined WebAssembly configurations


```zig
pub const PredefinedWASMConfigs = struct {
```

- fn `highPerformance`

High-performance WebAssembly configuration


```zig
pub fn highPerformance(allocator: std.mem.Allocator) !WASMConfig {
```

- fn `sizeOptimized`

Size-optimized WebAssembly configuration


```zig
pub fn sizeOptimized(allocator: std.mem.Allocator) !WASMConfig {
```

- fn `balanced`

Balanced WebAssembly configuration


```zig
pub fn balanced(allocator: std.mem.Allocator) !WASMConfig {
```

- fn `debug`

Debug WebAssembly configuration


```zig
pub fn debug(allocator: std.mem.Allocator) !WASMConfig {
```

- fn `logWASMConfig`

Log WebAssembly configuration


```zig
pub fn logWASMConfig(config: *const WASMConfig) void {
```

## src/features/monitoring/health.zig

- type `HealthStatus`

Health status levels


```zig
pub const HealthStatus = enum {
```

- fn `toString`

```zig
pub fn toString(self: HealthStatus) []const u8 {
```

- type `HealthCheck`

Individual health check result


```zig
pub const HealthCheck = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, status: HealthStatus, message: []const u8, response_time_ms: u64) !HealthCheck {
```

- fn `deinit`

```zig
pub fn deinit(self: *HealthCheck, allocator: std.mem.Allocator) void {
```

- fn `addMetadata`

```zig
pub fn addMetadata(self: *HealthCheck, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `HealthConfig`

Health checker configuration


```zig
pub const HealthConfig = struct {
```

- type `SystemHealth`

Overall system health status


```zig
pub const SystemHealth = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) SystemHealth {
```

- fn `deinit`

```zig
pub fn deinit(self: *SystemHealth) void {
```

- fn `updateOverallStatus`

```zig
pub fn updateOverallStatus(self: *SystemHealth) void {
```

- fn `addCheck`

```zig
pub fn addCheck(self: *SystemHealth, check: HealthCheck) !void {
```

- type `HealthChecker`

Comprehensive health checker


```zig
pub const HealthChecker = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: HealthConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

```zig
pub fn start(self: *Self) !void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) void {
```

- fn `setHealthChangeCallback`

```zig
pub fn setHealthChangeCallback(self: *Self, callback: *const fn (SystemHealth) void) void {
```

- fn `getCurrentHealth`

```zig
pub fn getCurrentHealth(self: *Self) *const SystemHealth {
```

- fn `exportHealthStatus`

Export health status as JSON


```zig
pub fn exportHealthStatus(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

## src/features/monitoring/memory_tracker.zig

- type `AllocationRecord`

Memory allocation record


```zig
pub const AllocationRecord = struct {
```

- fn `memoryUsage`

Calculate memory usage for this allocation


```zig
pub fn memoryUsage(self: AllocationRecord) usize {
```

- fn `age`

Get allocation age in nanoseconds


```zig
pub fn age(self: AllocationRecord, current_time: u64) u64 {
```

- fn `isPotentialLeak`

Check if allocation is a potential leak


```zig
pub fn isPotentialLeak(self: AllocationRecord, current_time: u64, leak_threshold_ns: u64) bool {
```

- type `MemoryStats`

Memory statistics snapshot


```zig
pub const MemoryStats = struct {
```

- fn `currentUsage`

Calculate current memory usage


```zig
pub fn currentUsage(self: MemoryStats) usize {
```

- fn `efficiency`

Calculate memory efficiency (1.0 = no waste, lower = more fragmentation)


```zig
pub fn efficiency(self: MemoryStats) f64 {
```

- fn `allocationSuccessRate`

Get allocation success rate


```zig
pub fn allocationSuccessRate(self: MemoryStats) f64 {
```

- type `MemoryProfilerConfig`

Memory profiler configuration


```zig
pub const MemoryProfilerConfig = struct {
```

- type `MemoryProfiler`

Memory profiler main structure


```zig
pub const MemoryProfiler = struct {
```

- fn `init`

Initialize memory profiler


```zig
pub fn init(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !*MemoryProfiler {
```

- fn `deinit`

Deinitialize memory profiler


```zig
pub fn deinit(self: *MemoryProfiler) void {
```

- fn `recordAllocation`

Record a memory allocation


```zig
pub fn recordAllocation(
```

- fn `recordDeallocation`

Record a memory deallocation


```zig
pub fn recordDeallocation(self: *MemoryProfiler, id: u64) void {
```

- fn `getStats`

Get current memory statistics


```zig
pub fn getStats(self: *MemoryProfiler) MemoryStats {
```

- fn `getPotentialLeaks`

Get potential memory leaks


```zig
pub fn getPotentialLeaks(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]AllocationRecord {
```

- fn `generateReport`

Generate memory usage report


```zig
pub fn generateReport(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]u8 {
```

- fn `resetStats`

Reset statistics


```zig
pub fn resetStats(self: *MemoryProfiler) void {
```

- fn `collectPeriodicStats`

Collect periodic statistics


```zig
pub fn collectPeriodicStats(self: *MemoryProfiler) void {
```

- fn `initGlobalProfiler`

Initialize global memory profiler


```zig
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !void {
```

- fn `deinitGlobalProfiler`

Deinitialize global memory profiler


```zig
pub fn deinitGlobalProfiler() void {
```

- fn `getGlobalProfiler`

Get global memory profiler instance


```zig
pub fn getGlobalProfiler() ?*MemoryProfiler {
```

- type `TrackedAllocator`

Tracked allocator that integrates with memory profiler


```zig
pub const TrackedAllocator = struct {
```

- fn `init`

Initialize tracked allocator


```zig
pub fn init(parent_allocator: std.mem.Allocator, profiler: *MemoryProfiler) TrackedAllocator {
```

- fn `allocator`

Get allocator interface


```zig
pub fn allocator(self: *TrackedAllocator) std.mem.Allocator {
```

- type `MemoryMonitor`

Memory usage monitor


```zig
pub const MemoryMonitor = struct {
```

- fn `init`

Initialize memory monitor


```zig
pub fn init(profiler: *MemoryProfiler) !*MemoryMonitor {
```

- fn `start`

Start monitoring thread


```zig
pub fn start(self: *MemoryMonitor) !void {
```

- fn `stop`

Stop monitoring


```zig
pub fn stop(self: *MemoryMonitor) void {
```

- fn `deinit`

Deinitialize monitor


```zig
pub fn deinit(self: *MemoryMonitor) void {
```

- type `PerformanceMonitor`

Performance monitoring utilities


```zig
pub const PerformanceMonitor = struct {
```

- fn `start`

Start performance measurement


```zig
pub fn start(self: *PerformanceMonitor) void {
```

- fn `end`

End performance measurement


```zig
pub fn end(self: *PerformanceMonitor) void {
```

- fn `elapsedTime`

Get elapsed time in nanoseconds


```zig
pub fn elapsedTime(self: PerformanceMonitor) u64 {
```

- fn `memoryDelta`

Get memory usage delta


```zig
pub fn memoryDelta(self: PerformanceMonitor) i64 {
```

- fn `generateReport`

Generate performance report


```zig
pub fn generateReport(self: PerformanceMonitor, allocator: std.mem.Allocator, operation_name: []const u8) ![]u8 {
```

- type `utils`

Utility functions for memory profiling


```zig
pub const utils = struct {
```

- fn `simpleConfig`

Create a simple memory profiler configuration


```zig
pub fn simpleConfig() MemoryProfilerConfig {
```

- fn `developmentConfig`

Create a development configuration with more detailed tracking


```zig
pub fn developmentConfig() MemoryProfilerConfig {
```

- fn `productionConfig`

Create a production configuration with minimal overhead


```zig
pub fn productionConfig() MemoryProfilerConfig {
```

## src/features/monitoring/mod.zig

- const `health`

```zig
pub const health = @import("health.zig");
```

- const `performance`

```zig
pub const performance = @import("performance.zig");
```

- const `performance_profiler`

```zig
pub const performance_profiler = @import("performance_profiler.zig");
```

- const `memory_tracker`

```zig
pub const memory_tracker = @import("memory_tracker.zig");
```

- const `tracing`

```zig
pub const tracing = @import("tracing.zig");
```

- const `sampling`

```zig
pub const sampling = @import("sampling.zig");
```

- const `prometheus`

```zig
pub const prometheus = @import("prometheus.zig");
```

- const `regression`

```zig
pub const regression = @import("regression.zig");
```

- const `mod`

```zig
pub const mod = @import("mod.zig");
```

## src/features/monitoring/performance.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `PerformanceError`

Performance monitoring specific error types


```zig
pub const PerformanceError = error{
```

- type `MetricType`

Performance metric types


```zig
pub const MetricType = enum {
```

- const `MetricValue`

Performance metric value


```zig
pub const MetricValue = union(MetricType) {
```

- type `HistogramData`

Histogram data for latency measurements


```zig
pub const HistogramData = struct {
```

- fn `record`

```zig
pub fn record(self: *HistogramData, value: f64) void {
```

- fn `percentile`

```zig
pub fn percentile(self: *const HistogramData, p: f64) f64 {
```

- type `TimerData`

Timer data for duration measurements


```zig
pub const TimerData = struct {
```

- fn `start`

```zig
pub fn start(self: *TimerData) void {
```

- fn `stop`

```zig
pub fn stop(self: *TimerData) void {
```

- fn `averageDuration`

```zig
pub fn averageDuration(self: *const TimerData) f64 {
```

- type `Metric`

Performance metric entry


```zig
pub const Metric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, value: MetricValue) !Metric {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- fn `addLabel`

```zig
pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `CPUProfiler`

CPU profiler with sampling


```zig
pub const CPUProfiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, sampling_rate: u32) CPUProfiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *CPUProfiler) void {
```

- fn `start`

```zig
pub fn start(self: *CPUProfiler) !void {
```

- fn `stop`

```zig
pub fn stop(self: *CPUProfiler) void {
```

- type `MemoryTracker`

Memory allocation tracker


```zig
pub const MemoryTracker = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !MemoryTracker {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryTracker) void {
```

- fn `recordAllocation`

```zig
pub fn recordAllocation(self: *MemoryTracker, ptr: usize, size: usize) void {
```

- fn `recordDeallocation`

```zig
pub fn recordDeallocation(self: *MemoryTracker, ptr: usize) void {
```

- fn `getCurrentUsage`

```zig
pub fn getCurrentUsage(self: *const MemoryTracker) u64 {
```

- fn `getPeakUsage`

```zig
pub fn getPeakUsage(self: *const MemoryTracker) u64 {
```

- type `PerformanceMonitor`

Global performance monitoring system


```zig
pub const PerformanceMonitor = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(self: *PerformanceMonitor, name: []const u8, value: MetricValue) !void {
```

- fn `startProfiling`

```zig
pub fn startProfiling(self: *PerformanceMonitor) !void {
```

- fn `stopProfiling`

```zig
pub fn stopProfiling(self: *PerformanceMonitor) void {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *PerformanceMonitor, name: []const u8) ?Metric {
```

- type `TracyProfiler`

Tracy profiler integration (when enabled)


```zig
pub const TracyProfiler = struct {
```

- fn `zoneName`

```zig
pub fn zoneName(comptime name: []const u8) void {
```

- fn `zoneStart`

```zig
pub fn zoneStart() void {
```

- fn `zoneEnd`

```zig
pub fn zoneEnd() void {
```

- fn `plot`

```zig
pub fn plot(name: []const u8, value: f64) void {
```

- fn `init`

```zig
pub fn init() !void {
```

- fn `deinit`

```zig
pub fn deinit() void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(name: []const u8, value: f64) void {
```

- fn `recordCounter`

```zig
pub fn recordCounter(name: []const u8, value: u64) void {
```

- fn `recordLatency`

```zig
pub fn recordLatency(name: []const u8, duration_ns: u64) void {
```

- type `Timer`

Timer utility for measuring execution time


```zig
pub const Timer = struct {
```

- fn `start`

```zig
pub fn start(comptime name: []const u8) Timer {
```

- fn `stop`

```zig
pub fn stop(self: Timer) void {
```

- fn `timed`

Convenient macro for timing function execution


```zig
pub fn timed(comptime name: []const u8, func: anytype) @TypeOf(func()) {
```

## src/features/monitoring/performance_profiler.zig

- type `ProfilingConfig`

Performance profiling configuration


```zig
pub const ProfilingConfig = struct {
```

- type `CallRecord`

Function call record (for call tracing and call tree)


```zig
pub const CallRecord = struct {
```

- fn `duration`

Calculate call duration (nanoseconds)


```zig
pub fn duration(self: CallRecord) u64 {
```

- fn `isComplete`

Check if call is complete (has exit time)


```zig
pub fn isComplete(self: CallRecord) bool {
```

- type `PerformanceCounter`

Performance counter (for custom and built-in metrics)


```zig
pub const PerformanceCounter = struct {
```

- fn `increment`

```zig
pub fn increment(self: *PerformanceCounter) void {
```

- fn `add`

```zig
pub fn add(self: *PerformanceCounter, delta: u64) void {
```

- fn `set`

```zig
pub fn set(self: *PerformanceCounter, new_value: u64) void {
```

- fn `reset`

```zig
pub fn reset(self: *PerformanceCounter) void {
```

- type `PerformanceProfile`

Performance profile data (per session)


```zig
pub const PerformanceProfile = struct {
```

- fn `duration`

```zig
pub fn duration(self: PerformanceProfile) u64 {
```

- fn `durationSeconds`

```zig
pub fn durationSeconds(self: PerformanceProfile) f64 {
```

- fn `cpuUtilization`

```zig
pub fn cpuUtilization(self: PerformanceProfile) f64 {
```

- type `FunctionProfiler`

Function profiler for instrumenting and aggregating function stats


```zig
pub const FunctionProfiler = struct {
```

- fn `enter`

```zig
pub fn enter(self: *FunctionProfiler) u64 {
```

- fn `exit`

```zig
pub fn exit(self: *FunctionProfiler, entry_time: u64) void {
```

- fn `averageExecutionTime`

```zig
pub fn averageExecutionTime(self: FunctionProfiler) u64 {
```

- type `PerformanceProfiler`

Main performance profiler


```zig
pub const PerformanceProfiler = struct {
```

- fn `init`

Initialize performance profiler


```zig
pub fn init(allocator: std.mem.Allocator, config: ProfilingConfig) !*PerformanceProfiler {
```

- fn `deinit`

Deinitialize performance profiler and free all resources


```zig
pub fn deinit(self: *PerformanceProfiler) void {
```

- fn `startSession`

Start profiling session


```zig
pub fn startSession(self: *PerformanceProfiler, session_name: []const u8) !void {
```

- fn `endSession`

End profiling session and return report


```zig
pub fn endSession(self: *PerformanceProfiler) ![]u8 {
```

- fn `startFunctionCall`

Start function call (for call tracing)


```zig
pub fn startFunctionCall(self: *PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) !u64 {
```

- fn `endFunctionCall`

End function call (for call tracing)


```zig
pub fn endFunctionCall(self: *PerformanceProfiler, entry_time: u64) void {
```

- fn `updateCounter`

Update or create a performance counter


```zig
pub fn updateCounter(self: *PerformanceProfiler, name: []const u8, delta: u64) void {
```

- fn `getFunctionStats`

Get function profiler statistics (sorted by total_time descending)


```zig
pub fn getFunctionStats(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]FunctionProfiler {
```

- fn `stop`

Stop profiling thread


```zig
pub fn stop(self: *PerformanceProfiler) void {
```

- fn `setMemoryTracker`

Integrate with memory tracker


```zig
pub fn setMemoryTracker(self: *PerformanceProfiler, tracker: *memory_tracker.MemoryProfiler) void {
```

- fn `createScope`

Create performance scope for measuring code blocks


```zig
pub fn createScope(self: *PerformanceProfiler, name: []const u8) Scope {
```

- type `Scope`

Performance measurement scope (RAII-style)


```zig
pub const Scope = struct {
```

- fn `end`

End the scope and record measurements


```zig
pub fn end(self: Scope) void {
```

- fn `initGlobalProfiler`

Initialize global performance profiler


```zig
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: ProfilingConfig) !void {
```

- fn `deinitGlobalProfiler`

Deinitialize global performance profiler


```zig
pub fn deinitGlobalProfiler() void {
```

- fn `getGlobalProfiler`

Get global performance profiler instance


```zig
pub fn getGlobalProfiler() ?*PerformanceProfiler {
```

- fn `startScope`

Convenience function to start a performance scope


```zig
pub fn startScope(name: []const u8) ?Scope {
```

- fn `profileFunctionCall`

Convenience function for profiling function calls (to be used with defer)


```zig
pub fn profileFunctionCall(profiler: ?*PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) FunctionCall {
```

- type `FunctionCall`

Function call scope for automatic profiling (RAII-style)


```zig
pub const FunctionCall = struct {
```

- fn `end`

```zig
pub fn end(self: FunctionCall) void {
```

- type `utils`

Performance monitoring utilities and presets


```zig
pub const utils = struct {
```

- fn `developmentConfig`

Create a development profiling configuration


```zig
pub fn developmentConfig() ProfilingConfig {
```

- fn `productionConfig`

Create a production profiling configuration


```zig
pub fn productionConfig() ProfilingConfig {
```

- fn `minimalConfig`

Create a minimal profiling configuration


```zig
pub fn minimalConfig() ProfilingConfig {
```

## src/features/monitoring/prometheus.zig

- type `MetricType`

Prometheus metric types


```zig
pub const MetricType = enum {
```

- type `Metric`

Individual metric definition


```zig
pub const Metric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, help: []const u8, metric_type: MetricType) !Metric {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- fn `addLabel`

```zig
pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- fn `setValue`

```zig
pub fn setValue(self: *Metric, value: f64) void {
```

- fn `increment`

```zig
pub fn increment(self: *Metric) void {
```

- fn `add`

```zig
pub fn add(self: *Metric, value: f64) void {
```

- type `MetricsCollector`

Metrics collector and registry


```zig
pub const MetricsCollector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `registerMetric`

```zig
pub fn registerMetric(self: *Self, name: []const u8, help: []const u8, metric_type: MetricType) !*Metric {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *Self, name: []const u8) ?*Metric {
```

- fn `recordDatabaseOperation`

```zig
pub fn recordDatabaseOperation(self: *Self, operation: []const u8, duration_ns: u64) !void {
```

- fn `updateDatabaseStats`

```zig
pub fn updateDatabaseStats(self: *Self, vectors_stored: u64, compression_ratio: f64) void {
```

- fn `recordSearch`

```zig
pub fn recordSearch(self: *Self, duration_ns: u64, results_count: usize) !void {
```

- fn `recordHttpRequest`

```zig
pub fn recordHttpRequest(self: *Self, method: []const u8, path: []const u8, status_code: u16, duration_ns: u64, response_size: usize) !void {
```

- fn `updateSystemMetrics`

```zig
pub fn updateSystemMetrics(self: *Self, cpu_percent: f64, memory_used: u64, memory_available: u64, disk_used: u64) void {
```

- fn `updateProcessMetrics`

```zig
pub fn updateProcessMetrics(self: *Self, cpu_seconds: f64, memory_bytes: u64, thread_count: u32) void {
```

- fn `exportPrometheusFormat`

Export metrics in Prometheus format


```zig
pub fn exportPrometheusFormat(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

- type `PrometheusServer`

Prometheus HTTP server for metrics export


```zig
pub const PrometheusServer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, metrics_collector: *MetricsCollector, host: []const u8, port: u16, path: []const u8) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

```zig
pub fn start(self: *Self) !void {
```

## src/features/monitoring/regression.zig

- type `RegressionSensitivity`

Regression sensitivity levels


```zig
pub const RegressionSensitivity = enum {
```

- type `PerformanceMetric`

Performance metric for regression analysis


```zig
pub const PerformanceMetric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, value: f64) !PerformanceMetric {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMetric, allocator: std.mem.Allocator) void {
```

- fn `addMetadata`

```zig
pub fn addMetadata(self: *PerformanceMetric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `PerformanceBaseline`

Statistical performance baseline


```zig
pub const PerformanceBaseline = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, metric_name: []const u8) !PerformanceBaseline {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceBaseline, allocator: std.mem.Allocator) void {
```

- fn `updateWithValue`

```zig
pub fn updateWithValue(self: *PerformanceBaseline, value: f64) void {
```

- fn `isRegression`

```zig
pub fn isRegression(self: *const PerformanceBaseline, value: f64, sensitivity: RegressionSensitivity) bool {
```

- fn `getRegressionSeverity`

```zig
pub fn getRegressionSeverity(self: *const PerformanceBaseline, value: f64) RegressionSensitivity {
```

- type `RegressionAlert`

Regression alert information


```zig
pub const RegressionAlert = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, metric_name: []const u8, current_value: f64, baseline_mean: f64, severity: RegressionSensitivity) !RegressionAlert {
```

- fn `deinit`

```zig
pub fn deinit(self: *RegressionAlert, allocator: std.mem.Allocator) void {
```

- type `RegressionConfig`

Configuration for regression detection


```zig
pub const RegressionConfig = struct {
```

- type `RegressionDetector`

Performance regression detector


```zig
pub const RegressionDetector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: RegressionConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `setRegressionCallback`

```zig
pub fn setRegressionCallback(self: *Self, callback: *const fn (RegressionAlert) void) void {
```

- fn `recordMetric`

Record a performance metric and check for regressions


```zig
pub fn recordMetric(self: *Self, metric: PerformanceMetric) !void {
```

- fn `getBaseline`

Get baseline for a metric


```zig
pub fn getBaseline(self: *Self, metric_name: []const u8) ?*const PerformanceBaseline {
```

- fn `getRecentAlerts`

Get recent alerts


```zig
pub fn getRecentAlerts(self: *Self, limit: ?usize) []const RegressionAlert {
```

- fn `exportStats`

Export regression detection statistics


```zig
pub fn exportStats(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `updateAllBaselines`

Force update all baselines (useful for testing)


```zig
pub fn updateAllBaselines(self: *Self) !void {
```

## src/features/monitoring/sampling.zig

- type `PerformanceSample`

Performance sample data point


```zig
pub const PerformanceSample = struct {
```

- fn `init`

```zig
pub fn init() PerformanceSample {
```

- fn `calculateMemoryUsagePercent`

```zig
pub fn calculateMemoryUsagePercent(self: *const PerformanceSample) f64 {
```

- fn `calculateProcessMemoryPercent`

```zig
pub fn calculateProcessMemoryPercent(self: *const PerformanceSample) f64 {
```

- type `SamplerConfig`

Performance sampler configuration


```zig
pub const SamplerConfig = struct {
```

- fn `getCpuUsage`

Get system CPU usage percentage


```zig
pub fn getCpuUsage() !f64 {
```

- fn `getMemoryInfo`

Get system memory information


```zig
pub fn getMemoryInfo() !struct { total: u64, used: u64, available: u64 } {
```

- fn `getProcessInfo`

Get process-specific information


```zig
pub fn getProcessInfo() !struct { cpu_percent: f64, memory_rss: u64, memory_vms: u64, threads: u32, uptime: u64 } {
```

- fn `getDiskInfo`

Get disk usage information


```zig
pub fn getDiskInfo() !struct { read_bytes: u64, write_bytes: u64, usage_percent: f64 } {
```

- type `PerformanceSampler`

Performance sampler with periodic monitoring


```zig
pub const PerformanceSampler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: SamplerConfig) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

```zig
pub fn start(self: *Self) !void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) void {
```

- fn `setAlertCallbacks`

```zig
pub fn setAlertCallbacks(self: *Self, cpu_callback: ?*const fn (f64) void, memory_callback: ?*const fn (f64) void, disk_callback: ?*const fn (f64) void) void {
```

- fn `getCurrentSample`

```zig
pub fn getCurrentSample(self: *Self) ?PerformanceSample {
```

- fn `getAverageCpuUsage`

```zig
pub fn getAverageCpuUsage(self: *Self, duration_seconds: u32) f64 {
```

- fn `getAverageMemoryUsage`

```zig
pub fn getAverageMemoryUsage(self: *Self, duration_seconds: u32) f64 {
```

- fn `getSamplesInRange`

```zig
pub fn getSamplesInRange(self: *Self, start_time: i64, end_time: i64, allocator: std.mem.Allocator) ![]PerformanceSample {
```

- fn `exportStats`

Export performance statistics


```zig
pub fn exportStats(self: *Self, allocator: std.mem.Allocator) ![]const u8 {
```

## src/features/monitoring/tracing.zig

- const `TracingError`

Tracing error types


```zig
pub const TracingError = error{
```

- type `TraceId`

Trace ID - unique identifier for a trace


```zig
pub const TraceId = struct {
```

- fn `init`

```zig
pub fn init() TraceId {
```

- fn `toString`

```zig
pub fn toString(self: TraceId, allocator: std.mem.Allocator) ![]u8 {
```

- fn `fromString`

```zig
pub fn fromString(str: []const u8) !TraceId {
```

- const `SpanId`

Span ID - unique identifier for a span within a trace


```zig
pub const SpanId = u64;
```

- type `SpanKind`

Span kind enumeration


```zig
pub const SpanKind = enum {
```

- type `SpanStatus`

Span status


```zig
pub const SpanStatus = enum {
```

- type `Span`

Trace span representing a single operation


```zig
pub const Span = struct {
```

- type `SpanEvent`

Span event for annotations


```zig
pub const SpanEvent = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *SpanEvent, allocator: std.mem.Allocator) void {
```

- fn `init`

```zig
pub fn init(
```

- fn `deinit`

```zig
pub fn deinit(self: *Span, allocator: std.mem.Allocator) void {
```

- fn `end`

End the span


```zig
pub fn end(self: *Span) void {
```

- fn `setStatus`

Set span status


```zig
pub fn setStatus(self: *Span, status: SpanStatus) void {
```

- fn `setAttribute`

Add an attribute to the span


```zig
pub fn setAttribute(self: *Span, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- fn `addEvent`

Add an event to the span


```zig
pub fn addEvent(self: *Span, allocator: std.mem.Allocator, name: []const u8) !void {
```

- fn `duration`

Get span duration in nanoseconds


```zig
pub fn duration(self: Span) ?i128 {
```

- fn `isActive`

Check if span is still active


```zig
pub fn isActive(self: Span) bool {
```

- type `TraceContext`

Trace context for propagating tracing information


```zig
pub const TraceContext = struct {
```

- fn `init`

Create a new trace context


```zig
pub fn init() TraceContext {
```

- fn `child`

Create child context


```zig
pub fn child(self: TraceContext) TraceContext {
```

- fn `toString`

Serialize context to string


```zig
pub fn toString(self: TraceContext, allocator: std.mem.Allocator) ![]u8 {
```

- fn `fromString`

Deserialize context from string


```zig
pub fn fromString(str: []const u8) !TraceContext {
```

- type `Tracer`

Tracer - main tracing interface


```zig
pub const Tracer = struct {
```

- type `TracerConfig`

```zig
pub const TracerConfig = struct {
```

- const `Sampler`

Sampling strategy


```zig
pub const Sampler = union(enum) {
```

- fn `shouldSample`

```zig
pub fn shouldSample(self: *Sampler) bool {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: TracerConfig) !*Tracer {
```

- fn `deinit`

```zig
pub fn deinit(self: *Tracer) void {
```

- fn `startSpan`

Start a new span


```zig
pub fn startSpan(self: *Tracer, name: []const u8, kind: SpanKind, context: ?TraceContext) !*Span {
```

- fn `endSpan`

End a span


```zig
pub fn endSpan(self: *Tracer, span: *Span) void {
```

- fn `getSpan`

Get active span by ID


```zig
pub fn getSpan(self: *Tracer, span_id: SpanId) ?*Span {
```

- fn `exportToJson`

Export traces to JSON (simplified)


```zig
pub fn exportToJson(self: *Tracer, allocator: std.mem.Allocator) ![]u8 {
```

- fn `initGlobalTracer`

Initialize global tracer


```zig
pub fn initGlobalTracer(allocator: std.mem.Allocator, config: Tracer.TracerConfig) !void {
```

- fn `deinitGlobalTracer`

Deinitialize global tracer


```zig
pub fn deinitGlobalTracer() void {
```

- fn `getGlobalTracer`

Get global tracer instance


```zig
pub fn getGlobalTracer() ?*Tracer {
```

- fn `startSpan`

Helper function to start a span with global tracer


```zig
pub fn startSpan(name: []const u8, kind: SpanKind, context: ?TraceContext) !*Span {
```

- fn `endSpan`

Helper function to end a span with global tracer


```zig
pub fn endSpan(span: *Span) void {
```

- fn `traceFunction`

Convenience macro-like function for tracing function calls


```zig
pub fn traceFunction(comptime func_name: []const u8, context: ?TraceContext) !*Span {
```

- fn `integrateWithPerformance`

Integration with performance monitoring


```zig
pub fn integrateWithPerformance(tracer: *Tracer, perf_monitor: *performance.PerformanceMonitor) void {
```

## src/features/web/c_api.zig

- type `WdbxResult`

```zig
pub const WdbxResult = extern struct {
```

## src/features/web/curl_wrapper.zig

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) CurlResponse {
```

- fn `deinit`

```zig
pub fn deinit(self: *CurlResponse) void {
```

- type `CurlHttpClient`

Libcurl HTTP client implementation


```zig
pub const CurlHttpClient = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: http_client.HttpClientConfig) !Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `request`

Make HTTP request using libcurl


```zig
pub fn request(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !http_client.HttpResponse {
```

## src/features/web/enhanced_web_server.zig

- type `EnhancedWebServer`

Enhanced web server with production-ready features


```zig
pub const EnhancedWebServer = struct {
```

- fn `init`

Initialize the enhanced web server


```zig
pub fn init(allocator: std.mem.Allocator, server_config: WebServerConfig) FrameworkError!*Self {
```

- fn `deinit`

Deinitialize the enhanced web server


```zig
pub fn deinit(self: *Self) void {
```

- fn `start`

Start the web server


```zig
pub fn start(self: *Self) FrameworkError!void {
```

- fn `stop`

Stop the web server


```zig
pub fn stop(self: *Self) void {
```

- fn `addRoute`

Add a route to the server


```zig
pub fn addRoute(self: *Self, route: Route) FrameworkError!void {
```

- fn `addMiddleware`

Add middleware to the server


```zig
pub fn addMiddleware(self: *Self, middleware: Middleware) FrameworkError!void {
```

- fn `getStats`

Get server statistics


```zig
pub fn getStats(self: *const Self) ServerStats {
```

- fn `healthCheck`

Health check for the server


```zig
pub fn healthCheck(self: *const Self) ServerHealthStatus {
```

- type `ServerState`

Server state management


```zig
pub const ServerState = enum {
```

- fn `canTransitionTo`

```zig
pub fn canTransitionTo(self: ServerState, new_state: ServerState) bool {
```

- type `HttpMethod`

HTTP methods


```zig
pub const HttpMethod = enum {
```

- type `Route`

Route definition


```zig
pub const Route = struct {
```

- const `RouteHandler`

Route handler function type


```zig
pub const RouteHandler = *const fn (request: *Request, response: *Response) anyerror!void;
```

- type `Middleware`

Middleware definition


```zig
pub const Middleware = struct {
```

- const `MiddlewareHandler`

Middleware handler function type


```zig
pub const MiddlewareHandler = *const fn (request: *Request, response: *Response, next: *const fn (*Request, *Response) anyerror!void) anyerror!void;
```

- type `RateLimit`

Rate limiting configuration


```zig
pub const RateLimit = struct {
```

- type `Request`

HTTP request structure


```zig
pub const Request = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Request {
```

- fn `deinit`

```zig
pub fn deinit(self: *Request) void {
```

- fn `getHeader`

```zig
pub fn getHeader(self: *const Request, name: []const u8) ?[]const u8 {
```

- fn `getQueryParam`

```zig
pub fn getQueryParam(self: *const Request, name: []const u8) ?[]const u8 {
```

- fn `hasHeader`

```zig
pub fn hasHeader(self: *const Request, name: []const u8) bool {
```

- fn `hasQueryParam`

```zig
pub fn hasQueryParam(self: *const Request, name: []const u8) bool {
```

- type `Response`

HTTP response structure


```zig
pub const Response = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Response {
```

- fn `deinit`

```zig
pub fn deinit(self: *Response) void {
```

- fn `setHeader`

```zig
pub fn setHeader(self: *Response, name: []const u8, value: []const u8) !void {
```

- fn `getHeader`

```zig
pub fn getHeader(self: *const Response, name: []const u8) ?[]const u8 {
```

- fn `setContentType`

```zig
pub fn setContentType(self: *Response, content_type: []const u8) void {
```

- fn `setBody`

```zig
pub fn setBody(self: *Response, body: []const u8) void {
```

- fn `send`

```zig
pub fn send(self: *Response, status: u16, body: []const u8) !void {
```

- fn `sendJson`

```zig
pub fn sendJson(self: *Response, status: u16, data: anytype) !void {
```

- fn `sendError`

```zig
pub fn sendError(self: *Response, status: u16, message: []const u8) !void {
```

- type `ErrorResponse`

Error response structure


```zig
pub const ErrorResponse = struct {
```

- type `ServerStats`

Server statistics


```zig
pub const ServerStats = struct {
```

- type `ServerHealthStatus`

Server health status


```zig
pub const ServerHealthStatus = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *ServerHealthStatus) void {
```

- type `ComponentHealth`

Component health status


```zig
pub const ComponentHealth = struct {
```

- type `HealthStatus`

Health status levels


```zig
pub const HealthStatus = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WebServerConfig) !*HttpServer {
```

- fn `deinit`

```zig
pub fn deinit(self: *HttpServer) void {
```

- fn `start`

```zig
pub fn start(self: *HttpServer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *HttpServer) void {
```

- fn `getActiveConnections`

```zig
pub fn getActiveConnections(self: *const HttpServer) u32 {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const HttpServer) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WebServerConfig) !*WebSocketServer {
```

- fn `deinit`

```zig
pub fn deinit(self: *WebSocketServer) void {
```

- fn `start`

```zig
pub fn start(self: *WebSocketServer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *WebSocketServer) void {
```

- fn `getActiveConnections`

```zig
pub fn getActiveConnections(self: *const WebSocketServer) u32 {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const WebSocketServer) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*RouteRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *RouteRegistry) void {
```

- fn `registerRoute`

```zig
pub fn registerRoute(self: *RouteRegistry, route: Route) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*RequestPool {
```

- fn `deinit`

```zig
pub fn deinit(self: *RequestPool) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ResponsePool {
```

- fn `deinit`

```zig
pub fn deinit(self: *ResponsePool) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*AgentRouter {
```

- fn `deinit`

```zig
pub fn deinit(self: *AgentRouter) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const AgentRouter) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*AuthManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *AuthManager) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *AuthManager) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const AuthManager) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*RateLimiter {
```

- fn `deinit`

```zig
pub fn deinit(self: *RateLimiter) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *RateLimiter) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const RateLimiter) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SecurityManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityManager) void {
```

- fn `initializePolicies`

```zig
pub fn initializePolicies(self: *SecurityManager) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const SecurityManager) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *PerformanceMonitor) !void {
```

- fn `start`

```zig
pub fn start(self: *PerformanceMonitor) !void {
```

- fn `stop`

```zig
pub fn stop(self: *PerformanceMonitor) void {
```

- fn `getTotalRequests`

```zig
pub fn getTotalRequests(self: *const PerformanceMonitor) u64 {
```

- fn `getAverageResponseTime`

```zig
pub fn getAverageResponseTime(self: *const PerformanceMonitor) f64 {
```

- fn `getErrorRate`

```zig
pub fn getErrorRate(self: *const PerformanceMonitor) f32 {
```

- fn `getMemoryUsage`

```zig
pub fn getMemoryUsage(self: *const PerformanceMonitor) usize {
```

- fn `getCpuUsage`

```zig
pub fn getCpuUsage(self: *const PerformanceMonitor) f32 {
```

- fn `registerCallback`

```zig
pub fn registerCallback(self: *PerformanceMonitor, callback: anytype) !void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const PerformanceMonitor) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*LoadBalancer {
```

- fn `deinit`

```zig
pub fn deinit(self: *LoadBalancer) void {
```

- fn `start`

```zig
pub fn start(self: *LoadBalancer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *LoadBalancer) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const LoadBalancer) ComponentHealth {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ClusterManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *ClusterManager) void {
```

- fn `start`

```zig
pub fn start(self: *ClusterManager) !void {
```

- fn `stop`

```zig
pub fn stop(self: *ClusterManager) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const ClusterManager) ComponentHealth {
```

## src/features/web/http_client.zig

- type `HttpClientConfig`

HTTP client configuration


```zig
pub const HttpClientConfig = struct {
```

- type `HttpResponse`

HTTP response structure


```zig
pub const HttpResponse = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *HttpResponse) void {
```

- type `HttpClient`

HTTP client with libcurl integration and fallback


```zig
pub const HttpClient = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: HttpClientConfig) Self {
```

- fn `get`

Make HTTP GET request with retry and backoff


```zig
pub fn get(self: *Self, url: []const u8) !HttpResponse {
```

- fn `post`

Make HTTP POST request with retry and backoff


```zig
pub fn post(self: *Self, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !HttpResponse {
```

- fn `request`

Make HTTP request with automatic retry and exponential backoff


```zig
pub fn request(self: *Self, method: []const u8, url: []const u8, content_type: ?[]const u8, body: ?[]const u8) !HttpResponse {
```

- fn `testConnectivity`

Test connectivity with enhanced diagnostics


```zig
pub fn testConnectivity(self: *Self, url: []const u8) !bool {
```

- type `ConnectivityTester`

Enhanced connectivity tester with comprehensive diagnostics


```zig
pub const ConnectivityTester = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: HttpClientConfig) Self {
```

- fn `runDiagnostics`

Run comprehensive connectivity tests


```zig
pub fn runDiagnostics(self: *Self, base_url: []const u8) !void {
```

## src/features/web/mod.zig

- const `web_server`

```zig
pub const web_server = @import("web_server.zig");
```

- const `enhanced_web_server`

```zig
pub const enhanced_web_server = @import("enhanced_web_server.zig");
```

- const `http_client`

```zig
pub const http_client = @import("http_client.zig");
```

- const `curl_wrapper`

```zig
pub const curl_wrapper = @import("curl_wrapper.zig");
```

- const `wdbx_http`

```zig
pub const wdbx_http = @import("wdbx_http.zig");
```

- const `weather`

```zig
pub const weather = @import("weather.zig");
```

- const `c_api`

```zig
pub const c_api = @import("c_api.zig");
```

- const `mod`

```zig
pub const mod = @import("mod.zig");
```

## src/features/web/wdbx_http.zig

- const `HttpError`

```zig
pub const HttpError = error{
```

- type `Response`

```zig
pub const Response = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: Response, allocator: std.mem.Allocator) void {
```

- type `ServerConfig`

```zig
pub const ServerConfig = struct {
```

- type `WdbxHttpServer`

```zig
pub const WdbxHttpServer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: ServerConfig) !*WdbxHttpServer {
```

- fn `deinit`

```zig
pub fn deinit(self: *WdbxHttpServer) void {
```

- fn `openDatabase`

```zig
pub fn openDatabase(self: *WdbxHttpServer, path: []const u8) !void {
```

- fn `start`

Mark the server as running. A real TCP listener can be layered on top of
the current request handler in a future iteration.


```zig
pub fn start(self: *WdbxHttpServer) !void {
```

- fn `stop`

```zig
pub fn stop(self: *WdbxHttpServer) void {
```

- fn `respond`

High level request handler used by the CLI and tests.


```zig
pub fn respond(
```

## src/features/web/weather.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `WeatherError`

Weather-specific error types


```zig
pub const WeatherError = error{
```

- type `WeatherData`

```zig
pub const WeatherData = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *WeatherData, allocator: std.mem.Allocator) void {
```

- type `WeatherConfig`

```zig
pub const WeatherConfig = struct {
```

- fn `fromEnv`

```zig
pub fn fromEnv(allocator: std.mem.Allocator, base: WeatherConfig) WeatherConfig {
```

- type `WeatherService`

```zig
pub const WeatherService = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: WeatherConfig) !WeatherService {
```

- fn `deinit`

```zig
pub fn deinit(self: *WeatherService) void {
```

- fn `getCurrentWeather`

```zig
pub fn getCurrentWeather(self: *WeatherService, city: []const u8) !WeatherData {
```

- type `WeatherUtils`

```zig
pub const WeatherUtils = struct {
```

- fn `kelvinToCelsius`

```zig
pub fn kelvinToCelsius(kelvin: f32) f32 {
```

- fn `testParseWeatherResponse`

```zig
pub fn testParseWeatherResponse(self: *WeatherService, json_str: []const u8) !WeatherData {
```

- fn `testParseForecastResponse`

```zig
pub fn testParseForecastResponse(self: *WeatherService, json_str: []const u8) ![]WeatherData {
```

- fn `celsiusToFahrenheit`

```zig
pub fn celsiusToFahrenheit(celsius: f32) f32 {
```

- fn `fahrenheitToCelsius`

```zig
pub fn fahrenheitToCelsius(fahrenheit: f32) f32 {
```

- fn `getWindDirection`

```zig
pub fn getWindDirection(degrees: u16) []const u8 {
```

- fn `formatWeatherJson`

```zig
pub fn formatWeatherJson(weather: WeatherData, allocator: std.mem.Allocator) ![]u8 {
```

- fn `getWeatherEmoji`

```zig
pub fn getWeatherEmoji(icon: []const u8) []const u8 {
```

## src/features/web/web_server.zig

- const `Allocator`

Re-export commonly used types for convenience


```zig
pub const Allocator = std.mem.Allocator;
```

- type `WebConfig`

Configuration settings for the web server.

This structure contains all the configurable parameters that control
the server's behavior, including network settings, security options,
and feature toggles.


```zig
pub const WebConfig = struct {
```

- type `WebServer`

Main web server instance that handles HTTP/WebSocket connections.

The WebServer manages network connections, routes requests to appropriate handlers,
and integrates with the AI agent system for intelligent request processing.
It supports both traditional HTTP endpoints and real-time WebSocket communication.


```zig
pub const WebServer = struct {
```

- fn `init`

Initializes a new WebServer instance with the specified configuration.

This function sets up the server with the provided configuration and
initializes the integrated AI agent for intelligent request processing.

Parameters:
- `allocator`: Memory allocator for the server and its components
- `config`: Configuration settings controlling server behavior

Returns:
- A pointer to the initialized WebServer instance

Errors:
- Returns an error if memory allocation fails
- Returns an error if AI agent initialization fails


```zig
pub fn init(allocator: std.mem.Allocator, config: WebConfig) !*WebServer {
```

- fn `deinit`

Properly releases all resources held by the WebServer.

This function should be called when the server is no longer needed
to prevent memory leaks and properly close network connections.


```zig
pub fn deinit(self: *WebServer) void {
```

- fn `start`

Starts the web server and begins accepting connections.

This function binds to the configured address and port, then enters
an infinite loop accepting and handling incoming connections.
Each connection is processed synchronously in the current implementation.

Errors:
- Returns an error if the address cannot be parsed
- Returns an error if the server cannot bind to the specified port


```zig
pub fn start(self: *WebServer) !void {
```

- fn `parseWebSocketFrame`

Parses a WebSocket frame from raw bytes.

This function implements the WebSocket frame parsing algorithm
according to RFC 6455, handling variable-length payload lengths
and masking (though masking is typically only used by clients).

Parameters:
- `data`: Raw frame data received from the WebSocket connection

Returns:
- A parsed WebSocketFrame structure

Errors:
- Returns InvalidFrame if the frame data is malformed or incomplete


```zig
pub fn parseWebSocketFrame(_: *WebServer, data: []const u8) !WebSocketFrame {
```

- fn `startOnce`

Starts the server for a single connection cycle (testing utility).

This function is intended for testing scenarios where you need to
accept exactly one connection, handle it, and then stop the server.
It's useful for unit tests and development scenarios.

Errors:
- Returns an error if the server cannot start
- Returns an error if connection handling fails


```zig
pub fn startOnce(self: *WebServer) !void {
```

- fn `handlePathForTest`

Test helper function for routing requests by path.

This function provides a simple way to test routing logic without
requiring actual network connections. It returns the response body
that would be sent for a given path.

Parameters:
- `path`: The request path to route
- `allocator`: Memory allocator for the response

Returns:
- The response body as a string

Errors:
- Returns an error if memory allocation fails


```zig
pub fn handlePathForTest(self: *WebServer, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
```

## src/main.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/ml/localml.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = core.Allocator;
```

- const `MLError`

LocalML-specific error types


```zig
pub const MLError = error{
```

- type `DataRow`

```zig
pub const DataRow = struct {
```

- fn `validate`

```zig
pub fn validate(self: DataRow) MLError!void {
```

- fn `fromArray`

```zig
pub fn fromArray(values: []const f64) MLError!DataRow {
```

- type `Model`

```zig
pub const Model = struct {
```

- fn `init`

```zig
pub fn init() Model {
```

- fn `predict`

```zig
pub fn predict(self: Model, row: DataRow) MLError!f64 {
```

- fn `train`

```zig
pub fn train(self: *Model, data: []const DataRow, learning_rate: f64, epochs: usize) MLError!void {
```

- fn `main`

```zig
pub fn main() !void {
```

## src/ml/neural.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- type `LayerType`

Neural network layer types


```zig
pub const LayerType = enum {
```

- type `Activation`

Activation functions with mixed precision support


```zig
pub const Activation = enum {
```

- fn `apply`

Apply activation function to a f32 value


```zig
pub fn apply(self: Activation, x: f32) f32 {
```

- fn `applyF16`

Apply activation function to a f16 value


```zig
pub fn applyF16(self: Activation, x: f16) f16 {
```

- fn `derivative`

Derivative of activation function (f32)


```zig
pub fn derivative(self: Activation, x: f32) f32 {
```

- fn `derivativeF16`

Derivative of activation function (f16)


```zig
pub fn derivativeF16(self: Activation, x: f16) f16 {
```

- type `Precision`

Precision mode for computations


```zig
pub const Precision = enum {
```

- type `TrainingConfig`

Neural network training configuration with enhanced memory options


```zig
pub const TrainingConfig = struct {
```

- type `LayerConfig`

Layer configuration


```zig
pub const LayerConfig = struct {
```

- type `NetworkConfig`

Complete neural network configuration


```zig
pub const NetworkConfig = struct {
```

- type `MemoryPool`

Memory pool for efficient buffer reuse


```zig
pub const MemoryPool = struct {
```

- type `PoolConfig`

Memory pool configuration


```zig
pub const PoolConfig = struct {
```

- type `PooledBuffer`

Pooled buffer


```zig
pub const PooledBuffer = struct {
```

- fn `release`

Return buffer to pool


```zig
pub fn release(self: *PooledBuffer) void {
```

- fn `slice`

Get buffer as slice of requested size


```zig
pub fn slice(self: *PooledBuffer, size: usize) []f32 {
```

- type `TrackedBuffer`

Enhanced buffer with liveness tracking


```zig
pub const TrackedBuffer = struct {
```

- fn `isStale`

Check if buffer is stale (not accessed recently)


```zig
pub fn isStale(self: TrackedBuffer, current_time: u64, stale_threshold_ns: u64) bool {
```

- fn `markAccessed`

Update access time


```zig
pub fn markAccessed(self: *TrackedBuffer, current_time: u64) void {
```

- type `LivenessConfig`

Liveness analysis configuration


```zig
pub const LivenessConfig = struct {
```

- fn `init`

Initialize memory pool


```zig
pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !*MemoryPool {
```

- fn `deinit`

Deinitialize memory pool


```zig
pub fn deinit(self: *MemoryPool) void {
```

- fn `allocBuffer`

Allocate buffer from pool or create new one


```zig
pub fn allocBuffer(self: *MemoryPool, size: usize) !*PooledBuffer {
```

- fn `returnBuffer`

Return buffer to pool for reuse


```zig
pub fn returnBuffer(self: *MemoryPool, buffer: *PooledBuffer) void {
```

- fn `getStats`

Get pool statistics


```zig
pub fn getStats(self: *MemoryPool) struct {
```

- fn `initLivenessAnalysis`

Initialize liveness analysis


```zig
pub fn initLivenessAnalysis(self: *MemoryPool, config: LivenessConfig) void {
```

- fn `recordBufferAccess`

Record buffer access for liveness analysis


```zig
pub fn recordBufferAccess(self: *MemoryPool, buffer: *PooledBuffer) void {
```

- fn `performLivenessCleanup`

Perform liveness-based cleanup


```zig
pub fn performLivenessCleanup(self: *MemoryPool, current_time: u64) void {
```

- fn `getLivenessStats`

Get liveness statistics


```zig
pub fn getLivenessStats(self: *MemoryPool) struct {
```

- type `Layer`

Neural network layer with enhanced memory safety and mixed precision support


```zig
pub const Layer = struct {
```

- fn `init`

Initialize a new layer with memory pool support


```zig
pub fn init(allocator: std.mem.Allocator, config: LayerConfig, memory_pool: ?*MemoryPool) !*Layer {
```

- fn `initF16`

Initialize f16 versions for mixed precision training


```zig
pub fn initF16(self: *Layer) !void {
```

- fn `syncToF32`

Synchronize f16 weights/biases back to f32 after training


```zig
pub fn syncToF32(self: *Layer) void {
```

- fn `forwardMixed`

Forward pass with mixed precision support


```zig
pub fn forwardMixed(self: *Layer, input: []const f32, use_f16: bool) ![]f32 {
```

- fn `backwardMixed`

Backward pass with mixed precision support


```zig
pub fn backwardMixed(
```

- fn `allocBuffer`

Allocate buffer using memory pool if available, fallback to allocator


```zig
pub fn allocBuffer(self: *Layer, size: usize) ![]f32 {
```

- fn `freeBuffer`

Free buffer using memory pool if available, fallback to allocator


```zig
pub fn freeBuffer(self: *Layer, buffer: []f32) void {
```

- fn `deinit`

Free layer resources with proper cleanup


```zig
pub fn deinit(self: *Layer) void {
```

- fn `forward`

Forward pass through the layer with memory pool support


```zig
pub fn forward(self: *Layer, input: []const f32) ![]f32 {
```

- fn `backward`

Backward pass through the layer with memory pool support


```zig
pub fn backward(
```

- type `CheckpointState`

Gradient checkpointing state


```zig
pub const CheckpointState = struct {
```

- type `NeuralNetwork`

Neural network for learning embeddings with enhanced memory safety


```zig
pub const NeuralNetwork = struct {
```

- fn `init`

Initialize a new neural network with optional memory pool


```zig
pub fn init(allocator: std.mem.Allocator, config: TrainingConfig) !*NeuralNetwork {
```

- fn `initDefault`

Initialize a new neural network with default configuration (backward compatibility)


```zig
pub fn initDefault(allocator: std.mem.Allocator) !*NeuralNetwork {
```

- fn `deinit`

Free network resources with proper cleanup


```zig
pub fn deinit(self: *NeuralNetwork) void {
```

- fn `deinitEnhanced`

Deinitialize with enhanced cleanup (for MemoryPool with liveness analysis)


```zig
pub fn deinitEnhanced(self: *MemoryPool) void {
```

- fn `addLayer`

Add a layer to the network with memory pool support


```zig
pub fn addLayer(self: *NeuralNetwork, config: LayerConfig) !void {
```

- fn `saveToFile`

Save network to file (basic implementation)


```zig
pub fn saveToFile(self: *NeuralNetwork, path: []const u8) !void {
```

- fn `loadFromFile`

Load network from file (basic implementation)


```zig
pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !*NeuralNetwork {
```

- fn `forward`

Forward pass through the network with memory optimization


```zig
pub fn forward(self: *NeuralNetwork, input: []const f32) ![]f32 {
```

- fn `forwardMixed`

Forward pass with mixed precision support


```zig
pub fn forwardMixed(self: *NeuralNetwork, input: []const f32) ![]f32 {
```

- fn `trainStep`

Train the network on a single sample with memory optimization


```zig
pub fn trainStep(
```

- fn `trainStepMixed`

Train the network on a single sample with mixed precision support


```zig
pub fn trainStepMixed(
```

## src/mod.zig

- const `ai`

AI/ML functionality and neural networks


```zig
pub const ai = @import("features/ai/mod.zig");
```

- const `gpu`

GPU acceleration and compute


```zig
pub const gpu = @import("features/gpu/mod.zig");
```

- const `database`

Vector databases and data persistence


```zig
pub const database = @import("features/database/mod.zig");
```

- const `web`

Web servers, HTTP clients, and services


```zig
pub const web = @import("features/web/mod.zig");
```

- const `monitoring`

System monitoring and observability


```zig
pub const monitoring = @import("features/monitoring/mod.zig");
```

- const `connectors`

External service integrations


```zig
pub const connectors = @import("features/connectors/mod.zig");
```

- const `utils`

Cross-cutting utilities and helpers


```zig
pub const utils = @import("shared/utils/mod.zig");
```

- const `core`

Core system functionality


```zig
pub const core = @import("shared/core/mod.zig");
```

- const `platform`

Platform-specific abstractions


```zig
pub const platform = @import("shared/platform/mod.zig");
```

- const `logging`

Logging and telemetry


```zig
pub const logging = @import("shared/logging/mod.zig");
```

- const `simd`

SIMD operations (moved to shared)


```zig
pub const simd = @import("shared/simd.zig");
```

- const `main`

Main application entry point


```zig
pub const main = @import("main.zig");
```

- const `root`

Root configuration


```zig
pub const root = @import("root.zig");
```

- fn `init`

Initialize the entire ABI framework


```zig
pub fn init(allocator: std.mem.Allocator) !void {
```

- fn `deinit`

Shutdown the entire ABI framework


```zig
pub fn deinit() void {
```

- fn `version`

Get framework version information


```zig
pub fn version() []const u8 {
```

## src/root.zig

- const `abi`

```zig
pub const abi = @import("mod.zig");
```

## src/shared/core/config.zig

- type `FrameworkConfig`

Main framework configuration


```zig
pub const FrameworkConfig = struct {
```

- fn `validate`

Validate the configuration


```zig
pub fn validate(self: FrameworkConfig) FrameworkError!void {
```

- fn `default`

Create a default configuration


```zig
pub fn default() FrameworkConfig {
```

- fn `minimal`

Create a minimal configuration for testing


```zig
pub fn minimal() FrameworkConfig {
```

- fn `production`

Create a production configuration


```zig
pub fn production() FrameworkConfig {
```

- type `AgentConfig`

Agent configuration


```zig
pub const AgentConfig = struct {
```

- fn `validate`

Validate agent configuration


```zig
pub fn validate(self: AgentConfig) FrameworkError!void {
```

- type `PersonaType`

Persona types for agents


```zig
pub const PersonaType = enum {
```

- type `AgentCapabilities`

Agent capabilities


```zig
pub const AgentCapabilities = packed struct(u64) {
```

- fn `validate`

Validate capability dependencies


```zig
pub fn validate(self: AgentCapabilities) bool {
```

- type `WebServerConfig`

Web server configuration


```zig
pub const WebServerConfig = struct {
```

- fn `validate`

Validate web server configuration


```zig
pub fn validate(self: WebServerConfig) FrameworkError!void {
```

- type `DatabaseConfig`

Database configuration


```zig
pub const DatabaseConfig = struct {
```

- fn `validate`

Validate database configuration


```zig
pub fn validate(self: DatabaseConfig) FrameworkError!void {
```

- type `IndexAlgorithm`

Index algorithms for vector database


```zig
pub const IndexAlgorithm = enum {
```

- type `PluginConfig`

Plugin configuration


```zig
pub const PluginConfig = struct {
```

- fn `validate`

Validate plugin configuration


```zig
pub fn validate(self: PluginConfig) FrameworkError!void {
```

- type `ConfigLoader`

Configuration loader for loading configurations from files


```zig
pub const ConfigLoader = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `loadFrameworkConfig`

Load framework configuration from JSON file


```zig
pub fn loadFrameworkConfig(self: *Self, path: []const u8) !FrameworkConfig {
```

- fn `saveFrameworkConfig`

Save framework configuration to JSON file


```zig
pub fn saveFrameworkConfig(self: *Self, config: FrameworkConfig, path: []const u8) !void {
```

- fn `loadAgentConfig`

Load agent configuration from JSON file


```zig
pub fn loadAgentConfig(self: *Self, path: []const u8) !AgentConfig {
```

- fn `saveAgentConfig`

Save agent configuration to JSON file


```zig
pub fn saveAgentConfig(self: *Self, config: AgentConfig, path: []const u8) !void {
```

## src/shared/core/core.zig

- const `AbiError`

```zig
pub const AbiError = error{
```

- fn `Result`

```zig
pub fn Result(comptime T: type) type {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) AbiError!void {
```

- fn `deinit`

```zig
pub fn deinit() void {
```

- fn `isInitialized`

```zig
pub fn isInitialized() bool {
```

- fn `getAllocator`

```zig
pub fn getAllocator() std.mem.Allocator {
```

## src/shared/core/errors.zig

- const `AbiError`

Unified error types for the entire framework.


```zig
pub const AbiError = error{
```

- const `FrameworkError`

Alias for legacy compatibility.


```zig
pub const FrameworkError = AbiError;
```

- fn `Result`

Result type for operations that can fail.


```zig
pub fn Result(comptime T: type) type {
```

- fn `ok`

Success result helper.


```zig
pub fn ok(comptime T: type, value: T) Result(T) {
```

- fn `err`

Error result helper.


```zig
pub fn err(comptime T: type, error_type: AbiError) Result(T) {
```

## src/shared/core/framework.zig

- type `Framework`

Main framework instance


```zig
pub const Framework = struct {
```

- fn `init`

Initialize the framework with the given configuration


```zig
pub fn init(allocator: std.mem.Allocator, framework_config: FrameworkConfig) FrameworkError!*Self {
```

- fn `deinit`

Deinitialize the framework and clean up resources


```zig
pub fn deinit(self: *Self) void {
```

- fn `getState`

Get the current framework state


```zig
pub fn getState(self: *const Self) FrameworkState {
```

- fn `getConfig`

Get framework configuration


```zig
pub fn getConfig(self: *const Self) FrameworkConfig {
```

- fn `getComponents`

Get component registry


```zig
pub fn getComponents(self: *Self) *ComponentRegistry {
```

- fn `getMetrics`

Get metrics collector


```zig
pub fn getMetrics(self: *Self) *MetricsCollector {
```

- fn `getLogger`

Get logger


```zig
pub fn getLogger(self: *Self) *Logger {
```

- fn `registerComponent`

Register a component with the framework


```zig
pub fn registerComponent(self: *Self, name: []const u8, component: anytype) !void {
```

- fn `getComponent`

Get a registered component


```zig
pub fn getComponent(self: *Self, name: []const u8) ?anyopaque {
```

- fn `healthCheck`

Health check for the framework


```zig
pub fn healthCheck(self: *const Self) HealthStatus {
```

- type `ComponentRegistry`

Component registry for managing framework components


```zig
pub const ComponentRegistry = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `register`

```zig
pub fn register(self: *Self, name: []const u8, component: anytype) !void {
```

- fn `get`

```zig
pub fn get(self: *Self, name: []const u8) ?*anyopaque {
```

- fn `unregister`

```zig
pub fn unregister(self: *Self, name: []const u8) bool {
```

- fn `list`

```zig
pub fn list(self: *const Self) []const []const u8 {
```

- type `HealthStatus`

Health status for the framework


```zig
pub const HealthStatus = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *HealthStatus) void {
```

- type `ComponentHealth`

Component health status


```zig
pub const ComponentHealth = struct {
```

- type `HealthLevel`

Health levels


```zig
pub const HealthLevel = enum {
```

- type `MetricsCollector`

Metrics collector for framework metrics


```zig
pub const MetricsCollector = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(self: *Self, name: []const u8, value: f64, tags: ?[]const []const u8) !void {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *const Self, name: []const u8) ?Metric {
```

- fn `getAllMetrics`

```zig
pub fn getAllMetrics(self: *const Self) []const Metric {
```

- type `Metric`

Individual metric


```zig
pub const Metric = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- type `Logger`

Logger for framework logging


```zig
pub const Logger = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, level: std.log.Level) !*Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `info`

```zig
pub fn info(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `debug`

```zig
pub fn debug(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `warn`

```zig
pub fn warn(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `err`

```zig
pub fn err(self: *const Self, comptime format: []const u8, args: anytype) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) ErrorHandler {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, _config: FrameworkConfig) ConfigManager {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) LifecycleManager {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*GPUBackendManager {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SIMDOperations {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*MemoryTracker {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceProfiler {
```

## src/shared/core/logging.zig

- type `LogLevel`

Log levels for the lightweight core logger.


```zig
pub const LogLevel = enum(u8) {
```

- fn `toString`

```zig
pub fn toString(self: LogLevel) []const u8 {
```

- type `log`

Lightweight logging system used before the structured logger is configured.


```zig
pub const log = struct {
```

- fn `init`

Initialize logging system.


```zig
pub fn init(alloc: std.mem.Allocator) errors.AbiError!void {
```

- fn `deinit`

Deinitialize logging system.


```zig
pub fn deinit() void {
```

- fn `setLevel`

Set log level.


```zig
pub fn setLevel(level: LogLevel) void {
```

- fn `debug`

Log a debug message.


```zig
pub fn debug(comptime format: []const u8, args: anytype) void {
```

- fn `info`

Log an info message.


```zig
pub fn info(comptime format: []const u8, args: anytype) void {
```

- fn `warn`

Log a warning message.


```zig
pub fn warn(comptime format: []const u8, args: anytype) void {
```

- fn `err`

Log an error message.


```zig
pub fn err(comptime format: []const u8, args: anytype) void {
```

- fn `fatal`

Log a fatal message.


```zig
pub fn fatal(comptime format: []const u8, args: anytype) void {
```

## src/shared/core/mod.zig

- const `core`

```zig
pub const core = @import("core.zig");
```

- const `framework`

```zig
pub const framework = @import("framework.zig");
```

- const `config`

```zig
pub const config = @import("config.zig");
```

- const `lifecycle`

```zig
pub const lifecycle = @import("lifecycle.zig");
```

- const `errors`

```zig
pub const errors = @import("errors.zig");
```

- const `logging`

```zig
pub const logging = @import("logging.zig");
```

- const `mod`

```zig
pub const mod = @import("mod.zig");
```

## src/shared/enhanced_plugin_system.zig

- type `EnhancedPluginSystem`

Enhanced plugin system with production-ready features


```zig
pub const EnhancedPluginSystem = struct {
```

- fn `init`

Initialize the enhanced plugin system


```zig
pub fn init(allocator: std.mem.Allocator) FrameworkError!*Self {
```

- fn `deinit`

Deinitialize the enhanced plugin system


```zig
pub fn deinit(self: *Self) void {
```

- fn `loadPlugin`

Load a plugin from file


```zig
pub fn loadPlugin(self: *Self, path: []const u8) FrameworkError!*Plugin {
```

- fn `unloadPlugin`

Unload a plugin


```zig
pub fn unloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
```

- fn `reloadPlugin`

Reload a plugin


```zig
pub fn reloadPlugin(self: *Self, name: []const u8) FrameworkError!void {
```

- fn `getPlugin`

Get a plugin by name


```zig
pub fn getPlugin(self: *Self, name: []const u8) ?*Plugin {
```

- fn `listPlugins`

List all loaded plugins


```zig
pub fn listPlugins(self: *const Self) []const []const u8 {
```

- fn `getPluginStats`

Get plugin statistics


```zig
pub fn getPluginStats(self: *const Self) PluginStats {
```

- fn `healthCheck`

Health check for all plugins


```zig
pub fn healthCheck(self: *const Self) PluginHealthStatus {
```

- type `PluginSystemConfig`

Plugin system configuration


```zig
pub const PluginSystemConfig = struct {
```

- fn `validate`

```zig
pub fn validate(self: PluginSystemConfig) FrameworkError!void {
```

- type `Plugin`

Enhanced plugin with production-ready features


```zig
pub const Plugin = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8, version: []const u8, description: []const u8, path: []const u8) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `initialize`

```zig
pub fn initialize(self: *Self, allocator: std.mem.Allocator) FrameworkError!void {
```

- fn `start`

```zig
pub fn start(self: *Self) FrameworkError!void {
```

- fn `stop`

```zig
pub fn stop(self: *Self) FrameworkError!void {
```

- fn `deinitialize`

```zig
pub fn deinitialize(self: *Self) void {
```

- fn `healthCheck`

```zig
pub fn healthCheck(self: *const Self) PluginHealth {
```

- fn `addService`

```zig
pub fn addService(self: *Self, service: PluginService) FrameworkError!void {
```

- fn `removeService`

```zig
pub fn removeService(self: *Self, service_name: []const u8) bool {
```

- fn `getService`

```zig
pub fn getService(self: *const Self, service_name: []const u8) ?PluginService {
```

- type `PluginState`

Plugin state management


```zig
pub const PluginState = enum {
```

- fn `canTransitionTo`

```zig
pub fn canTransitionTo(self: PluginState, new_state: PluginState) bool {
```

- type `PluginService`

Plugin service definition


```zig
pub const PluginService = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8, version: []const u8, description: []const u8, capabilities: PluginCapabilities, handler: *const fn ([]const u8) anyerror![]const u8) PluginService {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginService) void {
```

- type `PluginCapabilities`

Plugin capabilities


```zig
pub const PluginCapabilities = struct {
```

- type `PluginPerformanceMetrics`

Plugin performance metrics


```zig
pub const PluginPerformanceMetrics = struct {
```

- fn `updateResponseTime`

```zig
pub fn updateResponseTime(self: *PluginPerformanceMetrics, response_time_ms: f64) void {
```

- fn `recordSuccess`

```zig
pub fn recordSuccess(self: *PluginPerformanceMetrics) void {
```

- fn `recordFailure`

```zig
pub fn recordFailure(self: *PluginPerformanceMetrics) void {
```

- fn `getSuccessRate`

```zig
pub fn getSuccessRate(self: *const PluginPerformanceMetrics) f32 {
```

- type `PluginHealth`

Plugin health status


```zig
pub const PluginHealth = struct {
```

- type `HealthStatus`

Health status levels


```zig
pub const HealthStatus = enum {
```

- type `PluginStats`

Plugin statistics


```zig
pub const PluginStats = struct {
```

- type `PluginHealthStatus`

Plugin health status for all plugins


```zig
pub const PluginHealthStatus = struct {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginHealthStatus) void {
```

- type `PluginEventType`

Plugin event types


```zig
pub const PluginEventType = enum {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *PluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*NativePluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *NativePluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *NativePluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ScriptPluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *ScriptPluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *ScriptPluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*WebPluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *WebPluginLoader) void {
```

- fn `loadPlugin`

```zig
pub fn loadPlugin(self: *WebPluginLoader, path: []const u8) !*Plugin {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginRegistry) void {
```

- fn `registerPlugin`

```zig
pub fn registerPlugin(self: *PluginRegistry, plugin: *Plugin) !void {
```

- fn `unregisterPlugin`

```zig
pub fn unregisterPlugin(self: *PluginRegistry, name: []const u8) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginWatcher {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginWatcher) void {
```

- fn `start`

```zig
pub fn start(self: *PluginWatcher, directory: []const u8) !void {
```

- fn `stop`

```zig
pub fn stop(self: *PluginWatcher) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PluginManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginManager) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*ServiceDiscovery {
```

- fn `deinit`

```zig
pub fn deinit(self: *ServiceDiscovery) void {
```

- fn `registerService`

```zig
pub fn registerService(self: *ServiceDiscovery, service: PluginService) !void {
```

- fn `unregisterService`

```zig
pub fn unregisterService(self: *ServiceDiscovery, name: []const u8) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SecurityManager {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityManager) void {
```

- fn `validatePluginPath`

```zig
pub fn validatePluginPath(self: *SecurityManager, path: []const u8) !void {
```

- fn `validatePlugin`

```zig
pub fn validatePlugin(self: *SecurityManager, plugin: *Plugin) !void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SecurityContext {
```

- fn `deinit`

```zig
pub fn deinit(self: *SecurityContext) void {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `validatePlugin`

```zig
pub fn validatePlugin(self: *PerformanceMonitor, plugin: *Plugin) !void {
```

## src/shared/interface.zig

- type `PluginInterface`

Standard plugin interface using C-compatible function pointers
This vtable approach ensures compatibility across different compilation units


```zig
pub const PluginInterface = extern struct {
```

- fn `isValid`

```zig
pub fn isValid(self: *const PluginInterface) bool {
```

- type `Plugin`

Plugin wrapper that provides a safer Zig API around the C interface


```zig
pub const Plugin = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, interface: *const PluginInterface) !Plugin {
```

- fn `deinit`

```zig
pub fn deinit(self: *Plugin) void {
```

- fn `getInfo`

```zig
pub fn getInfo(self: *Plugin) *const PluginInfo {
```

- fn `initialize`

```zig
pub fn initialize(self: *Plugin, config: *PluginConfig) !void {
```

- fn `start`

```zig
pub fn start(self: *Plugin) !void {
```

- fn `stop`

```zig
pub fn stop(self: *Plugin) !void {
```

- fn `pause`

```zig
pub fn pause(self: *Plugin) !void {
```

- fn `resumePlugin`

```zig
pub fn resumePlugin(self: *Plugin) !void {
```

- fn `process`

```zig
pub fn process(self: *Plugin, input: ?*anyopaque, output: ?*anyopaque) !void {
```

- fn `configure`

```zig
pub fn configure(self: *Plugin, config: *const PluginConfig) !void {
```

- fn `getStatus`

```zig
pub fn getStatus(self: *Plugin) i32 {
```

- fn `getMetrics`

```zig
pub fn getMetrics(self: *Plugin, buffer: []u8) !usize {
```

- fn `onEvent`

```zig
pub fn onEvent(self: *Plugin, event_type: u32, event_data: ?*anyopaque) !void {
```

- fn `getApi`

```zig
pub fn getApi(self: *Plugin, api_name: [:0]const u8) ?*anyopaque {
```

- fn `getState`

```zig
pub fn getState(self: *const Plugin) PluginState {
```

- fn `setState`

```zig
pub fn setState(self: *Plugin, new_state: PluginState) !void {
```

- const `PluginFactoryFn`

Plugin factory function type


```zig
pub const PluginFactoryFn = *const fn () callconv(.c) ?*const PluginInterface;
```

- const `PLUGIN_ENTRY_POINT`

Standard plugin entry point function name


```zig
pub const PLUGIN_ENTRY_POINT = "abi_plugin_create";
```

- const `PLUGIN_ABI_VERSION`

ABI version for plugin compatibility


```zig
pub const PLUGIN_ABI_VERSION = types.PluginVersion.init(1, 0, 0);
```

- fn `createPlugin`

Create a plugin from a loaded interface


```zig
pub fn createPlugin(allocator: std.mem.Allocator, interface: *const PluginInterface) !*Plugin {
```

- fn `destroyPlugin`

Destroy a plugin instance


```zig
pub fn destroyPlugin(allocator: std.mem.Allocator, plugin: *Plugin) void {
```

## src/shared/loader.zig

- type `PluginLoader`

Cross-platform plugin loader


```zig
pub const PluginLoader = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) PluginLoader {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginLoader) void {
```

- fn `addPluginPath`

Add a directory to search for plugins


```zig
pub fn addPluginPath(self: *PluginLoader, path: []const u8) !void {
```

- fn `removePluginPath`

Remove a plugin search path


```zig
pub fn removePluginPath(self: *PluginLoader, path: []const u8) void {
```

- fn `discoverPlugins`

Discover plugins in the search paths


```zig
pub fn discoverPlugins(self: *PluginLoader) !std.ArrayList([]const u8) {
```

- fn `loadPlugin`

Load a plugin from a file path


```zig
pub fn loadPlugin(self: *PluginLoader, plugin_path: []const u8) !*const PluginInterface {
```

- fn `unloadPlugin`

Unload a plugin


```zig
pub fn unloadPlugin(self: *PluginLoader, plugin_path: []const u8) !void {
```

- fn `getLoadedPlugins`

Get the list of loaded plugins


```zig
pub fn getLoadedPlugins(self: *PluginLoader) []const LoadedLibrary {
```

- fn `makeLibraryName`

Construct a platform-specific library filename


```zig
pub fn makeLibraryName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
```

- fn `createLoader`

Create a plugin loader instance


```zig
pub fn createLoader(allocator: std.mem.Allocator) PluginLoader {
```

## src/shared/logging/logging.zig

- type `LogLevel`

Log level enumeration


```zig
pub const LogLevel = enum(u8) {
```

- fn `toString`

```zig
pub fn toString(self: LogLevel) []const u8 {
```

- fn `color`

```zig
pub fn color(self: LogLevel) []const u8 {
```

- type `OutputFormat`

Output format enumeration


```zig
pub const OutputFormat = enum {
```

- type `LoggerConfig`

Logger configuration


```zig
pub const LoggerConfig = struct {
```

- type `Logger`

Structured Logger


```zig
pub const Logger = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: LoggerConfig) !*Logger {
```

- fn `deinit`

```zig
pub fn deinit(self: *Logger) void {
```

- fn `log`

Log a message with structured fields


```zig
pub fn log(
```

- fn `trace`

Convenience methods for different log levels


```zig
pub fn trace(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `debug`

```zig
pub fn debug(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `info`

```zig
pub fn info(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `warn`

```zig
pub fn warn(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `err`

```zig
pub fn err(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `fatal`

```zig
pub fn fatal(self: *Logger, comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `initGlobalLogger`

Initialize global logger


```zig
pub fn initGlobalLogger(allocator: std.mem.Allocator, config: LoggerConfig) !void {
```

- fn `deinitGlobalLogger`

Deinitialize global logger


```zig
pub fn deinitGlobalLogger() void {
```

- fn `getGlobalLogger`

Get global logger instance


```zig
pub fn getGlobalLogger() ?*Logger {
```

- fn `log`

Global logging functions


```zig
pub fn log(
```

- fn `trace`

```zig
pub fn trace(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `debug`

```zig
pub fn debug(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `info`

```zig
pub fn info(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `warn`

```zig
pub fn warn(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `err`

```zig
pub fn err(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

- fn `fatal`

```zig
pub fn fatal(comptime message: []const u8, fields: anytype, src: std.builtin.SourceLocation) !void {
```

## src/shared/logging/mod.zig

- const `logging`

```zig
pub const logging = @import("logging.zig");
```

## src/shared/mod.zig

- const `interface`

```zig
pub const interface = @import("interface.zig");
```

- const `loader`

```zig
pub const loader = @import("loader.zig");
```

- const `registry`

```zig
pub const registry = @import("registry.zig");
```

- const `types`

```zig
pub const types = @import("types.zig");
```

- const `Plugin`

```zig
pub const Plugin = interface.Plugin;
```

- const `PluginInterface`

```zig
pub const PluginInterface = interface.PluginInterface;
```

- const `PluginLoader`

```zig
pub const PluginLoader = loader.PluginLoader;
```

- const `PluginRegistry`

```zig
pub const PluginRegistry = registry.PluginRegistry;
```

- const `PluginError`

```zig
pub const PluginError = types.PluginError;
```

- const `PluginType`

```zig
pub const PluginType = types.PluginType;
```

- const `PluginInfo`

```zig
pub const PluginInfo = types.PluginInfo;
```

- const `PluginConfig`

```zig
pub const PluginConfig = types.PluginConfig;
```

- const `createLoader`

```zig
pub const createLoader = loader.createLoader;
```

- const `createRegistry`

```zig
pub const createRegistry = registry.createRegistry;
```

- const `registerBuiltinInterface`

```zig
pub const registerBuiltinInterface = registry.registerBuiltinInterface;
```

- fn `init`

Initialize the plugin system


```zig
pub fn init(allocator: std.mem.Allocator) !PluginRegistry {
```

- type `VERSION`

Plugin system version


```zig
pub const VERSION = struct {
```

- const `MAJOR`

```zig
pub const MAJOR = 1;
```

- const `MINOR`

```zig
pub const MINOR = 0;
```

- const `PATCH`

```zig
pub const PATCH = 0;
```

- fn `string`

```zig
pub fn string() []const u8 {
```

- fn `isCompatible`

```zig
pub fn isCompatible(major: u32, minor: u32) bool {
```

## src/shared/platform/mod.zig

- const `platform`

```zig
pub const platform = @import("platform.zig");
```

## src/shared/platform/platform.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- type `PlatformInfo`

Platform capabilities and configuration


```zig
pub const PlatformInfo = struct {
```

- fn `detect`

```zig
pub fn detect() PlatformInfo {
```

- fn `initializePlatform`

Platform-specific initialization


```zig
pub fn initializePlatform() !void {
```

- type `FileOps`

Cross-platform file operations


```zig
pub const FileOps = struct {
```

- fn `openFile`

```zig
pub fn openFile(path: []const u8) !std.fs.File {
```

- fn `createFile`

```zig
pub fn createFile(path: []const u8) !std.fs.File {
```

- fn `deleteFile`

```zig
pub fn deleteFile(path: []const u8) !void {
```

- fn `fileExists`

```zig
pub fn fileExists(path: []const u8) bool {
```

- type `MemoryOps`

Cross-platform memory operations


```zig
pub const MemoryOps = struct {
```

- fn `getPageSize`

```zig
pub fn getPageSize() usize {
```

- fn `alignToPageSize`

```zig
pub fn alignToPageSize(size: usize) usize {
```

- fn `getVirtualMemoryLimit`

```zig
pub fn getVirtualMemoryLimit() usize {
```

- type `ThreadOps`

Cross-platform threading utilities


```zig
pub const ThreadOps = struct {
```

- fn `getOptimalThreadCount`

```zig
pub fn getOptimalThreadCount() u32 {
```

- fn `setThreadPriority`

```zig
pub fn setThreadPriority(thread: std.Thread, priority: ThreadPriority) !void {
```

- type `ThreadPriority`

```zig
pub const ThreadPriority = enum {
```

- type `PerfOps`

Cross-platform performance utilities


```zig
pub const PerfOps = struct {
```

- fn `getCpuFrequency`

```zig
pub fn getCpuFrequency() u64 {
```

- fn `getCacheInfo`

```zig
pub fn getCacheInfo() CacheInfo {
```

- type `CacheInfo`

```zig
pub const CacheInfo = struct {
```

- type `Colors`

ANSI color support


```zig
pub const Colors = struct {
```

- const `reset`

```zig
pub const reset = "\x1b[0m";
```

- const `bold`

```zig
pub const bold = "\x1b[1m";
```

- const `red`

```zig
pub const red = "\x1b[31m";
```

- const `green`

```zig
pub const green = "\x1b[32m";
```

- const `yellow`

```zig
pub const yellow = "\x1b[33m";
```

- const `blue`

```zig
pub const blue = "\x1b[34m";
```

- const `magenta`

```zig
pub const magenta = "\x1b[35m";
```

- const `cyan`

```zig
pub const cyan = "\x1b[36m";
```

- const `white`

```zig
pub const white = "\x1b[37m";
```

- fn `print`

```zig
pub fn print(comptime color: []const u8, comptime fmt: []const u8, args: anytype) void {
```

- const `PlatformError`

Platform-specific error handling


```zig
pub const PlatformError = error{
```

- fn `getTempDir`

Get platform-specific temporary directory


```zig
pub fn getTempDir(allocator: std.mem.Allocator) ![]const u8 {
```

- fn `sleep`

Platform-specific sleep function


```zig
pub fn sleep(milliseconds: u64) void {
```

- fn `getSystemInfo`

Get system information as a formatted string


```zig
pub fn getSystemInfo(allocator: std.mem.Allocator) ![]const u8 {
```

## src/shared/registry.zig

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, plugin: *Plugin, load_order: u32) PluginEntry {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginEntry, allocator: std.mem.Allocator) void {
```

- type `PluginRegistry`

Centralized plugin registry


```zig
pub const PluginRegistry = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !PluginRegistry {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginRegistry) void {
```

- fn `addPluginPath`

Add a search path for plugins


```zig
pub fn addPluginPath(self: *PluginRegistry, path: []const u8) !void {
```

- fn `discoverPlugins`

Discover plugins in search paths


```zig
pub fn discoverPlugins(self: *PluginRegistry) !std.ArrayList([]const u8) {
```

- fn `loadPlugin`

Load a plugin from file


```zig
pub fn loadPlugin(self: *PluginRegistry, plugin_path: []const u8) !void {
```

- fn `unloadPlugin`

Unload a plugin


```zig
pub fn unloadPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
```

- fn `registerBuiltinInterface`

Register a plugin interface that is built into the process (no dynamic library)


```zig
pub fn registerBuiltinInterface(self: *PluginRegistry, plugin_interface: *const interface.PluginInterface) !void {
```

- fn `initializePlugin`

Initialize a plugin with configuration


```zig
pub fn initializePlugin(self: *PluginRegistry, plugin_name: []const u8, config: ?*PluginConfig) !void {
```

- fn `startPlugin`

Start a plugin


```zig
pub fn startPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
```

- fn `stopPlugin`

Stop a plugin


```zig
pub fn stopPlugin(self: *PluginRegistry, plugin_name: []const u8) !void {
```

- fn `startAllPlugins`

Start all plugins in dependency order


```zig
pub fn startAllPlugins(self: *PluginRegistry) !void {
```

- fn `stopAllPlugins`

Stop all plugins in reverse order


```zig
pub fn stopAllPlugins(self: *PluginRegistry) !void {
```

- fn `getPlugin`

Get plugin by name


```zig
pub fn getPlugin(self: *PluginRegistry, plugin_name: []const u8) ?*Plugin {
```

- fn `getPluginsByType`

Get plugins by type


```zig
pub fn getPluginsByType(self: *PluginRegistry, plugin_type: PluginType) !std.ArrayList(*Plugin) {
```

- fn `getPluginNames`

Get all plugin names


```zig
pub fn getPluginNames(self: *PluginRegistry) !std.ArrayList([]const u8) {
```

- fn `getPluginCount`

Get plugin count


```zig
pub fn getPluginCount(self: *PluginRegistry) usize {
```

- fn `getPluginInfo`

Get plugin information


```zig
pub fn getPluginInfo(self: *PluginRegistry, plugin_name: []const u8) ?*const PluginInfo {
```

- fn `configurePlugin`

Configure a plugin


```zig
pub fn configurePlugin(self: *PluginRegistry, plugin_name: []const u8, config: *const PluginConfig) !void {
```

- fn `broadcastEvent`

Send event to all interested plugins


```zig
pub fn broadcastEvent(self: *PluginRegistry, event_type: u32, event_data: ?*anyopaque) !void {
```

- fn `registerEventHandler`

Register an event handler


```zig
pub fn registerEventHandler(self: *PluginRegistry, event_type: u32, handler_fn: *const fn (event_data: ?*anyopaque) void) !void {
```

- fn `createRegistry`

Create a plugin registry instance


```zig
pub fn createRegistry(allocator: std.mem.Allocator) !PluginRegistry {
```

## src/shared/simd.zig

- fn `distance`

SIMD-accelerated operations for high‑performance computing
Simple utility functions for tests and basic API
Distance between two equal‑length slices


```zig
pub fn distance(a: []const f32, b: []const f32) f32 {
```

- fn `cosineSimilarity`

Cosine similarity between two equal‑length slices


```zig
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
```

- type `PerformanceMonitor`

Dummy performance monitor struct for tests


```zig
pub const PerformanceMonitor = struct {};
```

- fn `getPerformanceMonitor`

```zig
pub fn getPerformanceMonitor() PerformanceMonitor {
```

- type `SIMDOpts`

```zig
pub const SIMDOpts = struct {
```

- type `VectorOps`

SIMD-accelerated vector operations


```zig
pub const VectorOps = struct {
```

- fn `dotProduct`

SIMD-accelerated dot product


```zig
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
```

- fn `matrixVectorMultiply`

SIMD-accelerated matrix‑vector multiplication


```zig
pub fn matrixVectorMultiply(matrix: []const f32, vector: []const f32, result: []f32, rows: usize, cols: usize) void {
```

- fn `vectorizedRelu`

SIMD-accelerated ReLU activation


```zig
pub fn vectorizedRelu(data: []f32) void {
```

- fn `vectorizedLeakyRelu`

SIMD-accelerated Leaky ReLU activation


```zig
pub fn vectorizedLeakyRelu(data: []f32, slope: f32) void {
```

- fn `vectorMax`

SIMD-accelerated element‑wise maximum


```zig
pub fn vectorMax(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `vectorAdd`

SIMD-accelerated element‑wise addition


```zig
pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `vectorMul`

SIMD-accelerated element‑wise multiplication


```zig
pub fn vectorMul(a: []const f32, b: []const f32, result: []f32) void {
```

- fn `normalize`

SIMD-accelerated vector normalization (L2 norm)


```zig
pub fn normalize(data: []f32, result: []f32) void {
```

- fn `matrixMultiply`

SIMD-accelerated matrix multiplication (basic implementation)


```zig
pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, rows: usize, cols: usize, inner_dim: usize) void {
```

- fn `shouldUseSimd`

Check if SIMD is available and beneficial for given size


```zig
pub fn shouldUseSimd(size: usize) bool {
```

## src/shared/types.zig

- const `PluginError`

Plugin system errors


```zig
pub const PluginError = error{
```

- type `PluginType`

Plugin types supported by the system


```zig
pub const PluginType = enum {
```

- fn `toString`

```zig
pub fn toString(self: PluginType) []const u8 {
```

- fn `fromString`

```zig
pub fn fromString(s: []const u8) ?PluginType {
```

- type `PluginVersion`

Plugin version information


```zig
pub const PluginVersion = struct {
```

- fn `init`

```zig
pub fn init(major: u32, minor: u32, patch: u32) PluginVersion {
```

- fn `isCompatible`

```zig
pub fn isCompatible(self: PluginVersion, required: PluginVersion) bool {
```

- fn `format`

```zig
pub fn format(
```

- type `PluginInfo`

Plugin metadata and information


```zig
pub const PluginInfo = struct {
```

- fn `isCompatible`

```zig
pub fn isCompatible(self: PluginInfo, framework_abi: PluginVersion) bool {
```

- type `PluginConfig`

Plugin configuration


```zig
pub const PluginConfig = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) PluginConfig {
```

- fn `deinit`

```zig
pub fn deinit(self: *PluginConfig) void {
```

- fn `setParameter`

```zig
pub fn setParameter(self: *PluginConfig, key: []const u8, value: []const u8) !void {
```

- fn `getParameter`

```zig
pub fn getParameter(self: *PluginConfig, key: []const u8) ?[]const u8 {
```

- type `PluginState`

Plugin state tracking


```zig
pub const PluginState = enum {
```

- fn `toString`

```zig
pub fn toString(self: PluginState) []const u8 {
```

- fn `canTransitionTo`

```zig
pub fn canTransitionTo(self: PluginState, new_state: PluginState) bool {
```

- type `PluginContext`

Plugin execution context


```zig
pub const PluginContext = struct {
```

- fn `log`

```zig
pub fn log(self: *PluginContext, level: u8, message: []const u8) void {
```

- fn `getService`

```zig
pub fn getService(self: *PluginContext, service_name: []const u8) ?*anyopaque {
```

## src/shared/utils/http/mod.zig

- type `HttpStatus`

HTTP status codes as defined in RFC 7231


```zig
pub const HttpStatus = enum(u16) {
```

- fn `phrase`

Get the human-readable phrase for this status code


```zig
pub fn phrase(self: HttpStatus) []const u8 {
```

- fn `code`

Get the numeric code for this status


```zig
pub fn code(self: HttpStatus) u16 {
```

- fn `isSuccess`

Check if this is a 2xx success status


```zig
pub fn isSuccess(self: HttpStatus) bool {
```

- fn `isRedirect`

Check if this is a 3xx redirection status


```zig
pub fn isRedirect(self: HttpStatus) bool {
```

- fn `isClientError`

Check if this is a 4xx client error status


```zig
pub fn isClientError(self: HttpStatus) bool {
```

- fn `isServerError`

Check if this is a 5xx server error status


```zig
pub fn isServerError(self: HttpStatus) bool {
```

- fn `isError`

Check if this is any kind of error status (4xx or 5xx)


```zig
pub fn isError(self: HttpStatus) bool {
```

- type `HttpMethod`

HTTP method types as defined in RFC 7231


```zig
pub const HttpMethod = enum {
```

- fn `fromString`

Parse an HTTP method from a string (case-insensitive)


```zig
pub fn fromString(method: []const u8) ?HttpMethod {
```

- fn `toString`

Convert HTTP method to string representation


```zig
pub fn toString(self: HttpMethod) []const u8 {
```

- fn `isSafe`

Check if this method is considered "safe" (RFC 7231)
Safe methods do not modify server state


```zig
pub fn isSafe(self: HttpMethod) bool {
```

- fn `isIdempotent`

Check if this method is idempotent (RFC 7231)
Idempotent methods can be called multiple times with the same effect


```zig
pub fn isIdempotent(self: HttpMethod) bool {
```

- fn `allowsBody`

Check if this method typically allows a request body


```zig
pub fn allowsBody(self: HttpMethod) bool {
```

- type `Headers`

HTTP header management with case-insensitive operations


```zig
pub const Headers = struct {
```

- fn `init`

Initialize a new Headers instance


```zig
pub fn init(allocator: std.mem.Allocator) Headers {
```

- fn `deinit`

Clean up resources used by Headers


```zig
pub fn deinit(self: *Headers) void {
```

- fn `set`

Set a header value (overwrites existing values)


```zig
pub fn set(self: *Headers, name: []const u8, value: []const u8) !void {
```

- fn `get`

Get a header value


```zig
pub fn get(self: *Headers, name: []const u8) ?[]const u8 {
```

- fn `remove`

Remove a header


```zig
pub fn remove(self: *Headers, name: []const u8) bool {
```

- fn `getOr`

Get a header value or return a default if not found


```zig
pub fn getOr(self: *Headers, name: []const u8, default_value: []const u8) []const u8 {
```

- type `HttpRequest`

HTTP request structure


```zig
pub const HttpRequest = struct {
```

- fn `init`

Initialize a new HTTP request


```zig
pub fn init(allocator: std.mem.Allocator, method: HttpMethod, path: []const u8) HttpRequest {
```

- fn `deinit`

Clean up resources used by the request


```zig
pub fn deinit(self: *HttpRequest) void {
```

- type `HttpResponse`

HTTP response structure


```zig
pub const HttpResponse = struct {
```

- fn `init`

Initialize a new HTTP response


```zig
pub fn init(allocator: std.mem.Allocator, status: HttpStatus) HttpResponse {
```

- fn `deinit`

Clean up resources used by the response


```zig
pub fn deinit(self: *HttpResponse) void {
```

- fn `setContentType`

Set the Content-Type header


```zig
pub fn setContentType(self: *HttpResponse, content_type: []const u8) !void {
```

- fn `setJson`

Set Content-Type to application/json


```zig
pub fn setJson(self: *HttpResponse) !void {
```

- fn `setText`

Set Content-Type to text/plain


```zig
pub fn setText(self: *HttpResponse) !void {
```

- fn `setHtml`

Set Content-Type to text/html


```zig
pub fn setHtml(self: *HttpResponse) !void {
```

## src/shared/utils/json/mod.zig

- const `JsonValue`

Simple JSON value types for basic JSON operations


```zig
pub const JsonValue = union(enum) {
```

- fn `deinit`

Clean up resources used by this JsonValue


```zig
pub fn deinit(self: *const JsonValue, allocator: std.mem.Allocator) void {
```

- type `JsonUtils`

JSON utility functions for parsing and serialization


```zig
pub const JsonUtils = struct {
```

- fn `parse`

Parse JSON string into JsonValue


```zig
pub fn parse(allocator: std.mem.Allocator, json_str: []const u8) !JsonValue {
```

- fn `stringify`

Serialize JsonValue to JSON string (simplified version)


```zig
pub fn stringify(allocator: std.mem.Allocator, value: JsonValue) ![]u8 {
```

- fn `parseInto`

Parse JSON string into typed struct using std.json


```zig
pub fn parseInto(allocator: std.mem.Allocator, comptime T: type, json_str: []const u8) !T {
```

- fn `stringifyFrom`

Serialize struct to JSON string (simplified)


```zig
pub fn stringifyFrom(allocator: std.mem.Allocator, value: anytype) ![]u8 {
```

- type `JsonOps`

High-level JSON operations for common use cases


```zig
pub const JsonOps = struct {
```

- fn `getValue`

Extract a value from a JSON object by key path (dot notation)


```zig
pub fn getValue(json: JsonValue, path: []const u8) ?JsonValue {
```

- fn `getString`

Get a string value from JSON by path


```zig
pub fn getString(json: JsonValue, path: []const u8) ?[]const u8 {
```

- fn `getInt`

Get an integer value from JSON by path


```zig
pub fn getInt(json: JsonValue, path: []const u8) ?i64 {
```

- fn `getBool`

Get a boolean value from JSON by path


```zig
pub fn getBool(json: JsonValue, path: []const u8) ?bool {
```

## src/shared/utils/math/mod.zig

- type `MathUtils`

Basic mathematical utility functions


```zig
pub const MathUtils = struct {
```

- fn `clamp`

Clamp value between min and max


```zig
pub fn clamp(comptime T: type, value: T, min: T, max: T) T {
```

- fn `lerp`

Linear interpolation between a and b


```zig
pub fn lerp(a: f64, b: f64, t: f64) f64 {
```

- fn `percentage`

Calculate percentage (value / total * 100)


```zig
pub fn percentage(value: f64, total: f64) f64 {
```

- fn `roundToDecimal`

Round to specified decimal places


```zig
pub fn roundToDecimal(value: f64, decimals: usize) f64 {
```

- fn `isPowerOfTwo`

Check if number is power of 2


```zig
pub fn isPowerOfTwo(value: usize) bool {
```

- fn `nextPowerOfTwo`

Find next power of 2 greater than or equal to value


```zig
pub fn nextPowerOfTwo(value: usize) usize {
```

- fn `factorial`

Calculate factorial


```zig
pub fn factorial(n: u64) u64 {
```

- fn `gcd`

Calculate greatest common divisor (GCD)


```zig
pub fn gcd(a: usize, b: usize) usize {
```

- fn `lcm`

Calculate least common multiple (LCM)


```zig
pub fn lcm(a: usize, b: usize) usize {
```

- type `Statistics`

Statistical calculation functions


```zig
pub const Statistics = struct {
```

- fn `mean`

Calculate mean (average)


```zig
pub fn mean(values: []const f64) f64 {
```

- fn `standardDeviation`

Calculate standard deviation


```zig
pub fn standardDeviation(values: []const f64) f64 {
```

- fn `median`

Calculate median


```zig
pub fn median(allocator: std.mem.Allocator, values: []const f64) !f64 {
```

- fn `variance`

Calculate variance


```zig
pub fn variance(values: []const f64) f64 {
```

- fn `min`

Calculate minimum value


```zig
pub fn min(values: []const f64) f64 {
```

- fn `max`

Calculate maximum value


```zig
pub fn max(values: []const f64) f64 {
```

- fn `range`

Calculate range (max - min)


```zig
pub fn range(values: []const f64) f64 {
```

- type `Geometry`

Geometric and distance calculation functions


```zig
pub const Geometry = struct {
```

- fn `distance2D`

Calculate distance between two points (2D)


```zig
pub fn distance2D(x1: f64, y1: f64, x2: f64, y2: f64) f64 {
```

- fn `distance3D`

Calculate distance between two points (3D)


```zig
pub fn distance3D(x1: f64, y1: f64, z1: f64, x2: f64, y2: f64, z2: f64) f64 {
```

- fn `distance`

Calculate Euclidean distance (N-dimensional)


```zig
pub fn distance(a: []const f64, b: []const f64) f64 {
```

- fn `manhattanDistance`

Calculate Manhattan distance (L1 norm)


```zig
pub fn manhattanDistance(a: []const f64, b: []const f64) f64 {
```

- fn `cosineSimilarity`

Calculate cosine similarity


```zig
pub fn cosineSimilarity(a: []const f64, b: []const f64) f64 {
```

- type `Angles`

Angular conversion utilities


```zig
pub const Angles = struct {
```

- fn `degreesToRadians`

Convert degrees to radians


```zig
pub fn degreesToRadians(degrees: f64) f64 {
```

- fn `radiansToDegrees`

Convert radians to degrees


```zig
pub fn radiansToDegrees(radians: f64) f64 {
```

- fn `normalizeRadians`

Normalize angle to [0, 2π) range


```zig
pub fn normalizeRadians(angle: f64) f64 {
```

- fn `normalizeDegrees`

Normalize angle to [-180, 180] degree range


```zig
pub fn normalizeDegrees(angle: f64) f64 {
```

- type `Random`

Random number generation utilities


```zig
pub const Random = struct {
```

- fn `intRange`

Generate random integer in range [min, max)


```zig
pub fn intRange(random: std.rand.Random, min: i64, max: i64) i64 {
```

- fn `floatRange`

Generate random float in range [min, max)


```zig
pub fn floatRange(random: std.rand.Random, min: f64, max: f64) f64 {
```

- fn `boolean`

Generate random boolean with given probability


```zig
pub fn boolean(random: std.rand.Random, probability: f64) bool {
```

- fn `choice`

Select random element from slice


```zig
pub fn choice(comptime T: type, random: std.rand.Random, items: []const T) ?T {
```

- fn `shuffle`

Shuffle slice in place


```zig
pub fn shuffle(comptime T: type, random: std.rand.Random, items: []T) void {
```

## src/shared/utils/mod.zig

- const `http`

```zig
pub const http = @import("http/mod.zig");
```

- const `json`

```zig
pub const json = @import("json/mod.zig");
```

- const `string`

```zig
pub const string = @import("string/mod.zig");
```

- const `math`

```zig
pub const math = @import("math/mod.zig");
```

- const `encoding`

```zig
pub const encoding = @import("encoding/mod.zig");
```

- const `fs`

```zig
pub const fs = @import("fs/mod.zig");
```

- const `net`

```zig
pub const net = @import("net/mod.zig");
```

- const `crypto`

```zig
pub const crypto = @import("crypto/mod.zig");
```

- const `utils`

```zig
pub const utils = @import("utils.zig");
```

## src/shared/utils/string/mod.zig

- type `StringUtils`

String manipulation utilities


```zig
pub const StringUtils = struct {
```

- fn `isEmptyOrWhitespace`

Check if string is empty or whitespace only


```zig
pub fn isEmptyOrWhitespace(str: []const u8) bool {
```

- fn `toLower`

Convert string to lowercase (allocates new string)


```zig
pub fn toLower(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
```

- fn `toUpper`

Convert string to uppercase (allocates new string)


```zig
pub fn toUpper(allocator: std.mem.Allocator, str: []const u8) ![]u8 {
```

- fn `trim`

Trim whitespace from both ends of string


```zig
pub fn trim(str: []const u8) []const u8 {
```

- fn `split`

Split string by delimiter and return ArrayList


```zig
pub fn split(allocator: std.mem.Allocator, str: []const u8, delimiter: []const u8) !std.ArrayList([]const u8) {
```

- fn `join`

Join array of strings with delimiter


```zig
pub fn join(allocator: std.mem.Allocator, strings: []const []const u8, delimiter: []const u8) ![]u8 {
```

- fn `startsWith`

Check if string starts with prefix


```zig
pub fn startsWith(str: []const u8, prefix: []const u8) bool {
```

- fn `endsWith`

Check if string ends with suffix


```zig
pub fn endsWith(str: []const u8, suffix: []const u8) bool {
```

- fn `replace`

Replace all occurrences of a substring


```zig
pub fn replace(allocator: std.mem.Allocator, str: []const u8, old: []const u8, new: []const u8) ![]u8 {
```

- type `ArrayUtils`

Array manipulation utilities


```zig
pub const ArrayUtils = struct {
```

- fn `contains`

Check if array contains element


```zig
pub fn contains(comptime T: type, haystack: []const T, needle: T) bool {
```

- fn `indexOf`

Find index of element in array


```zig
pub fn indexOf(comptime T: type, haystack: []const T, needle: T) ?usize {
```

- fn `removeAt`

Remove element at index (shifts remaining elements)


```zig
pub fn removeAt(comptime T: type, array: []T, index: usize) void {
```

- fn `insertAt`

Insert element at index (shifts elements to make room)


```zig
pub fn insertAt(comptime T: type, array: []T, index: usize, value: T) void {
```

- fn `reverse`

Reverse array in place


```zig
pub fn reverse(comptime T: type, array: []T) void {
```

- fn `fill`

Fill array with value


```zig
pub fn fill(comptime T: type, array: []T, value: T) void {
```

- fn `count`

Count occurrences of element in array


```zig
pub fn count(comptime T: type, array: []const T, value: T) usize {
```

- type `TimeUtils`

Time and duration utilities


```zig
pub const TimeUtils = struct {
```

- fn `nowMs`

Get current timestamp in milliseconds


```zig
pub fn nowMs() i64 {
```

- fn `nowUs`

Get current timestamp in microseconds


```zig
pub fn nowUs() i64 {
```

- fn `nowNs`

Get current timestamp in nanoseconds


```zig
pub fn nowNs() i64 {
```

- fn `formatDuration`

Format duration in human readable format


```zig
pub fn formatDuration(allocator: std.mem.Allocator, duration_ns: u64) ![]u8 {
```

- fn `parseDuration`

Parse duration string (e.g., "1.5s", "500ms", "30μs")


```zig
pub fn parseDuration(duration_str: []const u8) !u64 {
```

## src/shared/utils/utils.zig

- const `VERSION`

Project version information


```zig
pub const VERSION = .{
```

- fn `versionString`

Render version as semantic version string: "major.minor.patch[-pre]"


```zig
pub fn versionString(allocator: std.mem.Allocator) ![]u8 {
```

- const `http`

HTTP utilities (status codes, methods, headers, requests/responses)


```zig
pub const http = @import("http/mod.zig");
```

- const `json`

JSON parsing, serialization, and manipulation utilities


```zig
pub const json = @import("json/mod.zig");
```

- const `string`

String manipulation, array operations, and time utilities


```zig
pub const string = @import("string/mod.zig");
```

- const `math`

Mathematical functions, statistics, geometry, and random numbers


```zig
pub const math = @import("math/mod.zig");
```

- type `Config`

Common configuration struct (maintained for backward compatibility)


```zig
pub const Config = struct {
```

- fn `init`

```zig
pub fn init(name: []const u8) Config {
```

- type `DefinitionType`

Definition types used throughout the project


```zig
pub const DefinitionType = enum {
```

- fn `toString`

```zig
pub fn toString(self: DefinitionType) []const u8 {
```

- const `HttpStatus`

Legacy HTTP types - redirect to new modules


```zig
pub const HttpStatus = http.HttpStatus;
```

- const `HttpMethod`

```zig
pub const HttpMethod = http.HttpMethod;
```

- const `Headers`

```zig
pub const Headers = http.Headers;
```

- const `HttpRequest`

```zig
pub const HttpRequest = http.HttpRequest;
```

- const `HttpResponse`

```zig
pub const HttpResponse = http.HttpResponse;
```

- const `StringUtils`

Legacy string and array utilities - redirect to new modules


```zig
pub const StringUtils = string.StringUtils;
```

- const `ArrayUtils`

```zig
pub const ArrayUtils = string.ArrayUtils;
```

- const `TimeUtils`

```zig
pub const TimeUtils = string.TimeUtils;
```

- const `JsonUtils`

Legacy JSON utilities - redirect to new modules


```zig
pub const JsonUtils = json.JsonUtils;
```

- const `JsonValue`

```zig
pub const JsonValue = json.JsonValue;
```

- const `MathUtils`

Legacy math utilities - redirect to new modules


```zig
pub const MathUtils = math.MathUtils;
```

## src/simd.zig

- const `SIMDOpts`

```zig
pub const SIMDOpts = shared.SIMDOpts;
```

- const `getPerformanceMonitor`

```zig
pub const getPerformanceMonitor = shared.getPerformanceMonitor;
```

- const `getPerformanceMonitorDetails`

```zig
pub const getPerformanceMonitorDetails = shared.getPerformanceMonitorDetails;
```

- const `getVectorOps`

```zig
pub const getVectorOps = shared.getVectorOps;
```

- const `text`

```zig
pub const text = shared.text;
```

- fn `dotProductSIMD`

```zig
pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: shared.SIMDOpts) f32 {
```

- fn `vectorAddSIMD`

```zig
pub fn vectorAddSIMD(a: []const f32, b: []const f32, result: []f32) void {
```

## src/tests/integration/comprehensive_test_suite.zig

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) TestConfig {
```

- fn `format`

```zig
pub fn format(
```

- type `ComprehensiveTestRunner`

Comprehensive Test Runner


```zig
pub const ComprehensiveTestRunner = struct {
```

- fn `init`

```zig
pub fn init(config: TestConfig) !*ComprehensiveTestRunner {
```

- fn `deinit`

```zig
pub fn deinit(self: *ComprehensiveTestRunner) void {
```

- fn `runUnitTests`

Run all unit tests


```zig
pub fn runUnitTests(self: *ComprehensiveTestRunner) !void {
```

- fn `runIntegrationTests`

Run integration tests


```zig
pub fn runIntegrationTests(self: *ComprehensiveTestRunner) !void {
```

- fn `runPerformanceTests`

Run performance tests


```zig
pub fn runPerformanceTests(self: *ComprehensiveTestRunner) !void {
```

- fn `runSecurityTests`

Run security tests


```zig
pub fn runSecurityTests(self: *ComprehensiveTestRunner) !void {
```

- fn `runAllTests`

Run all tests


```zig
pub fn runAllTests(self: *ComprehensiveTestRunner) !void {
```

- fn `main`

Main test function


```zig
pub fn main() !void {
```

## src/tests/integration/integration_test_suite.zig

- fn `main`

Comprehensive integration test suite for ABI
Tests cross-module functionality and system integration


```zig
pub fn main() !void {
```

## src/tests/integration/main.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tests/integration/simple_tests.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tests/mod.zig

- fn `main`

Main test entry point


```zig
pub fn main() !void {
```

## src/tests/unit/simd.zig

- const `SIMDOpts`

```zig
pub const SIMDOpts = shared.SIMDOpts;
```

- const `getPerformanceMonitor`

```zig
pub const getPerformanceMonitor = shared.getPerformanceMonitor;
```

- const `getPerformanceMonitorDetails`

```zig
pub const getPerformanceMonitorDetails = shared.getPerformanceMonitorDetails;
```

- const `getVectorOps`

```zig
pub const getVectorOps = shared.getVectorOps;
```

- const `text`

```zig
pub const text = shared.text;
```

- fn `dotProductSIMD`

```zig
pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: shared.SIMDOpts) f32 {
```

- fn `vectorAddSIMD`

```zig
pub fn vectorAddSIMD(a: []const f32, b: []const f32, result: []f32) void {
```

## src/tests/unit/test_http_server.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tests/unit/test_refactoring.zig

- const `core`

```zig
pub const core = @import("core/mod.zig");
```

- const `simd`

```zig
pub const simd = @import("simd/mod.zig");
```

- const `db`

```zig
pub const db = @import("db/mod.zig");
```

- const `AbiError`

```zig
pub const AbiError = core.AbiError;
```

- const `DbError`

```zig
pub const DbError = db.DbError;
```

- const `WdbxHeader`

```zig
pub const WdbxHeader = db.WdbxHeader;
```

- fn `init`

Initialize the ABI framework


```zig
pub fn init(allocator: std.mem.Allocator) !void {
```

- fn `deinit`

Deinitialize the ABI framework


```zig
pub fn deinit() void {
```

- fn `isInitialized`

Check if the framework is initialized


```zig
pub fn isInitialized() bool {
```

- type `SystemInfo`

System information structure


```zig
pub const SystemInfo = struct {
```

- fn `getSystemInfo`

Get system information


```zig
pub fn getSystemInfo() SystemInfo {
```

## src/tests/unit/test_windows_networking.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/advanced_code_analyzer.zig

- type `CodeQualityMetrics`

Code Quality Metrics


```zig
pub const CodeQualityMetrics = struct {
```

- fn `format`

```zig
pub fn format(
```

- type `SecurityIssueType`

Security Issue Types


```zig
pub const SecurityIssueType = enum {
```

- fn `getSeverity`

```zig
pub fn getSeverity(self: SecurityIssueType) enum { low, medium, high, critical } {
```

- fn `getDescription`

```zig
pub fn getDescription(self: SecurityIssueType) []const u8 {
```

- type `PerformanceIssueType`

Performance Issue Types


```zig
pub const PerformanceIssueType = enum {
```

- fn `getImpact`

```zig
pub fn getImpact(self: PerformanceIssueType) enum { low, medium, high } {
```

- fn `getDescription`

```zig
pub fn getDescription(self: PerformanceIssueType) []const u8 {
```

- type `CodeQualityIssue`

Code Quality Issue


```zig
pub const CodeQualityIssue = struct {
```

- fn `format`

```zig
pub fn format(
```

- type `AdvancedCodeAnalyzer`

Advanced Code Analyzer


```zig
pub const AdvancedCodeAnalyzer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*AdvancedCodeAnalyzer {
```

- fn `deinit`

```zig
pub fn deinit(self: *AdvancedCodeAnalyzer) void {
```

- fn `analyzeFile`

Analyze a Zig source file for code quality issues


```zig
pub fn analyzeFile(self: *AdvancedCodeAnalyzer, file_path: []const u8) !void {
```

- fn `generateReport`

Generate comprehensive code quality report


```zig
pub fn generateReport(self: *AdvancedCodeAnalyzer, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `exportToJson`

Export issues to JSON format


```zig
pub fn exportToJson(self: *AdvancedCodeAnalyzer, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `main`

Main function for command-line usage


```zig
pub fn main() !void {
```

## src/tools/basic_code_analyzer.zig

- type `BasicMetrics`

Basic Code Quality Metrics


```zig
pub const BasicMetrics = struct {
```

- fn `print`

```zig
pub fn print(self: BasicMetrics) void {
```

- type `BasicCodeAnalyzer`

Basic Code Analyzer


```zig
pub const BasicCodeAnalyzer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*BasicCodeAnalyzer {
```

- fn `deinit`

```zig
pub fn deinit(self: *BasicCodeAnalyzer) void {
```

- fn `analyzeFile`

Analyze a Zig source file


```zig
pub fn analyzeFile(self: *BasicCodeAnalyzer, file_path: []const u8) !void {
```

- fn `printReport`

Print report to console


```zig
pub fn printReport(self: *BasicCodeAnalyzer) void {
```

- fn `main`

Main function for command-line usage


```zig
pub fn main() !void {
```

## src/tools/continuous_monitor.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/docs_generator.zig

- fn `main`

Documentation generator for ABI project
Generates comprehensive API documentation from source code with enhanced GitHub Pages support


```zig
pub fn main() !void {
```

## src/tools/generate_api_docs.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/generate_index.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/http_smoke.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/main.zig

- fn `main`

```zig
pub fn main() !void {
```

- fn `deinit`

```zig
pub fn deinit(self: *TrainingData) void {
```

## src/tools/memory_tracker.zig

- type `AllocationRecord`

Memory allocation record


```zig
pub const AllocationRecord = struct {
```

- fn `memoryUsage`

Calculate memory usage for this allocation


```zig
pub fn memoryUsage(self: AllocationRecord) usize {
```

- fn `age`

Get allocation age in nanoseconds


```zig
pub fn age(self: AllocationRecord, current_time: u64) u64 {
```

- fn `isPotentialLeak`

Check if allocation is a potential leak


```zig
pub fn isPotentialLeak(self: AllocationRecord, current_time: u64, leak_threshold_ns: u64) bool {
```

- type `MemoryStats`

Memory statistics snapshot


```zig
pub const MemoryStats = struct {
```

- fn `currentUsage`

Calculate current memory usage


```zig
pub fn currentUsage(self: MemoryStats) usize {
```

- fn `efficiency`

Calculate memory efficiency (1.0 = no waste, lower = more fragmentation)


```zig
pub fn efficiency(self: MemoryStats) f64 {
```

- fn `allocationSuccessRate`

Get allocation success rate


```zig
pub fn allocationSuccessRate(self: MemoryStats) f64 {
```

- type `MemoryProfilerConfig`

Memory profiler configuration


```zig
pub const MemoryProfilerConfig = struct {
```

- type `MemoryProfiler`

Memory profiler main structure


```zig
pub const MemoryProfiler = struct {
```

- fn `init`

Initialize memory profiler


```zig
pub fn init(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !*MemoryProfiler {
```

- fn `deinit`

Deinitialize memory profiler


```zig
pub fn deinit(self: *MemoryProfiler) void {
```

- fn `recordAllocation`

Record a memory allocation


```zig
pub fn recordAllocation(
```

- fn `recordDeallocation`

Record a memory deallocation


```zig
pub fn recordDeallocation(self: *MemoryProfiler, id: u64) void {
```

- fn `getStats`

Get current memory statistics


```zig
pub fn getStats(self: *MemoryProfiler) MemoryStats {
```

- fn `getPotentialLeaks`

Get potential memory leaks


```zig
pub fn getPotentialLeaks(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]AllocationRecord {
```

- fn `generateReport`

Generate memory usage report


```zig
pub fn generateReport(self: *MemoryProfiler, allocator: std.mem.Allocator) ![]u8 {
```

- fn `resetStats`

Reset statistics


```zig
pub fn resetStats(self: *MemoryProfiler) void {
```

- fn `collectPeriodicStats`

Collect periodic statistics


```zig
pub fn collectPeriodicStats(self: *MemoryProfiler) void {
```

- fn `initGlobalProfiler`

Initialize global memory profiler


```zig
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: MemoryProfilerConfig) !void {
```

- fn `deinitGlobalProfiler`

Deinitialize global memory profiler


```zig
pub fn deinitGlobalProfiler() void {
```

- fn `getGlobalProfiler`

Get global memory profiler instance


```zig
pub fn getGlobalProfiler() ?*MemoryProfiler {
```

- type `TrackedAllocator`

Tracked allocator that integrates with memory profiler


```zig
pub const TrackedAllocator = struct {
```

- fn `init`

Initialize tracked allocator


```zig
pub fn init(parent_allocator: std.mem.Allocator, profiler: *MemoryProfiler) TrackedAllocator {
```

- fn `allocator`

Get allocator interface


```zig
pub fn allocator(self: *TrackedAllocator) std.mem.Allocator {
```

- type `MemoryMonitor`

Memory usage monitor


```zig
pub const MemoryMonitor = struct {
```

- fn `init`

Initialize memory monitor


```zig
pub fn init(profiler: *MemoryProfiler) !*MemoryMonitor {
```

- fn `start`

Start monitoring thread


```zig
pub fn start(self: *MemoryMonitor) !void {
```

- fn `stop`

Stop monitoring


```zig
pub fn stop(self: *MemoryMonitor) void {
```

- fn `deinit`

Deinitialize monitor


```zig
pub fn deinit(self: *MemoryMonitor) void {
```

- type `PerformanceMonitor`

Performance monitoring utilities


```zig
pub const PerformanceMonitor = struct {
```

- fn `start`

Start performance measurement


```zig
pub fn start(self: *PerformanceMonitor) void {
```

- fn `end`

End performance measurement


```zig
pub fn end(self: *PerformanceMonitor) void {
```

- fn `elapsedTime`

Get elapsed time in nanoseconds


```zig
pub fn elapsedTime(self: PerformanceMonitor) u64 {
```

- fn `memoryDelta`

Get memory usage delta


```zig
pub fn memoryDelta(self: PerformanceMonitor) i64 {
```

- fn `generateReport`

Generate performance report


```zig
pub fn generateReport(self: PerformanceMonitor, allocator: std.mem.Allocator, operation_name: []const u8) ![]u8 {
```

- type `utils`

Utility functions for memory profiling


```zig
pub const utils = struct {
```

- fn `simpleConfig`

Create a simple memory profiler configuration


```zig
pub fn simpleConfig() MemoryProfilerConfig {
```

- fn `developmentConfig`

Create a development configuration with more detailed tracking


```zig
pub fn developmentConfig() MemoryProfilerConfig {
```

- fn `productionConfig`

Create a production configuration with minimal overhead


```zig
pub fn productionConfig() MemoryProfilerConfig {
```

## src/tools/perf_guard.zig

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/performance.zig

- const `Allocator`

Re-export commonly used types


```zig
pub const Allocator = std.mem.Allocator;
```

- const `PerformanceError`

Performance monitoring specific error types


```zig
pub const PerformanceError = error{
```

- type `MetricType`

Performance metric types


```zig
pub const MetricType = enum {
```

- const `MetricValue`

Performance metric value


```zig
pub const MetricValue = union(MetricType) {
```

- type `HistogramData`

Histogram data for latency measurements


```zig
pub const HistogramData = struct {
```

- fn `record`

```zig
pub fn record(self: *HistogramData, value: f64) void {
```

- fn `percentile`

```zig
pub fn percentile(self: *const HistogramData, p: f64) f64 {
```

- type `TimerData`

Timer data for duration measurements


```zig
pub const TimerData = struct {
```

- fn `start`

```zig
pub fn start(self: *TimerData) void {
```

- fn `stop`

```zig
pub fn stop(self: *TimerData) void {
```

- fn `averageDuration`

```zig
pub fn averageDuration(self: *const TimerData) f64 {
```

- type `Metric`

Performance metric entry


```zig
pub const Metric = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, name: []const u8, value: MetricValue) !Metric {
```

- fn `deinit`

```zig
pub fn deinit(self: *Metric, allocator: std.mem.Allocator) void {
```

- fn `addLabel`

```zig
pub fn addLabel(self: *Metric, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
```

- type `CPUProfiler`

CPU profiler with sampling


```zig
pub const CPUProfiler = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, sampling_rate: u32) CPUProfiler {
```

- fn `deinit`

```zig
pub fn deinit(self: *CPUProfiler) void {
```

- fn `start`

```zig
pub fn start(self: *CPUProfiler) !void {
```

- fn `stop`

```zig
pub fn stop(self: *CPUProfiler) void {
```

- type `MemoryTracker`

Memory allocation tracker


```zig
pub const MemoryTracker = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !MemoryTracker {
```

- fn `deinit`

```zig
pub fn deinit(self: *MemoryTracker) void {
```

- fn `recordAllocation`

```zig
pub fn recordAllocation(self: *MemoryTracker, ptr: usize, size: usize) void {
```

- fn `recordDeallocation`

```zig
pub fn recordDeallocation(self: *MemoryTracker, ptr: usize) void {
```

- fn `getCurrentUsage`

```zig
pub fn getCurrentUsage(self: *const MemoryTracker) u64 {
```

- fn `getPeakUsage`

```zig
pub fn getPeakUsage(self: *const MemoryTracker) u64 {
```

- type `PerformanceMonitor`

Global performance monitoring system


```zig
pub const PerformanceMonitor = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*PerformanceMonitor {
```

- fn `deinit`

```zig
pub fn deinit(self: *PerformanceMonitor) void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(self: *PerformanceMonitor, name: []const u8, value: MetricValue) !void {
```

- fn `startProfiling`

```zig
pub fn startProfiling(self: *PerformanceMonitor) !void {
```

- fn `stopProfiling`

```zig
pub fn stopProfiling(self: *PerformanceMonitor) void {
```

- fn `getMetric`

```zig
pub fn getMetric(self: *PerformanceMonitor, name: []const u8) ?Metric {
```

- type `TracyProfiler`

Tracy profiler integration (when enabled)


```zig
pub const TracyProfiler = struct {
```

- fn `zoneName`

```zig
pub fn zoneName(comptime name: []const u8) void {
```

- fn `zoneStart`

```zig
pub fn zoneStart() void {
```

- fn `zoneEnd`

```zig
pub fn zoneEnd() void {
```

- fn `plot`

```zig
pub fn plot(name: []const u8, value: f64) void {
```

- fn `init`

```zig
pub fn init() !void {
```

- fn `deinit`

```zig
pub fn deinit() void {
```

- fn `recordMetric`

```zig
pub fn recordMetric(name: []const u8, value: f64) void {
```

- fn `recordCounter`

```zig
pub fn recordCounter(name: []const u8, value: u64) void {
```

- fn `recordLatency`

```zig
pub fn recordLatency(name: []const u8, duration_ns: u64) void {
```

- type `Timer`

Timer utility for measuring execution time


```zig
pub const Timer = struct {
```

- fn `start`

```zig
pub fn start(comptime name: []const u8) Timer {
```

- fn `stop`

```zig
pub fn stop(self: Timer) void {
```

- fn `timed`

Convenient macro for timing function execution


```zig
pub fn timed(comptime name: []const u8, func: anytype) @TypeOf(func()) {
```

## src/tools/performance_ci.zig

- type `PerformanceThresholds`

Performance threshold configuration with environment variable support


```zig
pub const PerformanceThresholds = struct {
```

- fn `loadFromEnv`

Load performance thresholds from environment variables with comprehensive validation


```zig
pub fn loadFromEnv(allocator: std.mem.Allocator) !PerformanceThresholds {
```

- fn `validate`

Validate threshold configuration with comprehensive checks


```zig
pub fn validate(self: PerformanceThresholds) !void {
```

- type `PerformanceMetrics`

Enhanced performance metrics with statistical analysis and system resource tracking


```zig
pub const PerformanceMetrics = struct {
```

- fn `init`

Initialize performance metrics with sensible defaults


```zig
pub fn init(_: std.mem.Allocator) PerformanceMetrics {
```

- fn `calculateStatistics`

Calculate comprehensive statistical metrics from timing data


```zig
pub fn calculateStatistics(self: *PerformanceMetrics, search_times: []const u64) void {
```

- fn `toJson`

Export metrics to structured JSON format


```zig
pub fn toJson(self: *const PerformanceMetrics, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `fromJson`

Import metrics from JSON format (production implementation would use proper JSON parser)


```zig
pub fn fromJson(allocator: std.mem.Allocator, json_str: []const u8) !PerformanceMetrics {
```

- type `PerformanceBenchmarkRunner`

Enhanced performance benchmark runner with comprehensive analysis and CI/CD integration


```zig
pub const PerformanceBenchmarkRunner = struct {
```

- fn `init`

Initialize benchmark runner with validated configuration


```zig
pub fn init(allocator: std.mem.Allocator, thresholds: PerformanceThresholds, output_dir: []const u8) !*Self {
```

- fn `deinit`

Clean up all allocated resources


```zig
pub fn deinit(self: *Self) void {
```

- fn `runBenchmarkSuite`

Execute comprehensive performance benchmark suite with statistical analysis


```zig
pub fn runBenchmarkSuite(self: *Self) !PerformanceMetrics {
```

- type `RegressionResult`

Comprehensive regression detection result with detailed analysis


```zig
pub const RegressionResult = struct {
```

- fn `init`

Initialize regression result with default values


```zig
pub fn init(_: std.mem.Allocator) RegressionResult {
```

- fn `deinit`

Clean up allocated resources


```zig
pub fn deinit(self: *RegressionResult, allocator: std.mem.Allocator) void {
```

- fn `main`

Enhanced main entry point with comprehensive error handling and configuration


```zig
pub fn main() !void {
```

## src/tools/performance_profiler.zig

- type `ProfilingConfig`

Performance profiling configuration


```zig
pub const ProfilingConfig = struct {
```

- type `CallRecord`

Function call record (for call tracing and call tree)


```zig
pub const CallRecord = struct {
```

- fn `duration`

Calculate call duration (nanoseconds)


```zig
pub fn duration(self: CallRecord) u64 {
```

- fn `isComplete`

Check if call is complete (has exit time)


```zig
pub fn isComplete(self: CallRecord) bool {
```

- type `PerformanceCounter`

Performance counter (for custom and built-in metrics)


```zig
pub const PerformanceCounter = struct {
```

- fn `increment`

```zig
pub fn increment(self: *PerformanceCounter) void {
```

- fn `add`

```zig
pub fn add(self: *PerformanceCounter, delta: u64) void {
```

- fn `set`

```zig
pub fn set(self: *PerformanceCounter, new_value: u64) void {
```

- fn `reset`

```zig
pub fn reset(self: *PerformanceCounter) void {
```

- type `PerformanceProfile`

Performance profile data (per session)


```zig
pub const PerformanceProfile = struct {
```

- fn `duration`

```zig
pub fn duration(self: PerformanceProfile) u64 {
```

- fn `durationSeconds`

```zig
pub fn durationSeconds(self: PerformanceProfile) f64 {
```

- fn `cpuUtilization`

```zig
pub fn cpuUtilization(self: PerformanceProfile) f64 {
```

- type `FunctionProfiler`

Function profiler for instrumenting and aggregating function stats


```zig
pub const FunctionProfiler = struct {
```

- fn `enter`

```zig
pub fn enter(self: *FunctionProfiler) u64 {
```

- fn `exit`

```zig
pub fn exit(self: *FunctionProfiler, entry_time: u64) void {
```

- fn `averageExecutionTime`

```zig
pub fn averageExecutionTime(self: FunctionProfiler) u64 {
```

- type `PerformanceProfiler`

Main performance profiler


```zig
pub const PerformanceProfiler = struct {
```

- fn `init`

Initialize performance profiler


```zig
pub fn init(allocator: std.mem.Allocator, config: ProfilingConfig) !*PerformanceProfiler {
```

- fn `deinit`

Deinitialize performance profiler and free all resources


```zig
pub fn deinit(self: *PerformanceProfiler) void {
```

- fn `startSession`

Start profiling session


```zig
pub fn startSession(self: *PerformanceProfiler, session_name: []const u8) !void {
```

- fn `endSession`

End profiling session and return report


```zig
pub fn endSession(self: *PerformanceProfiler) ![]u8 {
```

- fn `startFunctionCall`

Start function call (for call tracing)


```zig
pub fn startFunctionCall(self: *PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) !u64 {
```

- fn `endFunctionCall`

End function call (for call tracing)


```zig
pub fn endFunctionCall(self: *PerformanceProfiler, entry_time: u64) void {
```

- fn `updateCounter`

Update or create a performance counter


```zig
pub fn updateCounter(self: *PerformanceProfiler, name: []const u8, delta: u64) void {
```

- fn `getFunctionStats`

Get function profiler statistics (sorted by total_time descending)


```zig
pub fn getFunctionStats(self: *PerformanceProfiler, allocator: std.mem.Allocator) ![]FunctionProfiler {
```

- fn `stop`

Stop profiling thread


```zig
pub fn stop(self: *PerformanceProfiler) void {
```

- fn `setMemoryTracker`

Integrate with memory tracker


```zig
pub fn setMemoryTracker(self: *PerformanceProfiler, tracker: *memory_tracker.MemoryProfiler) void {
```

- fn `createScope`

Create performance scope for measuring code blocks


```zig
pub fn createScope(self: *PerformanceProfiler, name: []const u8) Scope {
```

- type `Scope`

Performance measurement scope (RAII-style)


```zig
pub const Scope = struct {
```

- fn `end`

End the scope and record measurements


```zig
pub fn end(self: Scope) void {
```

- fn `initGlobalProfiler`

Initialize global performance profiler


```zig
pub fn initGlobalProfiler(allocator: std.mem.Allocator, config: ProfilingConfig) !void {
```

- fn `deinitGlobalProfiler`

Deinitialize global performance profiler


```zig
pub fn deinitGlobalProfiler() void {
```

- fn `getGlobalProfiler`

Get global performance profiler instance


```zig
pub fn getGlobalProfiler() ?*PerformanceProfiler {
```

- fn `startScope`

Convenience function to start a performance scope


```zig
pub fn startScope(name: []const u8) ?Scope {
```

- fn `profileFunctionCall`

Convenience function for profiling function calls (to be used with defer)


```zig
pub fn profileFunctionCall(profiler: ?*PerformanceProfiler, function_name: []const u8, file: []const u8, line: u32) FunctionCall {
```

- type `FunctionCall`

Function call scope for automatic profiling (RAII-style)


```zig
pub const FunctionCall = struct {
```

- fn `end`

```zig
pub fn end(self: FunctionCall) void {
```

- type `utils`

Performance monitoring utilities and presets


```zig
pub const utils = struct {
```

- fn `developmentConfig`

Create a development profiling configuration


```zig
pub fn developmentConfig() ProfilingConfig {
```

- fn `productionConfig`

Create a production profiling configuration


```zig
pub fn productionConfig() ProfilingConfig {
```

- fn `minimalConfig`

Create a minimal profiling configuration


```zig
pub fn minimalConfig() ProfilingConfig {
```

## src/tools/simple_code_analyzer.zig

- type `SimpleMetrics`

Simple Code Quality Metrics


```zig
pub const SimpleMetrics = struct {
```

- fn `format`

```zig
pub fn format(
```

- type `SimpleCodeAnalyzer`

Simple Code Analyzer


```zig
pub const SimpleCodeAnalyzer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) !*SimpleCodeAnalyzer {
```

- fn `deinit`

```zig
pub fn deinit(self: *SimpleCodeAnalyzer) void {
```

- fn `analyzeFile`

Analyze a Zig source file


```zig
pub fn analyzeFile(self: *SimpleCodeAnalyzer, file_path: []const u8) !void {
```

- fn `generateReport`

Generate simple report


```zig
pub fn generateReport(self: *SimpleCodeAnalyzer, allocator: std.mem.Allocator) ![]const u8 {
```

- fn `main`

Main function for command-line usage


```zig
pub fn main() !void {
```

## src/tools/static_analysis.zig

- fn `toString`

```zig
pub fn toString(self: Severity) []const u8 {
```

- fn `getColor`

```zig
pub fn getColor(self: Severity) []const u8 {
```

- fn `getResetColor`

```zig
pub fn getResetColor() []const u8 {
```

- fn `format`

```zig
pub fn format(self: Finding, enable_colors: bool) void {
```

- fn `getScore`

```zig
pub fn getScore(self: Finding) u32 {
```

- fn `fromEnv`

```zig
pub fn fromEnv(allocator: std.mem.Allocator) !AnalysisConfig {
```

- type `StaticAnalyzer`

Enhanced static analyzer with comprehensive analysis capabilities


```zig
pub const StaticAnalyzer = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: AnalysisConfig) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `analyzeFile`

```zig
pub fn analyzeFile(self: *Self, file_path: []const u8) !void {
```

- fn `generateReport`

```zig
pub fn generateReport(self: *Self) !void {
```

- fn `analyzeDirectory`

```zig
pub fn analyzeDirectory(self: *Self, dir_path: []const u8) !void {
```

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/stress_test.zig

- fn `fromEnv`

```zig
pub fn fromEnv(allocator: std.mem.Allocator) !StressTestConfig {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) StressTestMetrics {
```

- fn `deinit`

```zig
pub fn deinit(self: *StressTestMetrics) void {
```

- fn `recordOperation`

```zig
pub fn recordOperation(self: *StressTestMetrics, response_time_ns: u64, success: bool) void {
```

- fn `recordError`

```zig
pub fn recordError(self: *StressTestMetrics, error_type: []const u8) !void {
```

- fn `getSuccessRate`

```zig
pub fn getSuccessRate(self: *StressTestMetrics) f32 {
```

- fn `getAverageResponseTime`

```zig
pub fn getAverageResponseTime(self: *StressTestMetrics) f64 {
```

- fn `getOperationsPerSecond`

```zig
pub fn getOperationsPerSecond(self: *StressTestMetrics) f64 {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, thread_count: usize) !StressTestThreadPool {
```

- fn `deinit`

```zig
pub fn deinit(self: *StressTestThreadPool) void {
```

- fn `submitWork`

```zig
pub fn submitWork(self: *StressTestThreadPool, work_fn: *const fn (*StressTestMetrics) void, metrics: *StressTestMetrics) !void {
```

- fn `getActiveWorkers`

```zig
pub fn getActiveWorkers(self: *StressTestThreadPool) usize {
```

- type `StressTester`

Enhanced stress test framework with adaptive load management


```zig
pub const StressTester = struct {
```

- fn `init`

```zig
pub fn init(config: StressTestConfig) AdaptiveController {
```

- fn `shouldAdjustLoad`

```zig
pub fn shouldAdjustLoad(self: *AdaptiveController, metrics: *StressTestMetrics) bool {
```

- fn `adjustLoadFactor`

```zig
pub fn adjustLoadFactor(self: *AdaptiveController, current_factor: f32, metrics: *StressTestMetrics) f32 {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: StressTestConfig) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `runStressTest`

```zig
pub fn runStressTest(self: *Self) !void {
```

- fn `main`

```zig
pub fn main() !void {
```

## src/tools/windows_network_test.zig

- fn `fromEnv`

```zig
pub fn fromEnv(allocator: std.mem.Allocator) !NetworkTestConfig {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator) NetworkTestMetrics {
```

- fn `deinit`

```zig
pub fn deinit(self: *NetworkTestMetrics) void {
```

- fn `recordLatency`

```zig
pub fn recordLatency(self: *NetworkTestMetrics, latency_ns: u64) void {
```

- fn `recordBandwidth`

```zig
pub fn recordBandwidth(self: *NetworkTestMetrics, bytes_transferred: usize, duration_ns: u64) void {
```

- fn `recordError`

```zig
pub fn recordError(self: *NetworkTestMetrics, error_type: []const u8) !void {
```

- fn `recordSocketError`

```zig
pub fn recordSocketError(self: *NetworkTestMetrics, error_code: i32) !void {
```

- fn `getTcpSuccessRate`

```zig
pub fn getTcpSuccessRate(self: *NetworkTestMetrics) f32 {
```

- fn `getUdpPacketLossRate`

```zig
pub fn getUdpPacketLossRate(self: *NetworkTestMetrics) f32 {
```

- fn `format`

```zig
pub fn format(self: NetworkAdapter, allocator: std.mem.Allocator) ![]u8 {
```

- type `WindowsNetworkTester`

Enhanced Windows network testing framework


```zig
pub const WindowsNetworkTester = struct {
```

- fn `init`

```zig
pub fn init(allocator: std.mem.Allocator, config: NetworkTestConfig) Self {
```

- fn `deinit`

```zig
pub fn deinit(self: *Self) void {
```

- fn `runComprehensiveTests`

```zig
pub fn runComprehensiveTests(self: *Self) !void {
```

- fn `main`

```zig
pub fn main() !void {
```

