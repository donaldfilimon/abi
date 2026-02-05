# Feature Modules Completion Plan - Production Ready (90%)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring all 6 feature modules from current state to 90% production-ready.

**Current State:**
| Module | Current % | Target % | Gap |
|--------|-----------|----------|-----|
| AI | 78% | 90% | Vision, Documents, Embeddings |
| Database | 75% | 90% | GPU dispatch, DiskANN, Raft snapshots |
| GPU | 60% | 90% | CUDA/Metal backends, DSL codegen |
| Network | 87% | 90% | Raft snapshots, membership changes |
| Observability | 88% | 90% | System info, span processor |
| Web | 65% | 90% | **CRITICAL: server/, middleware/, handlers** |

**Total Estimated Time:** ~150 hours (3-4 weeks)

---

## Phase 1: Web Module (CRITICAL - 30 tasks)

Web is the most critical gap - cannot serve HTTP requests without server/middleware.

### Task 1.1: Create Server Core Types
**Files:** `src/features/web/server/types.zig` (create)
**Implement:**
```zig
pub const ServerConfig = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    max_connections: usize = 1024,
    read_timeout_ms: u64 = 30000,
    write_timeout_ms: u64 = 30000,
};
pub const ServerState = enum { stopped, starting, running, stopping };
pub const Connection = struct { stream: std.net.Stream, address: std.net.Address, allocator: std.mem.Allocator };
```
**Verify:** `zig test src/features/web/server/types.zig`

### Task 1.2: Create HTTP Server Wrapper
**Files:** `src/features/web/server/http_server.zig` (create)
**Implement:**
```zig
pub const Server = struct {
    allocator: std.mem.Allocator,
    config: types.ServerConfig,
    state: types.ServerState,
    io_backend: std.Io.Threaded,
    
    pub fn init(allocator: std.mem.Allocator, config: types.ServerConfig) !Server;
    pub fn listen(self: *Server) !void;
    pub fn accept(self: *Server) !types.Connection;
    pub fn handleConnection(self: *Server, conn: types.Connection) !void;
    pub fn shutdown(self: *Server) void;
    pub fn deinit(self: *Server) void;
};
```
**Dependencies:** Task 1.1
**Verify:** `zig test src/features/web/server/http_server.zig`

### Task 1.3: Implement Request Parser
**Files:** `src/features/web/server/request_parser.zig` (create)
**Implement:**
```zig
pub const ParsedRequest = struct {
    method: std.http.Method,
    path: []const u8,
    query: ?[]const u8,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8,
};
pub fn parseRequest(allocator: std.mem.Allocator, reader: *std.Io.Reader) !ParsedRequest;
pub fn extractPathParams(path: []const u8, pattern: []const u8) !std.StringHashMap([]const u8);
```
**Dependencies:** Task 1.1
**Verify:** `zig test src/features/web/server/request_parser.zig`

### Task 1.4: Create Response Builder
**Files:** `src/features/web/server/response_builder.zig` (create)
**Implement:**
```zig
pub const ResponseBuilder = struct {
    status: std.http.Status = .ok,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8 = null,
    
    pub fn init(allocator: std.mem.Allocator) ResponseBuilder;
    pub fn setStatus(self: *ResponseBuilder, status: std.http.Status) *ResponseBuilder;
    pub fn setHeader(self: *ResponseBuilder, key: []const u8, value: []const u8) !*ResponseBuilder;
    pub fn json(self: *ResponseBuilder, data: anytype) !*ResponseBuilder;
    pub fn send(self: *ResponseBuilder, writer: *std.Io.Writer) !void;
};
```
**Dependencies:** Task 1.1
**Verify:** `zig test src/features/web/server/response_builder.zig`

### Task 1.5: Create Server Module Entry
**Files:** `src/features/web/server/mod.zig` (create)
**Implement:**
```zig
pub const types = @import("types.zig");
pub const HttpServer = @import("http_server.zig").Server;
pub const RequestParser = @import("request_parser.zig");
pub const ResponseBuilder = @import("response_builder.zig").ResponseBuilder;
```
**Dependencies:** Tasks 1.1-1.4
**Verify:** `zig build test`

### Task 1.6: Create Middleware Types
**Files:** `src/features/web/middleware/types.zig` (create)
**Implement:**
```zig
pub const MiddlewareContext = struct {
    request: *server.RequestParser.ParsedRequest,
    response: *server.ResponseBuilder,
    state: std.StringHashMap([]const u8),
};
pub const MiddlewareFn = *const fn(ctx: *MiddlewareContext) anyerror!void;
pub const MiddlewareChain = struct {
    middlewares: std.ArrayList(MiddlewareFn),
    pub fn use(self: *MiddlewareChain, middleware: MiddlewareFn) !void;
    pub fn execute(self: *MiddlewareChain, ctx: *MiddlewareContext) !void;
};
```
**Dependencies:** Tasks 1.3, 1.4
**Verify:** `zig test src/features/web/middleware/types.zig`

### Task 1.7: Implement Logging Middleware
**Files:** `src/features/web/middleware/logging.zig` (create)
**Implement:**
```zig
pub fn loggingMiddleware(ctx: *types.MiddlewareContext) !void;
pub const LogFormat = enum { common, combined, json };
pub fn createLogger(format: LogFormat) types.MiddlewareFn;
```
**Dependencies:** Task 1.6
**Verify:** Integration test with sample requests

### Task 1.8: Implement CORS Middleware
**Files:** `src/features/web/middleware/cors.zig` (create)
**Implement:**
```zig
pub const CorsConfig = struct {
    allowed_origins: []const []const u8 = &.{"*"},
    allowed_methods: []const std.http.Method = &.{.GET, .POST, .PUT, .DELETE},
    allowed_headers: []const []const u8 = &.{"Content-Type", "Authorization"},
};
pub fn corsMiddleware(config: CorsConfig) types.MiddlewareFn;
```
**Dependencies:** Task 1.6
**Verify:** Test with OPTIONS requests

### Task 1.9: Implement Auth Middleware
**Files:** `src/features/web/middleware/auth.zig` (create)
**Implement:**
```zig
pub const AuthConfig = struct { jwt_secret: []const u8, token_header: []const u8 = "Authorization" };
pub fn authMiddleware(config: AuthConfig) types.MiddlewareFn;
pub fn extractToken(ctx: *types.MiddlewareContext, config: AuthConfig) ?[]const u8;
pub fn validateToken(token: []const u8, secret: []const u8) !bool;
```
**Dependencies:** Task 1.6
**Verify:** Test with valid/invalid tokens

### Task 1.10: Implement Error Handler Middleware
**Files:** `src/features/web/middleware/error_handler.zig` (create)
**Implement:**
```zig
pub const ErrorResponse = struct { status: u16, message: []const u8, error_code: ?[]const u8 };
pub fn errorHandlerMiddleware(ctx: *types.MiddlewareContext) !void;
pub fn handleError(ctx: *types.MiddlewareContext, err: anyerror) !void;
pub fn toHttpStatus(err: anyerror) std.http.Status;
```
**Dependencies:** Task 1.6
**Verify:** Test with various error types

### Task 1.11: Create Middleware Module Entry
**Files:** `src/features/web/middleware/mod.zig` (create)
**Implement:**
```zig
pub const types = @import("types.zig");
pub const logging = @import("logging.zig");
pub const cors = @import("cors.zig");
pub const auth = @import("auth.zig");
pub const error_handler = @import("error_handler.zig");
pub fn defaultStack(allocator: std.mem.Allocator) !types.MiddlewareChain;
```
**Dependencies:** Tasks 1.6-1.10
**Verify:** `zig build test`

### Task 1.12: Create Router Types
**Files:** `src/features/web/router/types.zig` (create)
**Implement:**
```zig
pub const HandlerFn = *const fn(ctx: *middleware.types.MiddlewareContext) anyerror!void;
pub const Route = struct { method: std.http.Method, path: []const u8, handler: HandlerFn };
pub const RouteMatch = struct { handler: HandlerFn, params: std.StringHashMap([]const u8) };
```
**Dependencies:** Task 1.6
**Verify:** `zig test src/features/web/router/types.zig`

### Task 1.13: Implement Route Matcher
**Files:** `src/features/web/router/matcher.zig` (create)
**Implement:**
```zig
pub fn matchRoute(routes: []const types.Route, method: std.http.Method, path: []const u8) !?types.RouteMatch;
pub fn extractParams(pattern: []const u8, path: []const u8) !std.StringHashMap([]const u8);
pub fn isPathMatch(pattern: []const u8, path: []const u8) bool;
```
**Dependencies:** Task 1.12
**Verify:** Test with `/users/:id` patterns

### Task 1.14: Implement Router
**Files:** `src/features/web/router/router.zig` (create)
**Implement:**
```zig
pub const Router = struct {
    routes: std.ArrayList(types.Route),
    pub fn init(allocator: std.mem.Allocator) Router;
    pub fn get(self: *Router, path: []const u8, handler: types.HandlerFn) !void;
    pub fn post(self: *Router, path: []const u8, handler: types.HandlerFn) !void;
    pub fn dispatch(self: *Router, req: *ParsedRequest, resp: *ResponseBuilder) !void;
};
```
**Dependencies:** Tasks 1.12, 1.13
**Verify:** Integration test with route registration

### Task 1.15: Create Router Module Entry
**Files:** `src/features/web/router/mod.zig` (create)
**Implement:**
```zig
pub const types = @import("types.zig");
pub const matcher = @import("matcher.zig");
pub const Router = @import("router.zig").Router;
```
**Dependencies:** Tasks 1.12-1.14
**Verify:** `zig build test`

### Task 1.16: Implement Health Handler
**Files:** `src/features/web/handlers/health.zig` (create)
**Implement:**
```zig
pub const HealthStatus = struct { status: []const u8, version: []const u8, uptime_ms: u64 };
pub fn healthHandler(ctx: *middleware.types.MiddlewareContext) !void;
pub fn livenessHandler(ctx: *middleware.types.MiddlewareContext) !void;
pub fn readinessHandler(ctx: *middleware.types.MiddlewareContext) !void;
```
**Dependencies:** Task 1.6
**Verify:** `curl localhost:8080/health`

### Task 1.17: Implement Metrics Handler
**Files:** `src/features/web/handlers/metrics.zig` (create)
**Implement:**
```zig
pub fn metricsHandler(ctx: *middleware.types.MiddlewareContext) !void;
pub fn prometheusHandler(ctx: *middleware.types.MiddlewareContext) !void;
```
**Integration:** Uses `src/features/observability/metrics/prometheus.zig`
**Dependencies:** Task 1.6
**Verify:** `curl localhost:8080/metrics`

### Task 1.18: Implement Auth Handlers
**Files:** `src/features/web/handlers/auth.zig` (create)
**Implement:**
```zig
pub const LoginRequest = struct { username: []const u8, password: []const u8 };
pub const LoginResponse = struct { token: []const u8, expires_in: u64 };
pub fn loginHandler(ctx: *middleware.types.MiddlewareContext) !void;
pub fn logoutHandler(ctx: *middleware.types.MiddlewareContext) !void;
pub fn refreshTokenHandler(ctx: *middleware.types.MiddlewareContext) !void;
```
**Dependencies:** Tasks 1.6, 1.9
**Verify:** Login flow tests

### Task 1.19: Update Handlers Module Entry
**Files:** `src/features/web/handlers/mod.zig` (modify)
**Add:**
```zig
pub const health = @import("health.zig");
pub const metrics = @import("metrics.zig");
pub const auth = @import("auth.zig");
pub const chat = @import("chat.zig");
```
**Dependencies:** Tasks 1.16-1.18
**Verify:** `zig build test`

### Task 1.20: Update Web Module Entry
**Files:** `src/features/web/mod.zig` (modify)
**Add:**
```zig
pub const server = @import("server/mod.zig");
pub const middleware = @import("middleware/mod.zig");
pub const router = @import("router/mod.zig");
pub const handlers = @import("handlers/mod.zig");
pub fn createServer(allocator: std.mem.Allocator, config: server.types.ServerConfig) !*server.HttpServer;
```
**Dependencies:** Tasks 1.5, 1.11, 1.15, 1.19
**Verify:** `zig build test`

### Task 1.21: Update Web Module Stub
**Files:** `src/features/web/stub.zig` (modify)
**Add:** Stub implementations for server, middleware, router, handlers
**Dependencies:** Task 1.20
**Verify:** `zig build -Denable-web=false test`

### Task 1.22-1.25: Web Integration Tests
**Files:** `src/features/web/tests/*.zig` (create)
- Task 1.22: Server integration test
- Task 1.23: Middleware integration test
- Task 1.24: Router integration test
- Task 1.25: E2E server test
**Verify:** `zig test src/features/web/tests/e2e_test.zig`

---

## Phase 2: AI Module (40 tasks)

### Vision Submodule (Tasks 2.1-2.10)

#### Task 2.1: Create JPEG Decoder
**Files:** `src/features/ai/vision/io/jpeg.zig` (create)
**Implement:** `JpegDecoder.decode()`, `JpegDecoder.decodeFile()`
**Verify:** `zig test src/features/ai/vision/io/jpeg.zig`

#### Task 2.2: Create PNG Decoder
**Files:** `src/features/ai/vision/io/png.zig` (create)
**Implement:** `PngDecoder.decode()`, DEFLATE decompression, filter reconstruction
**Verify:** `zig test src/features/ai/vision/io/png.zig`

#### Task 2.3: Create Image Loader
**Files:** `src/features/ai/vision/io/loader.zig`, `src/features/ai/vision/io/mod.zig` (create)
**Implement:** `ImageLoader.loadFromFile()`, `detectFormat()`
**Dependencies:** Tasks 2.1, 2.2
**Verify:** `zig test src/features/ai/vision/io/loader.zig`

#### Task 2.4: Implement Image Resize
**Files:** `src/features/ai/vision/preprocessing/resize.zig` (create)
**Implement:** `ResizeOp.resize()` with nearest, bilinear, bicubic methods
**Verify:** `zig test src/features/ai/vision/preprocessing/resize.zig`

#### Task 2.5: Implement Normalization
**Files:** `src/features/ai/vision/preprocessing/normalize.zig` (create)
**Implement:** `normalize()`, `toTensor()`, `channelsFirst()` (HWC → CHW)
**Verify:** `zig test src/features/ai/vision/preprocessing/normalize.zig`

#### Task 2.6: Implement Patch Extraction
**Files:** `src/features/ai/vision/preprocessing/patches.zig` (create)
**Implement:** `extractPatches()`, `flattenPatches()` for ViT
**Dependencies:** Task 2.5
**Verify:** `zig test src/features/ai/vision/preprocessing/patches.zig`

#### Task 2.7: ViT Embeddings & Positional Encoding
**Files:** `src/features/ai/vision/vit/embeddings.zig` (create)
**Implement:** `PatchEmbedding`, `PositionalEncoding`, `CLSToken`
**Dependencies:** Task 2.6
**Verify:** `zig test src/features/ai/vision/vit/embeddings.zig`

#### Task 2.8: ViT Encoder Block
**Files:** `src/features/ai/vision/vit/encoder.zig` (create)
**Implement:** `EncoderBlock`, `MultiHeadAttention`, `MLP`, `LayerNorm`
**Dependencies:** Task 2.7
**Verify:** `zig test src/features/ai/vision/vit/encoder.zig`

#### Task 2.9: ViT Inference Pipeline
**Files:** `src/features/ai/vision/vit/inference.zig` (create)
**Implement:** `ViTInference.predict()`, `ClassifierHead`, top-k predictions
**Dependencies:** Tasks 2.3-2.8
**Verify:** `zig test src/features/ai/vision/vit/inference.zig`

#### Task 2.10: Update Vision mod.zig & stub.zig
**Files:** `src/features/ai/vision/mod.zig`, `src/features/ai/vision/stub.zig` (modify)
**Dependencies:** Tasks 2.1-2.9
**Verify:** `zig build test -Denable-vision=true && zig build test -Denable-vision=false`

### Documents Submodule (Tasks 2.11-2.17)

#### Task 2.11: PDF Parser
**Files:** `src/features/ai/documents/pdf/parser.zig` (create)
**Implement:** `PdfParser.parseFile()`, xref table, trailer, catalog, pages
**Verify:** `zig test src/features/ai/documents/pdf/parser.zig`

#### Task 2.12: PDF Text Extraction
**Files:** `src/features/ai/documents/pdf/text_extractor.zig` (create)
**Implement:** `TextExtractor.extractText()`, Tj/TJ operators
**Dependencies:** Task 2.11
**Verify:** `zig test src/features/ai/documents/pdf/text_extractor.zig`

#### Task 2.13: HTML Parser
**Files:** `src/features/ai/documents/html/parser.zig` (create)
**Implement:** `HtmlParser.parse()`, DOM tree construction
**Verify:** `zig test src/features/ai/documents/html/parser.zig`

#### Task 2.14: HTML Text Extraction
**Files:** `src/features/ai/documents/html/text_extractor.zig` (create)
**Implement:** `HtmlTextExtractor.extractText()`, skip scripts/styles
**Dependencies:** Task 2.13
**Verify:** `zig test src/features/ai/documents/html/text_extractor.zig`

#### Task 2.15: Unified Document Interface
**Files:** `src/features/ai/documents/document.zig` (create)
**Implement:** `Document`, `DocumentLoader.load()`, format auto-detection
**Dependencies:** Tasks 2.12, 2.14
**Verify:** `zig test src/features/ai/documents/document.zig`

#### Task 2.16: Document QA Foundation
**Files:** `src/features/ai/documents/qa.zig` (create)
**Implement:** `DocumentQA.answer()`, context extraction, RAG integration
**Dependencies:** Task 2.15
**Verify:** `zig test src/features/ai/documents/qa.zig`

#### Task 2.17: Update Documents mod.zig & stub.zig
**Files:** `src/features/ai/documents/mod.zig`, `src/features/ai/documents/stub.zig` (modify)
**Dependencies:** Tasks 2.11-2.16
**Verify:** `zig build test`

### Embeddings Submodule (Tasks 2.18-2.25)

#### Task 2.18: Complete OpenAI Backend
**Files:** `src/features/ai/embeddings/backends/openai.zig` (modify)
**Implement:** Full HTTP client integration, batch processing, error handling
**Verify:** `zig test src/features/ai/embeddings/backends/openai.zig`

#### Task 2.19: Add Ollama Backend
**Files:** `src/features/ai/embeddings/backends/ollama.zig` (create)
**Implement:** `OllamaBackend.embed()`, local model support
**Verify:** `zig test src/features/ai/embeddings/backends/ollama.zig`

#### Task 2.20: Add Local Model Backend
**Files:** `src/features/ai/embeddings/backends/local.zig` (create)
**Implement:** `LocalBackend` using LLM module for embedding extraction
**Verify:** `zig test src/features/ai/embeddings/backends/local.zig`

#### Task 2.21: Backend Factory
**Files:** `src/features/ai/embeddings/backends/factory.zig` (create)
**Implement:** `BackendFactory.create()`, auto-detection, fallback chain
**Dependencies:** Tasks 2.18-2.20
**Verify:** `zig test src/features/ai/embeddings/backends/factory.zig`

#### Task 2.22: Embedding Cache Layer
**Files:** `src/features/ai/embeddings/cache.zig` (modify)
**Implement:** WDBX integration, LRU eviction, persistent cache
**Verify:** `zig test src/features/ai/embeddings/cache.zig`

#### Task 2.23: Batch Processing Optimization
**Files:** `src/features/ai/embeddings/batch.zig` (create)
**Implement:** `BatchProcessor.processBatch()`, parallel execution
**Dependencies:** Task 2.21
**Verify:** `zig test src/features/ai/embeddings/batch.zig`

#### Task 2.24: RAG Integration
**Files:** `src/features/ai/rag/retriever.zig` (modify)
**Implement:** Deep embeddings context integration, semantic search
**Dependencies:** Task 2.21
**Verify:** `zig test src/features/ai/rag/retriever.zig`

#### Task 2.25: Update Embeddings mod.zig & stub.zig
**Files:** `src/features/ai/embeddings/mod.zig`, `src/features/ai/embeddings/stub.zig` (modify)
**Dependencies:** Tasks 2.18-2.24
**Verify:** `zig build test`

### Streaming Submodule (Tasks 2.26-2.32)

#### Task 2.26: WebSocket Frame Parser
**Files:** `src/features/ai/streaming/websocket/frame.zig` (create)
**Implement:** Frame parsing (RFC 6455), masking, fragmentation
**Verify:** `zig test src/features/ai/streaming/websocket/frame.zig`

#### Task 2.27: WebSocket Connection Handler
**Files:** `src/features/ai/streaming/websocket/connection.zig` (create)
**Implement:** Handshake, message send/receive, ping/pong
**Dependencies:** Task 2.26
**Verify:** `zig test src/features/ai/streaming/websocket/connection.zig`

#### Task 2.28: WebSocket Server
**Files:** `src/features/ai/streaming/websocket/server.zig` (create)
**Implement:** `WebSocketServer.accept()`, upgrade handling
**Dependencies:** Task 2.27
**Verify:** `zig test src/features/ai/streaming/websocket/server.zig`

#### Task 2.29: HTTP SSE Handler
**Files:** `src/features/ai/streaming/sse/handler.zig` (create)
**Implement:** Server-Sent Events formatting, keep-alive
**Verify:** `zig test src/features/ai/streaming/sse/handler.zig`

#### Task 2.30: LLM Streaming Endpoint
**Files:** `src/features/ai/streaming/endpoints/llm.zig` (create)
**Implement:** `/v1/chat/completions` streaming, token-by-token output
**Dependencies:** Tasks 2.28, 2.29
**Verify:** Integration test with LLM module

#### Task 2.31: Backpressure & Rate Limiting
**Files:** `src/features/ai/streaming/backpressure.zig` (modify)
**Implement:** Token bucket, connection limits, queue management
**Verify:** `zig test src/features/ai/streaming/backpressure.zig`

#### Task 2.32: Update Streaming mod.zig & stub.zig
**Files:** `src/features/ai/streaming/mod.zig`, `src/features/ai/streaming/stub.zig` (modify)
**Dependencies:** Tasks 2.26-2.31
**Verify:** `zig build test`

---

## Phase 3: GPU Module (50 tasks)

### CUDA Backend (Tasks 3.1-3.15)

#### Task 3.1: CUDA Memory Operations
**Files:** `src/features/gpu/backends/cuda/memory.zig` (modify)
**Implement:** `cudaMalloc`, `cudaMemcpy`, `cudaFree` via FFI
**Verify:** `zig test src/features/gpu/backends/cuda/memory.zig`

#### Task 3.2: CUDA Stream Management
**Files:** `src/features/gpu/backends/cuda/stream.zig` (modify)
**Implement:** `cudaStreamCreate`, synchronization, multi-stream execution
**Dependencies:** Task 3.1
**Verify:** `zig test src/features/gpu/backends/cuda/stream.zig`

#### Task 3.3: CUDA Context Creation
**Files:** `src/features/gpu/backends/cuda/native.zig` (modify)
**Implement:** Device enumeration, context initialization
**Dependencies:** Tasks 3.1, 3.2
**Verify:** `zig test src/features/gpu/backends/cuda/native.zig`

#### Task 3.4: cuBLAS Integration
**Files:** `src/features/gpu/backends/cuda/cublas.zig` (modify)
**Implement:** SGEMM, GEMV, batched operations
**Dependencies:** Task 3.3
**Verify:** Matrix multiply benchmarks

#### Task 3.5: LLM Kernels - Attention
**Files:** `src/features/gpu/backends/cuda/llm_kernels.zig` (modify)
**Implement:** Flash attention, KV cache operations
**Dependencies:** Task 3.4
**Verify:** Attention accuracy tests

#### Task 3.6: LLM Kernels - Normalization
**Files:** `src/features/gpu/backends/cuda/llm_kernels.zig` (modify)
**Implement:** RMSNorm, LayerNorm CUDA kernels
**Dependencies:** Task 3.3
**Verify:** Normalization accuracy tests

#### Task 3.7: LLM Kernels - Activations
**Files:** `src/features/gpu/backends/cuda/llm_kernels.zig` (modify)
**Implement:** SiLU, GELU, Softmax CUDA kernels
**Dependencies:** Task 3.3
**Verify:** Activation accuracy tests

#### Task 3.8-3.15: Additional CUDA tasks
- Task 3.8: Quantized Kernels (INT8, INT4)
- Task 3.9: NVRTC Runtime Compilation
- Task 3.10: Multi-GPU Support (NCCL)
- Task 3.11: Memory Pool
- Task 3.12: Profiling Integration
- Task 3.13: Error Handling
- Task 3.14: Vtable Implementation
- Task 3.15: Integration Tests

### Metal Backend (Tasks 3.16-3.25)

#### Task 3.16-3.25: Metal implementation
- Task 3.16: Metal Device Setup
- Task 3.17: Metal Command Queues
- Task 3.18: MSL Shaders - MatMul
- Task 3.19: MSL Shaders - Attention
- Task 3.20: MSL Shaders - Normalization
- Task 3.21: Accelerate Framework Integration
- Task 3.22: Unified Memory Optimization
- Task 3.23: Metal Performance Shaders
- Task 3.24: Vtable Implementation
- Task 3.25: Integration Tests

### DSL Codegen (Tasks 3.26-3.35)

#### Task 3.26-3.35: Code generation
- Task 3.26: IR to CUDA Translation
- Task 3.27: IR to MSL Translation
- Task 3.28: Type Mapping
- Task 3.29: Memory Layout
- Task 3.30: Kernel Launch Config
- Task 3.31: NVRTC Compilation
- Task 3.32: Metal Library Compilation
- Task 3.33: Caching Layer
- Task 3.34: E2E DSL Tests
- Task 3.35: Update mod.zig & stub.zig

### Multi-GPU & Profiling (Tasks 3.36-3.50)

#### Task 3.36-3.50: Advanced features
- Task 3.36: Device Enumeration
- Task 3.37: Peer-to-Peer Transfers
- Task 3.38: Data Parallelism
- Task 3.39: All-Reduce Collective
- Task 3.40: Gradient Sync
- Task 3.41: Event Profiler
- Task 3.42: Kernel Metrics
- Task 3.43: Observability Integration
- Task 3.44: Health Monitoring
- Task 3.45: Performance Dashboard
- Task 3.46: Fallback Chain
- Task 3.47: Error Recovery
- Task 3.48: Update mod.zig
- Task 3.49: Update stub.zig
- Task 3.50: Integration Tests

---

## Phase 4: Database/Network/Observability (12 tasks)

### Database (Tasks 4.1-4.5)

#### Task 4.1: GPU Kernel Dispatch
**Files:** `src/features/database/gpu_accel.zig` (modify)
**Implement:** Connect to GPU module, kernel dispatch for distance computation
**Verify:** `zig test src/features/database/gpu_accel.zig`

#### Task 4.2: DiskANN Mmap I/O
**Files:** `src/features/database/diskann.zig` (modify)
**Implement:** Memory-mapped access, prefetching, I/O optimization
**Verify:** `zig test src/features/database/diskann.zig`

#### Task 4.3: Index Rebuild - Incremental
**Files:** `src/features/database/reindex.zig` (modify)
**Implement:** Incremental rebuild without full reconstruction
**Verify:** `zig test src/features/database/reindex.zig`

#### Task 4.4: Index Rebuild - Compaction
**Files:** `src/features/database/reindex.zig` (modify)
**Implement:** Compaction for deleted entries, fragmentation reduction
**Dependencies:** Task 4.3
**Verify:** `zig test src/features/database/reindex.zig`

#### Task 4.5: Database Raft Snapshot
**Files:** `src/features/database/distributed/raft_block_chain.zig` (modify)
**Implement:** `createSnapshot()`, `installSnapshot()`, snapshot transfer
**Verify:** `zig test src/features/database/distributed/raft_block_chain.zig`

### Network (Tasks 4.6-4.8)

#### Task 4.6: Raft Snapshot Installation
**Files:** `src/features/network/raft.zig` (modify)
**Implement:** `InstallSnapshot` RPC handler, state machine reset
**Dependencies:** Task 4.5
**Verify:** `zig test src/features/network/raft.zig`

#### Task 4.7: Raft Membership Changes
**Files:** `src/features/network/raft.zig` (modify)
**Implement:** Joint consensus, configuration changes
**Verify:** `zig test src/features/network/raft.zig`

#### Task 4.8: Circuit Breaker + Failover Integration
**Files:** `src/features/network/failover.zig` (modify)
**Implement:** Connect circuit breaker metrics to failover decisions
**Verify:** `zig test src/features/network/failover.zig`

### Observability (Tasks 4.9-4.12)

#### Task 4.9: System Info Expansion
**Files:** `src/features/observability/system_info/mod.zig` (modify)
**Implement:** CPU usage, memory stats, disk I/O, network stats
**Verify:** `zig test src/features/observability/system_info/mod.zig`

#### Task 4.10: Span Batch Processor
**Files:** `src/features/observability/tracing.zig` (modify)
**Implement:** Batch span export, sampling, buffering
**Verify:** `zig test src/features/observability/tracing.zig`

#### Task 4.11: Database Integration Tests
**Files:** `src/features/database/tests/integration_test.zig` (create)
**Dependencies:** Tasks 4.1-4.5
**Verify:** `zig test src/features/database/tests/integration_test.zig`

#### Task 4.12: Observability Integration Tests
**Files:** `src/features/observability/tests/integration_test.zig` (create)
**Dependencies:** Tasks 4.9-4.10
**Verify:** `zig test src/features/observability/tests/integration_test.zig`

---

## Final Verification

After completing all phases:

```bash
# Full test suite
zig build test --summary all

# Format check
zig fmt . --check

# Build with all features
zig build -Denable-ai=true -Denable-gpu=true -Denable-database=true -Denable-network=true -Denable-web=true -Denable-observability=true

# Build with all features disabled (stub validation)
zig build -Denable-ai=false -Denable-gpu=false -Denable-database=false -Denable-network=false -Denable-web=false

# CLI smoke tests
zig build cli-tests
```

---

## Summary

| Phase | Tasks | Hours | Focus |
|-------|-------|-------|-------|
| 1 | 25 | 30-60 | Web module (CRITICAL) |
| 2 | 40 | 60 | AI module |
| 3 | 50 | 72-100 | GPU module |
| 4 | 12 | 18-20 | Database/Network/Observability |

**Total: 127 tasks, ~150-240 hours**

**Execution Order:** Phase 1 (Web) → Phase 4 (quick wins) → Phase 2 (AI) → Phase 3 (GPU)

---

**Last Updated:** 2026-02-04
