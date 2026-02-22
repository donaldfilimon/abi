# shared

> Shared utilities (SIMD, time, sync, security, etc.).

**Source:** [`src/services/shared/mod.zig`](../../src/services/shared/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-errors"></a>`pub const errors`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L62)

Error definitions and handling utilities for the ABI framework.
Provides standardized error types and conversion functions.

### <a id="pub-const-logging"></a>`pub const logging`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L66)

Logging infrastructure with configurable log levels and output destinations.
Supports structured logging with context and scoped loggers.

### <a id="pub-const-plugins"></a>`pub const plugins`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L70)

Plugin registry and lifecycle management.
Enables dynamic loading and management of framework extensions.

### <a id="pub-const-simd"></a>`pub const simd`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L75)

SIMD (Single Instruction, Multiple Data) vector operations.
Provides optimized vector math with automatic fallback to scalar operations
when SIMD is not available. Includes dot product, L2 norm, and cosine similarity.

### <a id="pub-const-utils"></a>`pub const utils`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L79)

General-purpose utility functions: time, math, string, lifecycle management.
See sub-modules for specialized utilities (crypto, encoding, fs, http, json, net).

### <a id="pub-const-os"></a>`pub const os`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L83)

Operating system abstraction layer.
Provides platform-independent access to OS features.

### <a id="pub-const-app-paths"></a>`pub const app_paths`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L86)

Cross-platform ABI app path resolver (primary config root).

### <a id="pub-const-time"></a>`pub const time`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L90)

Time utilities compatible with Zig 0.16.
Platform-aware implementations for unix timestamps, monotonic clocks, and sleep.

### <a id="pub-const-sync"></a>`pub const sync`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L94)

Synchronization primitives compatible with Zig 0.16.
Provides Mutex, RwLock, and other concurrency utilities.

### <a id="pub-const-io"></a>`pub const io`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L98)

I/O utilities and helpers for file and stream operations.
Designed for Zig 0.16's explicit I/O backend model.

### <a id="pub-const-stub-common"></a>`pub const stub_common`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L102)

Common stub utilities for feature-disabled builds.
Provides consistent error types and placeholder implementations.

### <a id="pub-const-matrix"></a>`pub const matrix`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L105)

Dense matrix operations with SIMD-accelerated multiply (v2).

### <a id="pub-const-tensor"></a>`pub const tensor`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L108)

Multi-dimensional tensor operations with broadcasting (v2).

### <a id="pub-const-security"></a>`pub const security`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L128)

Comprehensive security module providing authentication, authorization,
and encryption features. Includes:

- **API Keys**: Secure key generation with salted hashing
- **JWT**: Token-based authentication (HS256, HS384, HS512)
- **RBAC**: Role-based access control with permission caching
- **TLS**: Secure connection management (TLS 1.2/1.3)
- **mTLS**: Mutual TLS for bidirectional authentication
- **Secrets**: Encrypted credential storage with rotation
- **Rate Limiting**: Token bucket and sliding window algorithms
- **Encryption**: AES-256-GCM and ChaCha20-Poly1305
- **Audit**: Tamper-evident security event logging

See `security/mod.zig` for full API documentation.

### <a id="pub-const-resilience"></a>`pub const resilience`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L132)

Resilience patterns (circuit breaker, etc.) for fault-tolerant systems.
Shared implementations used by network, streaming, and gateway modules.

### <a id="pub-const-signal"></a>`pub const signal`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L136)

POSIX signal handling for graceful shutdown.
Sets a shared atomic flag on SIGINT/SIGTERM.

### <a id="pub-const-log"></a>`pub const log`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L148)

Log a message at the default scope. Shorthand for `logging.log`.
Usage: `log.info("message {}", .{value});`

### <a id="pub-const-logger"></a>`pub const Logger`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L152)

Scoped logger type for structured logging with context.
Create with `Logger.init(allocator, .{ .scope = "my_component" })`.

### <a id="pub-const-vectoradd"></a>`pub const vectorAdd`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L160)

Add two vectors element-wise using SIMD when available.
Falls back to scalar operations on platforms without SIMD support.

### <a id="pub-const-vectordot"></a>`pub const vectorDot`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L164)

Compute the dot product of two vectors using SIMD acceleration.
Returns the sum of element-wise products: sum(a[i] * b[i]).

### <a id="pub-const-vectorl2norm"></a>`pub const vectorL2Norm`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L168)

Compute the L2 (Euclidean) norm of a vector: sqrt(sum(v[i]^2)).
Uses SIMD for efficient computation on large vectors.

### <a id="pub-const-cosinesimilarity"></a>`pub const cosineSimilarity`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L173)

Compute cosine similarity between two vectors.
Returns a value in [-1, 1] where 1 indicates identical direction.
Formula: dot(a, b) / (norm(a) * norm(b))

### <a id="pub-const-hassimdsupport"></a>`pub const hasSimdSupport`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L177)

Check if the current platform supports SIMD operations.
Returns true if hardware SIMD is available and enabled.

### <a id="pub-const-simplemodulelifecycle"></a>`pub const SimpleModuleLifecycle`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L192)

Simple module lifecycle management with init/deinit callbacks.
Tracks initialization state and prevents double-init/deinit.

Example:
```zig
var lifecycle = SimpleModuleLifecycle{};
try lifecycle.init(myInitFn);
defer lifecycle.deinit(myDeinitFn);
```

### <a id="pub-const-lifecycleerror"></a>`pub const LifecycleError`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L198)

Errors that can occur during module lifecycle operations.
- `AlreadyInitialized`: Module was already initialized
- `NotInitialized`: Attempted operation on uninitialized module
- `InitFailed`: Initialization callback returned an error

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
