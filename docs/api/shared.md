# shared

> Shared utilities (SIMD, time, sync, security, etc.).

**Source:** [`src/services/shared/mod.zig`](../../src/services/shared/mod.zig)

**Availability:** Always enabled

---

Shared Utilities Module

Common utilities, helpers, and cross-cutting concerns used throughout the ABI framework.
This module consolidates logging, SIMD operations, platform utilities, and security.

# Overview

The shared module provides foundational building blocks that are used across all ABI
framework components. It is organized into several categories:

- **Core Utilities**: Error handling, logging, time, I/O operations
- **Security**: Authentication, authorization, encryption, secrets management
- **Performance**: SIMD operations, memory management, binary serialization
- **Networking**: HTTP client, network utilities, encoding/decoding

# Usage

Import the shared module and access components directly:

```zig
const shared = @import("shared");

// Logging
shared.log.info("Application started", .{});

// SIMD operations
const dot = shared.vectorDot(a, b);

// Security
var jwt_manager = shared.security.JwtManager.init(allocator, secret, .{});
```

# Security Components

The security sub-module provides comprehensive security features:

| Component | Description |
|-----------|-------------|
| `api_keys` | API key generation, validation, rotation |
| `jwt` | JSON Web Token creation and verification |
| `rbac` | Role-based access control |
| `tls` | TLS/SSL connection management |
| `secrets` | Encrypted secrets storage |
| `rate_limit` | Request rate limiting |
| `encryption` | Data encryption at rest |
| `audit` | Security audit logging |

# Thread Safety

Most components in this module are thread-safe when used with proper synchronization.
Security components like `JwtManager`, `RateLimiter`, and `SecretsManager` include
internal mutex protection for concurrent access.

---

## API

### `pub const errors`

<sup>**type**</sup>

Error definitions and handling utilities for the ABI framework.
Provides standardized error types and conversion functions.

### `pub const logging`

<sup>**type**</sup>

Logging infrastructure with configurable log levels and output destinations.
Supports structured logging with context and scoped loggers.

### `pub const plugins`

<sup>**type**</sup>

Plugin registry and lifecycle management.
Enables dynamic loading and management of framework extensions.

### `pub const simd`

<sup>**type**</sup>

SIMD (Single Instruction, Multiple Data) vector operations.
Provides optimized vector math with automatic fallback to scalar operations
when SIMD is not available. Includes dot product, L2 norm, and cosine similarity.

### `pub const utils`

<sup>**type**</sup>

General-purpose utility functions: time, math, string, lifecycle management.
See sub-modules for specialized utilities (crypto, encoding, fs, http, json, net).

### `pub const os`

<sup>**type**</sup>

Operating system abstraction layer.
Provides platform-independent access to OS features.

### `pub const time`

<sup>**type**</sup>

Time utilities compatible with Zig 0.16.
Platform-aware implementations for unix timestamps, monotonic clocks, and sleep.

### `pub const sync`

<sup>**type**</sup>

Synchronization primitives compatible with Zig 0.16.
Provides Mutex, RwLock, and other concurrency utilities.

### `pub const io`

<sup>**type**</sup>

I/O utilities and helpers for file and stream operations.
Designed for Zig 0.16's explicit I/O backend model.

### `pub const stub_common`

<sup>**type**</sup>

Common stub utilities for feature-disabled builds.
Provides consistent error types and placeholder implementations.

### `pub const matrix`

<sup>**type**</sup>

Dense matrix operations with SIMD-accelerated multiply (v2).

### `pub const tensor`

<sup>**type**</sup>

Multi-dimensional tensor operations with broadcasting (v2).

### `pub const security`

<sup>**type**</sup>

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

### `pub const resilience`

<sup>**type**</sup>

Resilience patterns (circuit breaker, etc.) for fault-tolerant systems.
Shared implementations used by network, streaming, and gateway modules.

### `pub const signal`

<sup>**type**</sup>

POSIX signal handling for graceful shutdown.
Sets a shared atomic flag on SIGINT/SIGTERM.

### `pub const log`

<sup>**const**</sup>

Log a message at the default scope. Shorthand for `logging.log`.
Usage: `log.info("message {}", .{value});`

### `pub const Logger`

<sup>**const**</sup>

Scoped logger type for structured logging with context.
Create with `Logger.init(allocator, .{ .scope = "my_component" })`.

### `pub const vectorAdd`

<sup>**const**</sup>

Add two vectors element-wise using SIMD when available.
Falls back to scalar operations on platforms without SIMD support.

### `pub const vectorDot`

<sup>**const**</sup>

Compute the dot product of two vectors using SIMD acceleration.
Returns the sum of element-wise products: sum(a[i] * b[i]).

### `pub const vectorL2Norm`

<sup>**const**</sup>

Compute the L2 (Euclidean) norm of a vector: sqrt(sum(v[i]^2)).
Uses SIMD for efficient computation on large vectors.

### `pub const cosineSimilarity`

<sup>**const**</sup>

Compute cosine similarity between two vectors.
Returns a value in [-1, 1] where 1 indicates identical direction.
Formula: dot(a, b) / (norm(a) * norm(b))

### `pub const hasSimdSupport`

<sup>**const**</sup>

Check if the current platform supports SIMD operations.
Returns true if hardware SIMD is available and enabled.

### `pub const SimpleModuleLifecycle`

<sup>**const**</sup>

Simple module lifecycle management with init/deinit callbacks.
Tracks initialization state and prevents double-init/deinit.

Example:
```zig
var lifecycle = SimpleModuleLifecycle{};
try lifecycle.init(myInitFn);
defer lifecycle.deinit(myDeinitFn);
```

### `pub const LifecycleError`

<sup>**const**</sup>

Errors that can occur during module lifecycle operations.
- `AlreadyInitialized`: Module was already initialized
- `NotInitialized`: Attempted operation on uninitialized module
- `InitFailed`: Initialization callback returned an error

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
