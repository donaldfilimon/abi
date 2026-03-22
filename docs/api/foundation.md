---
title: foundation API
purpose: Generated API reference for foundation
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# foundation

> Shared Utilities Module

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
// From external modules (CLI, tests):
const shared = @import("abi").foundation;
// From within the abi module:
// const shared = @import("../../services/shared/mod.zig");

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

**Source:** [`src/services/shared/mod.zig`](../../src/services/shared/mod.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-errors"></a>`pub const errors`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L65)

Error definitions and handling utilities for the ABI framework.
Provides standardized error types and conversion functions.

### <a id="pub-const-types"></a>`pub const types`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L69)

Foundational type definitions used across multiple domains.
Standardizes confidence, emotional context, and identity primitives.

### <a id="pub-const-logging"></a>`pub const logging`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L73)

Logging infrastructure with configurable log levels and output destinations.
Supports structured logging with context and scoped loggers.

### <a id="pub-const-plugins"></a>`pub const plugins`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L77)

Plugin registry and lifecycle management.
Enables dynamic loading and management of framework extensions.

### <a id="pub-const-simd"></a>`pub const simd`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L82)

SIMD (Single Instruction, Multiple Data) vector operations.
Provides optimized vector math with automatic fallback to scalar operations
when SIMD is not available. Includes dot product, L2 norm, and cosine similarity.

### <a id="pub-const-utils"></a>`pub const utils`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L86)

General-purpose utility functions: time, math, string, lifecycle management.
See sub-modules for specialized utilities (crypto, encoding, fs, http, json, net).

### <a id="pub-const-os"></a>`pub const os`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L90)

Operating system abstraction layer.
Provides platform-independent access to OS features.

### <a id="pub-const-app-paths"></a>`pub const app_paths`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L93)

Cross-platform ABI app path resolver (primary config root).

### <a id="pub-const-time"></a>`pub const time`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L97)

Time utilities compatible with Zig 0.16.
Platform-aware implementations for unix timestamps, monotonic clocks, and sleep.

### <a id="pub-const-sync"></a>`pub const sync`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L101)

Synchronization primitives compatible with Zig 0.16.
Provides Mutex, RwLock, and other concurrency utilities.

### <a id="pub-const-io"></a>`pub const io`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L105)

I/O utilities and helpers for file and stream operations.
Designed for Zig 0.16's explicit I/O backend model.

### <a id="pub-const-stub-common"></a>`pub const stub_common`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L109)

Common stub utilities for feature-disabled builds.
Provides consistent error types and placeholder implementations.

### <a id="pub-const-matrix"></a>`pub const matrix`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L112)

Dense matrix operations with SIMD-accelerated multiply (v2).

### <a id="pub-const-tensor"></a>`pub const tensor`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L115)

Multi-dimensional tensor operations with broadcasting (v2).

### <a id="pub-const-security"></a>`pub const security`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L135)

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

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L139)

Resilience patterns (circuit breaker, etc.) for fault-tolerant systems.
Shared implementations used by network, streaming, and gateway modules.

### <a id="pub-const-signal"></a>`pub const signal`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L143)

POSIX signal handling for graceful shutdown.
Sets a shared atomic flag on SIGINT/SIGTERM.

### <a id="pub-const-log"></a>`pub const log`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L155)

Log a message at the default scope. Shorthand for `logging.log`.
Usage: `log.info("message {}", .{value});`

### <a id="pub-const-scopedtimer"></a>`pub const ScopedTimer`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L159)

Scoped timer for profiling blocks of code.
Create with `ScopedTimer.start("label")` and call `.stop()` when done.

### <a id="pub-const-vectoradd"></a>`pub const vectorAdd`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L167)

Add two vectors element-wise using SIMD when available.
Falls back to scalar operations on platforms without SIMD support.

### <a id="pub-const-vectordot"></a>`pub const vectorDot`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L171)

Compute the dot product of two vectors using SIMD acceleration.
Returns the sum of element-wise products: sum(a[i] * b[i]).

### <a id="pub-const-vectorl2norm"></a>`pub const vectorL2Norm`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L175)

Compute the L2 (Euclidean) norm of a vector: sqrt(sum(v[i]^2)).
Uses SIMD for efficient computation on large vectors.

### <a id="pub-const-cosinesimilarity"></a>`pub const cosineSimilarity`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L180)

Compute cosine similarity between two vectors.
Returns a value in [-1, 1] where 1 indicates identical direction.
Formula: dot(a, b) / (norm(a) * norm(b))

### <a id="pub-const-hassimdsupport"></a>`pub const hasSimdSupport`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L184)

Check if the current platform supports SIMD operations.
Returns true if hardware SIMD is available and enabled.

### <a id="pub-const-simplemodulelifecycle"></a>`pub const SimpleModuleLifecycle`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L199)

Simple module lifecycle management with init/deinit callbacks.
Tracks initialization state and prevents double-init/deinit.

Example:
```zig
var lifecycle = SimpleModuleLifecycle{};
try lifecycle.init(myInitFn);
defer lifecycle.deinit(myDeinitFn);
```

### <a id="pub-const-lifecycleerror"></a>`pub const LifecycleError`

<sup>**const**</sup> | [source](../../src/services/shared/mod.zig#L205)

Errors that can occur during module lifecycle operations.
- `AlreadyInitialized`: Module was already initialized
- `NotInitialized`: Attempted operation on uninitialized module
- `InitFailed`: Initialization callback returned an error



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `zig test <file> -fno-emit-bin` as fallback evidence while replacing the toolchain.
