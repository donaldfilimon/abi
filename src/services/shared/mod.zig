//! Shared Utilities Module
//!
//! Common utilities, helpers, and cross-cutting concerns used throughout the ABI framework.
//! This module consolidates logging, SIMD operations, platform utilities, and security.
//!
//! # Overview
//!
//! The shared module provides foundational building blocks that are used across all ABI
//! framework components. It is organized into several categories:
//!
//! - **Core Utilities**: Error handling, logging, time, I/O operations
//! - **Security**: Authentication, authorization, encryption, secrets management
//! - **Performance**: SIMD operations, memory management, binary serialization
//! - **Networking**: HTTP client, network utilities, encoding/decoding
//!
//! # Usage
//!
//! Import the shared module and access components directly:
//!
//! ```zig
//! const shared = @import("shared");
//!
//! // Logging
//! shared.log.info("Application started", .{});
//!
//! // SIMD operations
//! const dot = shared.vectorDot(a, b);
//!
//! // Security
//! var jwt_manager = shared.security.JwtManager.init(allocator, secret, .{});
//! ```
//!
//! # Security Components
//!
//! The security sub-module provides comprehensive security features:
//!
//! | Component | Description |
//! |-----------|-------------|
//! | `api_keys` | API key generation, validation, rotation |
//! | `jwt` | JSON Web Token creation and verification |
//! | `rbac` | Role-based access control |
//! | `tls` | TLS/SSL connection management |
//! | `secrets` | Encrypted secrets storage |
//! | `rate_limit` | Request rate limiting |
//! | `encryption` | Data encryption at rest |
//! | `audit` | Security audit logging |
//!
//! # Thread Safety
//!
//! Most components in this module are thread-safe when used with proper synchronization.
//! Security components like `JwtManager`, `RateLimiter`, and `SecretsManager` include
//! internal mutex protection for concurrent access.

const std = @import("std");

// ============================================================================
// Core Shared Utilities
// ============================================================================

/// Error definitions and handling utilities for the ABI framework.
/// Provides standardized error types and conversion functions.
pub const errors = @import("errors.zig");

/// Logging infrastructure with configurable log levels and output destinations.
/// Supports structured logging with context and scoped loggers.
pub const logging = @import("logging.zig");

/// Plugin registry and lifecycle management.
/// Enables dynamic loading and management of framework extensions.
pub const plugins = @import("plugins.zig");

/// SIMD (Single Instruction, Multiple Data) vector operations.
/// Provides optimized vector math with automatic fallback to scalar operations
/// when SIMD is not available. Includes dot product, L2 norm, and cosine similarity.
pub const simd = @import("simd/mod.zig");

/// General-purpose utility functions: time, math, string, lifecycle management.
/// See sub-modules for specialized utilities (crypto, encoding, fs, http, json, net).
pub const utils = @import("utils.zig");

/// Operating system abstraction layer.
/// Provides platform-independent access to OS features.
pub const os = @import("os.zig");

/// Time utilities compatible with Zig 0.16.
/// Platform-aware implementations for unix timestamps, monotonic clocks, and sleep.
pub const time = @import("time.zig");

/// Synchronization primitives compatible with Zig 0.16.
/// Provides Mutex, RwLock, and other concurrency utilities.
pub const sync = @import("sync.zig");

/// I/O utilities and helpers for file and stream operations.
/// Designed for Zig 0.16's explicit I/O backend model.
pub const io = @import("io.zig");

/// Common stub utilities for feature-disabled builds.
/// Provides consistent error types and placeholder implementations.
pub const stub_common = @import("stub_common.zig");

/// Dense matrix operations with SIMD-accelerated multiply (v2).
pub const matrix = @import("matrix.zig");

/// Multi-dimensional tensor operations with broadcasting (v2).
pub const tensor = @import("tensor.zig");

// ============================================================================
// Security Sub-module
// ============================================================================

/// Comprehensive security module providing authentication, authorization,
/// and encryption features. Includes:
///
/// - **API Keys**: Secure key generation with salted hashing
/// - **JWT**: Token-based authentication (HS256, HS384, HS512)
/// - **RBAC**: Role-based access control with permission caching
/// - **TLS**: Secure connection management (TLS 1.2/1.3)
/// - **mTLS**: Mutual TLS for bidirectional authentication
/// - **Secrets**: Encrypted credential storage with rotation
/// - **Rate Limiting**: Token bucket and sliding window algorithms
/// - **Encryption**: AES-256-GCM and ChaCha20-Poly1305
/// - **Audit**: Tamper-evident security event logging
///
/// See `security/mod.zig` for full API documentation.
pub const security = @import("security/mod.zig");

/// Resilience patterns (circuit breaker, etc.) for fault-tolerant systems.
/// Shared implementations used by network, streaming, and gateway modules.
pub const resilience = @import("resilience/mod.zig");

/// POSIX signal handling for graceful shutdown.
/// Sets a shared atomic flag on SIGINT/SIGTERM.
pub const signal = @import("signal.zig");

// Utils sub-modules are accessible via `shared.utils.*` (e.g., `shared.utils.memory`,
// `shared.utils.json`, `shared.utils.crypto`). No direct re-exports here to avoid
// duplicate paths to the same modules.

// ============================================================================
// Commonly Used Re-exports
// ============================================================================

/// Log a message at the default scope. Shorthand for `logging.log`.
/// Usage: `log.info("message {}", .{value});`
pub const log = logging.log;

/// Scoped logger type for structured logging with context.
/// Create with `Logger.init(allocator, .{ .scope = "my_component" })`.
pub const Logger = logging.Logger;

// ============================================================================
// SIMD Re-exports for Convenience
// ============================================================================

/// Add two vectors element-wise using SIMD when available.
/// Falls back to scalar operations on platforms without SIMD support.
pub const vectorAdd = simd.vectorAdd;

/// Compute the dot product of two vectors using SIMD acceleration.
/// Returns the sum of element-wise products: sum(a[i] * b[i]).
pub const vectorDot = simd.vectorDot;

/// Compute the L2 (Euclidean) norm of a vector: sqrt(sum(v[i]^2)).
/// Uses SIMD for efficient computation on large vectors.
pub const vectorL2Norm = simd.vectorL2Norm;

/// Compute cosine similarity between two vectors.
/// Returns a value in [-1, 1] where 1 indicates identical direction.
/// Formula: dot(a, b) / (norm(a) * norm(b))
pub const cosineSimilarity = simd.cosineSimilarity;

/// Check if the current platform supports SIMD operations.
/// Returns true if hardware SIMD is available and enabled.
pub const hasSimdSupport = simd.hasSimdSupport;

// ============================================================================
// Lifecycle Utilities
// ============================================================================

/// Simple module lifecycle management with init/deinit callbacks.
/// Tracks initialization state and prevents double-init/deinit.
///
/// Example:
/// ```zig
/// var lifecycle = SimpleModuleLifecycle{};
/// try lifecycle.init(myInitFn);
/// defer lifecycle.deinit(myDeinitFn);
/// ```
pub const SimpleModuleLifecycle = utils.SimpleModuleLifecycle;

/// Errors that can occur during module lifecycle operations.
/// - `AlreadyInitialized`: Module was already initialized
/// - `NotInitialized`: Attempted operation on uninitialized module
/// - `InitFailed`: Initialization callback returned an error
pub const LifecycleError = utils.LifecycleError;

test "shared module" {
    // Basic smoke test
    try std.testing.expect(hasSimdSupport() or !hasSimdSupport());
}
