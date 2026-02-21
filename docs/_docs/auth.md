---
title: "Auth & Security"
description: "Authentication, authorization, and security infrastructure"
section: "Operations"
order: 1
---

# Auth & Security

The auth module provides a comprehensive security infrastructure for the ABI
framework, re-exporting 16 specialized security sub-modules from
`services/shared/security/`. It covers authentication, authorization, encryption,
rate limiting, input validation, and audit logging.

- **Build flag:** `-Denable-auth=true` (default: enabled)
- **Namespace:** `abi.auth`
- **Source:** `src/features/auth/` and `src/services/shared/security/`

## Overview

When the auth feature is enabled, all 16 security sub-modules are accessible
through the `abi.auth` namespace. The feature gate controls the high-level
Context lifecycle and auth functions (`createToken`, `verifyToken`,
`createSession`, `checkPermission`), while the underlying security sub-modules
are always compiled since they live in `services/shared/`.

Key capabilities:

- **JWT** -- JSON Web Tokens with HMAC-SHA256/384/512 and RS256
- **API Keys** -- Key generation, validation, revocation, and rotation with secure hashing
- **RBAC** -- Role-based access control with permission checking
- **Sessions** -- Session lifecycle management
- **Passwords** -- Secure hashing via Argon2id, PBKDF2, and scrypt
- **Rate Limiting** -- Token bucket, sliding window, leaky bucket, and fixed window algorithms
- **Encryption** -- AES-256-GCM and ChaCha20-Poly1305
- **TLS / mTLS** -- Transport security and mutual TLS
- **Certificates** -- X.509 certificate management
- **Secrets** -- Encrypted credential storage with `ABI_MASTER_KEY`
- **CORS** -- Cross-Origin Resource Sharing configuration
- **Audit** -- Tamper-evident security event logging
- **Validation** -- Input sanitization for emails, URLs, and strings
- **IP Filter** -- IP allow/deny lists
- **Security Headers** -- Middleware for CSP, HSTS, and other security headers
- **CSPRNG** -- Cryptographically secure random number generation

## Quick Start

```zig
const abi = @import("abi");

// Initialize the framework with auth defaults
var builder = abi.Framework.builder(allocator);
var framework = try builder
    .withAuthDefaults()
    .build();
defer framework.deinit();

// Check feature status
if (!abi.auth.isEnabled()) {
    // Auth is compiled out -- high-level functions return error.FeatureDisabled
    return;
}

// Create and verify tokens
const token = try abi.auth.createToken(allocator, "user-123");
const verified = try abi.auth.verifyToken(token.raw);

// Check permissions
const allowed = try abi.auth.checkPermission("user-123", .write);
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `AuthConfig` | Configuration for the auth module |
| `Context` | Feature context for Framework integration |
| `AuthError` | Error set: `FeatureDisabled`, `InvalidCredentials`, `TokenExpired`, `Unauthorized`, `OutOfMemory` |
| `Token` | JWT token with raw string and claims (`sub`, `exp`, `iat`) |
| `Session` | Session record with `id`, `user_id`, timestamps |
| `Permission` | Permission level enum: `read`, `write`, `admin` |

### Key Functions

| Function | Description |
|----------|-------------|
| `init(allocator, AuthConfig)` | Initialize the auth module |
| `deinit()` | Tear down the auth module |
| `isEnabled()` | Returns `true` when auth feature is compiled in |
| `createToken(allocator, user_id)` | Create a new JWT token |
| `verifyToken(raw)` | Verify and decode a token |
| `createSession(allocator, user_id)` | Start a new session |
| `checkPermission(user_id, Permission)` | Check if a user has a permission |

### Security Sub-modules (16 modules)

| Sub-module | Namespace | Description |
|------------|-----------|-------------|
| `jwt` | `abi.auth.jwt` | JSON Web Tokens (HS256, HS384, HS512, RS256) |
| `api_keys` | `abi.auth.api_keys` | API key management with secure hashing |
| `rbac` | `abi.auth.rbac` | Role-based access control |
| `session` | `abi.auth.session` | Session management |
| `password` | `abi.auth.password` | Argon2id, PBKDF2, scrypt password hashing |
| `rate_limit` | `abi.auth.rate_limit` | Token bucket, sliding window, leaky bucket |
| `encryption` | `abi.auth.encryption` | AES-256-GCM, ChaCha20-Poly1305 |
| `tls` | `abi.auth.tls` | Transport layer security |
| `mtls` | `abi.auth.mtls` | Mutual TLS |
| `certificates` | `abi.auth.certificates` | X.509 certificate management |
| `secrets` | `abi.auth.secrets` | Encrypted credential storage |
| `audit` | `abi.auth.audit` | Tamper-evident event logging |
| `cors` | `abi.auth.cors` | Cross-Origin Resource Sharing |
| `validation` | `abi.auth.validation` | Input sanitization |
| `ip_filter` | `abi.auth.ip_filter` | IP allow/deny lists |
| `headers` | `abi.auth.headers` | Security headers middleware |

## Configuration

The `ABI_MASTER_KEY` environment variable is used by the secrets sub-module
for encrypted credential storage in production.

```zig
// JWT Manager
var jwt_manager = abi.auth.jwt.JwtManager.init(allocator, "secret-key-32bytes!!", .{
    .token_lifetime = 3600,   // 1 hour
    .issuer = "my-service",
});
defer jwt_manager.deinit();

// API Key Manager
var key_manager = abi.auth.api_keys.ApiKeyManager.init(allocator, .{});
defer key_manager.deinit();

// RBAC Manager
var rbac_manager = try abi.auth.rbac.RbacManager.init(allocator, .{});
defer rbac_manager.deinit();

// Rate Limiter (100 requests per 60-second window)
var limiter = abi.auth.rate_limit.RateLimiter.init(allocator, .{
    .enabled = true,
    .requests = 100,
    .window_seconds = 60,
});
defer limiter.deinit();

// Input Validator
var validator = abi.auth.validation.Validator.init(allocator, .{});
const email_ok = validator.validateEmail("user@example.com");
const url_ok = validator.validateUrl("https://api.example.com/v1");
```

## Examples

See `examples/auth.zig` for a complete working example that demonstrates JWT,
API keys, RBAC, rate limiting, and input validation.

```bash
zig build run-auth
```

## Disabling at Build Time

```bash
zig build -Denable-auth=false
```

When disabled, `abi.auth.isEnabled()` returns `false` and all high-level
functions (`createToken`, `verifyToken`, `createSession`, `checkPermission`)
return `error.FeatureDisabled`. The 16 security sub-modules remain accessible
since they are always compiled as part of `services/shared/security/`.

## Related

- [Connectors](connectors.html) -- LLM provider connectors use API key auth
- [Deployment](deployment.html) -- Production secrets and `ABI_MASTER_KEY`
- [Observability](observability.html) -- Security event metrics

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
