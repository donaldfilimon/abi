---
title: Security Module Guide
description: JWT, RBAC, session management, encryption, TLS/mTLS, certificate management, API keys, secrets, and audit logging
section: Guides
order: 92
permalink: /security-guide/
---

# Security Module Guide
## Summary
JWT, RBAC, session management, encryption, TLS/mTLS, certificate management, API keys, secrets, and audit logging

## Module Map

The security subsystem is exposed via `abi.auth.*` when the `auth` feature is
enabled. All sub-modules live in `src/services/shared/security/` and are
re-exported through `src/features/auth/mod.zig`.

| Sub-module | Import path | Purpose |
|---|---|---|
| JWT | `abi.auth.jwt` | HMAC-SHA256/384/512 token create and verify |
| API Keys | `abi.auth.api_keys` | Secure key generation, hashing, and validation |
| RBAC | `abi.auth.rbac` | Role-based access control with hierarchical permissions |
| Sessions | `abi.auth.session` | Server-side session management with expiry |
| Passwords | `abi.auth.password` | Argon2id, PBKDF2, scrypt hashing |
| CORS | `abi.auth.cors` | Cross-Origin Resource Sharing policy |
| Rate Limit | `abi.auth.rate_limit` | Token bucket, sliding window, leaky bucket |
| Encryption | `abi.auth.encryption` | AES-256-GCM, ChaCha20-Poly1305 |
| TLS | `abi.auth.tls` | Transport Layer Security configuration |
| mTLS | `abi.auth.mtls` | Mutual TLS client certificate verification |
| Certificates | `abi.auth.certificates` | X.509 certificate management |
| Secrets | `abi.auth.secrets` | Encrypted credential storage (Vault integration) |
| Audit | `abi.auth.audit` | Tamper-evident security event logging |
| Validation | `abi.auth.validation` | Input sanitization |
| IP Filter | `abi.auth.ip_filter` | IP allow/deny lists |
| Headers | `abi.auth.headers` | Security headers middleware |

## JWT (JSON Web Tokens)

Create and verify tokens using HMAC-SHA256/384/512:

```zig
const jwt = abi.auth.jwt;

// Create a token
const token = try jwt.create(allocator, .{
    .sub = "user-123",
    .exp = now + 3600,  // 1 hour
    .iat = now,
}, secret_key);
defer allocator.free(token);

// Verify and decode
const claims = try jwt.verify(allocator, token, secret_key);
```

Tokens are compact, URL-safe strings suitable for HTTP `Authorization: Bearer`
headers.

## RBAC (Role-Based Access Control)

Define roles with hierarchical permissions:

```zig
const rbac = abi.auth.rbac;

var authz = try rbac.Authorizer.init(allocator);
defer authz.deinit();

try authz.addRole("admin", &.{ .read, .write, .admin });
try authz.addRole("viewer", &.{ .read });

// Check permissions
const allowed = authz.check("viewer", .write);  // false
```

## Session Management

Server-side sessions with configurable TTL:

```zig
const session = abi.auth.session;

var store = try session.SessionStore.init(allocator, .{
    .ttl_seconds = 3600,
    .max_sessions = 10_000,
});
defer store.deinit();

const sess = try store.create("user-123");
// Later...
const found = store.get(sess.id);
```

## Encryption

Symmetric encryption using AES-256-GCM or ChaCha20-Poly1305:

```zig
const encryption = abi.auth.encryption;

// Encrypt
const ciphertext = try encryption.encrypt(allocator, plaintext, key, .aes_256_gcm);
defer allocator.free(ciphertext);

// Decrypt
const recovered = try encryption.decrypt(allocator, ciphertext, key, .aes_256_gcm);
defer allocator.free(recovered);
```

## TLS and mTLS

Configure transport security for servers and clients:

```zig
const tls = abi.auth.tls;

const config = tls.Config{
    .cert_file = "server.crt",
    .key_file = "server.key",
    .ca_file = "ca.crt",         // For mTLS
    .verify_client = true,        // Enable mTLS
    .min_version = .tls_1_3,
};
```

## Certificate Management

X.509 certificate utilities:

```zig
const certs = abi.auth.certificates;

const cert = try certs.load(allocator, "server.crt");
defer cert.deinit();

const is_valid = cert.isValid(now);
const days_left = cert.daysUntilExpiry(now);
```

## API Keys

Generate and validate API keys with secure hashing:

```zig
const api_keys = abi.auth.api_keys;

// Generate a new key
const key = try api_keys.generate(allocator, .{ .prefix = "abi_" });

// Validate
const valid = try api_keys.validate(allocator, key.raw, key.hash);
```

## Secrets Management

Encrypted credential storage with optional HashiCorp Vault integration:

```zig
const secrets = abi.auth.secrets;

var vault = try secrets.Store.init(allocator, .{
    .backend = .local_encrypted,  // or .vault
    .vault_addr = "https://vault.example.com:8200",
});
defer vault.deinit();

try vault.put("db-password", secret_bytes);
const value = try vault.get(allocator, "db-password");
```

## Audit Logging

Tamper-evident security event log:

```zig
const audit = abi.auth.audit;

var logger = try audit.Logger.init(allocator, .{
    .output_path = "audit.log",
    .tamper_detection = true,
});
defer logger.deinit();

try logger.log(.authentication, .{
    .user = "admin",
    .action = "login",
    .result = .success,
    .ip = "10.0.0.1",
});
```

## Generated Reference
## Overview

This guide is generated from repository metadata for **Guides** coverage and stays deterministic across runs.

## Build Snapshot

- Zig pin: `0.16.0-dev.2653+784e89fd4`
- Main tests: `1290` pass / `6` skip / `1296` total
- Feature tests: `2360` pass / `2365` total

## Feature Coverage

- **auth** — Authentication and security
  - Build flag: `enable_auth`
  - Source: `src/features/auth/mod.zig`

## Module Coverage

- `src/services/connectors/mod.zig` ([api](../api/connectors.html))

## Command Entry Points

- `abi agent` — Run AI agent (interactive or one-shot)
- `abi embed` — Generate embeddings from text (openai, mistral, cohere, ollama)
- `abi llm` — LLM inference (run, session, serve, providers, plugins, discover)
- `abi model` — Model management (list, download, remove, search)
- `abi ralph` — Ralph orchestrator (init, run, super, multi, status, gate, improve, skills)
- `abi train` — Training pipeline (run, llm, vision, auto, self, resume, info)

## Validation Commands

- `zig build typecheck`
- `zig build check-docs`
- `zig build run -- gendocs --check`

## Navigation

- API Reference: [../api/](../api/)
- API App: [../api-app/](../api-app/)
- Plans Index: [../plans/index.md](../plans/index.md)
- Source Root: [GitHub src tree](https://github.com/donaldfilimon/abi/tree/master/src)

## Maintenance Notes
- This page is generated by `zig build gendocs`.
- Edit template source in `tools/gendocs/templates/docs/` for structural changes.
- Edit generator logic in `tools/gendocs/` for data model or rendering changes.


---

*Generated automatically by `zig build gendocs`*


## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
