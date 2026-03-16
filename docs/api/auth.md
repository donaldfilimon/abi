---
title: auth API
purpose: Generated API reference for auth
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# auth

> Auth Module

Authentication and security infrastructure for the ABI framework.
Re-exports the full security infrastructure from `services/shared/security/`.

When the `auth` feature is enabled, all security sub-modules are available:
- `abi.auth.jwt` — JSON Web Tokens (HMAC-SHA256/384/512)
- `abi.auth.api_keys` — API key management with secure hashing
- `abi.auth.rbac` — Role-based access control
- `abi.auth.session` — Session management
- `abi.auth.password` — Secure password hashing (Argon2id, PBKDF2, scrypt)
- `abi.auth.cors` — Cross-Origin Resource Sharing
- `abi.auth.rate_limit` — Token bucket, sliding window, leaky bucket
- `abi.auth.encryption` — AES-256-GCM, ChaCha20-Poly1305
- `abi.auth.tls` / `abi.auth.mtls` — Transport security
- `abi.auth.certificates` — X.509 certificate management
- `abi.auth.secrets` — Encrypted credential storage
- `abi.auth.audit` — Tamper-evident security event logging
- `abi.auth.validation` — Input sanitization
- `abi.auth.ip_filter` — IP allow/deny lists
- `abi.auth.headers` — Security headers middleware

**Source:** [`src/features/auth/mod.zig`](../../src/features/auth/mod.zig)

**Build flag:** `-Dfeat_auth=true`

---

## API

### <a id="pub-fn-init-std-mem-allocator-config-authconfig-autherror-void"></a>`pub fn init(_: std.mem.Allocator, config: AuthConfig) AuthError!void`

<sup>**fn**</sup> | [source](../../src/features/auth/mod.zig#L103)

Initialise the auth module with a caller-provided config.
If `config.jwt_secret` is set, it will be used for all subsequent token
operations.  Otherwise the default dev secret is used and a warning is
printed to stderr.

### <a id="pub-fn-createtoken-allocator-std-mem-allocator-user-id-const-u8-autherror-token"></a>`pub fn createToken( allocator: std.mem.Allocator, user_id: []const u8, ) AuthError!Token`

<sup>**fn**</sup> | [source](../../src/features/auth/mod.zig#L134)

Create a signed JWT token for the given user_id.
Delegates to `jwt.JwtManager.createToken` from `services/shared/security/jwt.zig`.

### <a id="pub-fn-verifytoken-token-str-const-u8-autherror-token"></a>`pub fn verifyToken(token_str: []const u8) AuthError!Token`

<sup>**fn**</sup> | [source](../../src/features/auth/mod.zig#L163)

Verify a JWT token string and return parsed token info.
Delegates to `jwt.JwtManager.verifyToken` from `services/shared/security/jwt.zig`.
Uses the default dev secret; production callers should use `jwt.JwtManager`
directly with their own secret.

Note: The returned Token's `.claims.sub` is heap-allocated via page_allocator
when non-empty and should be freed by the caller if needed. The `.raw` field
points to the input `token_str` (caller-owned, not duped).

### <a id="pub-fn-createsession-allocator-std-mem-allocator-user-id-const-u8-autherror-session"></a>`pub fn createSession( allocator: std.mem.Allocator, user_id: []const u8, ) AuthError!Session`

<sup>**fn**</sup> | [source](../../src/features/auth/mod.zig#L212)

Create a new session for the given user_id.
Delegates to `session.SessionManager.create` from
`services/shared/security/session.zig`.

The returned `Session.id` and `Session.user_id` are heap-allocated via the
provided allocator; callers should free them when done (or use an arena).

### <a id="pub-fn-checkpermission-user-id-const-u8-permission-permission-autherror-bool"></a>`pub fn checkPermission(user_id: []const u8, permission: Permission) AuthError!bool`

<sup>**fn**</sup> | [source](../../src/features/auth/mod.zig#L267)

Check if a user has a given permission.
Delegates to `rbac.RbacManager.hasPermission` from
`services/shared/security/rbac.zig`.

Maps the auth-level `Permission` enum to the RBAC module's `Permission`.
Creates an ephemeral RbacManager with default roles. Without explicit
role assignment the user will have no permissions, so this returns false
by default — callers needing real RBAC should use `rbac.RbacManager`
directly and assign roles.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` / `zig build check-docs` on supported hosts. On Darwin 25+ / macOS 26+, ABI expects a host-built or otherwise known-good Zig matching `.zigversion`. If stock prebuilt Zig is linker-blocked, record `zig fmt --check ...` plus `./tools/scripts/run_build.sh typecheck --summary all` as fallback evidence while replacing the toolchain.
