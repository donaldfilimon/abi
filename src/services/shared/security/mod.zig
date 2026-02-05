//! Security Module
//!
//! Comprehensive authentication, authorization, and security features for the ABI framework.
//!
//! # Overview
//!
//! This module provides production-ready security primitives organized into these categories:
//!
//! ## Authentication
//! - **API Keys**: Secure key generation with salted hashing (SHA-256, SHA-512, BLAKE3)
//! - **JWT**: JSON Web Tokens with HMAC (HS256/384/512) and blacklist support
//! - **Passwords**: Secure hashing with Argon2id, PBKDF2, or scrypt
//! - **Sessions**: Session management with configurable expiration
//!
//! ## Authorization
//! - **RBAC**: Role-based access control with permission caching
//! - **IP Filtering**: Allow/deny lists with CIDR range support
//!
//! ## Transport Security
//! - **TLS**: TLS 1.2/1.3 connection management
//! - **mTLS**: Mutual TLS for bidirectional authentication
//! - **Certificates**: X.509 certificate management and rotation
//!
//! ## Data Protection
//! - **Encryption**: AES-256-GCM and ChaCha20-Poly1305 for data at rest
//! - **Secrets**: Encrypted credential storage with rotation support
//! - **Secure Delete**: Multi-pass file overwriting
//!
//! ## Request Security
//! - **Rate Limiting**: Token bucket, sliding window, and leaky bucket algorithms
//! - **Validation**: Input sanitization and validation
//! - **Security Headers**: CSP, HSTS, X-Frame-Options, etc.
//! - **CORS**: Cross-Origin Resource Sharing configuration
//!
//! ## Monitoring
//! - **Audit Logging**: Tamper-evident security event logging with HMAC chains
//!
//! # Quick Start
//!
//! ```zig
//! const security = @import("shared").security;
//!
//! // JWT authentication
//! var jwt_mgr = security.JwtManager.init(allocator, secret_key, .{
//!     .token_lifetime = 3600,
//!     .issuer = "my-app",
//! });
//! defer jwt_mgr.deinit();
//!
//! const token = try jwt_mgr.createToken(.{ .sub = "user123" });
//! const verified = try jwt_mgr.verifyToken(token);
//!
//! // Rate limiting
//! var limiter = security.RateLimiter.init(allocator, security.RateLimitPresets.standard);
//! defer limiter.deinit();
//!
//! const status = limiter.check(client_ip);
//! if (!status.allowed) {
//!     return error.RateLimited;
//! }
//! ```
//!
//! # Security Considerations
//!
//! - **Master Keys**: Set `ABI_MASTER_KEY` environment variable for production secrets management
//! - **JWT "none"**: The `allow_none_algorithm` option is disabled by default; enabling it logs a warning
//! - **Rate Limiting**: Enable on all public endpoints to prevent abuse
//! - **Audit Logging**: Enable HMAC chains for tamper detection in compliance scenarios
//! - **TLS**: Always verify certificates in production (`verify_certificates = true`)
//!
//! See `SECURITY.md` in the repository root for complete security guidelines.

// ============================================================================
// Core Security Modules
// ============================================================================

/// API key management with secure hashing and rotation.
/// Supports SHA-256, SHA-512, and BLAKE3 hashing algorithms with configurable
/// iteration counts for key stretching. Includes timing-safe comparison to
/// prevent timing attacks.
pub const api_keys = @import("api_keys.zig");

/// Mutual TLS (mTLS) for bidirectional certificate authentication.
/// Extends TLS with client certificate verification for service-to-service
/// authentication in zero-trust architectures.
pub const mtls = @import("mtls.zig");

/// Role-based access control (RBAC) with permission caching.
/// Provides hierarchical roles, permission checks, and cache invalidation.
/// Default roles: admin, user, readonly, metrics, manager, memory_user, link_admin.
pub const rbac = @import("rbac.zig");

/// TLS/SSL connection management supporting TLS 1.2 and 1.3.
/// Includes certificate validation, hostname verification, and ALPN negotiation.
pub const tls = @import("tls.zig");

// ============================================================================
// Extended Security Modules
// ============================================================================

/// Security audit logging with tamper-evident hash chains.
/// Provides structured event logging with severity levels, categories,
/// and optional HMAC-based integrity verification for compliance.
pub const audit = @import("audit.zig");

/// X.509 certificate management and rotation.
/// Handles certificate lifecycle including generation, validation,
/// expiration tracking, and automatic rotation.
pub const certificates = @import("certificates.zig");

/// Cross-Origin Resource Sharing (CORS) configuration.
/// Configurable allowed origins, methods, headers, and credentials handling.
/// Includes presets for common configurations.
pub const cors = @import("cors.zig");

/// Encryption at rest using AEAD algorithms.
/// Supports AES-256-GCM and ChaCha20-Poly1305 with key derivation
/// (HKDF, PBKDF2, Argon2id, scrypt) and envelope encryption.
pub const encryption = @import("encryption.zig");

/// Security headers middleware for HTTP responses.
/// Configures Content-Security-Policy, HSTS, X-Frame-Options,
/// X-Content-Type-Options, Referrer-Policy, and more.
pub const headers = @import("headers.zig");

/// IP address filtering with allow/deny lists.
/// Supports IPv4/IPv6 addresses and CIDR ranges.
/// Includes automatic blocking with configurable ban durations.
pub const ip_filter = @import("ip_filter.zig");

/// JSON Web Token (JWT) authentication.
/// Supports HMAC algorithms (HS256, HS384, HS512) with token creation,
/// verification, refresh, and blacklist-based revocation.
pub const jwt = @import("jwt.zig");

/// Secure password hashing with Argon2id, PBKDF2, and scrypt.
/// Includes password strength analysis and secure random generation.
pub const password = @import("password.zig");

/// Rate limiting with multiple algorithms.
/// Token bucket (allows bursts), sliding window (smooth distribution),
/// fixed window (simple), and leaky bucket (constant rate).
/// Supports per-IP, per-user, per-API-key, and global scopes.
pub const rate_limit = @import("rate_limit.zig");

/// Secrets management with encrypted storage.
/// Supports environment variables, encrypted files, and vault backends.
/// Includes automatic rotation and secure memory handling.
pub const secrets = @import("secrets.zig");

/// Session management for user sessions.
/// Configurable expiration, secure token generation, and session invalidation.
pub const session = @import("session.zig");

/// Input validation and sanitization.
/// Prevents injection attacks through configurable validation rules
/// and automatic sanitization of user input.
pub const validation = @import("validation.zig");

// ============================================================================
// API Key Management Types
// ============================================================================

/// Manager for API key lifecycle: generation, validation, rotation, revocation.
/// Thread-safe with atomic key ID generation.
pub const ApiKeyManager = api_keys.ApiKeyManager;

/// Configuration for API key generation and validation.
/// Includes key length, prefix, hash algorithm, and rotation settings.
pub const ApiKeyConfig = api_keys.ApiKeyConfig;

/// Represents a stored API key with metadata.
/// Contains hashed key, salt, scopes, expiration, and usage tracking.
pub const ApiKey = api_keys.ApiKey;

/// Result of API key generation containing the key ID and plaintext key.
/// The plaintext key should be shown to the user once and never stored.
pub const GeneratedKey = api_keys.GeneratedKey;

/// Errors that can occur during API key operations.
pub const ApiKeyError = api_keys.ApiKeyError;

/// Hash algorithms available for API key hashing.
/// Options: sha256, sha512, blake3 (default).
pub const HashAlgorithm = api_keys.HashAlgorithm;

/// Salt length in bytes for API key hashing (16 bytes).
pub const SALT_LENGTH = api_keys.SALT_LENGTH;

/// Default number of hash iterations for key derivation (100,000).
pub const DEFAULT_HASH_ITERATIONS = api_keys.DEFAULT_HASH_ITERATIONS;

// ============================================================================
// RBAC Types
// ============================================================================

/// Manager for role-based access control.
/// Handles role creation, user-role assignments, and permission checks
/// with internal caching for performance.
pub const RbacManager = rbac.RbacManager;

/// Configuration for RBAC behavior.
/// Controls default roles, custom role creation, and cache settings.
pub const RbacConfig = rbac.RbacConfig;

/// Represents a role with a set of permissions.
/// Includes name, description, and whether it's a system-defined role.
pub const Role = rbac.Role;

/// Records a role assignment to a user.
/// Tracks who granted the role and when, with optional expiration.
pub const RoleAssignment = rbac.RoleAssignment;

/// Available permissions in the system.
/// Includes standard permissions (read, write, delete, admin, execute)
/// and specialized permissions (memory_*, link_*) for unified memory/linking.
pub const Permission = rbac.Permission;

// ============================================================================
// TLS Types
// ============================================================================

/// TLS connection for secure communication.
/// Supports both client and server modes with handshake management.
pub const TlsConnection = tls.TlsConnection;

/// Configuration for TLS connections.
/// Includes version constraints, cipher suites, certificate paths, and ALPN.
pub const TlsConfig = tls.TlsConfig;

/// Supported TLS protocol versions.
/// Options: tls10, tls11, tls12, tls13.
pub const TlsVersion = tls.TlsVersion;

/// X.509 certificate representation.
/// Contains DER encoding, validity period, and subject information.
pub const TlsCertificate = tls.TlsCertificate;

/// Errors that can occur during TLS operations.
pub const TlsError = tls.TlsError;

/// Store for trusted certificates and revocation tracking.
/// Manages CA certificates and certificate revocation lists.
pub const CertificateStore = tls.CertificateStore;

/// TLS handshake state machine states.
/// Tracks progress through the TLS handshake protocol.
pub const HandshakeState = tls.HandshakeState;

// ============================================================================
// mTLS Types
// ============================================================================

/// Mutual TLS connection requiring client certificates.
/// Extends TLS with mandatory client authentication.
pub const MtlsConnection = mtls.MtlsConnection;

/// Configuration for mutual TLS.
/// Specifies required client certificate attributes and validation rules.
pub const MtlsConfig = mtls.MtlsConfig;

/// Information extracted from a validated client certificate.
/// Used for authorization decisions based on certificate attributes.
pub const ClientCertificateInfo = mtls.ClientCertificateInfo;

// ============================================================================
// JWT Types
// ============================================================================

/// JWT manager for token creation, verification, and revocation.
/// Thread-safe with internal mutex protection and token statistics.
pub const JwtManager = jwt.JwtManager;

/// Configuration for JWT operations.
/// Includes algorithm, lifetimes, issuer, audience, and blacklist settings.
pub const JwtConfig = jwt.JwtConfig;

/// JWT signing algorithms.
/// HMAC: hs256, hs384, hs512. RSA: rs256 (not yet implemented).
/// "none" algorithm available but disabled by default for security.
pub const JwtAlgorithm = jwt.Algorithm;

/// Standard and custom JWT claims.
/// Standard: iss, sub, aud, exp, nbf, iat, jti.
/// Custom claims stored in a string hash map.
pub const JwtClaims = jwt.Claims;

/// Decoded and verified JWT token.
/// Contains header, claims, signature, and verification status.
pub const JwtToken = jwt.Token;

/// Extract bearer token from an Authorization header.
/// Returns the token string if the header starts with "Bearer ".
pub const extractBearerToken = jwt.extractBearerToken;

/// Generate a cryptographically secure random secret key.
/// Returns a byte slice of the specified length.
pub const generateSecretKey = jwt.generateSecretKey;

// ============================================================================
// Password Hashing Types
// ============================================================================

/// Password hasher supporting multiple algorithms.
/// Argon2id (recommended), PBKDF2, scrypt with configurable parameters.
pub const PasswordHasher = password.PasswordHasher;

/// Configuration for password hashing.
/// Specifies algorithm and algorithm-specific parameters.
pub const PasswordConfig = password.PasswordConfig;

/// Password strength classification.
/// Levels: very_weak, weak, fair, strong, very_strong.
pub const PasswordStrength = password.PasswordStrength;

/// Detailed password strength analysis results.
/// Includes score, entropy estimate, and improvement suggestions.
pub const StrengthAnalysis = password.StrengthAnalysis;

/// Result of password hashing operation.
/// Contains the hash, salt, and algorithm used for verification.
pub const HashedPassword = password.HashedPassword;

/// Analyze password strength and provide feedback.
/// Checks length, character diversity, common patterns, and dictionary words.
pub const analyzePasswordStrength = password.analyzeStrength;

/// Generate a random password with configurable requirements.
/// Supports length, character classes, and exclusion patterns.
pub const generatePassword = password.generatePassword;

/// Argon2 algorithm parameters.
/// Memory cost, time cost, and parallelism.
pub const Argon2Params = password.Argon2Params;

/// PBKDF2 algorithm parameters.
/// Hash function and iteration count.
pub const Pbkdf2Params = password.Pbkdf2Params;

/// scrypt algorithm parameters.
/// N (CPU/memory cost), r (block size), p (parallelism).
pub const ScryptParams = password.ScryptParams;

// ============================================================================
// Session Management Types
// ============================================================================

/// Session manager for user session lifecycle.
/// Handles creation, validation, renewal, and invalidation.
pub const SessionManager = session.SessionManager;

/// Configuration for session management.
/// Includes timeout, renewal policy, and storage backend.
pub const SessionConfig = session.SessionConfig;

/// Represents an active user session.
/// Contains session ID, user ID, creation time, and metadata.
pub const Session = session.Session;

/// Errors that can occur during session operations.
pub const SessionError = session.SessionError;

// ============================================================================
// Audit Logging Types
// ============================================================================

/// Security audit logger with optional HMAC chain integrity.
/// Provides structured logging for security events with filtering,
/// alerting, and export capabilities.
pub const AuditLogger = audit.AuditLogger;

/// Configuration for audit logging.
/// Controls severity threshold, hash chain, retention, and alerting.
pub const AuditConfig = audit.AuditConfig;

/// Complete security audit event record.
/// Contains timestamp, severity, category, actor, target, and context.
pub const AuditEvent = audit.AuditEvent;

/// Severity levels for security events.
/// critical (0), high (1), medium (2), low (3), info (4).
pub const AuditSeverity = audit.Severity;

/// Categories of security events.
/// authentication, authorization, data_access, data_modification,
/// system, network, crypto, session, rate_limit, input_validation,
/// secrets, compliance.
pub const AuditEventCategory = audit.EventCategory;

/// Outcome of a security event.
/// success, failure, blocked, warning, unknown.
pub const AuditEventOutcome = audit.EventOutcome;

/// Information about who performed a security-relevant action.
/// Includes user/service ID, IP address, user agent, session/API key.
pub const AuditActor = audit.Actor;

/// Target resource of a security event.
/// Contains resource type, ID, path, and metadata.
pub const AuditTarget = audit.Target;

/// Builder pattern for creating audit events.
/// Provides fluent API for setting event properties.
pub const AuditEventBuilder = audit.EventBuilder;

/// Predefined event type constants for common security events.
/// LOGIN_SUCCESS, LOGIN_FAILURE, ACCESS_DENIED, RATE_LIMIT_EXCEEDED, etc.
pub const AuditEventTypes = audit.EventTypes;

// ============================================================================
// Input Validation Types
// ============================================================================

/// Input validator with configurable rules.
/// Validates and sanitizes user input to prevent injection attacks.
pub const Validator = validation.Validator;

/// Configuration for input validation.
/// Defines validation rules and sanitization options.
pub const ValidatorConfig = validation.ValidatorConfig;

/// Result of input validation.
/// Contains validity status and any validation errors.
pub const ValidationResult = validation.ValidationResult;

/// Errors that can occur during validation.
pub const ValidationError = validation.ValidationError;

/// Input sanitizer for cleaning potentially dangerous input.
/// Removes or escapes HTML, SQL, shell metacharacters, etc.
pub const Sanitizer = validation.Sanitizer;

// ============================================================================
// Security Headers Types
// ============================================================================

/// HTTP security headers manager.
/// Applies security headers to HTTP responses.
pub const SecurityHeaders = headers.SecurityHeaders;

/// Configuration for security headers.
/// Controls all header values with sensible defaults.
pub const SecurityHeadersConfig = headers.SecurityHeadersConfig;

/// Content-Security-Policy configuration.
/// Defines allowed sources for scripts, styles, images, etc.
pub const CspConfig = headers.CspConfig;

/// HTTP Strict Transport Security configuration.
/// Controls max-age, includeSubDomains, and preload.
pub const HstsConfig = headers.HstsConfig;

/// X-Frame-Options header values.
/// DENY, SAMEORIGIN, or ALLOW-FROM with specified origin.
pub const FrameOptions = headers.FrameOptions;

/// Referrer-Policy header values.
/// Controls how much referrer information is sent with requests.
pub const ReferrerPolicy = headers.ReferrerPolicy;

/// Preset security header configurations.
/// strict, moderate, minimal for different security postures.
pub const SecurityHeaderPresets = headers.Presets;

// ============================================================================
// Secrets Management Types
// ============================================================================

/// Secrets manager for encrypted credential storage.
/// Supports environment variables, encrypted files, and vault backends.
/// Thread-safe with caching and rotation support.
pub const SecretsManager = secrets.SecretsManager;

/// Configuration for secrets management.
/// Specifies provider, master key, validation rules, and caching.
pub const SecretsConfig = secrets.SecretsConfig;

/// Encrypted secret value with metadata.
/// Contains encrypted data, nonce, authentication tag, and access tracking.
pub const SecretValue = secrets.SecretValue;

/// Metadata about a stored secret.
/// Includes creation time, access count, expiration, and rotation schedule.
pub const SecretMetadata = secrets.SecretMetadata;

/// Classification of secret types.
/// generic, api_key, password, certificate, private_key,
/// database_credential, oauth_token, encryption_key, signing_key.
pub const SecretType = secrets.SecretType;

/// Memory-safe string that securely wipes on deallocation.
/// Use for handling sensitive data in memory.
pub const SecureString = secrets.SecureString;

/// Errors that can occur during secrets operations.
pub const SecretsError = secrets.SecretsError;

// ============================================================================
// Rate Limiting Types
// ============================================================================

/// Rate limiter with multiple algorithm support.
/// Thread-safe with automatic cleanup of expired entries.
pub const RateLimiter = rate_limit.RateLimiter;

/// Configuration for rate limiting.
/// Specifies requests per window, algorithm, scope, and ban settings.
pub const RateLimitConfig = rate_limit.RateLimitConfig;

/// Status returned from rate limit check.
/// Contains allowed/blocked status, remaining quota, and retry-after.
pub const RateLimitStatus = rate_limit.RateLimitStatus;

/// Rate limiting algorithms.
/// token_bucket (allows bursts), sliding_window (smooth),
/// fixed_window (simple), leaky_bucket (constant rate).
pub const RateLimitAlgorithm = rate_limit.Algorithm;

/// Scope for rate limit application.
/// ip, user, api_key, endpoint, global, or custom.
pub const RateLimitScope = rate_limit.Scope;

/// Multi-tier rate limiter combining multiple limits.
/// Allows different limits per scope (e.g., per-IP and per-user).
pub const MultiTierRateLimiter = rate_limit.MultiTierRateLimiter;

/// Preset rate limit configurations.
/// standard (100/min), strict (10/min + ban), lenient (1000/min), login (5/5min).
pub const RateLimitPresets = rate_limit.Presets;

// ============================================================================
// IP Filtering Types
// ============================================================================

/// IP filter for allow/deny list enforcement.
/// Supports IPv4, IPv6, and CIDR ranges with automatic blocking.
pub const IpFilter = ip_filter.IpFilter;

/// Configuration for IP filtering.
/// Specifies allow/deny lists and blocking behavior.
pub const IpFilterConfig = ip_filter.IpFilterConfig;

/// Parsed IP address (v4 or v6).
pub const IpAddress = ip_filter.IpAddress;

/// CIDR range for subnet matching.
/// e.g., 192.168.1.0/24 or 2001:db8::/32.
pub const CidrRange = ip_filter.CidrRange;

/// Reason for blocking an IP address.
/// deny_list, rate_limit, suspicious_activity, manual_block.
pub const BlockReason = ip_filter.BlockReason;

/// Record of a blocked IP with expiration.
pub const BlockEntry = ip_filter.BlockEntry;

/// Errors that can occur during IP filtering.
pub const IpFilterError = ip_filter.IpFilterError;

// ============================================================================
// Certificate Management Types
// ============================================================================

/// Certificate manager for X.509 lifecycle.
/// Handles generation, loading, validation, and rotation.
pub const CertificateManager = certificates.CertificateManager;

/// Configuration for certificate management.
/// Specifies paths, validity periods, and rotation policy.
pub const CertificateConfig = certificates.CertificateConfig;

/// Information extracted from an X.509 certificate.
/// Subject, issuer, validity period, key usage, and extensions.
pub const CertificateInfo = certificates.CertificateInfo;

/// Types of certificates.
/// root_ca, intermediate_ca, server, client, code_signing.
pub const CertificateType = certificates.CertificateType;

/// Key algorithms for certificate generation.
/// rsa_2048, rsa_4096, ecdsa_p256, ecdsa_p384, ed25519.
pub const KeyAlgorithm = certificates.KeyAlgorithm;

/// Generate a self-signed certificate for testing/development.
/// Not recommended for production use.
pub const generateSelfSignedCert = certificates.generateSelfSigned;

/// Errors that can occur during certificate operations.
pub const CertificateError = certificates.CertificateError;

// ============================================================================
// Encryption at Rest Types
// ============================================================================

/// Encryptor for data at rest using AEAD algorithms.
/// Supports both key-based and password-based encryption.
pub const Encryptor = encryption.Encryptor;

/// Configuration for encryption operations.
/// Specifies algorithm, KDF, and serialization options.
pub const EncryptionConfig = encryption.EncryptionConfig;

/// Encrypted data with header, ciphertext, and authentication tag.
/// Self-describing format that includes algorithm metadata.
pub const EncryptedData = encryption.EncryptedData;

/// Encryption algorithms available.
/// aes_256_gcm (recommended), chacha20_poly1305, xchacha20_poly1305.
pub const EncryptionAlgorithm = encryption.Algorithm;

/// Key wrapper for envelope encryption.
/// Encrypts data encryption keys (DEKs) with a master key (KEK).
pub const KeyWrapper = encryption.KeyWrapper;

/// Generate a cryptographically secure 256-bit encryption key.
pub const generateEncryptionKey = encryption.generateKey;

/// Securely delete a file by overwriting with random data.
/// Multiple passes with final zero-fill for forensic resistance.
pub const secureDelete = encryption.secureDelete;

/// Errors that can occur during encryption operations.
pub const EncryptionError = encryption.EncryptionError;

/// Key derivation functions for password-based encryption.
/// hkdf_sha256, pbkdf2_sha256, argon2id (recommended), scrypt.
pub const Kdf = encryption.Kdf;

// ============================================================================
// CORS Types
// ============================================================================

/// CORS handler for cross-origin request validation.
/// Validates origins, methods, and headers for preflight and actual requests.
pub const CorsHandler = cors.CorsHandler;

/// Configuration for CORS handling.
/// Specifies allowed origins, methods, headers, and credentials.
pub const CorsConfig = cors.CorsConfig;

/// CORS response headers to add to HTTP responses.
pub const CorsHeader = cors.CorsHeader;

/// Result of preflight request validation.
/// Indicates whether the request is allowed and which headers to add.
pub const PreflightResult = cors.PreflightResult;

/// Preset CORS configurations.
/// permissive (allow all), restrictive (same-origin only), custom templates.
pub const CorsPresets = cors.Presets;

test {
    @import("std").testing.refAllDecls(@This());
}
