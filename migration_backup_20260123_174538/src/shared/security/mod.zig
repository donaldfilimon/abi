//! Security module providing comprehensive authentication, authorization, and security features.
//!
//! This module consolidates security-related functionality including:
//! - API key management with secure hashing and rotation
//! - Role-based access control (RBAC)
//! - TLS/SSL support for secure communication
//! - mTLS (Mutual TLS) for bidirectional certificate authentication
//! - JWT token authentication
//! - Password hashing (Argon2, PBKDF2, scrypt)
//! - Session management
//! - Input validation and sanitization
//! - Security headers middleware
//! - Secrets management
//! - Rate limiting
//! - IP filtering (allow/deny lists)
//! - Certificate management and rotation
//! - Encryption at rest
//! - CORS configuration
//! - Security audit logging

// Core security modules
pub const api_keys = @import("api_keys.zig");
pub const mtls = @import("mtls.zig");
pub const rbac = @import("rbac.zig");
pub const tls = @import("tls.zig");

// New security modules
pub const audit = @import("audit.zig");
pub const certificates = @import("certificates.zig");
pub const cors = @import("cors.zig");
pub const encryption = @import("encryption.zig");
pub const headers = @import("headers.zig");
pub const ip_filter = @import("ip_filter.zig");
pub const jwt = @import("jwt.zig");
pub const password = @import("password.zig");
pub const rate_limit = @import("rate_limit.zig");
pub const secrets = @import("secrets.zig");
pub const session = @import("session.zig");
pub const validation = @import("validation.zig");

// ============================================================================
// API key management types
// ============================================================================
pub const ApiKeyManager = api_keys.ApiKeyManager;
pub const ApiKeyConfig = api_keys.ApiKeyConfig;
pub const ApiKey = api_keys.ApiKey;
pub const GeneratedKey = api_keys.GeneratedKey;
pub const ApiKeyError = api_keys.ApiKeyError;
pub const HashAlgorithm = api_keys.HashAlgorithm;
pub const SALT_LENGTH = api_keys.SALT_LENGTH;
pub const DEFAULT_HASH_ITERATIONS = api_keys.DEFAULT_HASH_ITERATIONS;

// ============================================================================
// RBAC types
// ============================================================================
pub const RbacManager = rbac.RbacManager;
pub const RbacConfig = rbac.RbacConfig;
pub const Role = rbac.Role;
pub const RoleAssignment = rbac.RoleAssignment;
pub const Permission = rbac.Permission;

// ============================================================================
// TLS types
// ============================================================================
pub const TlsConnection = tls.TlsConnection;
pub const TlsConfig = tls.TlsConfig;
pub const TlsVersion = tls.TlsVersion;
pub const TlsCertificate = tls.TlsCertificate;
pub const TlsError = tls.TlsError;
pub const CertificateStore = tls.CertificateStore;
pub const HandshakeState = tls.HandshakeState;

// ============================================================================
// mTLS types
// ============================================================================
pub const MtlsConnection = mtls.MtlsConnection;
pub const MtlsConfig = mtls.MtlsConfig;
pub const ClientCertificateInfo = mtls.ClientCertificateInfo;

// ============================================================================
// JWT types
// ============================================================================
pub const JwtManager = jwt.JwtManager;
pub const JwtConfig = jwt.JwtConfig;
pub const JwtAlgorithm = jwt.Algorithm;
pub const JwtClaims = jwt.Claims;
pub const JwtToken = jwt.Token;
pub const extractBearerToken = jwt.extractBearerToken;
pub const generateSecretKey = jwt.generateSecretKey;

// ============================================================================
// Password hashing types
// ============================================================================
pub const PasswordHasher = password.PasswordHasher;
pub const PasswordConfig = password.PasswordConfig;
pub const PasswordStrength = password.PasswordStrength;
pub const StrengthAnalysis = password.StrengthAnalysis;
pub const HashedPassword = password.HashedPassword;
pub const analyzePasswordStrength = password.analyzeStrength;
pub const generatePassword = password.generatePassword;
pub const Argon2Params = password.Argon2Params;
pub const Pbkdf2Params = password.Pbkdf2Params;
pub const ScryptParams = password.ScryptParams;

// ============================================================================
// Session management types
// ============================================================================
pub const SessionManager = session.SessionManager;
pub const SessionConfig = session.SessionConfig;
pub const Session = session.Session;
pub const SessionError = session.SessionError;

// ============================================================================
// Audit logging types
// ============================================================================
pub const AuditLogger = audit.AuditLogger;
pub const AuditConfig = audit.AuditConfig;
pub const AuditEvent = audit.AuditEvent;
pub const AuditSeverity = audit.Severity;
pub const AuditEventCategory = audit.EventCategory;
pub const AuditEventOutcome = audit.EventOutcome;
pub const AuditActor = audit.Actor;
pub const AuditTarget = audit.Target;
pub const AuditEventBuilder = audit.EventBuilder;
pub const AuditEventTypes = audit.EventTypes;

// ============================================================================
// Input validation types
// ============================================================================
pub const Validator = validation.Validator;
pub const ValidatorConfig = validation.ValidatorConfig;
pub const ValidationResult = validation.ValidationResult;
pub const ValidationError = validation.ValidationError;
pub const Sanitizer = validation.Sanitizer;

// ============================================================================
// Security headers types
// ============================================================================
pub const SecurityHeaders = headers.SecurityHeaders;
pub const SecurityHeadersConfig = headers.SecurityHeadersConfig;
pub const CspConfig = headers.CspConfig;
pub const HstsConfig = headers.HstsConfig;
pub const FrameOptions = headers.FrameOptions;
pub const ReferrerPolicy = headers.ReferrerPolicy;
pub const SecurityHeaderPresets = headers.Presets;

// ============================================================================
// Secrets management types
// ============================================================================
pub const SecretsManager = secrets.SecretsManager;
pub const SecretsConfig = secrets.SecretsConfig;
pub const SecretValue = secrets.SecretValue;
pub const SecretMetadata = secrets.SecretMetadata;
pub const SecretType = secrets.SecretType;
pub const SecureString = secrets.SecureString;
pub const SecretsError = secrets.SecretsError;

// ============================================================================
// Rate limiting types
// ============================================================================
pub const RateLimiter = rate_limit.RateLimiter;
pub const RateLimitConfig = rate_limit.RateLimitConfig;
pub const RateLimitStatus = rate_limit.RateLimitStatus;
pub const RateLimitAlgorithm = rate_limit.Algorithm;
pub const RateLimitScope = rate_limit.Scope;
pub const MultiTierRateLimiter = rate_limit.MultiTierRateLimiter;
pub const RateLimitPresets = rate_limit.Presets;

// ============================================================================
// IP filtering types
// ============================================================================
pub const IpFilter = ip_filter.IpFilter;
pub const IpFilterConfig = ip_filter.IpFilterConfig;
pub const IpAddress = ip_filter.IpAddress;
pub const CidrRange = ip_filter.CidrRange;
pub const BlockReason = ip_filter.BlockReason;
pub const BlockEntry = ip_filter.BlockEntry;
pub const IpFilterError = ip_filter.IpFilterError;

// ============================================================================
// Certificate management types
// ============================================================================
pub const CertificateManager = certificates.CertificateManager;
pub const CertificateConfig = certificates.CertificateConfig;
pub const CertificateInfo = certificates.CertificateInfo;
pub const CertificateType = certificates.CertificateType;
pub const KeyAlgorithm = certificates.KeyAlgorithm;
pub const generateSelfSignedCert = certificates.generateSelfSigned;
pub const CertificateError = certificates.CertificateError;

// ============================================================================
// Encryption at rest types
// ============================================================================
pub const Encryptor = encryption.Encryptor;
pub const EncryptionConfig = encryption.EncryptionConfig;
pub const EncryptedData = encryption.EncryptedData;
pub const EncryptionAlgorithm = encryption.Algorithm;
pub const KeyWrapper = encryption.KeyWrapper;
pub const generateEncryptionKey = encryption.generateKey;
pub const secureDelete = encryption.secureDelete;
pub const EncryptionError = encryption.EncryptionError;
pub const Kdf = encryption.Kdf;

// ============================================================================
// CORS types
// ============================================================================
pub const CorsHandler = cors.CorsHandler;
pub const CorsConfig = cors.CorsConfig;
pub const CorsHeader = cors.CorsHeader;
pub const PreflightResult = cors.PreflightResult;
pub const CorsPresets = cors.Presets;

test {
    @import("std").testing.refAllDecls(@This());
}
