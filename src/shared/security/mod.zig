//! Security module providing authentication, authorization, and TLS support.
//!
//! This module consolidates security-related functionality including:
//! - API key management with secure hashing and rotation
//! - Role-based access control (RBAC)
//! - TLS/SSL support for secure communication
//! - mTLS (Mutual TLS) for bidirectional certificate authentication

pub const api_keys = @import("api_keys.zig");
pub const mtls = @import("mtls.zig");
pub const rbac = @import("rbac.zig");
pub const tls = @import("tls.zig");

// API key management types
pub const ApiKeyManager = api_keys.ApiKeyManager;
pub const ApiKeyConfig = api_keys.ApiKeyConfig;
pub const ApiKey = api_keys.ApiKey;
pub const GeneratedKey = api_keys.GeneratedKey;
pub const ApiKeyError = api_keys.ApiKeyError;
pub const HashAlgorithm = api_keys.HashAlgorithm;
pub const SALT_LENGTH = api_keys.SALT_LENGTH;
pub const DEFAULT_HASH_ITERATIONS = api_keys.DEFAULT_HASH_ITERATIONS;

// RBAC types
pub const RbacManager = rbac.RbacManager;
pub const RbacConfig = rbac.RbacConfig;
pub const Role = rbac.Role;
pub const RoleAssignment = rbac.RoleAssignment;
pub const Permission = rbac.Permission;

// TLS types
pub const TlsConnection = tls.TlsConnection;
pub const TlsConfig = tls.TlsConfig;
pub const TlsVersion = tls.TlsVersion;
pub const TlsCertificate = tls.TlsCertificate;
pub const TlsError = tls.TlsError;
pub const CertificateStore = tls.CertificateStore;
pub const HandshakeState = tls.HandshakeState;

// mTLS types
pub const MtlsConnection = mtls.MtlsConnection;
pub const MtlsConfig = mtls.MtlsConfig;
pub const ClientCertificateInfo = mtls.ClientCertificateInfo;

test {
    @import("std").testing.refAllDecls(@This());
}
