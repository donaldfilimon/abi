# security API Reference

> TLS, mTLS, API keys, and RBAC

**Source:** [`src/shared/security/mod.zig`](../../src/shared/security/mod.zig)

---

Security module providing comprehensive authentication, authorization, and security features.

This module consolidates security-related functionality including:
- API key management with secure hashing and rotation
- Role-based access control (RBAC)
- TLS/SSL support for secure communication
- mTLS (Mutual TLS) for bidirectional certificate authentication
- JWT token authentication
- Password hashing (Argon2, PBKDF2, scrypt)
- Session management
- Input validation and sanitization
- Security headers middleware
- Secrets management
- Rate limiting
- IP filtering (allow/deny lists)
- Certificate management and rotation
- Encryption at rest
- CORS configuration
- Security audit logging

---

## API

---

*Generated automatically by `zig build gendocs`*
