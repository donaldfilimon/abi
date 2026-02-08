---
title: "SECURITY"
tags: [security, policy]
---
# Security Policy
> **Codebase Status:** Synced with repository as of the latest release tag.

<p align="center">
  <img src="https://img.shields.io/badge/Security-Priority-critical?style=for-the-badge" alt="Security Priority"/>
  <img src="https://img.shields.io/badge/Reporting-GitHub_Advisories-blue?style=for-the-badge" alt="Reporting"/>
</p>

## Security Advisories

### [CVE-NOT-ASSIGNED] Path Traversal in Backup/Restore Endpoints (2025-12-27)

**Severity**: CRITICAL (CWE-22: Improper Limitation of a Pathname to a Restricted Directory)

**Affected Versions**: All versions prior to 0.2.0

**Vulnerability**:
The database backup and restore HTTP endpoints in `src/features/database/unified.zig` did not validate user-provided filenames. An attacker could craft requests with path traversal sequences (e.g., `../`) to read arbitrary files on the server filesystem or write backup files to arbitrary locations.

**Attack Scenario**:
```http
# Read arbitrary files
GET /api/database/restore?file=../../etc/passwd

# Write backup to arbitrary location
GET /api/database/backup?file=../../malicious_backup.db
```

**Fix**:
- Added path validation functions in `src/services/shared/utils/fs/mod.zig`
- Restricted backup/restore operations to the `backups/` directory only
- Added validation to reject:
  - Path traversal sequences (`..`)
  - Absolute paths
  - Windows drive letters
  - Empty filenames

**Implementation**:
- `isSafeBackupPath()` validates filename safety
- `normalizeBackupPath()` resolves paths to `backups/` directory
- Returns `PathValidationError` on invalid input
- Created `backups/` directory automatically if it doesn't exist

**Mitigation**:
Upgrade to version 0.2.0 or later. If unable to upgrade immediately:
- Restrict network access to backup/restore endpoints
- Run the service in a sandboxed environment
- Use reverse proxies to block requests with `..` in parameters

**References**:
- Fix in `src/features/database/unified.zig:68-95`
- Fix in `src/services/shared/utils/fs/mod.zig:1-90`
- Fix in `src/features/database/http.zig:48`

## Supported Versions
| Version | Supported |
| ------- | --------- |
| 0.3.x   | Yes       |
| 0.2.x   | Security fixes only |
| 0.1.x   | No        |

## Reporting a Vulnerability
Please report security issues privately using GitHub Security Advisories.
Include reproduction steps, impact assessment, and suggested fixes.

## Best Practices
- Use the latest supported version.
- Keep Zig updated.
- Validate untrusted inputs and sandbox untrusted code.
- **Secrets Management**: Use `src/services/shared/security/secrets.zig` for all credential handling. Secrets are encrypted in memory, audited on access, and never logged. Configure `SecretsConfig` with an appropriate `env_prefix` and enable `audit_logging`.
- Avoid printing secret values or their hashes to logs; the `SecretsManager` ensures decryption occurs only in controlled code paths.

## Additional Details
The CLI is minimal by design; most deployments should embed ABI as a library.

## See Also

- [CONTRIBUTING.md](CONTRIBUTING.md) - Maintainer contact information and development workflow
- [AGENTS.md](AGENTS.md) - Baseline coding and testing conventions
- [CLAUDE.md](CLAUDE.md) - Architectural context for coding agents
