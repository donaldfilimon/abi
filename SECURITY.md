---
title: Security Policy
purpose: Vulnerability reporting and security guidelines
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2934+47d2e5de9
---

# Security Policy

If you find a security issue in ABI, avoid filing a public bug with exploit details first.

## Reporting

1. Prefer a private disclosure path through the repository hosting platform if one is available.
2. If no private advisory flow is available, contact the maintainer directly before public disclosure.
3. Include affected versions, impact, reproduction steps, and any suggested mitigation.

## Scope

Security issues include memory safety bugs with realistic exploitability, authentication or authorization flaws, secret leakage, path traversal, unsafe file handling, and remote code execution paths.

For non-sensitive bugs, use the normal issue tracker instead.
