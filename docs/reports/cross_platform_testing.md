# Cross-Platform Testing Reference

This page captures the consolidated matrix, workflows, and follow-up tasks for the project's cross-platform guarantees. It merges the previous testing guide and enhancement summary into a single home.

## Test Matrix
- **Operating systems**: Windows Server 2019/2022, macOS 13/14, Ubuntu 18.04/20.04/22.04.
- **Architectures**: x86_64 everywhere, plus Apple Silicon (aarch64) coverage on macOS and Linux.
- **Zig toolchains**: 0.16.0-dev (baseline), 0.16.0 release, and nightly master builds.

## Platform-Specific Notes
### Windows
- Prefer Winsock2 networking and handle path casing/ACL semantics.
- Exercise service management, registry interactions, and file locking behavior.

### macOS
- Validate kqueue-powered event loops and both Intel/Apple Silicon binaries.
- Track framework entitlements when invoking Foundation/CoreFoundation APIs.

### Linux
- Use epoll for scalable I/O and test against multiple libc variants (glibc, musl).
- Account for containerized environments where `/proc` and cgroup limits may differ.

## Best Practices
- Gate platform logic with `builtin.os.tag` checks and centralize conditional compilation.
- Use `std.fs.path` helpers for filesystem joins and normalization.
- Exercise both IPv4/IPv6 sockets and TLS pathways during regression runs.
- Capture environment fingerprints (OS, architecture, Zig version) in test logs for traceability.

## CI & Automation
- GitHub Actions matrix runs the full suite across all OS/architecture/Zig combinations with caching tuned per runner.
- Dedicated test files exist in `tests/cross-platform/{windows,macos,linux}.zig`, each guarding platform-only features and failure modes.
- Container builds validate Linux deployments via Docker and publish artifacts for downstream smoke tests.

## Troubleshooting Checklist
1. Confirm platform detection and feature flags before diving into failure logs.
2. Re-run isolated suites (`zig build test-windows`, `zig build test-macos`, `zig build test-linux`) to reproduce deterministically.
3. Inspect permissions/path handling issues; normalize separators and encoding where needed.
4. Monitor CI artifacts for system metadata, panic traces, and performance regressions.

## Enhancement Summary
- Expanded the CI matrix to 48+ combinations covering OS, architecture, and compiler deltas.
- Added platform-focused regression suites and container smoke coverage.
- Produced this merged reference so future updates live in one maintained location.
- Next steps: monitor nightly regressions, expand hardware diversity, and feed learnings back into developer onboarding materials.
