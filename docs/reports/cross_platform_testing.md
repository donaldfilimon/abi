# Cross-Platform Testing Reference

Generated on: 2025-09-28 06:25:31 UTC

This reference merges the retired cross-platform testing guide and enhancement summary so every team can rely on the same coverage matrix, workflows, and follow-up plan.

## Coverage Matrix

| Dimension | Current Scope | Notes |
| --- | --- | --- |
| Operating systems | Windows Server 2019/2022, macOS 13/14, Ubuntu 18.04/20.04/22.04 | Mirrors CI runners and staging fleet |
| Architectures | x86_64 (all platforms), aarch64 (macOS Apple Silicon, Linux ARM64) | Capture Rosetta + native binaries |
| Zig toolchains | 0.16.0-dev baseline, 0.16.0 release, nightly `master` | Nightly protects against upstream regressions |
| Test types | Unit, integration, performance, security, GPU smoke, container smoke | Driven by `tests/cross-platform/*.zig` |
| CI combinations | ~48+ (7 OS targets × 2 architectures × 3 Zig versions) | 4× increase over prior matrix |

### Operating Systems
- **Windows**: Windows Server 2019 and 2022 runners with validated PowerShell deployment scripts.
- **macOS**: macOS 13 (Ventura) and macOS 14 (Sonoma) on Intel and Apple Silicon hardware.
- **Linux**: Ubuntu 18.04, 20.04, and 22.04 images covering glibc and musl libc variants.

### Architectures
- **x86_64**: Primary architecture for CI, staging, and production workloads.
- **aarch64**: Required for Apple Silicon and ARM server validation; binaries run natively and under Rosetta.

### Zig Toolchains
- `0.16.0-dev` pin (repository baseline).
- Latest 0.16.0 release candidate for stability checks.
- Nightly `master` builds to surface upstream breakage early.

---

## CI Automation & Artifacts
- `.github/workflows/ci.yml` provisions matrix builds across OS, architecture, and toolchain combinations with cache tuning per runner.
- Automation scripts generate `/tmp/ci_analysis.md` and `/tmp/enhanced_matrix.md` snapshots when evaluating coverage.
- Container smoke tests build Docker images (e.g., `ubuntu:22.04`) and publish artifacts for downstream validation.
- Platform-specific suites live in `tests/cross-platform/{windows,macos,linux}.zig` and skip gracefully on other OS targets.

---

## Platform Playbooks

### Windows
- Prefer Winsock2 networking and ensure file path comparisons honor case preservation and ACL semantics.
- Validate service management flows, registry interactions, and file locking across Server 2019/2022 images.
- Exercise deployment scripts (`deploy/scripts/deploy-staging.ps1`) and ensure NTFS permission adjustments succeed.

### macOS
- Exercise kqueue-driven event loops on both Intel and Apple Silicon hardware, verifying event delivery parity.
- Track entitlements whenever invoking Foundation/CoreFoundation APIs and document signing requirements.
- Validate universal binaries and Rosetta fallbacks, especially for GPU toolchains.

### Linux
- Use epoll for scalable I/O and test on both glibc and musl images.
- Account for containerized environments where `/proc` access or cgroup limits may differ; add guards for missing filesystems.
- Validate shell scripts (`deploy/scripts/deploy-staging.sh`) under bash and dash to ensure portability.

---

## Best Practices & Snippets

### Conditional Compilation
```zig
const builtin = @import("builtin");

if (builtin.os.tag == .windows) {
    // Windows-specific code paths
} else if (builtin.os.tag == .macos) {
    // macOS-specific code paths
} else if (builtin.os.tag == .linux) {
    // Linux-specific code paths
}
```

### Platform Detection Helpers
```zig
const is_windows = builtin.os.tag == .windows;
const is_macos = builtin.os.tag == .macos;
const is_linux = builtin.os.tag == .linux;
```

### Cross-Platform Paths
```zig
const path = try std.fs.path.join(allocator, &[_][]const u8{"dir", "file.txt"});
```

### Network Coverage
```zig
const address4 = try std.net.Address.parseIp4("127.0.0.1", 8080);
const address6 = try std.net.Address.parseIp6("::1", 8080);
```

### Environment Fingerprint
```zig
const stdout = std.io.getStdOut().writer();
try stdout.print("platform={s} arch={s} zig={s}\n", .{
    @tagName(builtin.os.tag),
    @tagName(builtin.cpu.arch),
    builtin.zig_version_string,
});
```

---

## Running the Matrix
```bash
# Run entire suite
zig build test

# Focused cross-platform aggregator
zig build test-cross-platform

# OS-specific runs
zig build test-windows
zig build test-macos
zig build test-linux
```

---

## Troubleshooting & Debugging
1. Confirm `builtin.os.tag` and feature flags match expectations before triaging failures.
2. Re-run isolated suites (`zig build test-windows`, etc.) to reproduce deterministically.
3. Inspect file path normalization and permission handling; normalize separators/encodings as needed.
4. Review CI artifacts for system metadata, panic traces, and performance regressions.
5. Capture environment fingerprints in logs to accelerate cross-team debugging.

---

## Container Guidance
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl xz-utils
# Install Zig toolchain and run tests
```
- Validate container images with both glibc and musl bases when possible.
- Ensure `/proc` reads guard against restricted environments; skip tests gracefully when resources are unavailable.

---

## Enhancement Summary

### Completed
- Updated CI workflow with the expanded Zig/OS matrix and architecture coverage.
- Created platform-specific regression suites under `tests/cross-platform/` with allocator-aware skips.
- Generated this consolidated reference that merges the historical guide and automation summary.

### Coverage Improvements
| Category | Before | After | Improvement |
| --- | --- | --- | --- |
| OS Versions | 3 | 7 | +133% |
| Zig Versions | 3 | 4 | +33% |
| Architectures | 1 | 2 | +100% |
| Total Combinations | ~12 | ~48+ | +300% |

### Next Steps
1. Monitor CI runs across all platforms and investigate regressions promptly.
2. Extend performance benchmarking to stress platform-specific hot paths (IOCP, kqueue, epoll).
3. Keep this reference updated as new OS/Zig releases land or hardware diversity grows.

### Files Created / Updated by Automation
- `.github/workflows/ci.yml`
- `tests/cross-platform/windows.zig`
- `tests/cross-platform/macos.zig`
- `tests/cross-platform/linux.zig`
- `docs/reports/cross_platform_testing.md`

### Benefits
- Improved reliability via proactive coverage.
- Earlier detection of OS/toolchain regressions in CI.
- Reduced support burden thanks to consistent behavior across environments.

---

## Automation Notes
- Generated by `scripts/enhance_cross_platform_testing.sh`.
- CI matrix recommendations recorded in `/tmp/enhanced_matrix.md`.
- Regenerated platform-specific regression suites under `tests/cross-platform/`.
- Review GitHub Actions workflow changes before merging automated edits.
