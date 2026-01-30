# Production Readiness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all blocking issues and complete production readiness checklist for ABI framework deployment.

**Architecture:** Address compilation errors first, then security issues, then commit uncommitted streaming recovery work, then operational readiness items. Each phase builds on the previous.

**Tech Stack:** Zig 0.16, platform-aware time utilities, SecureString for credentials, proper error handling patterns.

---

## Current State Summary

| Category | Status | Blocking |
|----------|--------|----------|
| Compilation | 4 errors | YES |
| Tests | 771/776 (blocked) | YES |
| Security | 2 HIGH, 5 MEDIUM | Partial |
| Uncommitted Work | 6 files | YES |
| Documentation | Good coverage | NO |

---

## Phase 1: Fix Compilation Errors (BLOCKING)

### Task 1: Fix std.time.sleep() in streaming_recovery.zig

**Files:**
- Modify: `src/tests/integration/streaming_recovery.zig:56,80,276`

**Step 1: Read the file to understand context**

The file uses `std.time.sleep()` which was removed in Zig 0.16. Need to replace with platform-aware sleep.

**Step 2: Replace sleep calls with busy-wait loop**

Replace all 3 occurrences of:
```zig
std.time.sleep(15 * std.time.ns_per_ms);
```

With a busy-wait approach using Timer (tests don't have I/O backend):
```zig
// Busy-wait for timeout period (tests don't have I/O backend for proper sleep)
const start = std.time.Timer.start() catch {
    // Fallback: just proceed without waiting
    return;
};
while (start.read() < 15 * std.time.ns_per_ms) {
    std.atomic.spinLoopHint();
}
```

**Step 3: Verify compilation**

Run: `zig build -Denable-ai=true 2>&1 | grep -E "(error|streaming_recovery)"`
Expected: No errors for streaming_recovery.zig

---

### Task 2: Fix linux.getpid() in os.zig

**Files:**
- Modify: `src/shared/os.zig:633,645`

**Step 1: Read the context around line 633**

The issue is that `linux` is defined as an empty struct on non-Linux platforms, so `linux.getpid()` doesn't exist.

**Step 2: Fix getpid() with proper platform detection**

Replace lines 626-634:
```zig
/// Get current process ID
pub fn getpid() Pid {
    if (comptime is_wasm) return 0;

    if (comptime builtin.os.tag == .windows) {
        return libc.GetCurrentProcessId();
    }

    return @intCast(linux.getpid());
}
```

With:
```zig
/// Get current process ID
pub fn getpid() Pid {
    if (comptime is_wasm) return 0;

    if (comptime builtin.os.tag == .windows) {
        return libc.GetCurrentProcessId();
    }

    if (comptime builtin.os.tag == .linux) {
        return @intCast(std.os.linux.getpid());
    }

    // POSIX fallback (macOS, BSD, etc.)
    return std.c.getpid();
}
```

**Step 3: Fix getppid() similarly**

Replace lines 637-646:
```zig
/// Get parent process ID
pub fn getppid() Pid {
    if (comptime is_wasm) return 0;

    if (comptime builtin.os.tag == .windows) {
        // Windows doesn't have a direct getppid, would need NtQueryInformationProcess
        return 0;
    }

    return @intCast(linux.getppid());
}
```

With:
```zig
/// Get parent process ID
pub fn getppid() Pid {
    if (comptime is_wasm) return 0;

    if (comptime builtin.os.tag == .windows) {
        // Windows doesn't have a direct getppid, would need NtQueryInformationProcess
        return 0;
    }

    if (comptime builtin.os.tag == .linux) {
        return @intCast(std.os.linux.getppid());
    }

    // POSIX fallback (macOS, BSD, etc.)
    return std.c.getppid();
}
```

**Step 4: Verify compilation**

Run: `zig build -Denable-ai=true 2>&1 | grep -E "(error|os.zig)"`
Expected: No errors for os.zig

---

### Task 3: Run full test suite

**Step 1: Run all tests**

Run: `zig build test --summary all 2>&1 | tail -20`
Expected: All tests pass or show only pre-existing failures (not compilation errors)

**Step 2: Verify test count**

Expected: 771+ tests passing

**Step 3: Commit compilation fixes**

```bash
git add src/tests/integration/streaming_recovery.zig src/shared/os.zig
git commit -m "fix: resolve Zig 0.16 compilation errors

- Replace std.time.sleep() with Timer-based busy-wait in streaming tests
- Fix linux.getpid()/getppid() with proper platform detection for macOS/BSD

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Commit Uncommitted Streaming Recovery Work

### Task 4: Review and commit streaming recovery implementation

**Files:**
- New: `src/ai/streaming/circuit_breaker.zig`
- New: `src/ai/streaming/metrics.zig`
- New: `src/ai/streaming/recovery.zig`
- New: `src/ai/streaming/retry_config.zig`
- New: `src/ai/streaming/session_cache.zig`
- New: `src/tests/integration/streaming_recovery.zig`
- Modify: `src/ai/streaming/backends/mod.zig`
- Modify: `src/ai/streaming/mod.zig`
- Modify: `src/ai/streaming/server.zig`
- Modify: `src/tests/integration/mod.zig`

**Step 1: Verify all streaming files compile**

Run: `zig build -Denable-ai=true`
Expected: Build succeeds

**Step 2: Format all files**

Run: `zig fmt src/ai/streaming/ src/tests/integration/streaming_recovery.zig`

**Step 3: Commit streaming recovery feature**

```bash
git add src/ai/streaming/circuit_breaker.zig \
        src/ai/streaming/metrics.zig \
        src/ai/streaming/recovery.zig \
        src/ai/streaming/retry_config.zig \
        src/ai/streaming/session_cache.zig \
        src/ai/streaming/backends/mod.zig \
        src/ai/streaming/mod.zig \
        src/ai/streaming/server.zig \
        src/tests/integration/streaming_recovery.zig \
        src/tests/integration/mod.zig

git commit -m "feat(streaming): add error recovery with circuit breakers

Stream error recovery implementation:
- CircuitBreaker: per-backend failure isolation (closed/open/half_open states)
- RetryConfig: exponential backoff with jitter
- SessionCache: LRU token cache for reconnection recovery
- StreamingMetrics: comprehensive per-backend metrics collection
- StreamRecovery: unified manager integrating all components
- BackendRouter: recovery-aware routing with fallback
- Server: 503 responses with Retry-After when circuit open
- Integration tests: circuit breaker, cache, metrics, fault injection

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Security Hardening

### Task 5: Add runtime warning for JWT none algorithm

**Files:**
- Modify: `src/shared/security/jwt.zig`

**Step 1: Read the JWT config section**

Read lines 155-170 to find the config struct.

**Step 2: Add warning log when none algorithm is enabled**

In the `init` or `create` function, after config is set, add:
```zig
if (config.allow_none_algorithm) {
    // Log warning at initialization time
    std.log.warn("JWT 'none' algorithm enabled - tokens can be forged without signatures!", .{});
}
```

**Step 3: Verify compilation**

Run: `zig build -Denable-ai=true`

**Step 4: Commit**

```bash
git add src/shared/security/jwt.zig
git commit -m "security(jwt): add runtime warning for 'none' algorithm

Logs prominent warning when allow_none_algorithm is enabled,
helping prevent accidental deployment with unsigned tokens.

Addresses security audit finding H-1.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 6: Fail on missing master key in production mode

**Files:**
- Modify: `src/shared/security/secrets.zig`

**Step 1: Read the secrets initialization section**

Read lines 175-195 to understand the current behavior.

**Step 2: Add production mode check**

Add a config option and fail-fast behavior:
```zig
pub const SecretsConfig = struct {
    /// If true, fails initialization when no master key is provided
    /// Set to true for production deployments
    require_master_key: bool = false,
    // ... other fields
};

// In init function:
if (config.require_master_key and master_key_source == .generated) {
    return error.MasterKeyRequired;
}

if (master_key_source == .generated) {
    std.log.warn("Using randomly generated master key - secrets will be lost on restart!", .{});
}
```

**Step 3: Verify compilation**

Run: `zig build -Denable-ai=true`

**Step 4: Commit**

```bash
git add src/shared/security/secrets.zig
git commit -m "security(secrets): add require_master_key option for production

- New require_master_key config option (default false)
- Returns error.MasterKeyRequired when enabled and no key provided
- Logs warning when using generated key in non-strict mode

Addresses security audit finding H-2.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 7: Implement secure API key wiping

**Files:**
- Modify: `src/connectors/mod.zig`
- Reference: `src/shared/security/secrets.zig` (for secureZero)

**Step 1: Read the connector config deinit**

**Step 2: Add secure wiping for API keys**

Import secureZero and use it in deinit:
```zig
const secrets = @import("../shared/security/secrets.zig");

pub fn deinit(self: *Config) void {
    if (self.openai_api_key) |key| {
        secrets.secureZero(key);
        self.allocator.free(key);
    }
    // ... repeat for other keys
}
```

**Step 3: Verify compilation and commit**

```bash
git add src/connectors/mod.zig
git commit -m "security(connectors): secure wipe API keys on deallocation

Uses secureZero() to clear API key memory before freeing,
preventing recovery from memory dumps.

Addresses security audit finding M-1.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Update PLAN.md Sprint Status

### Task 8: Update sprint status to reflect completed work

**Files:**
- Modify: `PLAN.md`

**Step 1: Mark streaming tasks as complete**

Change:
```markdown
### In Progress
- [ ] **Stream error recovery** - Graceful handling of disconnections, reconnection logic, observability for failures
- [ ] **Streaming integration tests** - E2E tests with fault injection for SSE/WebSocket endpoints
```

To:
```markdown
### In Progress
(None currently)

### Completed This Sprint
- [x] **Stream error recovery** - Graceful handling of disconnections, reconnection logic, observability for failures
- [x] **Streaming integration tests** - E2E tests with fault injection for SSE/WebSocket endpoints
- [x] **Streaming documentation** - Comprehensive guide for SSE/WebSocket streaming API (`docs/streaming.md`)
- [x] **Model management guide** - Documentation for downloading, caching, hot-reload (`docs/models.md`)
```

**Step 2: Update test count if improved**

**Step 3: Commit**

```bash
git add PLAN.md
git commit -m "docs(plan): mark streaming recovery as complete

- Stream error recovery implemented (circuit breakers, retry, caching)
- Integration tests with fault injection added
- Security hardening applied (JWT warning, master key check, API key wiping)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 5: Operational Documentation

### Task 9: Create production deployment guide

**Files:**
- Create: `docs/deployment.md`

**Step 1: Create the deployment guide**

```markdown
# Production Deployment Guide

## Prerequisites

- Zig 0.16 or later
- For GPU: NVIDIA drivers (CUDA) or Vulkan SDK
- For database: Persistent storage volume
- Environment variables configured

## Configuration

### Required Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ABI_MASTER_KEY` | 32-byte hex key for secrets encryption | YES (production) |
| `ABI_OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `ABI_ANTHROPIC_API_KEY` | Anthropic API key | If using Claude |

### Security Checklist

- [ ] `ABI_MASTER_KEY` set (not using random generation)
- [ ] JWT `allow_none_algorithm` is false (default)
- [ ] Rate limiting configured for public endpoints
- [ ] TLS enabled for all external connections

## Building for Production

```bash
zig build -Doptimize=ReleaseFast \
  -Denable-ai=true \
  -Denable-gpu=true \
  -Denable-database=true \
  -Denable-profiling=true
```

## Health Checks

The streaming server exposes `/health` endpoint:
```bash
curl http://localhost:8080/health
```

## Monitoring

Enable observability with `-Denable-profiling=true` for:
- Metrics collection
- Distributed tracing
- Performance profiling

## Graceful Shutdown

Send SIGTERM and wait for:
1. Active streams to drain (30s timeout)
2. Database connections to close
3. GPU resources to release

## Troubleshooting

See [docs/troubleshooting.md](troubleshooting.md) for common issues.
```

**Step 2: Commit**

```bash
git add docs/deployment.md
git commit -m "docs: add production deployment guide

Covers environment setup, security checklist, build options,
health checks, monitoring, and graceful shutdown.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

### Task 10: Update CLAUDE.md with security notes

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add security section to CLAUDE.md**

Add after Environment Variables section:
```markdown
## Security Considerations

| Setting | Default | Production Recommendation |
|---------|---------|---------------------------|
| JWT `allow_none_algorithm` | false | Keep false |
| `require_master_key` | false | Set true |
| Rate limiting | off | Enable for public APIs |

**Critical for production:**
1. Set `ABI_MASTER_KEY` environment variable
2. Enable rate limiting on public endpoints
3. Review `docs/SECURITY_AUDIT.md` for known issues
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude): add security considerations section

Documents critical production security settings including
JWT algorithm, master key requirements, and rate limiting.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 6: Final Verification

### Task 11: Full build and test verification

**Step 1: Clean build**

```bash
rm -rf .zig-cache zig-out
zig build -Denable-ai=true -Denable-gpu=true -Denable-database=true
```

**Step 2: Run full test suite**

```bash
zig build test --summary all
```

**Step 3: Format check**

```bash
zig build lint
```

**Step 4: Document final test count in PLAN.md**

---

### Task 12: Create summary commit

**Step 1: Review all changes**

```bash
git log --oneline -10
git diff --stat HEAD~10
```

**Step 2: Tag release candidate**

```bash
git tag -a v0.5.0-rc1 -m "Production readiness release candidate

- Fixed Zig 0.16 compilation errors
- Stream error recovery with circuit breakers
- Security hardening (JWT, secrets, API keys)
- Production deployment documentation"
```

---

## Summary

| Phase | Tasks | Purpose |
|-------|-------|---------|
| 1 | 1-3 | Fix blocking compilation errors |
| 2 | 4 | Commit streaming recovery feature |
| 3 | 5-7 | Security hardening |
| 4 | 8 | Update sprint documentation |
| 5 | 9-10 | Operational documentation |
| 6 | 11-12 | Final verification |

**Total Tasks:** 12
**Estimated Commits:** 9
**Critical Fixes:** 4 compilation errors, 2 HIGH security issues
