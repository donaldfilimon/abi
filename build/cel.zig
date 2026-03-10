const std = @import("std");
const builtin = @import("builtin");

/// Zig bootstrap bridge integration for ABI's CEL transition. This module keeps
/// the legacy `.cel` implementation alive as the backing store, but all new
/// user-facing guidance should point at `.zig-bootstrap`.
///
/// The backing Zig toolchain is still built from `.cel` and fixes:
///   - `__availability_version_check` / `_arc4random_buf` undefined symbols
///   - `__CONST_ZIG` segment ordering issues (upstream #25521)
///   - SDK version detection mismatches in BUILD_VERSION load commands
///
/// Detection order:
///   1. `.cel/bin/zig` (repo-local backing build)
///   2. `$CEL_ZIG` environment variable (external CEL install)
///   3. Fall through to stock `zig` on PATH
/// Whether the build is running on a Darwin host known to be blocked by the
/// upstream Zig linker incompatibility.
pub const is_blocked_darwin = builtin.os.tag == .macos and
    builtin.os.version_range.semver.min.major >= 26;

/// Zig bootstrap bridge status detected at build time.
pub const CelStatus = enum {
    /// Backing `.cel/bin/zig` exists and is executable
    available,
    /// Wrapper exists but backing binaries still need a build
    needs_build,
    /// No bootstrap wrapper or backing bridge present
    not_present,
    /// Not on a blocked platform (bootstrap Zig not needed)
    not_needed,
};

/// Detect Zig bootstrap availability at build time by probing the build root.
pub fn detectCelStatus(b: *std.Build) CelStatus {
    if (!is_blocked_darwin) return .not_needed;

    // Check for the backing binary; the wrapper itself always exists once
    // tracked, so it is not enough to prove readiness.
    b.build_root.handle.access(b.graph.io, ".cel/bin/zig", .{}) catch {
        b.build_root.handle.access(b.graph.io, ".zig-bootstrap/build.sh", .{}) catch {
            return .not_present;
        };
        return .needs_build;
    };
    return .available;
}

fn addBootstrapCheckStepNamed(b: *std.Build, name: []const u8, desc: []const u8) *std.Build.Step {
    const step = b.step(name, desc);

    const cmd = b.addSystemCommand(&.{
        "sh", "-c",
        \\set -e
        \\printf '\n'
        \\printf '  Zig Bootstrap Status\n'
        \\printf '  ====================\n\n'
        \\
        \\REPO_ROOT="$(pwd)"
        \\BOOTSTRAP_DIR="$REPO_ROOT/.zig-bootstrap"
        \\CEL_DIR="$REPO_ROOT/.cel"
        \\BOOTSTRAP_ZIG="$BOOTSTRAP_DIR/bin/zig"
        \\BOOTSTRAP_ZLS="$BOOTSTRAP_DIR/bin/zls"
        \\CEL_ZIG="$CEL_DIR/bin/zig"
        \\CEL_ZLS="$CEL_DIR/bin/zls"
        \\BOOTSTRAP_HOST_ZIG="$REPO_ROOT/zig-bootstrap-emergency/out/host/bin/zig"
        \\EXPECTED="$(tr -d '[:space:]' < "$REPO_ROOT/.zigversion" 2>/dev/null || true)"
        \\
        \\# Platform check
        \\OS_VER="$(sw_vers -productVersion 2>/dev/null || echo 'unknown')"
        \\printf '  Platform:  macOS %s\n' "$OS_VER"
        \\
        \\# Check if blocked
        \\MAJOR="$(echo "$OS_VER" | cut -d. -f1)"
        \\if [ "$MAJOR" -ge 26 ] 2>/dev/null; then
        \\  printf '  Status:    BLOCKED (macOS 26+ linker incompatibility)\n'
        \\  printf '  Bootstrap: RECOMMENDED\n\n'
        \\else
        \\  printf '  Status:    OK (stock Zig should work)\n'
        \\  printf '  Bootstrap: Optional\n\n'
        \\fi
        \\
        \\# Check bootstrap wrapper and backing binaries
        \\if [ -x "$CEL_ZIG" ]; then
        \\  CEL_VER="$("$CEL_ZIG" version 2>/dev/null || echo 'unknown')"
        \\  printf '  .zig-bootstrap/bin/zig: FOUND (backed by %s)\n' "$CEL_VER"
        \\  
        \\  if [ -f "$REPO_ROOT/.zigversion" ]; then
        \\    EXPECTED="$(cat "$REPO_ROOT/.zigversion" | tr -d '[:space:]')"
        \\    if [ "$CEL_VER" = "$EXPECTED" ]; then
        \\      printf '  Version match: YES\n'
        \\    else
        \\      printf '  Version match: NO (zig=%s, expected=%s)\n' "$CEL_VER" "$EXPECTED"
        \\    fi
        \\  fi
        \\elif [ -f "$BOOTSTRAP_DIR/build.sh" ]; then
        \\  printf '  .zig-bootstrap/bin/zig: NOT BUILT\n'
        \\  printf '  Action:                 Run ./.zig-bootstrap/build.sh to build the backing Zig bridge\n'
        \\else
        \\  printf '  .zig-bootstrap/bin/zig: NOT PRESENT\n'
        \\  printf '  Action:                 Zig bootstrap bridge not found in this checkout\n'
        \\fi
        \\if [ -x "$CEL_ZLS" ]; then
        \\  ZLS_VER="$("$CEL_ZLS" --version 2>/dev/null | head -n 1 || echo 'unknown')"
        \\  printf '  .zig-bootstrap/bin/zls: FOUND (%s)\n' "$ZLS_VER"
        \\elif [ -x "$CEL_ZIG" ]; then
        \\  printf '  .zig-bootstrap/bin/zls: NOT BUILT\n'
        \\fi
        \\if [ -x "$BOOTSTRAP_HOST_ZIG" ]; then
        \\  BOOTSTRAP_VER="$("$BOOTSTRAP_HOST_ZIG" version 2>/dev/null || echo 'unknown')"
        \\  printf '  bootstrap zig: FOUND (%s)\n' "$BOOTSTRAP_VER"
        \\elif [ -d "$REPO_ROOT/zig-bootstrap-emergency/zig" ]; then
        \\  printf '  bootstrap zig: source present, host binary not built\n'
        \\fi
        \\
        \\# Check patches
        \\if [ -d "$CEL_DIR/patches" ]; then
        \\  PATCH_COUNT="$(ls "$CEL_DIR/patches/"*.patch 2>/dev/null | wc -l | tr -d ' ')"
        \\  printf '  Patches:       %s patch file(s)\n' "$PATCH_COUNT"
        \\fi
        \\
        \\# Check stock zig
        \\if command -v zig >/dev/null 2>&1; then
        \\  STOCK_VER="$(zig version 2>/dev/null || echo 'unknown')"
        \\  STOCK_PATH="$(command -v zig)"
        \\  printf '  Stock zig:     %s (%s)\n' "$STOCK_VER" "$STOCK_PATH"
        \\  if [ -n "$EXPECTED" ] && [ "$STOCK_VER" = "$EXPECTED" ]; then
        \\    printf '  Stock pin:     matches repo pin\n'
        \\  elif [ -n "$EXPECTED" ]; then
        \\    printf '  Stock pin:     mismatch (expected %s)\n' "$EXPECTED"
        \\  fi
        \\  if BUILD_ERR="$(zig build --help 2>&1 1>/dev/null)"; then
        \\    printf '  Build runner:  stock zig can start ABI build steps\n'
        \\  elif printf '%s' "$BUILD_ERR" | grep -q '__availability_version_check\\|undefined symbol:'; then
        \\    printf '  Build runner:  BLOCKED by Darwin linker failure\n'
        \\  else
        \\    printf '  Build runner:  stock zig failed before ABI gates could run\n'
        \\  fi
        \\else
        \\  printf '  Stock zig:     NOT FOUND\n'
        \\fi
        \\
        \\printf '\n'
        \\
        \\# Instructions
        \\if [ -x "$CEL_ZIG" ]; then
        \\  printf '  Next action:\n'
        \\  printf '    eval "$(./tools/scripts/use_zig_bootstrap.sh)"\n\n'
        \\elif [ "$MAJOR" -ge 26 ] 2>/dev/null && [ -x "$BOOTSTRAP_HOST_ZIG" ]; then
        \\  printf '  Next action:\n'
        \\  printf '    ./.zig-bootstrap/build.sh\n\n'
        \\elif [ "$MAJOR" -ge 26 ] 2>/dev/null && [ -d "$REPO_ROOT/zig-bootstrap-emergency/zig" ]; then
        \\  printf '  Next action:\n'
        \\  printf '    abi bootstrap-zig bootstrap\n\n'
        \\elif [ ! -x "$CEL_ZIG" ] && [ "$MAJOR" -ge 26 ] 2>/dev/null; then
        \\  printf '  To inspect bootstrap prerequisites first:\n'
        \\  printf '    ./tools/scripts/zig_bootstrap_migrate.sh --check\n\n'
        \\  printf '  To build the Zig bridge:\n'
        \\  printf '    ./.zig-bootstrap/build.sh\n\n'
        \\  printf '  To use it:\n'
        \\  printf '    eval "$(./tools/scripts/use_zig_bootstrap.sh)"\n'
        \\  printf '    — or —\n'
        \\  printf '    export PATH="$(pwd)/.zig-bootstrap/bin:$PATH"\n\n'
        \\fi
    });

    step.dependOn(&cmd.step);
    return step;
}

pub fn addZigBootstrapCheckStep(b: *std.Build) *std.Build.Step {
    return addBootstrapCheckStepNamed(b, "zig-bootstrap-check", "Check the repo-local Zig bootstrap bridge status for this platform");
}

/// Add a legacy `cel-check` alias step for one migration wave.
pub fn addCelCheckStep(b: *std.Build) *std.Build.Step {
    return addBootstrapCheckStepNamed(b, "cel-check", "Deprecated alias for zig-bootstrap-check");
}

fn addBootstrapShellStep(b: *std.Build, name: []const u8, desc: []const u8, flag: ?[]const u8) *std.Build.Step {
    const step = b.step(name, desc);

    const shell = if (flag) |f|
        std.fmt.comptimePrint(
            \\set -e
            \\if [ -x ".zig-bootstrap/build.sh" ]; then
            \\  exec ./.zig-bootstrap/build.sh {s}
            \\else
            \\  printf 'ERROR: .zig-bootstrap/build.sh not found\n' >&2
            \\  exit 1
            \\fi
        , .{f})
    else
        \\set -e
        \\if [ -x ".zig-bootstrap/build.sh" ]; then
        \\  exec ./.zig-bootstrap/build.sh
        \\else
        \\  printf 'ERROR: .zig-bootstrap/build.sh not found\n' >&2
        \\  exit 1
        \\fi
    ;

    const cmd = b.addSystemCommand(&.{ "sh", "-c", shell });
    step.dependOn(&cmd.step);
    return step;
}

pub fn addZigBootstrapBuildStep(b: *std.Build) *std.Build.Step {
    return addBootstrapShellStep(b, "zig-bootstrap-build", "Build the repo-local Zig bootstrap bridge and ZLS from source", null);
}

pub fn addCelBuildStep(b: *std.Build) *std.Build.Step {
    return addBootstrapShellStep(b, "cel-build", "Deprecated alias for zig-bootstrap-build", null);
}

pub fn addZigBootstrapStatusStep(b: *std.Build) *std.Build.Step {
    return addBootstrapShellStep(b, "zig-bootstrap-status", "Show detailed Zig bootstrap bridge status", "--status");
}

pub fn addCelStatusStep(b: *std.Build) *std.Build.Step {
    return addBootstrapShellStep(b, "cel-status", "Deprecated alias for zig-bootstrap-status", "--status");
}

pub fn addZigBootstrapVerifyStep(b: *std.Build) *std.Build.Step {
    return addBootstrapShellStep(b, "zig-bootstrap-verify", "Verify the repo-local Zig bootstrap bridge is ready", "--verify");
}

pub fn addCelVerifyStep(b: *std.Build) *std.Build.Step {
    return addBootstrapShellStep(b, "cel-verify", "Deprecated alias for zig-bootstrap-verify", "--verify");
}

/// Emit a build-time note suggesting CEL when on a blocked Darwin host.
pub fn emitCelSuggestion(b: *std.Build, status: CelStatus) void {
    if (!is_blocked_darwin) return;

    switch (status) {
        .available => {
            const note = b.addSystemCommand(&.{
                "sh", "-c",
                \\printf '\n  NOTE: Zig bootstrap bridge detected. If using stock zig fails:\n'
                \\printf '    eval "$(./tools/scripts/use_zig_bootstrap.sh)"\n\n'
            });
            b.getInstallStep().dependOn(&note.step);
        },
        .needs_build => {
            const note = b.addSystemCommand(&.{
                "sh", "-c",
                \\if [ -x "zig-bootstrap-emergency/out/host/bin/zig" ]; then
                \\  printf '\n  NOTE: macOS 26+ detected. Bootstrap host Zig is ready; build the Zig bridge:\n'
                \\  printf '    ./.zig-bootstrap/build.sh\n'
                \\elif [ -d "zig-bootstrap-emergency/zig" ]; then
                \\  printf '\n  NOTE: macOS 26+ detected. Refresh the bootstrap host Zig first:\n'
                \\  printf '    abi bootstrap-zig bootstrap\n'
                \\  printf '    ./.zig-bootstrap/build.sh\n'
                \\else
                \\  printf '\n  NOTE: macOS 26+ detected. Inspect prerequisites, then build the Zig bridge:\n'
                \\  printf '    ./tools/scripts/zig_bootstrap_migrate.sh --check\n'
                \\  printf '    ./.zig-bootstrap/build.sh\n'
                \\fi
                \\printf '    eval "$(./tools/scripts/use_zig_bootstrap.sh)"\n\n'
            });
            b.getInstallStep().dependOn(&note.step);
        },
        .not_present, .not_needed => {},
    }
}
