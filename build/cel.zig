const std = @import("std");
const builtin = @import("builtin");

/// CEL (Custom Environment Linker) toolchain integration for the ABI build
/// system. Detects the `.cel` patched Zig/ZLS toolchain and provides build-time
/// helpers to prefer it on blocked Darwin hosts (macOS 26+).
///
/// The `.cel` toolchain is a patched Zig built from source that fixes:
///   - `__availability_version_check` / `_arc4random_buf` undefined symbols
///   - `__CONST_ZIG` segment ordering issues (upstream #25521)
///   - SDK version detection mismatches in BUILD_VERSION load commands
///
/// Detection order:
///   1. `.cel/bin/zig` (repo-local patched build)
///   2. `$CEL_ZIG` environment variable (external CEL install)
///   3. Fall through to stock `zig` on PATH
/// Whether the build is running on a Darwin host known to be blocked by the
/// upstream Zig linker incompatibility.
pub const is_blocked_darwin = builtin.os.tag == .macos and
    builtin.os.version_range.semver.min.major >= 26;

/// CEL toolchain status detected at build time.
pub const CelStatus = enum {
    /// .cel/bin/zig exists and is executable
    available,
    /// .cel/ directory exists but no binary (needs build)
    needs_build,
    /// No .cel infrastructure present
    not_present,
    /// Not on a blocked platform (CEL not needed)
    not_needed,
};

/// Detect CEL toolchain availability at build time by probing the build root.
pub fn detectCelStatus(b: *std.Build) CelStatus {
    if (!is_blocked_darwin) return .not_needed;

    // Check for .cel/bin/zig
    b.build_root.handle.access(b.graph.io, ".cel/bin/zig", .{}) catch {
        // No binary — check if .cel/ directory exists (partial setup)
        b.build_root.handle.access(b.graph.io, ".cel/build.sh", .{}) catch {
            return .not_present;
        };
        return .needs_build;
    };
    return .available;
}

/// Add a `cel-check` build step that reports CEL toolchain status.
pub fn addCelCheckStep(b: *std.Build) *std.Build.Step {
    const step = b.step("cel-check", "Check .cel patched Zig/ZLS toolchain status for this platform");

    // Use a system command to report status since we need runtime output
    const cmd = b.addSystemCommand(&.{
        "sh", "-c",
        \\set -e
        \\printf '\n'
        \\printf '  .cel Toolchain Status\n'
        \\printf '  =====================\n\n'
        \\
        \\REPO_ROOT="$(pwd)"
        \\CEL_DIR="$REPO_ROOT/.cel"
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
        \\  printf '  CEL:       RECOMMENDED\n\n'
        \\else
        \\  printf '  Status:    OK (stock Zig should work)\n'
        \\  printf '  CEL:       Optional\n\n'
        \\fi
        \\
        \\# Check CEL binaries
        \\if [ -x "$CEL_ZIG" ]; then
        \\  CEL_VER="$("$CEL_ZIG" version 2>/dev/null || echo 'unknown')"
        \\  printf '  .cel/bin/zig:  FOUND (version %s)\n' "$CEL_VER"
        \\  
        \\  # Check version match
        \\  if [ -f "$REPO_ROOT/.zigversion" ]; then
        \\    EXPECTED="$(cat "$REPO_ROOT/.zigversion" | tr -d '[:space:]')"
        \\    if [ "$CEL_VER" = "$EXPECTED" ]; then
        \\      printf '  Version match: YES\n'
        \\    else
        \\      printf '  Version match: NO (cel=%s, expected=%s)\n' "$CEL_VER" "$EXPECTED"
        \\    fi
        \\  fi
        \\elif [ -f "$CEL_DIR/build.sh" ]; then
        \\  printf '  .cel/bin/zig:  NOT BUILT\n'
        \\  printf '  Action:        Run ./.cel/build.sh to build the patched toolchain\n'
        \\else
        \\  printf '  .cel/bin/zig:  NOT PRESENT\n'
        \\  printf '  Action:        CEL infrastructure not found in this checkout\n'
        \\fi
        \\if [ -x "$CEL_ZLS" ]; then
        \\  ZLS_VER="$("$CEL_ZLS" --version 2>/dev/null | head -n 1 || echo 'unknown')"
        \\  printf '  .cel/bin/zls:  FOUND (%s)\n' "$ZLS_VER"
        \\elif [ -x "$CEL_ZIG" ]; then
        \\  printf '  .cel/bin/zls:  NOT BUILT\n'
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
        \\  printf '    eval "$(./tools/scripts/use_cel.sh)"\n\n'
        \\elif [ "$MAJOR" -ge 26 ] 2>/dev/null && [ -x "$BOOTSTRAP_HOST_ZIG" ]; then
        \\  printf '  Next action:\n'
        \\  printf '    ./.cel/build.sh\n\n'
        \\elif [ "$MAJOR" -ge 26 ] 2>/dev/null && [ -d "$REPO_ROOT/zig-bootstrap-emergency/zig" ]; then
        \\  printf '  Next action:\n'
        \\  printf '    abi toolchain bootstrap\n\n'
        \\elif [ ! -x "$CEL_ZIG" ] && [ "$MAJOR" -ge 26 ] 2>/dev/null; then
        \\  printf '  To inspect CEL prerequisites first:\n'
        \\  printf '    ./tools/scripts/cel_migrate.sh --check\n\n'
        \\  printf '  To build the CEL toolchain:\n'
        \\  printf '    ./.cel/build.sh\n\n'
        \\  printf '  To use it:\n'
        \\  printf '    eval "$(./tools/scripts/use_cel.sh)"\n'
        \\  printf '    — or —\n'
        \\  printf '    export PATH="$(pwd)/.cel/bin:$PATH"\n\n'
        \\fi
    });

    step.dependOn(&cmd.step);
    return step;
}

/// Helper: create a build step that delegates to .cel/build.sh with an optional flag.
fn addCelShellStep(b: *std.Build, name: []const u8, desc: []const u8, flag: ?[]const u8) *std.Build.Step {
    const step = b.step(name, desc);

    const shell = if (flag) |f|
        std.fmt.comptimePrint(
            \\set -e
            \\if [ -x ".cel/build.sh" ]; then
            \\  exec ./.cel/build.sh {s}
            \\else
            \\  printf 'ERROR: .cel/build.sh not found\n' >&2
            \\  exit 1
            \\fi
        , .{f})
    else
        \\set -e
        \\if [ -x ".cel/build.sh" ]; then
        \\  exec ./.cel/build.sh
        \\else
        \\  printf 'ERROR: .cel/build.sh not found\n' >&2
        \\  exit 1
        \\fi
    ;

    const cmd = b.addSystemCommand(&.{ "sh", "-c", shell });
    step.dependOn(&cmd.step);
    return step;
}

/// Add a `cel-build` build step that triggers the CEL toolchain build.
pub fn addCelBuildStep(b: *std.Build) *std.Build.Step {
    return addCelShellStep(b, "cel-build", "Build the .cel patched Zig toolchain and ZLS from source", null);
}

/// Add a `cel-status` step that runs .cel/build.sh --status.
pub fn addCelStatusStep(b: *std.Build) *std.Build.Step {
    return addCelShellStep(b, "cel-status", "Show detailed .cel toolchain build status", "--status");
}

/// Add a `cel-verify` step that runs .cel/build.sh --verify.
pub fn addCelVerifyStep(b: *std.Build) *std.Build.Step {
    return addCelShellStep(b, "cel-verify", "Verify .cel patched Zig/ZLS binaries are ready", "--verify");
}

/// Emit a build-time note suggesting CEL when on a blocked Darwin host.
pub fn emitCelSuggestion(b: *std.Build, status: CelStatus) void {
    if (!is_blocked_darwin) return;

    switch (status) {
        .available => {
            const note = b.addSystemCommand(&.{
                "sh", "-c",
                \\printf '\n  NOTE: .cel toolchain detected. If using stock zig fails:\n'
                \\printf '    eval "$(./tools/scripts/use_cel.sh)"\n\n'
            });
            b.getInstallStep().dependOn(&note.step);
        },
        .needs_build => {
            const note = b.addSystemCommand(&.{
                "sh", "-c",
                \\if [ -x "zig-bootstrap-emergency/out/host/bin/zig" ]; then
                \\  printf '\n  NOTE: macOS 26+ detected. Bootstrap host Zig is ready; build the .cel toolchain:\n'
                \\  printf '    ./.cel/build.sh\n'
                \\elif [ -d "zig-bootstrap-emergency/zig" ]; then
                \\  printf '\n  NOTE: macOS 26+ detected. Refresh the bootstrap host Zig first:\n'
                \\  printf '    abi toolchain bootstrap\n'
                \\  printf '    ./.cel/build.sh\n'
                \\else
                \\  printf '\n  NOTE: macOS 26+ detected. Inspect prerequisites, then build the .cel toolchain:\n'
                \\  printf '    ./tools/scripts/cel_migrate.sh --check\n'
                \\  printf '    ./.cel/build.sh\n'
                \\fi
                \\printf '    eval "$(./tools/scripts/use_cel.sh)"\n\n'
            });
            b.getInstallStep().dependOn(&note.step);
        },
        .not_present, .not_needed => {},
    }
}
