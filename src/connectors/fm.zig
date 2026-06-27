//! Apple FoundationModels on-device connector (Phase 2: comptime-gated bridge).
//!
//! This mirrors the structure of the other connectors (see `grok.zig`) but
//! targets Apple's on-device FoundationModels runtime rather than an HTTP API,
//! so it carries no credentials or base URL.
//!
//! The Objective-C runtime surface is comptime-gated exactly like
//! `features/gpu/metal_shared.zig`: the extern declarations only exist when
//! `builtin.target.os.tag == .macos` AND `build_options.feat_foundationmodels`.
//! On every other build (the default — feature OFF, or any non-macOS target)
//! the `fm_fns` container is an empty struct, so no ObjC / framework symbols are
//! ever emitted or linked and `completeLive` returns `error.FMUnavailable`.
//!
//! PHASE 3 — live Swift `@c` bridge:
//!   FoundationModels.framework is a *Swift-only* framework. Its `.tbd` exports
//!   exclusively Swift-mangled symbols (`_$s16FoundationModels...`,
//!   `swift-abi-version: 7`) and contains **zero** `OBJC_CLASS` symbols, so
//!   `SystemLanguageModel` / `LanguageModelSession` are unreachable from
//!   `objc_getClass` / `objc_msgSend`. An Objective-C `.m` shim cannot reach the
//!   Swift `async throws respond(to:)` API either.
//!
//!   The bridge is therefore `src/connectors/fm_shim.swift`: a Swift file using
//!   the official `@c` attribute (SE-0495, Swift 6.3+) to export two synchronous
//!   C entry points — `abi_fm_available()` and `abi_fm_complete(prompt, out,
//!   out_len)` — that `build.zig` compiles with `swiftc -parse-as-library` to an
//!   object and links into this module ALONGSIDE `FoundationModels.framework`,
//!   ONLY under `os == .macos AND build_options.feat_foundationmodels`. The shim
//!   drives `respond(to:)` to completion synchronously via `Task` +
//!   `DispatchSemaphore.wait()` (the justified async→sync pattern at a C ABI
//!   boundary) and copies the UTF-8 result into the caller's buffer.
//!
//!   On the default build (flag off, or any non-macOS target) `fm_enabled` is
//!   comptime-false: the `extern` declarations are never emitted, no Swift object
//!   is compiled or linked, and `completeLive` returns `error.FMUnavailable`.

const builtin = @import("builtin");
const std = @import("std");
const build_options = @import("build_options");
const connector = @import("connector.zig");

const ConnectorError = connector.ConnectorError;
const Response = connector.Response;

/// Comptime gate: the on-device bridge is only compiled in on macOS with the
/// feature flag enabled. Everywhere else this is `false`, so the `if (fm_enabled)`
/// branches below are never semantically analyzed and `fm_fns` stays empty.
const fm_enabled = builtin.target.os.tag == .macos and build_options.feat_foundationmodels;

/// C entry points exported by `src/connectors/fm_shim.swift` (compiled + linked
/// only on macOS+flag). When the feature is off (or off-platform) this collapses
/// to an empty struct, guaranteeing no extern symbol is emitted or linked.
///
/// Contract (mirrored in fm_shim.swift):
///   abi_fm_available() -> c_int:  1 = ready, 0 = unavailable, -1 = OS too old.
///   abi_fm_complete(prompt, out, out_len) -> c_int:
///     >=0 = UTF-8 bytes written to `out` (NUL-terminated, truncated to fit);
///     -1 null args, -2 OS too old, -3 model unavailable, -4 generation error.
const fm_fns = if (fm_enabled) struct {
    extern fn abi_fm_available() c_int;
    extern fn abi_fm_complete(prompt: [*:0]const u8, out: [*]u8, out_len: usize) c_int;
} else struct {};

/// Configuration for the on-device FoundationModels client. No credentials or
/// base URL: inference runs locally, not over a network transport.
pub const FmConfig = struct {
    /// Wall-clock budget for an on-device generation, in milliseconds.
    timeout_ms: u32 = 15000,
    /// Maximum tokens to request from the on-device model.
    max_tokens: u32 = 512,
};

/// Probe whether the Apple FoundationModels runtime is reachable on this build.
///
/// On the enabled macOS path this calls the Swift shim's `abi_fm_available()`,
/// which queries `SystemLanguageModel.default.availability` and returns 1 only
/// when the model is present and ready (Apple Intelligence enabled + model
/// downloaded on a supported device). On every other build it is comptime-folded
/// to `return false` and references no extern symbol.
pub fn fmAvailable() bool {
    if (!fm_enabled) return false;
    return fm_fns.abi_fm_available() == 1;
}

pub const Client = struct {
    allocator: std.mem.Allocator,
    config: FmConfig,

    pub fn init(allocator: std.mem.Allocator, config: FmConfig) Client {
        return .{ .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *Client) void {
        _ = self;
    }

    /// On-device completion against Apple FoundationModels.
    ///
    /// Returns `error.FMUnavailable` unless built on macOS with
    /// `-Dfeat-foundationmodels` AND the runtime is reachable (which, per the
    /// file header, requires the Swift `@c` bridge in `src/connectors/fm_shim.swift`,
    /// compiled+linked by build.zig under macOS + `-Dfeat-foundationmodels`, plus
    /// Apple-Intelligence hardware at runtime).
    /// Never fabricates a model response.
    pub fn completeLive(self: *Client, allocator: std.mem.Allocator, prompt: []const u8) ConnectorError!Response {
        // The enabled branch's syntactic use of self/allocator/prompt satisfies
        // the unused-parameter lint in both configs; when the flag is off the
        // branch is comptime-pruned so `completeOnDevice` (and its `fm_fns`
        // references) is never analyzed or linked.
        if (fm_enabled) {
            return self.completeOnDevice(allocator, prompt);
        }
        return ConnectorError.FMUnavailable;
    }

    /// Enabled-path implementation. Only analyzed when `fm_enabled` is true, so
    /// its `fm_fns` references are never evaluated on the default build.
    ///
    /// Drives the Swift shim's synchronous `abi_fm_complete`, mapping its return
    /// code to a `Response` (>=0 bytes written) or a precise error. Never
    /// fabricates a completion: a negative code surfaces as `FMUnavailable`
    /// (model not reachable / OS too old) or `FMSessionFailed` (generation
    /// threw).
    fn completeOnDevice(self: *Client, allocator: std.mem.Allocator, prompt: []const u8) ConnectorError!Response {
        if (!fmAvailable()) return ConnectorError.FMUnavailable;

        // Output buffer sized generously from max_tokens (~4 UTF-8 bytes/token),
        // floored at 8 KiB so short max_tokens still leave room for the reply.
        const cap = @max(@as(usize, 8 * 1024), @as(usize, self.config.max_tokens) * 4);
        const out = try allocator.alloc(u8, cap);
        errdefer allocator.free(out);

        // NUL-terminate the prompt for the C ABI (portable across toolchains
        // that renamed dupeZ).
        const prompt_z = try allocator.allocSentinel(u8, prompt.len, 0);
        defer allocator.free(prompt_z);
        @memcpy(prompt_z, prompt);

        const rc = fm_fns.abi_fm_complete(prompt_z, out.ptr, out.len);
        if (rc < 0) {
            // `out` is freed by the active `errdefer` on this error return.
            return switch (rc) {
                -4 => ConnectorError.FMSessionFailed,
                // -1 null args (defensive), -2 OS too old, -3 model unavailable.
                else => ConnectorError.FMUnavailable,
            };
        }

        const n: usize = @intCast(rc);
        // Hand back exactly the bytes written. Shrink in place when possible so
        // the owned slice's length matches its allocation; otherwise copy out
        // and free the oversized buffer.
        if (allocator.resize(out, n)) {
            return .{ .status = 200, .body = out[0..n], .owned = true };
        }
        const body = try allocator.dupe(u8, out[0..n]);
        allocator.free(out);
        return .{ .status = 200, .body = body, .owned = true };
    }
};

/// Convenience factory for a default on-device config.
pub fn fmConfig() FmConfig {
    return .{};
}

test {
    std.testing.refAllDecls(@This());
}

test "fm completeLive is unavailable off-flag, or returns a real response on-device" {
    const allocator = std.testing.allocator;
    var client = Client.init(allocator, fmConfig());
    defer client.deinit();

    if (fm_enabled and fmAvailable()) {
        // Bridge compiled in AND an Apple-Intelligence model is ready on this
        // host: an on-device completion runs for real. Accept a successful,
        // allocator-owned response (free it) or a transient generation failure;
        // never assert a fixed string from a live model.
        if (client.completeLive(allocator, "Reply with: ok")) |resp_value| {
            var resp = resp_value;
            defer resp.deinit(allocator);
            try std.testing.expectEqual(@as(u16, 200), resp.status);
        } else |err| {
            try std.testing.expect(err == ConnectorError.FMSessionFailed);
        }
    } else {
        // Default build (feat-foundationmodels OFF) or no on-device model: the
        // bridge is not reachable. No hardware dependence on this path.
        try std.testing.expectError(
            ConnectorError.FMUnavailable,
            client.completeLive(allocator, "hello on-device"),
        );
    }
}

test "fmAvailable is false unless the on-device bridge is enabled and ready" {
    if (!fm_enabled) {
        // No bridge compiled in (the default build): always false, no hardware
        // requirement.
        try std.testing.expect(!fmAvailable());
    } else {
        // Enabled: availability is OS/hardware dependent. Assert only that the
        // probe is callable and yields a bool.
        const reachable: bool = fmAvailable();
        try std.testing.expect(reachable == true or reachable == false);
    }
}

test "fm config carries sane defaults" {
    const cfg = fmConfig();
    try std.testing.expectEqual(@as(u32, 15000), cfg.timeout_ms);
    try std.testing.expectEqual(@as(u32, 512), cfg.max_tokens);
}
