//! TLS configuration for the WDBX REST server (disclosed partial).
//!
//! Reads `ABI_WDBX_TLS_CERT` and `ABI_WDBX_TLS_KEY` environment variables to
//! load and validate TLS certificate/key file paths. When both are set the
//! REST server logs the configuration and advises deploying behind an external
//! TLS-terminating reverse proxy (nginx, Caddy, haproxy) for production HTTPS.
//!
//! **Native TLS termination is not linked in this build.** Zig 0.17 has no
//! `std.crypto.tls.Server` (only `Client`). The server continues to serve plain
//! HTTP over the local TCP socket. TLS configuration here is a **forward-looking
//! cap — the env vars are read and validated so that wiring is frictionless**
//! when native TLS arrives or when deploying behind a proxy.
//!
//! ## Security
//!
//! - Self-signed certificates are accepted.
//! - This is NOT production-audited TLS. No NAC, OCSP stapling, or HSTS is
//!   performed. For non-loopback exposure, always deploy a hardened reverse
//!   proxy with full TLS termination.
//! - Loopback-only (127.0.0.1) remains the **default** binding; TLS config alone
//!   does NOT bind to `0.0.0.0`.
//! - Bearer-token auth (`ABI_WDBX_REST_TOKEN`) and rate limiting remain fully
//!   orthogonal and continue to work alongside TLS configuration.

const std = @import("std");
const env = @import("../../foundation/env.zig");

pub const TLS_CERT_ENV = "ABI_WDBX_TLS_CERT";
pub const TLS_KEY_ENV = "ABI_WDBX_TLS_KEY";

/// TLS configuration loaded from environment variables.
///
/// All slices are borrowed from the process environment map (valid for the
/// entire process lifetime). Callers must not free them.
pub const TlsConfig = struct {
    /// Path to the PEM-encoded TLS certificate file.
    cert_path: []const u8,
    /// Path to the PEM-encoded TLS private key file.
    key_path: []const u8,

    /// Read and validate TLS configuration from environment variables.
    ///
    /// `io` is forwarded to the filesystem access check (Zig 0.17
    /// `std.Io.Dir.cwd().access(io, ...)` pattern).
    ///
    /// Returns `null` (and logs a warning) when:
    /// - Neither env var is set (TLS not configured — normal loopback mode).
    /// - Only one of the two env vars is set (misconfiguration).
    /// - The cert or key file does not exist or is not readable.
    ///
    /// Returns a `TlsConfig` when both env vars are set and both files are
    /// accessible. The caller should log the configuration for transparency.
    pub fn fromEnv(io: std.Io) ?TlsConfig {
        const cert = env.get(TLS_CERT_ENV) orelse {
            if (env.get(TLS_KEY_ENV) != null) {
                std.log.warn("TLS: {s} set without {s}; ignoring TLS config", .{ TLS_KEY_ENV, TLS_CERT_ENV });
            }
            return null;
        };
        const key = env.get(TLS_KEY_ENV) orelse {
            std.log.warn("TLS: {s} set without {s}; ignoring TLS config", .{ TLS_CERT_ENV, TLS_KEY_ENV });
            return null;
        };

        // Validate certificate file is accessible.
        const cwd = std.Io.Dir.cwd();
        cwd.access(io, cert, .{}) catch |err| {
            std.log.warn("TLS: cert file '{s}' inaccessible ({s}); TLS disabled", .{ cert, @errorName(err) });
            return null;
        };

        // Validate key file is accessible.
        cwd.access(io, key, .{}) catch |err| {
            std.log.warn("TLS: key file '{s}' inaccessible ({s}); TLS disabled", .{ key, @errorName(err) });
            return null;
        };

        return TlsConfig{
            .cert_path = cert,
            .key_path = key,
        };
    }
};

test "TlsConfig: returns null when env is unset" {
    // No env vars installed in test context, so fromEnv returns null.
    try std.testing.expect(TlsConfig.fromEnv(std.testing.io) == null);
}

test {
    std.testing.refAllDecls(@This());
}
