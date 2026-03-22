//! HTTP security headers middleware.
//!
//! This module provides:
//! - Content-Security-Policy (CSP)
//! - X-Frame-Options
//! - X-Content-Type-Options
//! - X-XSS-Protection
//! - Strict-Transport-Security (HSTS)
//! - Referrer-Policy
//! - Permissions-Policy
//! - Cache-Control security headers
//! - Custom header injection

const std = @import("std");
const csprng = @import("csprng.zig");

/// Security header configuration
pub const SecurityHeadersConfig = struct {
    /// Enable security headers
    enabled: bool = true,
    /// Content-Security-Policy settings
    csp: CspConfig = .{},
    /// X-Frame-Options setting
    frame_options: FrameOptions = .deny,
    /// Enable X-Content-Type-Options: nosniff
    nosniff: bool = true,
    /// X-XSS-Protection setting
    xss_protection: XssProtection = .block,
    /// HSTS settings
    hsts: HstsConfig = .{},
    /// Referrer-Policy setting
    referrer_policy: ReferrerPolicy = .strict_origin_when_cross_origin,
    /// Permissions-Policy settings
    permissions_policy: PermissionsPolicyConfig = .{},
    /// Cache-Control for sensitive responses
    cache_control: CacheControlConfig = .{},
    /// Remove server identification headers
    remove_server_header: bool = true,
    /// Remove powered-by headers
    remove_powered_by: bool = true,
    /// Cross-Origin-Embedder-Policy
    coep: ?[]const u8 = "require-corp",
    /// Cross-Origin-Opener-Policy
    coop: ?[]const u8 = "same-origin",
    /// Cross-Origin-Resource-Policy
    corp: ?[]const u8 = "same-origin",
};

/// Content-Security-Policy configuration
pub const CspConfig = struct {
    enabled: bool = true,
    /// Default source directive
    default_src: []const []const u8 = &.{"'self'"},
    /// Script sources
    script_src: []const []const u8 = &.{"'self'"},
    /// Style sources
    style_src: []const []const u8 = &.{ "'self'", "'unsafe-inline'" },
    /// Image sources
    img_src: []const []const u8 = &.{ "'self'", "data:" },
    /// Font sources
    font_src: []const []const u8 = &.{"'self'"},
    /// Connect sources (XHR, WebSocket, etc.)
    connect_src: []const []const u8 = &.{"'self'"},
    /// Frame sources
    frame_src: []const []const u8 = &.{"'none'"},
    /// Object sources (plugins)
    object_src: []const []const u8 = &.{"'none'"},
    /// Media sources
    media_src: []const []const u8 = &.{"'self'"},
    /// Base URI restriction
    base_uri: []const []const u8 = &.{"'self'"},
    /// Form action targets
    form_action: []const []const u8 = &.{"'self'"},
    /// Frame ancestors
    frame_ancestors: []const []const u8 = &.{"'none'"},
    /// Upgrade insecure requests
    upgrade_insecure_requests: bool = true,
    /// Block all mixed content
    block_all_mixed_content: bool = true,
    /// Report URI for CSP violations
    report_uri: ?[]const u8 = null,
    /// Report-To endpoint
    report_to: ?[]const u8 = null,
    /// Use nonces for scripts (dynamically generated)
    use_nonces: bool = false,
    /// Nonce prefix
    nonce_prefix: []const u8 = "csp-",
};

/// X-Frame-Options values
pub const FrameOptions = enum {
    deny,
    sameorigin,
    allow_from,

    pub fn toString(self: FrameOptions) []const u8 {
        return switch (self) {
            .deny => "DENY",
            .sameorigin => "SAMEORIGIN",
            .allow_from => "ALLOW-FROM",
        };
    }
};

/// X-XSS-Protection values
pub const XssProtection = enum {
    disabled,
    enabled,
    block,

    pub fn toString(self: XssProtection) []const u8 {
        return switch (self) {
            .disabled => "0",
            .enabled => "1",
            .block => "1; mode=block",
        };
    }
};

/// HSTS configuration
pub const HstsConfig = struct {
    enabled: bool = true,
    /// Max age in seconds
    max_age: u64 = 31536000, // 1 year
    /// Include subdomains
    include_subdomains: bool = true,
    /// Enable preload
    preload: bool = false,
};

/// Referrer-Policy values
pub const ReferrerPolicy = enum {
    no_referrer,
    no_referrer_when_downgrade,
    origin,
    origin_when_cross_origin,
    same_origin,
    strict_origin,
    strict_origin_when_cross_origin,
    unsafe_url,

    pub fn toString(self: ReferrerPolicy) []const u8 {
        return switch (self) {
            .no_referrer => "no-referrer",
            .no_referrer_when_downgrade => "no-referrer-when-downgrade",
            .origin => "origin",
            .origin_when_cross_origin => "origin-when-cross-origin",
            .same_origin => "same-origin",
            .strict_origin => "strict-origin",
            .strict_origin_when_cross_origin => "strict-origin-when-cross-origin",
            .unsafe_url => "unsafe-url",
        };
    }
};

/// Permissions-Policy configuration
pub const PermissionsPolicyConfig = struct {
    enabled: bool = true,
    /// Accelerometer permission
    accelerometer: []const u8 = "()",
    /// Ambient light sensor
    ambient_light_sensor: []const u8 = "()",
    /// Autoplay
    autoplay: []const u8 = "()",
    /// Battery status
    battery: []const u8 = "()",
    /// Camera
    camera: []const u8 = "()",
    /// Display capture
    display_capture: []const u8 = "()",
    /// Document domain
    document_domain: []const u8 = "()",
    /// Encrypted media
    encrypted_media: []const u8 = "()",
    /// Fullscreen
    fullscreen: []const u8 = "(self)",
    /// Geolocation
    geolocation: []const u8 = "()",
    /// Gyroscope
    gyroscope: []const u8 = "()",
    /// Magnetometer
    magnetometer: []const u8 = "()",
    /// Microphone
    microphone: []const u8 = "()",
    /// MIDI
    midi: []const u8 = "()",
    /// Payment
    payment: []const u8 = "()",
    /// Picture-in-picture
    picture_in_picture: []const u8 = "()",
    /// USB
    usb: []const u8 = "()",
    /// Web Share
    web_share: []const u8 = "(self)",
    /// XR spatial tracking
    xr_spatial_tracking: []const u8 = "()",
};

/// Cache-Control configuration for sensitive responses
pub const CacheControlConfig = struct {
    /// For sensitive/authenticated responses
    sensitive: []const u8 = "no-store, no-cache, must-revalidate, proxy-revalidate",
    /// For public static assets
    public_static: []const u8 = "public, max-age=31536000, immutable",
    /// For private but cacheable content
    private: []const u8 = "private, no-cache, must-revalidate",
    /// Disable caching completely
    no_cache: []const u8 = "no-store, no-cache, must-revalidate, private",
};

/// Security header name-value pair
pub const Header = struct {
    name: []const u8,
    value: []const u8,
};

/// Security headers builder
pub const SecurityHeaders = struct {
    allocator: std.mem.Allocator,
    config: SecurityHeadersConfig,
    /// Generated nonce for CSP (if enabled)
    current_nonce: ?[]const u8 = null,

    pub fn init(allocator: std.mem.Allocator, config: SecurityHeadersConfig) SecurityHeaders {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn deinit(self: *SecurityHeaders) void {
        if (self.current_nonce) |nonce| {
            self.allocator.free(nonce);
        }
    }

    /// Generate all security headers
    pub fn getHeaders(self: *SecurityHeaders) ![]Header {
        var headers = std.ArrayListUnmanaged(Header).empty;
        errdefer headers.deinit(self.allocator);

        if (!self.config.enabled) {
            return headers.toOwnedSlice(self.allocator);
        }

        // Content-Security-Policy
        if (self.config.csp.enabled) {
            const csp = try self.buildCsp();
            try headers.append(self.allocator, .{ .name = "Content-Security-Policy", .value = csp });
        }

        // X-Frame-Options
        try headers.append(self.allocator, .{
            .name = "X-Frame-Options",
            .value = self.config.frame_options.toString(),
        });

        // X-Content-Type-Options
        if (self.config.nosniff) {
            try headers.append(self.allocator, .{
                .name = "X-Content-Type-Options",
                .value = "nosniff",
            });
        }

        // X-XSS-Protection
        try headers.append(self.allocator, .{
            .name = "X-XSS-Protection",
            .value = self.config.xss_protection.toString(),
        });

        // Strict-Transport-Security
        if (self.config.hsts.enabled) {
            const hsts = try self.buildHsts();
            try headers.append(self.allocator, .{ .name = "Strict-Transport-Security", .value = hsts });
        }

        // Referrer-Policy
        try headers.append(self.allocator, .{
            .name = "Referrer-Policy",
            .value = self.config.referrer_policy.toString(),
        });

        // Permissions-Policy
        if (self.config.permissions_policy.enabled) {
            const pp = try self.buildPermissionsPolicy();
            try headers.append(self.allocator, .{ .name = "Permissions-Policy", .value = pp });
        }

        // Cross-Origin policies
        if (self.config.coep) |coep| {
            try headers.append(self.allocator, .{ .name = "Cross-Origin-Embedder-Policy", .value = coep });
        }
        if (self.config.coop) |coop| {
            try headers.append(self.allocator, .{ .name = "Cross-Origin-Opener-Policy", .value = coop });
        }
        if (self.config.corp) |corp| {
            try headers.append(self.allocator, .{ .name = "Cross-Origin-Resource-Policy", .value = corp });
        }

        return headers.toOwnedSlice(self.allocator);
    }

    /// Get headers that should be removed
    pub fn getHeadersToRemove(self: *SecurityHeaders) []const []const u8 {
        // Capacity 10 is sufficient: max 4 headers can be added (provably safe)
        var to_remove: std.StaticArrayList([]const u8, 10) = .{};

        if (self.config.remove_server_header) {
            to_remove.appendAssumeCapacity("Server");
        }
        if (self.config.remove_powered_by) {
            to_remove.appendAssumeCapacity("X-Powered-By");
            to_remove.appendAssumeCapacity("X-AspNet-Version");
            to_remove.appendAssumeCapacity("X-AspNetMvc-Version");
        }

        return to_remove.items;
    }

    /// Generate a new nonce for CSP
    pub fn generateNonce(self: *SecurityHeaders) ![]const u8 {
        // Free old nonce
        if (self.current_nonce) |old| {
            self.allocator.free(old);
        }

        // Generate 16 random bytes
        var random_bytes: [16]u8 = undefined;
        csprng.fillRandom(&random_bytes);

        // Encode as base64
        const encoder = std.base64.standard.Encoder;
        const size = encoder.calcSize(random_bytes.len);
        const nonce = try self.allocator.alloc(u8, self.config.csp.nonce_prefix.len + size);

        @memcpy(nonce[0..self.config.csp.nonce_prefix.len], self.config.csp.nonce_prefix);
        _ = encoder.encode(nonce[self.config.csp.nonce_prefix.len..], &random_bytes);

        self.current_nonce = nonce;
        return nonce;
    }

    /// Get current nonce (or generate one)
    pub fn getNonce(self: *SecurityHeaders) ![]const u8 {
        if (self.current_nonce) |nonce| {
            return nonce;
        }
        return self.generateNonce();
    }

    /// Get cache control header for response type
    pub fn getCacheControl(self: *SecurityHeaders, response_type: CacheResponseType) []const u8 {
        return switch (response_type) {
            .sensitive => self.config.cache_control.sensitive,
            .public_static => self.config.cache_control.public_static,
            .private => self.config.cache_control.private,
            .no_cache => self.config.cache_control.no_cache,
        };
    }

    // Private helpers

    fn buildCsp(self: *SecurityHeaders) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        const csp = self.config.csp;

        // Build directives
        try self.appendDirective(&buffer, "default-src", csp.default_src);
        try self.appendDirective(&buffer, "script-src", csp.script_src);
        try self.appendDirective(&buffer, "style-src", csp.style_src);
        try self.appendDirective(&buffer, "img-src", csp.img_src);
        try self.appendDirective(&buffer, "font-src", csp.font_src);
        try self.appendDirective(&buffer, "connect-src", csp.connect_src);
        try self.appendDirective(&buffer, "frame-src", csp.frame_src);
        try self.appendDirective(&buffer, "object-src", csp.object_src);
        try self.appendDirective(&buffer, "media-src", csp.media_src);
        try self.appendDirective(&buffer, "base-uri", csp.base_uri);
        try self.appendDirective(&buffer, "form-action", csp.form_action);
        try self.appendDirective(&buffer, "frame-ancestors", csp.frame_ancestors);

        if (csp.upgrade_insecure_requests) {
            try buffer.appendSlice(self.allocator, "upgrade-insecure-requests; ");
        }

        if (csp.block_all_mixed_content) {
            try buffer.appendSlice(self.allocator, "block-all-mixed-content; ");
        }

        if (csp.report_uri) |uri| {
            {
                const tmp = try std.fmt.allocPrint(self.allocator, "report-uri {s}; ", .{uri});
                defer self.allocator.free(tmp);
                try buffer.appendSlice(self.allocator, tmp);
            }
        }

        if (csp.report_to) |endpoint| {
            {
                const tmp = try std.fmt.allocPrint(self.allocator, "report-to {s}; ", .{endpoint});
                defer self.allocator.free(tmp);
                try buffer.appendSlice(self.allocator, tmp);
            }
        }

        // Remove trailing "; "
        if (buffer.items.len >= 2) {
            buffer.shrinkRetainingCapacity(buffer.items.len - 2);
        }

        return buffer.toOwnedSlice(self.allocator);
    }

    fn appendDirective(self: *SecurityHeaders, buffer: *std.ArrayListUnmanaged(u8), name: []const u8, sources: []const []const u8) !void {
        if (sources.len == 0) return;

        try buffer.appendSlice(self.allocator, name);
        try buffer.append(self.allocator, ' ');

        for (sources, 0..) |src, i| {
            if (i > 0) try buffer.append(self.allocator, ' ');
            try buffer.appendSlice(self.allocator, src);
        }

        try buffer.appendSlice(self.allocator, "; ");
    }

    fn buildHsts(self: *SecurityHeaders) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        {
            const tmp = try std.fmt.allocPrint(self.allocator, "max-age={d}", .{self.config.hsts.max_age});
            defer self.allocator.free(tmp);
            try buffer.appendSlice(self.allocator, tmp);
        }

        if (self.config.hsts.include_subdomains) {
            try buffer.appendSlice(self.allocator, "; includeSubDomains");
        }

        if (self.config.hsts.preload) {
            try buffer.appendSlice(self.allocator, "; preload");
        }

        return buffer.toOwnedSlice(self.allocator);
    }

    fn buildPermissionsPolicy(self: *SecurityHeaders) ![]const u8 {
        var buffer = std.ArrayListUnmanaged(u8).empty;
        errdefer buffer.deinit(self.allocator);

        const pp = self.config.permissions_policy;

        {
            const tmp = try std.fmt.allocPrint(self.allocator,
                \\accelerometer={s}, ambient-light-sensor={s}, autoplay={s}, battery={s}, camera={s}, display-capture={s}, document-domain={s}, encrypted-media={s}, fullscreen={s}, geolocation={s}, gyroscope={s}, magnetometer={s}, microphone={s}, midi={s}, payment={s}, picture-in-picture={s}, usb={s}, web-share={s}, xr-spatial-tracking={s}
            , .{
                pp.accelerometer,
                pp.ambient_light_sensor,
                pp.autoplay,
                pp.battery,
                pp.camera,
                pp.display_capture,
                pp.document_domain,
                pp.encrypted_media,
                pp.fullscreen,
                pp.geolocation,
                pp.gyroscope,
                pp.magnetometer,
                pp.microphone,
                pp.midi,
                pp.payment,
                pp.picture_in_picture,
                pp.usb,
                pp.web_share,
                pp.xr_spatial_tracking,
            });
            defer self.allocator.free(tmp);
            try buffer.appendSlice(self.allocator, tmp);
        }

        return buffer.toOwnedSlice(self.allocator);
    }
};

/// Cache response type
pub const CacheResponseType = enum {
    sensitive,
    public_static,
    private,
    no_cache,
};

/// Preset configurations
pub const Presets = struct {
    /// Strict security (recommended for sensitive applications)
    pub const strict: SecurityHeadersConfig = .{
        .csp = .{
            .script_src = &.{"'self'"},
            .style_src = &.{"'self'"},
            .frame_ancestors = &.{"'none'"},
            .upgrade_insecure_requests = true,
            .block_all_mixed_content = true,
        },
        .frame_options = .deny,
        .hsts = .{
            .max_age = 31536000,
            .include_subdomains = true,
            .preload = true,
        },
        .coep = "require-corp",
        .coop = "same-origin",
        .corp = "same-origin",
    };

    /// Balanced security (good defaults)
    pub const balanced: SecurityHeadersConfig = .{
        .csp = .{
            .script_src = &.{ "'self'", "'unsafe-inline'" },
            .style_src = &.{ "'self'", "'unsafe-inline'" },
        },
        .frame_options = .sameorigin,
        .hsts = .{
            .max_age = 15768000, // 6 months
            .include_subdomains = false,
            .preload = false,
        },
    };

    /// Minimal security (for development)
    pub const minimal: SecurityHeadersConfig = .{
        .enabled = true,
        .csp = .{ .enabled = false },
        .hsts = .{ .enabled = false },
        .coep = null,
        .coop = null,
        .corp = null,
    };

    /// API-focused security (for REST APIs)
    pub const api: SecurityHeadersConfig = .{
        .csp = .{ .enabled = false }, // Not relevant for APIs
        .frame_options = .deny,
        .nosniff = true,
        .cache_control = .{
            .sensitive = "no-store",
        },
    };
};

// Tests

test "build security headers" {
    const allocator = std.testing.allocator;
    var headers = SecurityHeaders.init(allocator, .{});
    defer headers.deinit();

    const result = try headers.getHeaders();
    defer {
        // Free allocated header values (CSP, HSTS, Permissions-Policy)
        for (result) |header| {
            if (std.mem.eql(u8, header.name, "Content-Security-Policy") or
                std.mem.eql(u8, header.name, "Strict-Transport-Security") or
                std.mem.eql(u8, header.name, "Permissions-Policy"))
            {
                allocator.free(header.value);
            }
        }
        allocator.free(result);
    }

    // Should have multiple headers
    try std.testing.expect(result.len > 0);

    // Check for expected headers
    var found_csp = false;
    var found_frame_options = false;
    var found_hsts = false;

    for (result) |header| {
        if (std.mem.eql(u8, header.name, "Content-Security-Policy")) found_csp = true;
        if (std.mem.eql(u8, header.name, "X-Frame-Options")) found_frame_options = true;
        if (std.mem.eql(u8, header.name, "Strict-Transport-Security")) found_hsts = true;
    }

    try std.testing.expect(found_csp);
    try std.testing.expect(found_frame_options);
    try std.testing.expect(found_hsts);
}

test "nonce generation" {
    const allocator = std.testing.allocator;
    var headers = SecurityHeaders.init(allocator, .{
        .csp = .{ .use_nonces = true },
    });
    defer headers.deinit();

    const nonce1 = try headers.generateNonce();
    // Save nonce1 before generating nonce2 (generateNonce frees the old nonce)
    const nonce1_copy = try allocator.dupe(u8, nonce1);
    defer allocator.free(nonce1_copy);

    const nonce2 = try headers.generateNonce();

    // Nonces should be different
    try std.testing.expect(!std.mem.eql(u8, nonce1_copy, nonce2));

    // Current nonce should be nonce2
    try std.testing.expectEqualStrings(nonce2, headers.current_nonce.?);
}

test "strict preset" {
    const allocator = std.testing.allocator;
    var headers = SecurityHeaders.init(allocator, Presets.strict);
    defer headers.deinit();

    const result = try headers.getHeaders();
    defer {
        for (result) |header| {
            if (std.mem.eql(u8, header.name, "Content-Security-Policy") or
                std.mem.eql(u8, header.name, "Strict-Transport-Security") or
                std.mem.eql(u8, header.name, "Permissions-Policy"))
            {
                allocator.free(header.value);
            }
        }
        allocator.free(result);
    }

    // Verify strict settings
    for (result) |header| {
        if (std.mem.eql(u8, header.name, "X-Frame-Options")) {
            try std.testing.expectEqualStrings("DENY", header.value);
        }
    }
}

test {
    std.testing.refAllDecls(@This());
}
