//! Platform stub for minimal/disabled builds
//!
//! Provides fallback implementations when platform detection is not needed.

pub const Os = enum {
    unknown,

    pub fn current() Os {
        return .unknown;
    }
};

pub const Arch = enum {
    unknown,

    pub fn current() Arch {
        return .unknown;
    }

    pub fn hasSimd(self: Arch) bool {
        _ = self;
        return false;
    }
};

pub const PlatformInfo = struct {
    os: Os = .unknown,
    arch: Arch = .unknown,
    max_threads: u32 = 1,

    pub fn detect() PlatformInfo {
        return .{};
    }
};

pub const is_threaded_target = false;

pub fn getCpuCountSafe() usize {
    return 1;
}

pub fn getPlatformInfo() PlatformInfo {
    return .{};
}

pub fn supportsThreading() bool {
    return false;
}

pub fn getCpuCount() usize {
    return 1;
}

pub fn getDescription() []const u8 {
    return "Unknown Platform";
}

pub fn hasSimd() bool {
    return false;
}

pub fn isAppleSilicon() bool {
    return false;
}

pub fn isDesktop() bool {
    return false;
}

pub fn isMobile() bool {
    return false;
}

pub fn isWasm() bool {
    return false;
}
