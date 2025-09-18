//! Mobile Platform Support Module
//!
//! This module provides specialized support for mobile platforms including
//! iOS with Metal and Android with Vulkan backends.

pub const mobile_platform_support = @import("mobile_platform_support.zig");

// Re-export key types and functions
pub const MobilePlatformManager = mobile_platform_support.MobilePlatformManager;
pub const MobileCapabilities = mobile_platform_support.MobileCapabilities;
pub const PowerManagement = mobile_platform_support.PowerManagement;
pub const ThermalManagement = mobile_platform_support.ThermalManagement;
