/// Mobile module configuration.
pub const MobileConfig = struct {
    platform: Platform = .auto,
    enable_sensors: bool = false,
    enable_notifications: bool = false,

    pub const Platform = enum { auto, ios, android };

    pub fn defaults() MobileConfig {
        return .{};
    }
};
