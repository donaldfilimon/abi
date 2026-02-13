pub const MessagingConfig = struct {
    max_channels: u32 = 256,
    buffer_size: u32 = 4096,
    enable_persistence: bool = false,

    pub fn defaults() MessagingConfig {
        return .{};
    }
};
