pub const StorageBackend = enum { memory, local, s3, gcs };

pub const StorageConfig = struct {
    backend: StorageBackend = .memory,
    base_path: []const u8 = "./storage",
    max_object_size_mb: u32 = 1024,

    pub fn defaults() StorageConfig {
        return .{};
    }
};
