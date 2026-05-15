// OS Control Whitelist & Limits
// This file defines the permissible operations for the ABI OS Controller agent.

pub const Command = enum {
    // Process Management
    list_processes,
    get_process_info,

    // Resource Monitoring
    get_cpu_usage,
    get_memory_usage,

    // System Config
    get_system_info,
};

pub const Limits = struct {
    max_memory_mb: u32 = 1024,
    max_cpu_percent: u8 = 50,
};
