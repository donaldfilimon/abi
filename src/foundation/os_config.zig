// OS Control Whitelist & Limits
// This file defines the permissible operations for the ABI OS Controller agent.

pub const Command = enum {
    // Process Management
    list_processes,
    get_process_info,
    get_pid,
    get_parent_pid,

    // Resource Monitoring
    get_cpu_usage,
    get_memory_usage,

    // System Config
    get_system_info,

    // File Operations
    read_file,
    write_file,
    stat_file,
    list_directory,

    // Environment
    get_env_var,
    get_env_vars,
    get_cwd,

    // Platform Detection
    get_platform,
    get_arch,
    get_hostname,
};

pub const Limits = struct {
    max_memory_mb: u32 = 1024,
    max_cpu_percent: u8 = 50,
};

pub const Platform = enum {
    macos,
    linux,
    windows,
    freebsd,
    unknown,
};

pub const Arch = enum {
    x86_64,
    aarch64,
    arm,
    riscv64,
    wasm32,
    unknown,
};

pub const FileInfo = struct {
    size: u64,
    is_dir: bool,
    is_file: bool,
    path: []const u8,
};

pub const ProcessInfo = struct {
    pid: u32,
    ppid: u32,
    name: []const u8,
};

pub const SystemInfo = struct {
    platform: Platform,
    arch: Arch,
    hostname: []const u8,
    cpu_count: u32,
    total_memory_mb: u64,
};
