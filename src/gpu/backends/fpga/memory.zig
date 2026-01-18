//! FPGA Memory Management
//!
//! Handles allocation and transfer of data to/from FPGA device memory,
//! supporting DDR, HBM, and on-chip memory regions.

const std = @import("std");
const build_options = @import("build_options");
const loader = @import("loader.zig");

pub const MemoryError = error{
    OutOfMemory,
    InvalidPointer,
    TransferFailed,
    DeviceNotReady,
    InvalidSize,
    AlignmentError,
    BankNotAvailable,
};

/// Memory bank type on the FPGA
pub const MemoryBank = enum {
    ddr0, // DDR bank 0
    ddr1, // DDR bank 1
    ddr2, // DDR bank 2
    ddr3, // DDR bank 3
    hbm, // High Bandwidth Memory (if available)
    plram, // On-chip PL RAM (fast, limited size)
    auto, // Automatic bank selection

    pub fn name(self: MemoryBank) []const u8 {
        return switch (self) {
            .ddr0 => "DDR[0]",
            .ddr1 => "DDR[1]",
            .ddr2 => "DDR[2]",
            .ddr3 => "DDR[3]",
            .hbm => "HBM",
            .plram => "PLRAM",
            .auto => "AUTO",
        };
    }
};

/// Memory allocation flags
pub const MemoryFlags = packed struct {
    /// Allocate in device memory (DDR/HBM)
    device: bool = true,
    /// Host-visible memory (for zero-copy access)
    host_visible: bool = false,
    /// Memory is coherent between host and device
    host_coherent: bool = false,
    /// Enable caching on device
    cached: bool = true,
    /// Read-only on device
    read_only: bool = false,
    /// Write-only on device
    write_only: bool = false,
    _padding: u2 = 0,
};

/// Buffer handle for FPGA memory
pub const FpgaBuffer = struct {
    allocator: std.mem.Allocator,
    device_index: u32,
    device_ptr: ?*anyopaque = null,
    host_ptr: ?[*]u8 = null,
    size: usize,
    bank: MemoryBank,
    flags: MemoryFlags,
    is_mapped: bool = false,

    /// Get the device pointer (for kernel arguments)
    pub fn getDevicePtr(self: *const FpgaBuffer) ?*anyopaque {
        return self.device_ptr;
    }

    /// Get the host pointer (if mapped or host-visible)
    pub fn getHostPtr(self: *const FpgaBuffer) ?[*]u8 {
        return self.host_ptr;
    }

    /// Check if buffer has valid device allocation
    pub fn isValid(self: *const FpgaBuffer) bool {
        return self.device_ptr != null;
    }
};

/// FPGA memory manager for a device
pub const FpgaMemory = struct {
    allocator: std.mem.Allocator,
    device_index: u32,
    platform: loader.Platform,

    // Memory tracking
    total_allocated: usize = 0,
    allocation_count: usize = 0,

    // Bank availability (bytes)
    ddr_available: [4]u64 = .{ 0, 0, 0, 0 },
    hbm_available: u64 = 0,
    plram_available: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, device_index: u32) MemoryError!FpgaMemory {
        const device_info = loader.getDeviceInfo(device_index) catch {
            return error.DeviceNotReady;
        };

        var mem = FpgaMemory{
            .allocator = allocator,
            .device_index = device_index,
            .platform = device_info.platform,
        };

        // Initialize available memory from device info
        const ddr_per_bank = device_info.ddr_size_bytes / 4;
        mem.ddr_available = .{ ddr_per_bank, ddr_per_bank, ddr_per_bank, ddr_per_bank };
        mem.hbm_available = device_info.hbm_size_bytes;
        mem.plram_available = 4 * 1024 * 1024; // Typical 4 MB PLRAM

        return mem;
    }

    pub fn deinit(self: *FpgaMemory) void {
        if (self.allocation_count > 0) {
            std.log.warn("FPGA memory: {d} allocations not freed", .{self.allocation_count});
        }
    }

    /// Allocate a buffer on the FPGA
    pub fn allocate(
        self: *FpgaMemory,
        size: usize,
        bank: MemoryBank,
        flags: MemoryFlags,
    ) MemoryError!FpgaBuffer {
        if (size == 0) return error.InvalidSize;

        // Align to 64-byte boundary (typical FPGA requirement)
        const aligned_size = (size + 63) & ~@as(usize, 63);

        // Select bank if auto
        const actual_bank = if (bank == .auto) self.selectBestBank(aligned_size) else bank;

        // Check availability
        if (!self.checkBankAvailability(actual_bank, aligned_size)) {
            return error.OutOfMemory;
        }

        // In a real implementation:
        // - Xilinx: xrt::bo constructor
        // - Intel: clCreateBuffer

        // For now, simulate with host memory for testing
        const host_ptr = self.allocator.alloc(u8, aligned_size) catch {
            return error.OutOfMemory;
        };

        // Update tracking
        self.updateBankUsage(actual_bank, aligned_size, true);
        self.total_allocated += aligned_size;
        self.allocation_count += 1;

        return FpgaBuffer{
            .allocator = self.allocator,
            .device_index = self.device_index,
            .device_ptr = @ptrCast(host_ptr.ptr),
            .host_ptr = host_ptr.ptr,
            .size = aligned_size,
            .bank = actual_bank,
            .flags = flags,
        };
    }

    /// Free a buffer
    pub fn free(self: *FpgaMemory, buffer: *FpgaBuffer) void {
        if (buffer.host_ptr) |ptr| {
            // In a real implementation, this would free device memory
            const slice = ptr[0..buffer.size];
            self.allocator.free(slice);
        }

        self.updateBankUsage(buffer.bank, buffer.size, false);
        self.total_allocated -= buffer.size;
        self.allocation_count -= 1;

        buffer.device_ptr = null;
        buffer.host_ptr = null;
    }

    /// Copy data from host to device buffer
    pub fn copyToDevice(self: *FpgaMemory, buffer: *FpgaBuffer, data: []const u8) MemoryError!void {
        _ = self;
        if (buffer.host_ptr == null) return error.InvalidPointer;
        if (data.len > buffer.size) return error.InvalidSize;

        // In a real implementation:
        // - Xilinx: xrt::bo::sync(XCL_BO_SYNC_BO_TO_DEVICE)
        // - Intel: clEnqueueWriteBuffer

        // For simulation, just copy to host memory
        @memcpy(buffer.host_ptr.?[0..data.len], data);
    }

    /// Copy data from device buffer to host
    pub fn copyFromDevice(self: *FpgaMemory, buffer: *FpgaBuffer, dest: []u8) MemoryError!void {
        _ = self;
        if (buffer.host_ptr == null) return error.InvalidPointer;
        if (dest.len > buffer.size) return error.InvalidSize;

        // In a real implementation:
        // - Xilinx: xrt::bo::sync(XCL_BO_SYNC_BO_FROM_DEVICE)
        // - Intel: clEnqueueReadBuffer

        // For simulation, just copy from host memory
        @memcpy(dest, buffer.host_ptr.?[0..dest.len]);
    }

    /// Map buffer for direct host access (if supported)
    pub fn map(self: *FpgaMemory, buffer: *FpgaBuffer) MemoryError![]u8 {
        _ = self;
        if (!buffer.flags.host_visible) return error.InvalidPointer;
        if (buffer.host_ptr == null) return error.InvalidPointer;

        buffer.is_mapped = true;
        return buffer.host_ptr.?[0..buffer.size];
    }

    /// Unmap a mapped buffer
    pub fn unmap(self: *FpgaMemory, buffer: *FpgaBuffer) void {
        _ = self;
        buffer.is_mapped = false;
    }

    // Internal helpers

    fn selectBestBank(self: *const FpgaMemory, size: usize) MemoryBank {
        // Prefer PLRAM for small allocations
        if (size <= 256 * 1024 and self.plram_available >= size) {
            return .plram;
        }

        // Prefer HBM for bandwidth-intensive operations
        if (self.hbm_available >= size) {
            return .hbm;
        }

        // Find DDR bank with most space
        var best_bank: MemoryBank = .ddr0;
        var best_available: u64 = self.ddr_available[0];

        for (1..4) |i| {
            if (self.ddr_available[i] > best_available) {
                best_available = self.ddr_available[i];
                best_bank = @enumFromInt(@as(u3, @intCast(i)));
            }
        }

        return best_bank;
    }

    fn checkBankAvailability(self: *const FpgaMemory, bank: MemoryBank, size: usize) bool {
        return switch (bank) {
            .ddr0 => self.ddr_available[0] >= size,
            .ddr1 => self.ddr_available[1] >= size,
            .ddr2 => self.ddr_available[2] >= size,
            .ddr3 => self.ddr_available[3] >= size,
            .hbm => self.hbm_available >= size,
            .plram => self.plram_available >= size,
            .auto => true, // Will be resolved to actual bank
        };
    }

    fn updateBankUsage(self: *FpgaMemory, bank: MemoryBank, size: usize, allocating: bool) void {
        const delta: i64 = if (allocating) -@as(i64, @intCast(size)) else @as(i64, @intCast(size));

        switch (bank) {
            .ddr0 => self.ddr_available[0] = @intCast(@as(i64, @intCast(self.ddr_available[0])) + delta),
            .ddr1 => self.ddr_available[1] = @intCast(@as(i64, @intCast(self.ddr_available[1])) + delta),
            .ddr2 => self.ddr_available[2] = @intCast(@as(i64, @intCast(self.ddr_available[2])) + delta),
            .ddr3 => self.ddr_available[3] = @intCast(@as(i64, @intCast(self.ddr_available[3])) + delta),
            .hbm => self.hbm_available = @intCast(@as(i64, @intCast(self.hbm_available)) + delta),
            .plram => self.plram_available = @intCast(@as(i64, @intCast(self.plram_available)) + delta),
            .auto => {},
        }
    }

    /// Get memory statistics
    pub fn getStats(self: *const FpgaMemory) MemoryStats {
        var total_ddr: u64 = 0;
        for (self.ddr_available) |avail| {
            total_ddr += avail;
        }

        return MemoryStats{
            .total_allocated = self.total_allocated,
            .allocation_count = self.allocation_count,
            .ddr_available = total_ddr,
            .hbm_available = self.hbm_available,
            .plram_available = self.plram_available,
        };
    }
};

pub const MemoryStats = struct {
    total_allocated: usize,
    allocation_count: usize,
    ddr_available: u64,
    hbm_available: u64,
    plram_available: u64,
};

test "fpga memory allocation" {
    if (comptime !build_options.gpu_fpga) return;

    const allocator = std.testing.allocator;

    // This test requires an FPGA device to be available
    if (loader.detectFpgaDevices() == 0) return;

    var mem = try FpgaMemory.init(allocator, 0);
    defer mem.deinit();

    var buffer = try mem.allocate(1024, .auto, .{});
    defer mem.free(&buffer);

    try std.testing.expect(buffer.isValid());
}
