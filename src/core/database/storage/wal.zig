//! Write-Ahead Log (WAL) types and writer.

pub const WalEntry = struct {
    /// Transaction sequence number
    seq: u64,
    /// Entry type
    entry_type: WalEntryType,
    /// Entry data
    data: []const u8,
    /// CRC32 of entry
    checksum: u32,
};

pub const WalEntryType = enum(u8) {
    insert = 0x01,
    update = 0x02,
    delete = 0x03,
    checkpoint = 0x10,
    commit = 0xFF,
};

pub const WalWriter = struct {
    file_path: []const u8,
    seq: u64,
    allocator: @import("std").mem.Allocator,

    pub fn init(allocator: @import("std").mem.Allocator, path: []const u8) WalWriter {
        return .{
            .file_path = path,
            .seq = 0,
            .allocator = allocator,
        };
    }

    pub fn append(self: *WalWriter, entry_type: WalEntryType, data: []const u8) !u64 {
        // Would write to WAL file
        self.seq += 1;
        _ = entry_type;
        _ = data;
        return self.seq;
    }

    pub fn checkpoint(self: *WalWriter) !void {
        _ = try self.append(.checkpoint, &.{});
    }
};
