const std = @import("std");
const types = @import("stub_types.zig");

pub const HASH_LEN = 32;
pub const GENESIS_HASH: [HASH_LEN]u8 = std.mem.zeroes([HASH_LEN]u8);

pub const BlockHeader = struct {
    hash: [HASH_LEN]u8 = undefined,
    prev_hash: [HASH_LEN]u8 = undefined,
    timestamp_ms: i64 = 0,
    sequence: u64 = 0,
};

pub const MvccBlock = struct {
    header: BlockHeader = .{},
    data: types.ConversationBlock = undefined,
    next: ?*MvccBlock = null,
    version: u64 = 0,

    pub fn deinit(self: *MvccBlock, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

pub fn computeBlockHash(prev_hash: [HASH_LEN]u8, timestamp_ms: i64, sequence: u64, profile: []const u8, metadata: []const u8) [HASH_LEN]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(&prev_hash);
    var buf: [32]u8 = undefined;
    std.mem.writeInt(i64, buf[0..8], timestamp_ms, .little);
    std.mem.writeInt(u64, buf[8..16], sequence, .little);
    hasher.update(buf[0..16]);
    hasher.update(profile);
    hasher.update(metadata);
    var out: [HASH_LEN]u8 = undefined;
    hasher.final(&out);
    return out;
}

pub const BlockChain = struct {
    pub fn init(allocator: std.mem.Allocator) BlockChain {
        _ = allocator;
        return .{};
    }

    pub fn deinit(self: *BlockChain) void {
        _ = self;
    }

    pub fn append(self: *BlockChain, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![HASH_LEN]u8 {
        _ = self;
        _ = query_id;
        _ = response_id;
        _ = metadata;
        if (profile.len == 0) return error.InvalidProfile;
        return error.FeatureDisabled;
    }

    pub fn appendAt(self: *BlockChain, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8, timestamp_ms: i64) ![HASH_LEN]u8 {
        _ = timestamp_ms;
        return self.append(profile, query_id, response_id, metadata);
    }

    pub fn getBlock(self: *BlockChain, hash: [HASH_LEN]u8) ?*const MvccBlock {
        _ = self;
        _ = hash;
        return null;
    }

    pub fn getSnapshot(self: *BlockChain) Snapshot {
        return .{ .head = null, .length = 0, .chain = self };
    }

    pub fn releaseSnapshot(self: *BlockChain) void {
        _ = self;
    }

    pub fn verifyChain(self: *BlockChain) bool {
        _ = self;
        return true;
    }

    pub fn len(self: *const BlockChain) usize {
        _ = self;
        return 0;
    }

    pub fn getTailHash(self: *BlockChain) ?[HASH_LEN]u8 {
        _ = self;
        return null;
    }

    pub fn iterator(self: *const BlockChain) Iterator {
        _ = self;
        return .{ .current = null };
    }

    pub fn releaseIterator(self: *const BlockChain) void {
        _ = self;
    }

    pub const Iterator = struct {
        current: ?*const MvccBlock,

        pub fn next(self: *Iterator) ?*const MvccBlock {
            _ = self;
            return null;
        }
    };

    pub const Snapshot = struct {
        head: ?*const MvccBlock,
        length: usize,
        chain: *const BlockChain,

        pub fn getBlock(self: *const Snapshot, hash: [HASH_LEN]u8) ?*const MvccBlock {
            _ = self;
            _ = hash;
            return null;
        }

        pub fn iterator(self: *const Snapshot) Iterator {
            _ = self;
            return .{ .current = null };
        }

        pub fn len(self: *const Snapshot) usize {
            return self.length;
        }
    };
};

test {
    std.testing.refAllDecls(@This());
}
