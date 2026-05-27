const std = @import("std");
const sync = @import("../../foundation/sync.zig");
const foundation_time = @import("../../foundation/time.zig");
const wdbx_mod = @import("mod.zig");

pub const HASH_LEN = 32;
pub const GENESIS_HASH: [HASH_LEN]u8 = std.mem.zeroes([HASH_LEN]u8);

pub const BlockHeader = struct {
    hash: [HASH_LEN]u8,
    prev_hash: [HASH_LEN]u8,
    timestamp_ms: i64,
    sequence: u64,
};

pub const MvccBlock = struct {
    header: BlockHeader,
    data: wdbx_mod.ConversationBlock,
    next: ?*MvccBlock = null,
    version: u64 = 0,
    lock: sync.SpinLock = .{},

    pub fn deinit(self: *MvccBlock, allocator: std.mem.Allocator) void {
        allocator.free(self.data.profile);
        allocator.free(self.data.metadata);
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
    allocator: std.mem.Allocator,
    head: ?*MvccBlock = null,
    tail: ?*MvccBlock = null,
    length: usize = 0,
    next_sequence: u64 = 0,
    write_lock: sync.SpinLock = .{},
    read_lock: sync.RwLock = .{},

    pub fn init(allocator: std.mem.Allocator) BlockChain {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *BlockChain) void {
        var current = self.head;
        while (current) |node| {
            const next = node.next;
            node.deinit(self.allocator);
            self.allocator.destroy(node);
            current = next;
        }
        self.head = null;
        self.tail = null;
        self.length = 0;
    }

    pub fn append(self: *BlockChain, profile: []const u8, query_id: u32, response_id: u32, metadata: []const u8) ![HASH_LEN]u8 {
        if (profile.len == 0) return error.InvalidProfile;

        self.write_lock.lock();
        defer self.write_lock.unlock();

        const prev_hash = if (self.tail) |t| t.header.hash else GENESIS_HASH;
        const sequence = self.next_sequence;
        const timestamp_ms = foundation_time.unixMs();

        const hash = computeBlockHash(prev_hash, timestamp_ms, sequence, profile, metadata);

        const owned_profile = try self.allocator.dupe(u8, profile);
        errdefer self.allocator.free(owned_profile);

        const owned_metadata = try self.allocator.dupe(u8, metadata);
        errdefer self.allocator.free(owned_metadata);

        const node = try self.allocator.create(MvccBlock);
        errdefer self.allocator.destroy(node);

        node.* = .{
            .header = .{
                .hash = hash,
                .prev_hash = prev_hash,
                .timestamp_ms = timestamp_ms,
                .sequence = sequence,
            },
            .data = .{
                .id = hash,
                .prev_id = prev_hash,
                .timestamp_ms = timestamp_ms,
                .profile = owned_profile,
                .query_id = query_id,
                .response_id = response_id,
                .metadata = owned_metadata,
            },
            .next = null,
            .version = sequence,
        };

        if (self.tail) |t| {
            t.lock.lock();
            defer t.lock.unlock();
            t.next = node;
        } else {
            self.head = node;
        }
        self.tail = node;
        self.length += 1;
        self.next_sequence += 1;

        return hash;
    }

    pub fn getBlock(self: *BlockChain, hash: [HASH_LEN]u8) ?*const MvccBlock {
        self.read_lock.lockRead();
        defer self.read_lock.unlockRead();

        var current = self.head;
        while (current) |node| {
            if (std.mem.eql(u8, &node.header.hash, &hash)) {
                return node;
            }
            current = node.next;
        }
        return null;
    }

    pub fn getSnapshot(self: *BlockChain) Snapshot {
        self.read_lock.lockRead();
        return .{
            .head = self.head,
            .length = self.length,
            .chain = self,
        };
    }

    pub fn releaseSnapshot(self: *BlockChain) void {
        self.read_lock.unlockRead();
    }

    pub fn verifyChain(self: *BlockChain) bool {
        self.read_lock.lockRead();
        defer self.read_lock.unlockRead();

        var current = self.head;
        var expected_prev = GENESIS_HASH;
        while (current) |node| {
            if (!std.mem.eql(u8, &node.header.prev_hash, &expected_prev)) {
                return false;
            }
            const recomputed = computeBlockHash(
                node.header.prev_hash,
                node.header.timestamp_ms,
                node.header.sequence,
                node.data.profile,
                node.data.metadata,
            );
            if (!std.mem.eql(u8, &node.header.hash, &recomputed)) {
                return false;
            }
            expected_prev = node.header.hash;
            current = node.next;
        }
        return true;
    }

    pub fn len(self: *const BlockChain) usize {
        const self_mut = @constCast(self);
        self_mut.read_lock.lockRead();
        defer self_mut.read_lock.unlockRead();
        return self.length;
    }

    pub fn getTailHash(self: *BlockChain) ?[HASH_LEN]u8 {
        self.read_lock.lockRead();
        defer self.read_lock.unlockRead();
        return if (self.tail) |t| t.header.hash else null;
    }

    pub fn iterator(self: *const BlockChain) Iterator {
        const self_mut = @constCast(self);
        self_mut.read_lock.lockRead();
        return .{ .current = self.head };
    }

    pub fn releaseIterator(self: *const BlockChain) void {
        const self_mut = @constCast(self);
        self_mut.read_lock.unlockRead();
    }

    pub const Iterator = struct {
        current: ?*const MvccBlock,

        pub fn next(self: *Iterator) ?*const MvccBlock {
            const node = self.current orelse return null;
            self.current = node.next;
            return node;
        }
    };

    pub const Snapshot = struct {
        head: ?*const MvccBlock,
        length: usize,
        chain: *const BlockChain,

        pub fn getBlock(self: *const Snapshot, hash: [HASH_LEN]u8) ?*const MvccBlock {
            var current = self.head;
            while (current) |node| {
                if (std.mem.eql(u8, &node.header.hash, &hash)) {
                    return node;
                }
                current = node.next;
            }
            return null;
        }

        pub fn iterator(self: *const Snapshot) Iterator {
            return .{ .current = self.head };
        }

        pub fn len(self: *const Snapshot) usize {
            return self.length;
        }
    };
};

test "BlockChain append and verify" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    const h1 = try chain.append("abbey", 1, 2, "first block");
    const h2 = try chain.append("aviva", 3, 4, "second block");

    try std.testing.expect(chain.len() == 2);
    try std.testing.expect(chain.verifyChain());

    const block1 = chain.getBlock(h1);
    try std.testing.expect(block1 != null);
    try std.testing.expectEqualStrings("abbey", block1.?.data.profile);

    const block2 = chain.getBlock(h2);
    try std.testing.expect(block2 != null);
    try std.testing.expectEqualStrings("aviva", block2.?.data.profile);
}

test "BlockChain genesis block" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    try std.testing.expect(chain.len() == 0);
    try std.testing.expect(chain.head == null);
    try std.testing.expect(chain.tail == null);

    _ = try chain.append("abi", 0, 0, "genesis");
    try std.testing.expect(chain.len() == 1);
    try std.testing.expect(chain.head != null);
    try std.testing.expect(chain.tail != null);
    try std.testing.expect(std.mem.eql(u8, &chain.head.?.header.prev_hash, &GENESIS_HASH));
}

test "BlockChain iterator" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    _ = try chain.append("a", 1, 1, "meta1");
    _ = try chain.append("b", 2, 2, "meta2");
    _ = try chain.append("c", 3, 3, "meta3");

    var it = chain.iterator();
    defer chain.releaseIterator();
    var count: usize = 0;
    while (it.next()) |node| {
        _ = node;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "BlockChain snapshot isolation" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    _ = try chain.append("abbey", 1, 2, "snap test");
    _ = try chain.append("aviva", 3, 4, "snap test 2");

    const snapshot = chain.getSnapshot();
    defer chain.releaseSnapshot();

    try std.testing.expectEqual(@as(usize, 2), snapshot.len());

    var it = snapshot.iterator();
    var count: usize = 0;
    while (it.next()) |node| {
        _ = node;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), count);
}

test "BlockChain invalid profile" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    try std.testing.expectError(
        error.InvalidProfile,
        chain.append("", 0, 0, "no profile"),
    );
}

test "BlockChain hash linking" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    const h1 = try chain.append("test", 1, 1, "block1");
    const h2 = try chain.append("test", 2, 2, "block2");

    const block2 = chain.getBlock(h2);
    try std.testing.expect(block2 != null);
    try std.testing.expect(std.mem.eql(u8, &block2.?.header.prev_hash, &h1));

    const block1 = chain.getBlock(h1);
    try std.testing.expect(block1 != null);
    try std.testing.expect(std.mem.eql(u8, &block1.?.header.prev_hash, &GENESIS_HASH));
}

test "BlockChain get non-existent" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    _ = try chain.append("test", 1, 1, "only block");

    var fake_hash: [HASH_LEN]u8 = undefined;
    @memset(&fake_hash, 0xFF);

    try std.testing.expect(chain.getBlock(fake_hash) == null);
}

test "BlockChain tail hash" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    try std.testing.expect(chain.getTailHash() == null);

    const h1 = try chain.append("a", 1, 1, "first");
    const tail1 = chain.getTailHash();
    try std.testing.expect(tail1 != null);
    try std.testing.expect(std.mem.eql(u8, &tail1.?, &h1));

    const h2 = try chain.append("b", 2, 2, "second");
    const tail2 = chain.getTailHash();
    try std.testing.expect(tail2 != null);
    try std.testing.expect(std.mem.eql(u8, &tail2.?, &h2));
}

test "BlockChain append owns profile and metadata" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    var profile = [_]u8{ 'a', 'b', 'i' };
    var metadata = [_]u8{ 'm', 'e', 't', 'a' };
    const block_id = try chain.append(&profile, 1, 2, &metadata);

    @memcpy(&profile, "zzz");
    @memcpy(&metadata, "xxxx");

    const block = chain.getBlock(block_id) orelse return error.MissingBlock;
    try std.testing.expectEqualStrings("abi", block.data.profile);
    try std.testing.expectEqualStrings("meta", block.data.metadata);
    try std.testing.expect(chain.verifyChain());
}

test "BlockChain verifyChain detects tampered metadata" {
    var chain = BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    _ = try chain.append("abi", 1, 2, "meta");
    const node = chain.head orelse return error.MissingBlock;
    const metadata = @constCast(node.data.metadata);
    metadata[0] = 'X';

    try std.testing.expect(!chain.verifyChain());
}

test "computeBlockHash deterministic" {
    const h1 = computeBlockHash(GENESIS_HASH, 1000, 0, "profile", "meta");
    const h2 = computeBlockHash(GENESIS_HASH, 1000, 0, "profile", "meta");
    try std.testing.expect(std.mem.eql(u8, &h1, &h2));

    const h3 = computeBlockHash(GENESIS_HASH, 1001, 0, "profile", "meta");
    try std.testing.expect(!std.mem.eql(u8, &h1, &h3));
}

test {
    std.testing.refAllDecls(@This());
}
