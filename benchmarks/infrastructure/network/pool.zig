//! Connection pool simulation benchmarks.

const std = @import("std");
const sync = @import("abi").services.shared.sync;

pub const ConnectionPool = struct {
    const Connection = struct {
        id: u32,
        in_use: bool,
        created_at: i64,
    };

    connections: std.ArrayListUnmanaged(Connection),
    mutex: sync.Mutex,
    max_size: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, max_size: usize) ConnectionPool {
        return .{ .connections = .{}, .mutex = .{}, .max_size = max_size, .allocator = allocator };
    }

    pub fn deinit(self: *ConnectionPool) void {
        self.connections.deinit(self.allocator);
    }

    pub fn acquire(self: *ConnectionPool) !?*Connection {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.connections.items) |*conn| {
            if (!conn.in_use) {
                conn.in_use = true;
                return conn;
            }
        }

        if (self.connections.items.len < self.max_size) {
            try self.connections.append(self.allocator, .{
                .id = @intCast(self.connections.items.len),
                .in_use = true,
                .created_at = @as(i64, @intCast(self.connections.items.len)),
            });
            return &self.connections.items[self.connections.items.len - 1];
        }
        return null;
    }

    pub fn release(_: *ConnectionPool, conn: *Connection) void {
        conn.in_use = false;
    }
};

pub fn benchConnectionPool(allocator: std.mem.Allocator, pool_size: usize, operations: usize) !u64 {
    var pool = ConnectionPool.init(allocator, pool_size);
    defer pool.deinit();

    var acquired: u64 = 0;
    for (0..operations) |_| {
        if (try pool.acquire()) |conn| {
            acquired += 1;
            pool.release(conn);
        }
    }
    return acquired;
}
