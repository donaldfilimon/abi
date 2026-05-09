//! Profile registry for capability-based lookup.

const std = @import("std");
const types = @import("types.zig");
const presets_mod = @import("presets.zig");
const Profile = types.Profile;
const Capability = types.Capability;
const Domain = types.Domain;

pub const ProfileRegistry = struct {
    allocator: std.mem.Allocator,
    profiles: std.StringHashMapUnmanaged(Profile),

    pub fn init(allocator: std.mem.Allocator) ProfileRegistry {
        return .{
            .allocator = allocator,
            .profiles = .{},
        };
    }

    pub fn deinit(self: *ProfileRegistry) void {
        self.profiles.deinit(self.allocator);
    }

    pub fn register(self: *ProfileRegistry, profile: Profile) !void {
        try self.profiles.put(self.allocator, profile.id, profile);
    }

    pub fn loadPresets(self: *ProfileRegistry) !void {
        for (presets_mod.presets.all) |p| {
            try self.register(p);
        }
    }

    pub fn get(self: *const ProfileRegistry, id: []const u8) ?Profile {
        return self.profiles.get(id);
    }

    pub fn findBestMatch(self: *const ProfileRegistry, required: []const Capability) ?Profile {
        var best: ?Profile = null;
        var best_score: u32 = 0;

        var iter = self.profiles.iterator();
        while (iter.next()) |entry| {
            const score = entry.value_ptr.matchScore(required);
            if (score > best_score) {
                best_score = score;
                best = entry.value_ptr.*;
            }
        }

        return best;
    }

    pub fn findByCapability(
        self: *const ProfileRegistry,
        allocator: std.mem.Allocator,
        cap: Capability,
    ) ![]const Profile {
        var results: std.ArrayListUnmanaged(Profile) = .empty;
        errdefer results.deinit(allocator);

        var iter = self.profiles.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.hasCapability(cap)) {
                try results.append(allocator, entry.value_ptr.*);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    pub fn findByDomain(
        self: *const ProfileRegistry,
        allocator: std.mem.Allocator,
        domain: Domain,
    ) ![]const Profile {
        var results: std.ArrayListUnmanaged(Profile) = .empty;
        errdefer results.deinit(allocator);

        var iter = self.profiles.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.domain == domain) {
                try results.append(allocator, entry.value_ptr.*);
            }
        }

        return results.toOwnedSlice(allocator);
    }

    pub fn count(self: *const ProfileRegistry) usize {
        return self.profiles.count();
    }
};

test {
    std.testing.refAllDecls(@This());
}
