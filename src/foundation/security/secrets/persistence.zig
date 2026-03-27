const std = @import("std");
const shared = @import("shared.zig");

const SecretValue = shared.SecretValue;
const SecretsError = shared.SecretsError;

pub fn Persistence(comptime Manager: type) type {
    return struct {
        pub fn packEncryptedValue(self: *Manager, encrypted: SecretValue) SecretsError![]u8 {
            const packed_len = 12 + 16 + encrypted.encrypted_value.len;
            const packed_data = try self.allocator.alloc(u8, packed_len);
            @memcpy(packed_data[0..12], &encrypted.nonce);
            @memcpy(packed_data[12..28], &encrypted.tag);
            @memcpy(packed_data[28..], encrypted.encrypted_value);
            return packed_data;
        }

        pub fn encodeBase64(self: *Manager, data: []const u8) SecretsError![]u8 {
            const encoder = std.base64.standard.Encoder;
            const b64_size = encoder.calcSize(data.len);
            const b64_data = try self.allocator.alloc(u8, b64_size);
            _ = encoder.encode(b64_data, data);
            return b64_data;
        }

        pub fn readSecretsFile(self: *Manager, secrets_path: []const u8) SecretsError![]u8 {
            const io = self.io_backend.io();
            return std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch return error.SecretNotFound;
        }

        pub fn readSecretsFileOrEmpty(self: *Manager, secrets_path: []const u8) SecretsError![]u8 {
            const io = self.io_backend.io();
            return std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch try self.allocator.dupe(u8, "{}");
        }

        pub fn buildJsonEntry(self: *Manager, name: []const u8, b64_data: []const u8) SecretsError![]u8 {
            return std.fmt.allocPrint(self.allocator, "\"{s}\":\"{s}\"", .{ name, b64_data });
        }

        pub fn buildUpdatedContent(
            self: *Manager,
            existing: []const u8,
            name: []const u8,
            new_entry: []const u8,
        ) SecretsError![]u8 {
            const trimmed_all = std.mem.trim(u8, existing, " \n\r\t");
            if (trimmed_all.len == 0 or std.mem.eql(u8, trimmed_all, "{}")) {
                return std.fmt.allocPrint(self.allocator, "{{{s}}}", .{new_entry});
            }

            const search_key = try std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{name});
            defer self.allocator.free(search_key);

            if (std.mem.indexOf(u8, existing, search_key)) |key_idx| {
                const value_start = key_idx + search_key.len;
                var value_end = value_start;
                while (value_end < existing.len and existing[value_end] != '"') : (value_end += 1) {}
                if (value_end >= existing.len) return error.InvalidSecretFormat;

                const prefix = existing[0..key_idx];
                const suffix = existing[value_end + 1 ..];
                return std.fmt.allocPrint(self.allocator, "{s}{s}{s}", .{ prefix, new_entry, suffix });
            }

            const trimmed = std.mem.trimEnd(u8, existing, " \n\r\t}");
            return std.fmt.allocPrint(self.allocator, "{s},{s}}}", .{ trimmed, new_entry });
        }

        pub fn writeSecretsFile(self: *Manager, secrets_path: []const u8, content: []const u8) SecretsError!void {
            const io = self.io_backend.io();
            var file = std.Io.Dir.cwd().createFile(io, secrets_path, .{ .truncate = true }) catch return error.FileWriteFailed;
            defer file.close(io);
            var write_buf: [4096]u8 = undefined;
            var writer = file.writer(io, &write_buf);
            writer.interface.writeAll(content) catch return error.FileWriteFailed;
            writer.flush() catch return error.FileWriteFailed;
        }

        pub fn parseVaultResponse(allocator: std.mem.Allocator, body: []const u8, name: []const u8) SecretsError![]u8 {
            const parsed = std.json.parseFromSlice(
                std.json.Value,
                allocator,
                body,
                .{ .ignore_unknown_fields = true },
            ) catch return error.InvalidSecretFormat;
            defer parsed.deinit();

            const root = parsed.value;

            const outer_data = switch (root) {
                .object => |obj| obj.get("data") orelse return error.SecretNotFound,
                else => return error.InvalidSecretFormat,
            };
            const inner_data = switch (outer_data) {
                .object => |obj| obj.get("data") orelse return error.SecretNotFound,
                else => return error.InvalidSecretFormat,
            };
            const data_obj = switch (inner_data) {
                .object => |obj| obj,
                else => return error.InvalidSecretFormat,
            };

            if (data_obj.get("value")) |val| {
                return extractStringValue(allocator, val);
            }

            if (data_obj.get(name)) |val| {
                return extractStringValue(allocator, val);
            }

            var it = data_obj.iterator();
            if (it.next()) |entry| {
                return extractStringValue(allocator, entry.value_ptr.*);
            }

            return error.SecretNotFound;
        }

        fn extractStringValue(allocator: std.mem.Allocator, val: std.json.Value) SecretsError![]u8 {
            return switch (val) {
                .string => |s| allocator.dupe(u8, s) catch return error.OutOfMemory,
                .integer => |i| std.fmt.allocPrint(allocator, "{d}", .{i}) catch return error.OutOfMemory,
                .float => |f| std.fmt.allocPrint(allocator, "{d}", .{f}) catch return error.OutOfMemory,
                .bool => |b| allocator.dupe(u8, if (b) "true" else "false") catch return error.OutOfMemory,
                else => error.InvalidSecretFormat,
            };
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
