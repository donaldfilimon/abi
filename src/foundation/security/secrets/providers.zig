const std = @import("std");
const time = @import("../../time.zig");
const shared = @import("shared.zig");

const SecretValue = shared.SecretValue;
const SecretsError = shared.SecretsError;

pub fn Providers(comptime Manager: type) type {
    return struct {
        const persistence = @import("persistence.zig").Persistence(Manager);

        pub fn loadFromEnv(self: *Manager, name: []const u8) SecretsError![]u8 {
            const env_name = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{
                self.config.env_prefix,
                name,
            });
            defer self.allocator.free(env_name);

            const env_name_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{env_name}, 0);
            defer self.allocator.free(env_name_z);
            const name_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{name}, 0);
            defer self.allocator.free(name_z);

            const value_ptr = std.c.getenv(env_name_z.ptr) orelse
                std.c.getenv(name_z.ptr) orelse
                return error.SecretNotFound;
            const value = std.mem.span(value_ptr);
            return self.allocator.dupe(u8, value);
        }

        pub fn loadFromFile(self: *Manager, name: []const u8) SecretsError![]u8 {
            const secrets_path = self.config.secrets_file orelse return error.SecretNotFound;

            const content = try persistence.readSecretsFile(self, secrets_path);
            defer self.allocator.free(content);

            const search_key = std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{name}) catch return error.OutOfMemory;
            defer self.allocator.free(search_key);

            const key_idx = std.mem.indexOf(u8, content, search_key) orelse return error.SecretNotFound;
            const value_start = key_idx + search_key.len;

            var value_end = value_start;
            while (value_end < content.len and content[value_end] != '"') : (value_end += 1) {}
            if (value_end <= value_start) return error.SecretNotFound;

            const encrypted_b64 = content[value_start..value_end];

            const decoder = std.base64.standard.Decoder;
            const decoded_size = decoder.calcSizeForSlice(encrypted_b64) catch return error.InvalidBase64;
            const encrypted_data = try self.allocator.alloc(u8, decoded_size);
            errdefer self.allocator.free(encrypted_data);

            decoder.decode(encrypted_data, encrypted_b64) catch return error.InvalidBase64;

            if (encrypted_data.len < 28) {
                self.allocator.free(encrypted_data);
                return error.InvalidSecretFormat;
            }

            const nonce = encrypted_data[0..12].*;
            const tag = encrypted_data[12..28].*;
            const ciphertext = encrypted_data[28..];

            const plaintext = try self.allocator.alloc(u8, ciphertext.len);
            errdefer self.allocator.free(plaintext);

            const aead = std.crypto.aead.aes_gcm.Aes256Gcm;
            aead.decrypt(plaintext, ciphertext, tag, &.{}, nonce, self.master_key) catch {
                self.allocator.free(encrypted_data);
                return error.DecryptionFailed;
            };

            self.allocator.free(encrypted_data);
            return plaintext;
        }

        pub fn loadFromMemory(self: *Manager, name: []const u8) SecretsError![]u8 {
            if (self.cache.get(name)) |cached| {
                var value = cached.value;
                return try value.decrypt(self.master_key);
            }
            return error.SecretNotFound;
        }

        pub fn loadFromVault(self: *Manager, name: []const u8) SecretsError![]u8 {
            return switch (self.config.vault_provider) {
                .hashicorp => loadFromHashicorpVault(self, name),
                .aws_secrets_manager => loadFromAws(self, name),
                .azure_key_vault => loadFromAzure(self, name),
            };
        }

        pub fn loadFromAws(_: *Manager, _: []const u8) SecretsError![]u8 {
            return error.NotImplemented;
        }

        pub fn loadFromAzure(_: *Manager, _: []const u8) SecretsError![]u8 {
            return error.NotImplemented;
        }

        pub fn loadFromHashicorpVault(self: *Manager, name: []const u8) SecretsError![]u8 {
            const vault_url = self.config.vault_url orelse return error.VaultUrlNotConfigured;
            const vault_token = self.config.vault_token orelse return error.VaultTokenNotConfigured;

            const cache_key = std.fmt.allocPrint(self.allocator, "vault:{s}", .{name}) catch return error.OutOfMemory;
            defer self.allocator.free(cache_key);

            if (self.cache.get(cache_key)) |cached| {
                var value = cached.value;
                return value.decrypt(self.master_key);
            }

            const url = std.fmt.allocPrint(self.allocator, "{s}/v1/secret/data/{s}", .{
                vault_url,
                name,
            }) catch return error.OutOfMemory;
            defer self.allocator.free(url);

            const async_http = @import("../../utils/http/async_http.zig");

            var client = async_http.AsyncHttpClient.init(self.allocator) catch |err| {
                std.log.err("Failed to initialize HTTP client for Vault: {}", .{err});
                return error.VaultConnectionFailed;
            };
            defer client.deinit();

            var request = async_http.HttpRequest.init(self.allocator, .get, url) catch |err| {
                std.log.err("Failed to create Vault HTTP request: {}", .{err});
                return error.VaultRequestFailed;
            };
            defer request.deinit();

            request.setHeader("X-Vault-Token", vault_token) catch return error.OutOfMemory;

            var response = client.fetchJson(&request) catch |err| {
                std.log.err("Vault HTTP request failed for secret '{s}': {}", .{ name, err });
                return error.VaultConnectionFailed;
            };
            defer response.deinit();

            if (!response.isSuccess()) {
                std.log.err("Vault returned HTTP {d} for secret '{s}'", .{ response.status_code, name });
                if (response.status_code == 404) return error.SecretNotFound;
                return error.VaultRequestFailed;
            }

            const secret_value = persistence.parseVaultResponse(self.allocator, response.body, name) catch |err| {
                std.log.err("Failed to parse Vault response for secret '{s}': {}", .{ name, err });
                return error.InvalidSecretFormat;
            };

            return secret_value;
        }

        pub fn saveToFile(self: *Manager, name: []const u8, encrypted: SecretValue) SecretsError!void {
            const secrets_path = self.config.secrets_file orelse return error.SecretsFileNotConfigured;

            const packed_data = try persistence.packEncryptedValue(self, encrypted);
            defer self.allocator.free(packed_data);

            const b64_data = try persistence.encodeBase64(self, packed_data);
            defer self.allocator.free(b64_data);

            const existing_content = try persistence.readSecretsFileOrEmpty(self, secrets_path);
            defer self.allocator.free(existing_content);

            const new_entry = try persistence.buildJsonEntry(self, name, b64_data);
            defer self.allocator.free(new_entry);

            const new_content = try persistence.buildUpdatedContent(self, existing_content, name, new_entry);
            defer self.allocator.free(new_content);

            try persistence.writeSecretsFile(self, secrets_path, new_content);
        }

        pub fn saveToVault(self: *Manager, name: []const u8, value: []const u8) SecretsError!void {
            const vault_url = self.config.vault_url orelse return error.VaultUrlNotConfigured;
            const vault_token = self.config.vault_token orelse return error.VaultTokenNotConfigured;

            const url = std.fmt.allocPrint(self.allocator, "{s}/v1/secret/data/{s}", .{
                vault_url,
                name,
            }) catch return error.OutOfMemory;
            defer self.allocator.free(url);

            const cache_key = try std.fmt.allocPrint(self.allocator, "vault:{s}", .{name});
            errdefer self.allocator.free(cache_key);
            const encrypted = try encryptSecret(self, value);

            try self.cache.put(self.allocator, cache_key, .{
                .value = encrypted,
                .cached_at = time.unixSeconds(),
            });

            const async_http = @import("../../utils/http/async_http.zig");

            var client = async_http.AsyncHttpClient.init(self.allocator) catch |err| {
                std.log.err("Failed to initialize HTTP client for Vault write: {}", .{err});
                return error.VaultConnectionFailed;
            };
            defer client.deinit();

            var request = async_http.HttpRequest.init(self.allocator, .post, url) catch |err| {
                std.log.err("Failed to create Vault write request: {}", .{err});
                return error.VaultRequestFailed;
            };
            defer request.deinit();

            request.setHeader("X-Vault-Token", vault_token) catch return error.OutOfMemory;

            const json_body = std.fmt.allocPrint(
                self.allocator,
                "{{\"data\":{{\"value\":\"{s}\"}}}}",
                .{value},
            ) catch return error.OutOfMemory;
            defer self.allocator.free(json_body);

            request.setJsonBody(json_body) catch return error.OutOfMemory;

            var response = client.fetch(&request) catch |err| {
                std.log.err("Vault write request failed for secret '{s}': {}", .{ name, err });
                return error.VaultConnectionFailed;
            };
            defer response.deinit();

            if (!response.isSuccess()) {
                std.log.err("Vault returned HTTP {d} on write for secret '{s}'", .{ response.status_code, name });
                return error.VaultRequestFailed;
            }

            std.log.info("Vault secret written successfully. Key: {s}", .{name});
        }

        pub fn deleteFromFile(self: *Manager, name: []const u8) SecretsError!void {
            const secrets_path = self.config.secrets_file orelse return error.SecretsFileNotConfigured;
            const io = self.io_backend.io();

            const content = std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch return;
            defer self.allocator.free(content);

            const search_key = try std.fmt.allocPrint(self.allocator, "\"{s}\":\"", .{name});
            defer self.allocator.free(search_key);

            const key_idx = std.mem.indexOf(u8, content, search_key) orelse return;
            const value_start = key_idx + search_key.len;
            var value_end = value_start;
            while (value_end < content.len and content[value_end] != '"') : (value_end += 1) {}
            value_end += 1;

            var remove_start = key_idx;
            var remove_end = value_end;

            if (remove_end < content.len and content[remove_end] == ',') {
                remove_end += 1;
            } else if (remove_start > 0 and content[remove_start - 1] == ',') {
                remove_start -= 1;
            }

            var new_content = std.ArrayListUnmanaged(u8).empty;
            defer new_content.deinit(self.allocator);

            try new_content.appendSlice(self.allocator, content[0..remove_start]);
            try new_content.appendSlice(self.allocator, content[remove_end..]);

            var write_file = std.Io.Dir.cwd().createFile(io, secrets_path, .{ .truncate = true }) catch return error.FileWriteFailed;
            defer write_file.close(io);
            var write_buf: [4096]u8 = undefined;
            var writer = write_file.writer(io, &write_buf);
            writer.interface.writeAll(new_content.items) catch return error.FileWriteFailed;
            writer.flush() catch return error.FileWriteFailed;
        }

        pub fn deleteFromVault(self: *Manager, name: []const u8) SecretsError!void {
            const vault_url = self.config.vault_url orelse return error.VaultUrlNotConfigured;
            const vault_token = self.config.vault_token orelse return error.VaultTokenNotConfigured;

            const cache_key = try std.fmt.allocPrint(self.allocator, "vault:{s}", .{name});
            defer self.allocator.free(cache_key);

            if (self.cache.fetchRemove(cache_key)) |kv| {
                self.allocator.free(kv.key);
                var v = kv.value;
                v.value.deinit();
            }

            const async_http = @import("../../utils/http/async_http.zig");

            const url = std.fmt.allocPrint(self.allocator, "{s}/v1/secret/data/{s}", .{
                vault_url,
                name,
            }) catch return error.OutOfMemory;
            defer self.allocator.free(url);

            var client = async_http.AsyncHttpClient.init(self.allocator) catch |err| {
                std.log.err("Failed to initialize HTTP client for Vault delete: {}", .{err});
                return error.VaultConnectionFailed;
            };
            defer client.deinit();

            var request = async_http.HttpRequest.init(self.allocator, .delete, url) catch |err| {
                std.log.err("Failed to create Vault delete request: {}", .{err});
                return error.VaultRequestFailed;
            };
            defer request.deinit();

            request.setHeader("X-Vault-Token", vault_token) catch return error.OutOfMemory;

            var response = client.fetch(&request) catch |err| {
                std.log.err("Vault delete request failed for secret '{s}': {}", .{ name, err });
                return error.VaultConnectionFailed;
            };
            defer response.deinit();

            if (!response.isSuccess()) {
                if (response.status_code != 404) {
                    std.log.err("Vault returned HTTP {d} on delete for secret '{s}'", .{ response.status_code, name });
                    return error.VaultRequestFailed;
                }
            }

            std.log.info("Vault secret deleted. Key: {s}", .{name});
        }

        pub fn envExists(self: *Manager, name: []const u8) bool {
            const env_name = std.fmt.allocPrint(self.allocator, "{s}{s}", .{
                self.config.env_prefix,
                name,
            }) catch return false;
            defer self.allocator.free(env_name);

            const env_name_z = std.fmt.allocPrintSentinel(self.allocator, "{s}", .{env_name}, 0) catch return false;
            defer self.allocator.free(env_name_z);
            const name_z = std.fmt.allocPrintSentinel(self.allocator, "{s}", .{name}, 0) catch return false;
            defer self.allocator.free(name_z);
            return std.c.getenv(env_name_z.ptr) != null or std.c.getenv(name_z.ptr) != null;
        }

        pub fn fileExists(self: *Manager, name: []const u8) bool {
            const secrets_path = self.config.secrets_file orelse return false;
            const io = self.io_backend.io();

            const content = std.Io.Dir.cwd().readFileAlloc(io, secrets_path, self.allocator, .limited(1024 * 1024)) catch return false;
            defer self.allocator.free(content);

            const search_key = std.fmt.allocPrint(self.allocator, "\"{s}\":", .{name}) catch return false;
            defer self.allocator.free(search_key);

            return std.mem.indexOf(u8, content, search_key) != null;
        }

        pub fn vaultExists(self: *Manager, name: []const u8) bool {
            const cache_key = std.fmt.allocPrint(self.allocator, "vault:{s}", .{name}) catch return false;
            defer self.allocator.free(cache_key);

            return self.cache.contains(cache_key);
        }

        fn encryptSecret(self: *Manager, value: []const u8) SecretsError!SecretValue {
            var nonce: [12]u8 = undefined;
            @import("../csprng.zig").fillRandom(&nonce) catch return error.DecryptionFailed;

            const ciphertext = try self.allocator.alloc(u8, value.len);
            errdefer self.allocator.free(ciphertext);

            var tag: [16]u8 = undefined;

            const aead = std.crypto.aead.aes_gcm.Aes256Gcm;
            aead.encrypt(ciphertext, &tag, value, &.{}, nonce, self.master_key);

            return SecretValue{
                .allocator = self.allocator,
                .encrypted_value = ciphertext,
                .nonce = nonce,
                .tag = tag,
                .metadata = .{
                    .created_at = time.unixSeconds(),
                },
            };
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}
