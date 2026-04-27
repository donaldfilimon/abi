const std = @import("std");
const shared = @import("shared.zig");

const SecretsError = shared.SecretsError;
const ValidationRule = shared.ValidationRule;

pub fn validateSecret(
    rules: []const ValidationRule,
    validation_failures: *u64,
    name: []const u8,
    value: []const u8,
) SecretsError!void {
    for (rules) |rule| {
        if (!matchesPattern(name, rule.name_pattern)) continue;

        if (rule.min_length) |min| {
            if (value.len < min) {
                validation_failures.* += 1;
                return error.SecretTooShort;
            }
        }

        if (rule.max_length) |max| {
            if (value.len > max) {
                validation_failures.* += 1;
                return error.SecretTooLong;
            }
        }

        if (rule.required_prefix) |prefix| {
            if (!std.mem.startsWith(u8, value, prefix)) {
                validation_failures.* += 1;
                return error.InvalidSecretFormat;
            }
        }

        if (rule.forbidden_chars) |forbidden| {
            for (value) |c| {
                if (std.mem.indexOfScalar(u8, forbidden, c) != null) {
                    validation_failures.* += 1;
                    return error.ForbiddenCharacter;
                }
            }
        }

        if (rule.must_be_base64) {
            _ = std.base64.standard.Decoder.calcSizeForSlice(value) catch {
                validation_failures.* += 1;
                return error.InvalidBase64;
            };
        }
    }
}

pub fn matchesPattern(name: []const u8, pattern: []const u8) bool {
    if (std.mem.eql(u8, pattern, "*")) return true;

    if (std.mem.indexOf(u8, pattern, "*")) |star_idx| {
        const prefix = pattern[0..star_idx];
        const suffix = pattern[star_idx + 1 ..];

        return std.mem.startsWith(u8, name, prefix) and
            std.mem.endsWith(u8, name, suffix);
    }

    return std.mem.eql(u8, name, pattern);
}

test {
    std.testing.refAllDecls(@This());
}
