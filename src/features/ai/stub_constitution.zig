const std = @import("std");
const types = @import("stub_types.zig");

pub const Principle = types.Principle;
pub const AuditResult = types.AuditResult;

pub const Constitution = struct {
    pub fn validate(response: []const u8) AuditResult {
        var result = AuditResult.init();
        if (response.len == 0) {
            result.passed = false;
            result.violations.set(@backingInt(Principle.truthfulness));
            result.scores[@backingInt(Principle.truthfulness)] = 0.0;
        }
        result.finalize();
        return result;
    }

    pub fn evaluateResponse(response: []const u8, principles: []const Principle) AuditResult {
        var result = AuditResult.init();
        if (response.len == 0) {
            result.passed = false;
            for (principles) |p| {
                result.violations.set(@backingInt(p));
                result.scores[@backingInt(p)] = 0.0;
            }
        }
        result.finalize();
        return result;
    }
};

test {
    std.testing.refAllDecls(@This());
}
