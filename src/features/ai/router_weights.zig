const std = @import("std");

pub const ProfileWeights = struct {
    w_abbey: f32,
    w_aviva: f32,
    w_abi: f32,

    pub fn normalize(self: *ProfileWeights) void {
        const total = self.w_abbey + self.w_aviva + self.w_abi;
        if (total > 0) {
            self.w_abbey /= total;
            self.w_aviva /= total;
            self.w_abi /= total;
        }
    }
};

/// Blend two ProfileWeights with alpha (0.0 = a only, 1.0 = b only).
pub fn blendWeights(a: ProfileWeights, b: ProfileWeights, alpha: f32) ProfileWeights {
    const a_alpha = 1.0 - alpha;
    return .{
        .w_abbey = a.w_abbey * a_alpha + b.w_abbey * alpha,
        .w_aviva = a.w_aviva * a_alpha + b.w_aviva * alpha,
        .w_abi = a.w_abi * a_alpha + b.w_abi * alpha,
    };
}

test {
    std.testing.refAllDecls(@This());
}
