const shared = @import("shared/simd");

pub const SIMDOpts = shared.SIMDOpts;
pub const getPerformanceMonitor = shared.getPerformanceMonitor;
pub const getPerformanceMonitorDetails = shared.getPerformanceMonitorDetails;
pub const getVectorOps = shared.getVectorOps;
pub const text = shared.text;

pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: shared.SIMDOpts) f32 {
    return shared.dotProductSIMD(a, b, opts);
}

pub fn vectorAddSIMD(a: []const f32, b: []const f32, result: []f32) void {
    shared.vectorAddSIMD(a, b, result);
}
