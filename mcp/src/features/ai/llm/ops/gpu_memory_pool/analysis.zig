const types = @import("types.zig");
const DefragmentationRecommendation = types.DefragmentationRecommendation;

/// Detailed fragmentation analysis result.
pub const FragmentationAnalysis = struct {
    /// Internal fragmentation ratio (wasted space in allocated blocks)
    internal_fragmentation_ratio: f64,
    /// External fragmentation ratio (unusable free blocks)
    external_fragmentation_ratio: f64,
    /// Combined fragmentation ratio
    total_fragmentation_ratio: f64,
    /// Bytes wasted to internal fragmentation
    internal_fragmentation_bytes: usize,
    /// Bytes wasted to external fragmentation
    external_fragmentation_bytes: usize,
    /// Total bytes in free lists
    total_free_list_bytes: usize,
    /// Number of unusable free blocks
    unusable_block_count: usize,
    /// Minimum allocation size seen (threshold for unusable blocks)
    min_request_size: usize,
    /// Defragmentation recommendation
    recommendation: DefragmentationRecommendation,

    /// Format as human-readable summary.
    pub fn format(self: FragmentationAnalysis, writer: anytype) !void {
        try writer.print("Fragmentation Analysis:\n", .{});
        try writer.print("  Internal: {d:.1}% ({d} bytes wasted in {d} allocated bytes)\n", .{
            self.internal_fragmentation_ratio * 100.0,
            self.internal_fragmentation_bytes,
            self.internal_fragmentation_bytes + self.total_free_list_bytes,
        });
        try writer.print("  External: {d:.1}% ({d} unusable blocks, {d} bytes)\n", .{
            self.external_fragmentation_ratio * 100.0,
            self.unusable_block_count,
            self.external_fragmentation_bytes,
        });
        try writer.print("  Total: {d:.1}%\n", .{self.total_fragmentation_ratio * 100.0});
        try writer.print("  Free list: {d} bytes total\n", .{self.total_free_list_bytes});
        try writer.print("  Min request size: {d} bytes\n", .{self.min_request_size});
        try writer.print("  Severity: {s}\n", .{self.recommendation.severity.toString()});
        try writer.print("  Recommendation: {s}\n", .{self.recommendation.getMessage()});
    }
};
