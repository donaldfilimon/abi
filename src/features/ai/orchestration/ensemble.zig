//! Ensemble Methods Module
//!
//! Provides ensemble methods for combining outputs from multiple models
//! to improve quality, reliability, and confidence in generated responses.

const std = @import("std");

// ============================================================================
// Types
// ============================================================================

/// Method for combining multiple model outputs.
pub const EnsembleMethod = enum {
    /// Select output by majority vote (for classification/discrete outputs).
    voting,
    /// Average numerical outputs or embeddings.
    averaging,
    /// Weighted combination based on model reliability.
    weighted_average,
    /// Select best output based on confidence scores.
    best_of_n,
    /// Concatenate all outputs with deduplication.
    concatenate,
    /// Use first successful response.
    first_success,
    /// Custom aggregation function.
    custom,

    pub fn toString(self: EnsembleMethod) []const u8 {
        return @tagName(self);
    }
};

/// Result from ensemble execution.
pub const EnsembleResult = struct {
    /// Combined or selected response.
    response: []u8,
    /// Number of models that contributed.
    model_count: usize,
    /// Confidence in the ensemble result (0.0 to 1.0).
    confidence: f64,
    /// Individual model responses (if retained).
    individual_responses: ?[]const ModelResponse = null,
    /// Aggregation metadata.
    metadata: ?AggregationMetadata = null,
};

/// Individual model response for ensemble.
pub const ModelResponse = struct {
    /// Model identifier.
    model_id: []const u8,
    /// Generated response.
    response: []const u8,
    /// Confidence score from this model.
    confidence: f64 = 1.0,
    /// Latency in milliseconds.
    latency_ms: u64 = 0,
    /// Whether this response was selected.
    selected: bool = false,
};

/// Metadata about the aggregation process.
pub const AggregationMetadata = struct {
    /// Method used for aggregation.
    method: EnsembleMethod,
    /// Total responses received.
    total_responses: usize,
    /// Responses that failed or were invalid.
    failed_responses: usize,
    /// Agreement ratio (for voting).
    agreement_ratio: f64 = 0.0,
    /// Average confidence across models.
    avg_confidence: f64 = 0.0,
    /// Standard deviation of responses (for numerical).
    std_deviation: f64 = 0.0,
};

// ============================================================================
// Ensemble
// ============================================================================

/// Ensemble manager for combining multiple model outputs.
pub const Ensemble = struct {
    allocator: std.mem.Allocator,
    method: EnsembleMethod,
    min_responses: usize = 2,
    max_responses: usize = 10,
    timeout_ms: u64 = 30000,
    retain_individual: bool = false,

    pub fn init(allocator: std.mem.Allocator, method: EnsembleMethod) Ensemble {
        return .{
            .allocator = allocator,
            .method = method,
        };
    }

    pub fn deinit(self: *Ensemble) void {
        _ = self;
        // No resources to clean up currently
    }

    /// Combine multiple responses using the configured method.
    pub fn combine(
        self: *Ensemble,
        responses: []const ModelResponse,
    ) !EnsembleResult {
        if (responses.len == 0) {
            return error.NoResponses;
        }

        return switch (self.method) {
            .voting => self.combineByVoting(responses),
            .averaging => self.combineByAveraging(responses),
            .weighted_average => self.combineByWeightedAverage(responses),
            .best_of_n => self.combineByBestOfN(responses),
            .concatenate => self.combineByConcatenation(responses),
            .first_success => self.combineByFirstSuccess(responses),
            .custom => error.CustomAggregatorRequired,
        };
    }

    /// Voting: Select the most common response.
    fn combineByVoting(self: *Ensemble, responses: []const ModelResponse) !EnsembleResult {
        // Count occurrences of each unique response
        var counts = std.StringHashMapUnmanaged(usize){};
        defer counts.deinit(self.allocator);

        for (responses) |resp| {
            const entry = counts.getOrPutValue(self.allocator, resp.response, 0) catch continue;
            entry.value_ptr.* += 1;
        }

        // Find the response with highest count
        var best_response: []const u8 = responses[0].response;
        var best_count: usize = 0;

        var it = counts.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* > best_count) {
                best_count = entry.value_ptr.*;
                best_response = entry.key_ptr.*;
            }
        }

        const agreement = @as(f64, @floatFromInt(best_count)) /
            @as(f64, @floatFromInt(responses.len));

        return EnsembleResult{
            .response = try self.allocator.dupe(u8, best_response),
            .model_count = responses.len,
            .confidence = agreement,
            .metadata = .{
                .method = .voting,
                .total_responses = responses.len,
                .failed_responses = 0,
                .agreement_ratio = agreement,
                .avg_confidence = self.calculateAvgConfidence(responses),
            },
        };
    }

    /// Averaging: For numerical outputs, compute the average.
    fn combineByAveraging(self: *Ensemble, responses: []const ModelResponse) !EnsembleResult {
        // Parse responses as floats if possible
        var sum: f64 = 0.0;
        var valid_count: usize = 0;
        var values = std.ArrayListUnmanaged(f64).empty;
        defer values.deinit(self.allocator);

        for (responses) |resp| {
            const trimmed = std.mem.trim(u8, resp.response, " \t\n\r");
            if (std.fmt.parseFloat(f64, trimmed)) |val| {
                sum += val;
                valid_count += 1;
                try values.append(self.allocator, val);
            } else |_| {
                // Not a numeric response
            }
        }

        if (valid_count == 0) {
            // Fall back to first success if no numeric responses
            return self.combineByFirstSuccess(responses);
        }

        const avg = sum / @as(f64, @floatFromInt(valid_count));

        // Calculate standard deviation
        var variance_sum: f64 = 0.0;
        for (values.items) |val| {
            const diff = val - avg;
            variance_sum += diff * diff;
        }
        const std_dev = std.math.sqrt(variance_sum / @as(f64, @floatFromInt(valid_count)));

        // Format average as response
        const response = try std.fmt.allocPrint(self.allocator, "{d:.6}", .{avg});

        return EnsembleResult{
            .response = response,
            .model_count = valid_count,
            .confidence = if (std_dev < 0.1) 0.95 else @max(0.5, 1.0 - std_dev),
            .metadata = .{
                .method = .averaging,
                .total_responses = responses.len,
                .failed_responses = responses.len - valid_count,
                .avg_confidence = self.calculateAvgConfidence(responses),
                .std_deviation = std_dev,
            },
        };
    }

    /// Weighted average: Combine based on model confidence/reliability.
    fn combineByWeightedAverage(self: *Ensemble, responses: []const ModelResponse) !EnsembleResult {
        // For text responses, weight by confidence and select best
        var best_response: ?[]const u8 = null;
        var best_weight: f64 = 0.0;
        var total_weight: f64 = 0.0;

        for (responses) |resp| {
            total_weight += resp.confidence;
            if (resp.confidence > best_weight) {
                best_weight = resp.confidence;
                best_response = resp.response;
            }
        }

        if (best_response == null) {
            return error.NoValidResponses;
        }

        // Normalize confidence
        const normalized_confidence = if (total_weight > 0)
            best_weight / total_weight * @as(f64, @floatFromInt(responses.len))
        else
            best_weight;

        return EnsembleResult{
            .response = try self.allocator.dupe(u8, best_response.?),
            .model_count = responses.len,
            .confidence = @min(normalized_confidence, 1.0),
            .metadata = .{
                .method = .weighted_average,
                .total_responses = responses.len,
                .failed_responses = 0,
                .avg_confidence = self.calculateAvgConfidence(responses),
            },
        };
    }

    /// Best of N: Select response with highest confidence.
    fn combineByBestOfN(self: *Ensemble, responses: []const ModelResponse) !EnsembleResult {
        var best: ?ModelResponse = null;

        for (responses) |resp| {
            if (best == null or resp.confidence > best.?.confidence) {
                best = resp;
            }
        }

        if (best == null) {
            return error.NoValidResponses;
        }

        return EnsembleResult{
            .response = try self.allocator.dupe(u8, best.?.response),
            .model_count = responses.len,
            .confidence = best.?.confidence,
            .metadata = .{
                .method = .best_of_n,
                .total_responses = responses.len,
                .failed_responses = 0,
                .avg_confidence = self.calculateAvgConfidence(responses),
            },
        };
    }

    /// Concatenation: Combine all unique responses.
    fn combineByConcatenation(self: *Ensemble, responses: []const ModelResponse) !EnsembleResult {
        var seen = std.StringHashMapUnmanaged(void){};
        defer seen.deinit(self.allocator);

        var combined = std.ArrayListUnmanaged(u8).empty;
        errdefer combined.deinit(self.allocator);

        var first = true;
        for (responses) |resp| {
            // Skip duplicates
            if (seen.contains(resp.response)) continue;
            try seen.put(self.allocator, resp.response, {});

            if (!first) {
                try combined.appendSlice(self.allocator, "\n---\n");
            }
            first = false;

            // Add model attribution
            try combined.appendSlice(self.allocator, "[");
            try combined.appendSlice(self.allocator, resp.model_id);
            try combined.appendSlice(self.allocator, "] ");
            try combined.appendSlice(self.allocator, resp.response);
        }

        const unique_count = seen.count();

        return EnsembleResult{
            .response = try combined.toOwnedSlice(self.allocator),
            .model_count = responses.len,
            .confidence = @as(f64, @floatFromInt(unique_count)) /
                @as(f64, @floatFromInt(responses.len)),
            .metadata = .{
                .method = .concatenate,
                .total_responses = responses.len,
                .failed_responses = 0,
                .avg_confidence = self.calculateAvgConfidence(responses),
            },
        };
    }

    /// First success: Use first valid response.
    fn combineByFirstSuccess(self: *Ensemble, responses: []const ModelResponse) !EnsembleResult {
        for (responses) |resp| {
            if (resp.response.len > 0) {
                return EnsembleResult{
                    .response = try self.allocator.dupe(u8, resp.response),
                    .model_count = 1,
                    .confidence = resp.confidence,
                    .metadata = .{
                        .method = .first_success,
                        .total_responses = responses.len,
                        .failed_responses = 0,
                        .avg_confidence = resp.confidence,
                    },
                };
            }
        }

        return error.NoValidResponses;
    }

    fn calculateAvgConfidence(self: *Ensemble, responses: []const ModelResponse) f64 {
        _ = self;
        if (responses.len == 0) return 0.0;

        var sum: f64 = 0.0;
        for (responses) |resp| {
            sum += resp.confidence;
        }
        return sum / @as(f64, @floatFromInt(responses.len));
    }
};

// ============================================================================
// Ensemble Configuration Presets
// ============================================================================

/// Configuration for different ensemble scenarios.
pub const EnsemblePreset = struct {
    method: EnsembleMethod,
    min_responses: usize,
    max_responses: usize,
    timeout_ms: u64,

    /// High quality: use more models, longer timeout.
    pub const highQuality = EnsemblePreset{
        .method = .best_of_n,
        .min_responses = 3,
        .max_responses = 5,
        .timeout_ms = 60000,
    };

    /// Fast: use fewer models, shorter timeout.
    pub const fast = EnsemblePreset{
        .method = .first_success,
        .min_responses = 1,
        .max_responses = 3,
        .timeout_ms = 15000,
    };

    /// Consensus: require agreement between models.
    pub const consensus = EnsemblePreset{
        .method = .voting,
        .min_responses = 3,
        .max_responses = 5,
        .timeout_ms = 45000,
    };

    /// Comprehensive: get diverse perspectives.
    pub const comprehensive = EnsemblePreset{
        .method = .concatenate,
        .min_responses = 2,
        .max_responses = 4,
        .timeout_ms = 60000,
    };
};

// ============================================================================
// Errors
// ============================================================================

pub const EnsembleError = error{
    NoResponses,
    NoValidResponses,
    CustomAggregatorRequired,
    InsufficientResponses,
    Timeout,
};

// ============================================================================
// Tests
// ============================================================================

test "voting ensemble" {
    var ens = Ensemble.init(std.testing.allocator, .voting);
    defer ens.deinit();

    const responses = [_]ModelResponse{
        .{ .model_id = "a", .response = "yes", .confidence = 0.8 },
        .{ .model_id = "b", .response = "yes", .confidence = 0.9 },
        .{ .model_id = "c", .response = "no", .confidence = 0.7 },
    };

    const result = try ens.combine(&responses);
    defer std.testing.allocator.free(result.response);

    try std.testing.expectEqualStrings("yes", result.response);
    try std.testing.expectEqual(@as(usize, 3), result.model_count);
    // 2 out of 3 agreed on "yes"
    try std.testing.expect(result.confidence > 0.6);
}

test "best of n ensemble" {
    var ens = Ensemble.init(std.testing.allocator, .best_of_n);
    defer ens.deinit();

    const responses = [_]ModelResponse{
        .{ .model_id = "a", .response = "low confidence", .confidence = 0.5 },
        .{ .model_id = "b", .response = "high confidence", .confidence = 0.95 },
        .{ .model_id = "c", .response = "medium confidence", .confidence = 0.7 },
    };

    const result = try ens.combine(&responses);
    defer std.testing.allocator.free(result.response);

    try std.testing.expectEqualStrings("high confidence", result.response);
    try std.testing.expect(result.confidence > 0.9);
}

test "first success ensemble" {
    var ens = Ensemble.init(std.testing.allocator, .first_success);
    defer ens.deinit();

    const responses = [_]ModelResponse{
        .{ .model_id = "a", .response = "first response", .confidence = 0.7 },
        .{ .model_id = "b", .response = "second response", .confidence = 0.9 },
    };

    const result = try ens.combine(&responses);
    defer std.testing.allocator.free(result.response);

    try std.testing.expectEqualStrings("first response", result.response);
}

test "averaging ensemble with numbers" {
    var ens = Ensemble.init(std.testing.allocator, .averaging);
    defer ens.deinit();

    const responses = [_]ModelResponse{
        .{ .model_id = "a", .response = "10.0", .confidence = 0.8 },
        .{ .model_id = "b", .response = "12.0", .confidence = 0.9 },
        .{ .model_id = "c", .response = "11.0", .confidence = 0.85 },
    };

    const result = try ens.combine(&responses);
    defer std.testing.allocator.free(result.response);

    // Average should be 11.0
    // SAFETY: The averaging ensemble method always produces a valid float string via allocPrint("{d:.6}", ...).
    // Test inputs are controlled numeric strings that produce a predictable numeric output.
    const avg = std.fmt.parseFloat(f64, result.response) catch unreachable;
    try std.testing.expect(@abs(avg - 11.0) < 0.001);
}
