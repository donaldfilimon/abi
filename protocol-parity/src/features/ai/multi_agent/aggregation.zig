//! Multi-Agent Result Aggregation Strategies
//!
//! Provides real implementations for combining outputs from multiple agents:
//!
//! - **hashVote**: Majority voting via Wyhash fingerprinting of responses.
//!   Each unique response is hashed, and the most frequent hash wins.
//!
//! - **sectionMerge**: Merges responses by markdown heading sections.
//!   Deduplicates identical sections and preserves unique content from each agent.
//!
//! - **weightedSelect**: Selects the best response using quality heuristics
//!   (length, structure, completeness).

const std = @import("std");

pub const AggregationError = error{
    /// No successful results to aggregate.
    NoResults,
    /// All responses were empty.
    EmptyResponses,
    /// Out of memory during aggregation.
    OutOfMemory,
};

/// Result from a single agent execution.
pub const AgentOutput = struct {
    response: []const u8,
    success: bool,
    agent_index: usize,
    duration_ns: u64 = 0,
};

// ============================================================================
// Hash-Based Majority Vote
// ============================================================================

/// Aggregate by majority vote using Wyhash fingerprinting.
///
/// Hashes each successful response. The response with the most identical
/// copies wins. Ties are broken by earliest occurrence.
///
/// This is useful when multiple agents should converge on the same answer
/// (e.g., classification, factual Q&A, code review consensus).
pub fn hashVote(allocator: std.mem.Allocator, outputs: []const AgentOutput) AggregationError![]u8 {
    // Count votes per unique response hash
    var vote_counts = std.AutoHashMapUnmanaged(u64, VoteEntry).empty;
    defer vote_counts.deinit(allocator);

    var successful: u32 = 0;

    for (outputs) |output| {
        if (!output.success or output.response.len == 0) continue;
        successful += 1;

        // Normalize: trim whitespace before hashing
        const trimmed = std.mem.trim(u8, output.response, " \t\n\r");
        if (trimmed.len == 0) continue;

        const hash = std.hash.Wyhash.hash(0, trimmed);

        const gop = vote_counts.getOrPut(allocator, hash) catch return AggregationError.OutOfMemory;
        if (gop.found_existing) {
            gop.value_ptr.count += 1;
        } else {
            gop.value_ptr.* = .{
                .count = 1,
                .response = output.response,
                .first_index = output.agent_index,
            };
        }
    }

    if (successful == 0) return AggregationError.NoResults;

    // Find the response with the most votes
    var best: ?*const VoteEntry = null;
    var iter = vote_counts.iterator();
    while (iter.next()) |entry| {
        if (best == null or entry.value_ptr.count > best.?.count) {
            best = entry.value_ptr;
        } else if (entry.value_ptr.count == best.?.count) {
            // Tie-break: prefer earlier agent (lower index)
            if (entry.value_ptr.first_index < best.?.first_index) {
                best = entry.value_ptr;
            }
        }
    }

    if (best) |b| {
        return allocator.dupe(u8, b.response) catch return AggregationError.OutOfMemory;
    }

    return AggregationError.EmptyResponses;
}

const VoteEntry = struct {
    count: u32,
    response: []const u8,
    first_index: usize,
};

// ============================================================================
// Section-Based Merge
// ============================================================================

/// Merge responses by extracting and deduplicating markdown sections.
///
/// Splits each response by `#` headings, then merges unique sections.
/// Identical sections (by heading + content hash) are deduplicated.
/// Non-headed content is concatenated with separators.
///
/// This is useful for synthesis tasks where each agent may cover
/// different aspects of a topic.
pub fn sectionMerge(allocator: std.mem.Allocator, outputs: []const AgentOutput) AggregationError![]u8 {
    var seen_hashes = std.AutoHashMapUnmanaged(u64, void).empty;
    defer seen_hashes.deinit(allocator);

    var result = std.ArrayListUnmanaged(u8).empty;
    errdefer result.deinit(allocator);

    var has_content = false;

    for (outputs) |output| {
        if (!output.success or output.response.len == 0) continue;

        // Split into sections by markdown heading lines
        var lines = std.mem.splitSequence(u8, output.response, "\n");
        var current_section = std.ArrayListUnmanaged(u8).empty;
        defer current_section.deinit(allocator);

        while (lines.next()) |line| {
            const is_heading = line.len > 0 and line[0] == '#';

            if (is_heading and current_section.items.len > 0) {
                // Flush previous section
                try flushSection(allocator, &result, &seen_hashes, current_section.items, &has_content);
                current_section.clearRetainingCapacity();
            }

            current_section.appendSlice(allocator, line) catch return AggregationError.OutOfMemory;
            current_section.append(allocator, '\n') catch return AggregationError.OutOfMemory;
        }

        // Flush final section
        if (current_section.items.len > 0) {
            try flushSection(allocator, &result, &seen_hashes, current_section.items, &has_content);
        }
    }

    if (!has_content) return AggregationError.EmptyResponses;

    return result.toOwnedSlice(allocator) catch return AggregationError.OutOfMemory;
}

fn flushSection(
    allocator: std.mem.Allocator,
    result: *std.ArrayListUnmanaged(u8),
    seen: *std.AutoHashMapUnmanaged(u64, void),
    section: []const u8,
    has_content: *bool,
) AggregationError!void {
    const trimmed = std.mem.trim(u8, section, " \t\n\r");
    if (trimmed.len == 0) return;

    const hash = std.hash.Wyhash.hash(0, trimmed);

    // Skip duplicate sections
    const gop = seen.getOrPut(allocator, hash) catch return AggregationError.OutOfMemory;
    if (gop.found_existing) return;

    if (has_content.*) {
        result.appendSlice(allocator, "\n") catch return AggregationError.OutOfMemory;
    }
    result.appendSlice(allocator, section) catch return AggregationError.OutOfMemory;
    has_content.* = true;
}

// ============================================================================
// Weighted Selection
// ============================================================================

/// Select the best response using quality heuristics.
///
/// Scores each response on:
/// - Length (proxy for completeness) — up to 40 points
/// - Structure (has headings, lists, code blocks) — up to 30 points
/// - Sentence count (more detail) — up to 30 points
pub fn weightedSelect(allocator: std.mem.Allocator, outputs: []const AgentOutput) AggregationError![]u8 {
    var best_score: u32 = 0;
    var best_response: ?[]const u8 = null;

    for (outputs) |output| {
        if (!output.success or output.response.len == 0) continue;

        const score = scoreResponse(output.response);
        if (score > best_score or best_response == null) {
            best_score = score;
            best_response = output.response;
        }
    }

    if (best_response) |resp| {
        return allocator.dupe(u8, resp) catch return AggregationError.OutOfMemory;
    }

    return AggregationError.NoResults;
}

fn scoreResponse(response: []const u8) u32 {
    var score: u32 = 0;

    // Length score (up to 40 points, 1 point per 100 chars, capped)
    score += @min(@as(u32, @intCast(response.len / 100)), 40);

    // Structure score (up to 30 points)
    if (std.mem.indexOf(u8, response, "# ") != null) score += 10; // has headings
    if (std.mem.indexOf(u8, response, "- ") != null) score += 5; // has lists
    if (std.mem.indexOf(u8, response, "```") != null) score += 10; // has code blocks
    if (std.mem.indexOf(u8, response, "**") != null) score += 5; // has bold

    // Sentence count (up to 30 points, 2 per sentence)
    var sentences: u32 = 0;
    for (response) |c| {
        if (c == '.' or c == '!' or c == '?') sentences += 1;
    }
    score += @min(sentences * 2, 30);

    return score;
}

// ============================================================================
// Tests
// ============================================================================

test "hashVote selects majority response" {
    const allocator = std.testing.allocator;

    const outputs = [_]AgentOutput{
        .{ .response = "The answer is 42", .success = true, .agent_index = 0 },
        .{ .response = "The answer is 42", .success = true, .agent_index = 1 },
        .{ .response = "The answer is 7", .success = true, .agent_index = 2 },
    };

    const result = try hashVote(allocator, &outputs);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("The answer is 42", result);
}

test "hashVote handles single response" {
    const allocator = std.testing.allocator;

    const outputs = [_]AgentOutput{
        .{ .response = "only one", .success = true, .agent_index = 0 },
    };

    const result = try hashVote(allocator, &outputs);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("only one", result);
}

test "hashVote skips failed agents" {
    const allocator = std.testing.allocator;

    const outputs = [_]AgentOutput{
        .{ .response = "good", .success = true, .agent_index = 0 },
        .{ .response = "error", .success = false, .agent_index = 1 },
        .{ .response = "good", .success = true, .agent_index = 2 },
    };

    const result = try hashVote(allocator, &outputs);
    defer allocator.free(result);

    try std.testing.expectEqualStrings("good", result);
}

test "hashVote returns error for all failures" {
    const outputs = [_]AgentOutput{
        .{ .response = "error", .success = false, .agent_index = 0 },
    };

    try std.testing.expectError(AggregationError.NoResults, hashVote(std.testing.allocator, &outputs));
}

test "sectionMerge deduplicates identical sections" {
    const allocator = std.testing.allocator;

    const outputs = [_]AgentOutput{
        .{ .response = "# Intro\nHello world\n# Details\nSome details", .success = true, .agent_index = 0 },
        .{ .response = "# Intro\nHello world\n# Extra\nBonus content", .success = true, .agent_index = 1 },
    };

    const result = try sectionMerge(allocator, &outputs);
    defer allocator.free(result);

    // Should contain "Intro", "Details", and "Extra" sections
    try std.testing.expect(std.mem.indexOf(u8, result, "# Intro") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "# Details") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "# Extra") != null);

    // "# Intro" should appear only once (deduplicated)
    const first_intro = std.mem.indexOf(u8, result, "# Intro").?;
    const rest = result[first_intro + 1 ..];
    const second_intro = std.mem.indexOf(u8, rest, "# Intro");
    try std.testing.expect(second_intro == null);
}

test "weightedSelect prefers structured responses" {
    const allocator = std.testing.allocator;

    const outputs = [_]AgentOutput{
        .{ .response = "short", .success = true, .agent_index = 0 },
        .{ .response = "# Title\n\nThis is a **detailed** response.\n\n- Point one\n- Point two\n\n```code```\n\nMore text.", .success = true, .agent_index = 1 },
    };

    const result = try weightedSelect(allocator, &outputs);
    defer allocator.free(result);

    // Should select the structured response (higher score)
    try std.testing.expect(std.mem.indexOf(u8, result, "# Title") != null);
}

test "scoreResponse calculates correctly" {
    // Empty
    try std.testing.expectEqual(@as(u32, 0), scoreResponse(""));

    // Simple text with a sentence
    const simple_score = scoreResponse("Hello world.");
    try std.testing.expect(simple_score >= 2); // at least sentence points

    // Structured text
    const structured_score = scoreResponse("# Title\n\nHello world.\n\n- Item\n\n```code```\n\n**Bold**.");
    try std.testing.expect(structured_score > simple_score);
}

test {
    std.testing.refAllDecls(@This());
}
