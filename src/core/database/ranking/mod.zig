//! Turns candidate sets into useful memory.

const std = @import("std");
const core = @import("../core/mod.zig");
const block = @import("../block/mod.zig");

pub const WeightProfile = struct {
    semantic_similarity: f32 = 1.0,
    recency: f32 = 1.0,
    trust_score: f32 = 1.0,
    user_pinning: f32 = 1.0,
    project_locality: f32 = 1.0,
    profile_preference: f32 = 1.0,
    contradiction_penalty: f32 = 1.0,
    past_usefulness: f32 = 1.0,

    /// Returns the sum of all weights for normalization.
    pub fn totalWeight(self: WeightProfile) f32 {
        return self.semantic_similarity +
            self.recency +
            self.trust_score +
            self.user_pinning +
            self.project_locality +
            self.profile_preference +
            self.contradiction_penalty +
            self.past_usefulness;
    }
};

/// Per-candidate scoring signals used to compute a final weighted score.
pub const CandidateSignals = struct {
    semantic_similarity: f32 = 0.0,
    recency: f32 = 0.0,
    trust_score: f32 = 0.0,
    user_pinning: f32 = 0.0,
    project_locality: f32 = 0.0,
    profile_preference: f32 = 0.0,
    contradiction_penalty: f32 = 0.0,
    past_usefulness: f32 = 0.0,
};

pub const RankedCandidate = struct {
    id: core.ids.BlockId,
    score: f32,
};

pub const ScoreTrace = struct {
    id: core.ids.BlockId,
    base_score: f32,
    recency_modifier: f32,
    trust_modifier: f32,
    final_score: f32,
};

pub const Scorer = struct {
    allocator: std.mem.Allocator,
    profile: WeightProfile,

    pub fn init(allocator: std.mem.Allocator, profile: WeightProfile) Scorer {
        return .{
            .allocator = allocator,
            .profile = profile,
        };
    }

    /// Compute a weighted score from per-candidate signals and the weight profile.
    pub fn computeWeightedScore(self: *Scorer, signals: CandidateSignals) f32 {
        const total_weight = self.profile.totalWeight();
        if (total_weight == 0.0) return 0.0;

        const weighted_sum =
            signals.semantic_similarity * self.profile.semantic_similarity +
            signals.recency * self.profile.recency +
            signals.trust_score * self.profile.trust_score +
            signals.user_pinning * self.profile.user_pinning +
            signals.project_locality * self.profile.project_locality +
            signals.profile_preference * self.profile.profile_preference +
            signals.past_usefulness * self.profile.past_usefulness -
            signals.contradiction_penalty * self.profile.contradiction_penalty;

        return weighted_sum / total_weight;
    }

    pub fn rankCandidates(
        self: *Scorer,
        candidates: []const core.ids.BlockId,
    ) ![]RankedCandidate {
        return self.rankCandidatesWithSignals(candidates, null);
    }

    /// Rank candidates using explicit per-candidate signals when available,
    /// falling back to ID-derived heuristics when signals are not provided.
    pub fn rankCandidatesWithSignals(
        self: *Scorer,
        candidates: []const core.ids.BlockId,
        signals: ?[]const CandidateSignals,
    ) ![]RankedCandidate {
        var ranked = std.ArrayList(RankedCandidate).empty;
        defer ranked.deinit(self.allocator);
        try ranked.ensureTotalCapacity(self.allocator, candidates.len);

        for (candidates, 0..) |id, i| {
            const candidate_signals = if (signals) |s|
                s[i]
            else
                deriveDefaultSignals(id);

            const score = self.computeWeightedScore(candidate_signals);
            ranked.appendAssumeCapacity(.{
                .id = id,
                .score = score,
            });
        }

        // Sort descending by score
        const items = ranked.items;
        std.mem.sort(RankedCandidate, items, {}, compareByScoreDesc);

        return ranked.toOwnedSlice(self.allocator);
    }

    /// Derive default signal values from a BlockId when no external metadata
    /// is available. Uses a hash of the ID bytes to produce deterministic
    /// but varied baseline scores in [0.0, 1.0].
    fn deriveDefaultSignals(id: core.ids.BlockId) CandidateSignals {
        // Use different byte ranges of the 32-byte ID to derive each signal,
        // giving varied but deterministic defaults.
        return .{
            .semantic_similarity = byteRangeToNorm(id.id[0..8]),
            .recency = byteRangeToNorm(id.id[4..12]),
            .trust_score = byteRangeToNorm(id.id[8..16]),
            .user_pinning = 0.0, // No pinning data without external input
            .project_locality = byteRangeToNorm(id.id[16..24]),
            .profile_preference = byteRangeToNorm(id.id[20..28]),
            .contradiction_penalty = 0.0, // No penalty without external input
            .past_usefulness = byteRangeToNorm(id.id[24..32]),
        };
    }

    fn byteRangeToNorm(bytes: *const [8]u8) f32 {
        const v = std.mem.readInt(u64, bytes, .little);
        return @as(f32, @floatFromInt(v >> 32)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
    }

    fn compareByScoreDesc(_: void, a: RankedCandidate, b: RankedCandidate) bool {
        return a.score > b.score;
    }
};
