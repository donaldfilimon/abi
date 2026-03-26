//! Pipeline WDBX Persistence
//!
//! Adapters for converting pipeline state into WDBX blocks and for
//! persisting/loading modulation state to/from WDBX.

const std = @import("std");
const types = @import("types.zig");
const ctx_mod = @import("context.zig");
const PipelineContext = ctx_mod.PipelineContext;
const BlockConfig = types.BlockConfig;
const BlockChain = types.BlockChain;
const ProfileTag = types.ProfileTag;
const RoutingWeights = types.RoutingWeights;
const IntentCategory = types.IntentCategory;
const PipelineStepTag = types.PipelineStepTag;
const StepKind = types.StepKind;

/// Converts pipeline step state into WDBX BlockConfig and records blocks.
pub const PipelineBlockAdapter = struct {
    /// Map StepKind to PipelineStepTag for block metadata.
    pub fn stepKindToTag(kind: StepKind) PipelineStepTag {
        return switch (kind) {
            .retrieve => .retrieve,
            .template => .template,
            .route => .route,
            .modulate => .modulate,
            .generate => .generate,
            .validate => .validate,
            .store => .store,
            .transform => .transform,
            .filter => .filter,
            .reason => .reason,
        };
    }

    /// Map StepKind to an appropriate IntentCategory.
    pub fn stepKindToIntent(kind: StepKind) IntentCategory {
        return switch (kind) {
            .retrieve => .factual_inquiry,
            .validate => .policy_check,
            .generate => .general,
            .route => .general,
            .modulate => .general,
            .template => .general,
            .store => .general,
            .transform => .general,
            .filter => .general,
            .reason => .factual_inquiry,
        };
    }

    /// Create a BlockConfig from the current pipeline context state.
    pub fn toBlockConfig(pctx: *const PipelineContext, kind: StepKind) BlockConfig {
        // Generate embedding from the most relevant text at this step
        const embed_text = if (pctx.generated_response) |r|
            r
        else if (pctx.rendered_prompt) |p|
            p
        else
            pctx.input;

        const embedding = PipelineContext.hashEmbedding(embed_text);

        // Build profile tag from routing state
        const profile_tag: ProfileTag = if (pctx.primary_profile) |pp|
            .{ .primary_profile = pp }
        else
            .{ .primary_profile = .abbey };

        // Use routing weights if available, else default
        const routing_weights = pctx.routing_weights orelse RoutingWeights{
            .abbey_weight = 1.0,
            .aviva_weight = 0.0,
            .abi_weight = 0.0,
        };

        // Get previous hash from the chain's head block
        var previous_hash: [32]u8 = .{0} ** 32;
        if (pctx.chain) |chain| {
            if (chain.current_head) |head_id| {
                if (chain.blocks.get(head_id)) |head_block| {
                    previous_hash = head_block.hash;
                }
            }
        }

        return .{
            .query_embedding = &embedding,
            .profile_tag = profile_tag,
            .routing_weights = routing_weights,
            .intent = stepKindToIntent(kind),
            .pipeline_step = stepKindToTag(kind),
            .pipeline_id = pctx.pipeline_id,
            .step_index = pctx.current_step,
            .previous_hash = previous_hash,
        };
    }

    /// Record a WDBX block for a pipeline step. Returns the block ID.
    pub fn recordStepBlock(pctx: *PipelineContext, kind: StepKind) !u64 {
        const chain = pctx.chain orelse return 0;
        const config = toBlockConfig(pctx, kind);
        return try chain.addBlock(config);
    }
};

/// Persistence adapter for modulation state in WDBX.
pub const ModulationPersistence = struct {
    /// Serialize a modulation profile preference into embedding values.
    pub fn preferencesToEmbedding(
        abbey_score: f32,
        aviva_score: f32,
        abi_score: f32,
        total_interactions: u32,
    ) [4]f32 {
        const normalized_total: f32 = @min(
            @as(f32, @floatFromInt(total_interactions)) / 1000.0,
            1.0,
        );
        return .{ abbey_score, aviva_score, abi_score, normalized_total };
    }

    /// Extract modulation scores from an embedding.
    pub fn embeddingToPreferences(embedding: []const f32) struct {
        abbey_score: f32,
        aviva_score: f32,
        abi_score: f32,
    } {
        if (embedding.len < 3) return .{ .abbey_score = 0.5, .aviva_score = 0.5, .abi_score = 0.5 };
        return .{
            .abbey_score = embedding[0],
            .aviva_score = embedding[1],
            .abi_score = embedding[2],
        };
    }
};
