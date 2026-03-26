//! Pipeline DSL Types
//!
//! Shared types used by both mod.zig and stub.zig for the Abbey Dynamic Model
//! prompt pipeline. Each pipeline step is a typed operation that reads/writes
//! WDBX blocks with cryptographic integrity.

const std = @import("std");
const build_options = @import("build_options");
const db_feature = if (build_options.feat_database) @import("../../database/mod.zig") else @import("../../database/stub.zig");
const block_chain = db_feature.memory.block_chain;
pub const BlockChain = block_chain.BlockChain;
pub const BlockConfig = block_chain.BlockConfig;
pub const ProfileTag = block_chain.ProfileTag;
pub const RoutingWeights = block_chain.RoutingWeights;
pub const IntentCategory = block_chain.IntentCategory;
pub const PolicyFlags = block_chain.PolicyFlags;
pub const PipelineStepTag = block_chain.PipelineStepTag;

/// Identifies each kind of pipeline step.
pub const StepKind = enum {
    retrieve,
    template,
    route,
    modulate,
    generate,
    validate,
    store,
    transform,
    filter,
    reason,
};

/// Source for retrieval steps.
pub const RetrieveSource = enum {
    wdbx,
};

/// Configuration for the retrieve step.
pub const RetrieveConfig = struct {
    source: RetrieveSource = .wdbx,
    k: u32 = 5,
    apply_recency_decay: bool = true,
};

/// Configuration for the template step.
pub const TemplateConfig = struct {
    template_str: []const u8,
};

/// Routing strategy selection.
pub const RouteStrategy = enum {
    heuristic,
    abi_backed,
    adaptive,
};

/// Configuration for the route step.
pub const RouteConfig = struct {
    strategy: RouteStrategy = .adaptive,
};

/// Generation mode.
pub const GenerateMode = enum {
    blocking,
    streaming,
};

/// Configuration for the generate step.
pub const GenerateConfig = struct {
    mode: GenerateMode = .blocking,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
};

/// Validation target.
pub const ValidateTarget = enum {
    constitution,
};

/// Configuration for the validate step.
pub const ValidateConfig = struct {
    target: ValidateTarget = .constitution,
    fallback_on_failure: bool = true,
};

/// Store target.
pub const StoreTarget = enum {
    wdbx,
};

/// Configuration for the store step.
pub const StoreConfig = struct {
    target: StoreTarget = .wdbx,
};

/// Configuration for the modulate step.
pub const ModulateConfig = struct {
    /// Override session_id (defaults to pipeline session_id).
    session_id_override: ?[]const u8 = null,
};

/// Configuration for the reason step.
pub const ReasonConfig = struct {
    max_steps: u32 = 10,
};

/// User-provided transform function type.
pub const TransformFn = *const fn ([]const u8, std.mem.Allocator) anyerror![]const u8;

/// Configuration for the transform step.
pub const TransformConfig = struct {
    transform_fn: TransformFn,
};

/// User-provided filter predicate type (forward-declared context).
pub const FilterFn = *const fn (*const anyopaque) bool;

/// Configuration for the filter step.
pub const FilterConfig = struct {
    predicate: FilterFn,
    halt_on_false: bool = true,
};

/// A single pipeline step with its configuration.
pub const Step = struct {
    kind: StepKind,
    config: StepConfig,
};

/// Tagged union of all step configurations.
pub const StepConfig = union(StepKind) {
    retrieve: RetrieveConfig,
    template: TemplateConfig,
    route: RouteConfig,
    modulate: ModulateConfig,
    generate: GenerateConfig,
    validate: ValidateConfig,
    store: StoreConfig,
    transform: TransformConfig,
    filter: FilterConfig,
    reason: ReasonConfig,
};

/// Result of a complete pipeline execution.
pub const PipelineResult = struct {
    /// The final generated response (if any).
    response: ?[]const u8 = null,
    /// IDs of all WDBX blocks created during execution.
    block_ids: []const u64 = &.{},
    /// The pipeline execution ID.
    pipeline_id: u64 = 0,
    /// Number of steps executed.
    steps_executed: u16 = 0,
    /// Whether constitution validation passed.
    validation_passed: bool = true,
    /// Total execution time in milliseconds.
    elapsed_ms: u64 = 0,

    allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: *PipelineResult) void {
        if (self.allocator) |alloc| {
            if (self.response) |r| alloc.free(r);
            if (self.block_ids.len > 0) alloc.free(self.block_ids);
        }
    }
};

/// Errors that can occur during pipeline execution.
pub const PipelineError = error{
    PipelineNotBuilt,
    StepFailed,
    RetrievalFailed,
    TemplateFailed,
    RoutingFailed,
    GenerationFailed,
    ValidationFailed,
    StoreFailed,
    FilterHalted,
    NoBlockChain,
    FeatureDisabled,
    OutOfMemory,
};
