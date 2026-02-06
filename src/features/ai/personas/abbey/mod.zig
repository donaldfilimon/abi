//! Abbey Persona - Empathetic Polymath
//!
//! Abbey combines emotional intelligence with deep technical expertise.
//! This module wraps the core Abbey implementation to conform to the
//! Multi-Persona Assistant interface.
//!
//! Enhanced Features:
//! - Emotion processing and detection
//! - Empathy injection for warm, supportive responses
//! - Step-by-step reasoning chains
//! - Tone adaptation based on user emotional state

const std = @import("std");
const time = @import("../../../../services/shared/time.zig");
const sync = @import("../../../../services/shared/sync.zig");
const types = @import("../types.zig");
const config = @import("../config.zig");
const core_types = @import("../../core/types.zig");
const abbey_core = @import("../../abbey/mod.zig");

// Import enhanced modules
pub const emotion_mod = @import("emotion.zig");
pub const empathy_mod = @import("empathy.zig");
pub const reasoning_mod = @import("reasoning.zig");

// Re-export key types
pub const EmotionProcessor = emotion_mod.EmotionProcessor;
pub const EmotionalResponse = emotion_mod.EmotionalResponse;
pub const ToneStyle = emotion_mod.ToneStyle;
pub const EmpathyInjector = empathy_mod.EmpathyInjector;
pub const EmpathyInjection = empathy_mod.EmpathyInjection;
pub const ReasoningEngine = reasoning_mod.ReasoningEngine;
pub const ReasoningChain = reasoning_mod.ReasoningChain;
pub const ReasoningStep = reasoning_mod.ReasoningStep;

/// Abbey persona implementation with enhanced emotion processing.
pub const AbbeyPersona = struct {
    allocator: std.mem.Allocator,
    config: config.AbbeyConfig,
    /// The underlying Abbey engine instance.
    engine: *abbey_core.Abbey,
    /// Emotion processor for detecting user emotions.
    emotion_processor: EmotionProcessor,
    /// Empathy injector for response enhancement.
    empathy_injector: EmpathyInjector,
    /// Reasoning engine for transparent thought processes.
    reasoning_engine: ReasoningEngine,
    /// Current emotional state tracking.
    current_emotional_state: core_types.EmotionalState,

    const Self = @This();

    /// Initialize the Abbey persona with configuration.
    pub fn init(allocator: std.mem.Allocator, cfg: config.AbbeyConfig) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Convert persona config to legacy config for the engine
        // In a full implementation, the engine would be refactored to use the new config.
        const engine_cfg = abbey_core.Abbey.LegacyConfig{
            .name = "Abbey",
            .enable_emotions = cfg.emotion_adaptation,
            .enable_reasoning_log = cfg.include_reasoning,
            .max_reasoning_steps = cfg.max_reasoning_steps,
        };

        const engine = try allocator.create(abbey_core.Abbey);
        errdefer allocator.destroy(engine);
        engine.* = try abbey_core.Abbey.init(allocator, engine_cfg);

        // Initialize emotion processor
        const emotion_config = emotion_mod.EmotionConfig{
            .min_intensity_threshold = 0.3,
            .recency_weight = 0.7,
            .track_trajectory = true,
        };

        // Initialize empathy injector
        const empathy_config = empathy_mod.EmpathyConfig{
            .min_acknowledgment_threshold = if (cfg.emotion_adaptation) 0.4 else 0.8,
            .include_transitions = true,
            .include_encouragement = true,
        };

        // Initialize reasoning engine
        const reasoning_config = reasoning_mod.ReasoningConfig{
            .max_steps = cfg.max_reasoning_steps,
            .emotion_aware = cfg.emotion_adaptation,
            .show_intermediate_steps = cfg.include_reasoning,
            .output_format = .detailed,
        };

        self.* = .{
            .allocator = allocator,
            .config = cfg,
            .engine = engine,
            .emotion_processor = EmotionProcessor.initWithConfig(allocator, emotion_config),
            .empathy_injector = EmpathyInjector.initWithConfig(allocator, empathy_config),
            .reasoning_engine = ReasoningEngine.initWithConfig(allocator, reasoning_config),
            .current_emotional_state = .{},
        };

        return self;
    }

    /// Shutdown the persona and free resources.
    pub fn deinit(self: *Self) void {
        self.emotion_processor.deinit();
        self.engine.deinit();
        self.allocator.destroy(self.engine);
        self.allocator.destroy(self);
    }

    pub fn getName(_: *const Self) []const u8 {
        return "Abbey";
    }

    pub fn getType(_: *const Self) types.PersonaType {
        return .abbey;
    }

    /// Process a request using Abbey's empathetic and technical logic.
    pub fn process(self: *Self, request: types.PersonaRequest) anyerror!types.PersonaResponse {
        var timer = time.Timer.start() catch {
            return error.TimerFailed;
        };

        // Step 1: Process emotions from the request
        const emotional_response = try self.emotion_processor.process(
            request.content,
            self.current_emotional_state,
        );

        // Update emotional state for tracking
        self.current_emotional_state.update(
            emotional_response.primary_emotion,
            emotional_response.intensity,
        );

        // Step 2: Generate empathy injection based on emotions
        var empathy_injection = try self.empathy_injector.inject(emotional_response, null);
        defer self.empathy_injector.freeInjection(&empathy_injection);

        // Step 3: Perform reasoning with emotional context
        var reasoning_chain: ?ReasoningChain = null;
        if (self.config.include_reasoning) {
            reasoning_chain = try self.reasoning_engine.reason(
                request.content,
                .{},
                emotional_response,
            );
        }
        defer if (reasoning_chain) |*rc| rc.deinit(self.allocator);

        // Step 4: Process through the core Abbey engine
        var input = request.content;
        var owned_input: ?[]const u8 = null;
        if (request.system_instruction) |instruction| {
            const combined = try std.fmt.allocPrint(
                self.allocator,
                "{s}\n\n{s}",
                .{ instruction, request.content },
            );
            owned_input = combined;
            input = combined;
        }
        defer if (owned_input) |buf| self.allocator.free(buf);

        const legacy_resp = try self.engine.process(input);

        // Step 5: Build the enhanced response
        var content_builder: std.ArrayListUnmanaged(u8) = .{};
        errdefer content_builder.deinit(self.allocator);

        // Add empathy prefix if configured and applicable
        if (self.config.emotion_adaptation and empathy_injection.prefix.len > 0) {
            try content_builder.appendSlice(self.allocator, empathy_injection.prefix);
        }

        // Add main content
        try content_builder.appendSlice(self.allocator, legacy_resp.content);

        // Add empathy suffix if applicable
        if (self.config.emotion_adaptation and empathy_injection.suffix.len > 0) {
            try content_builder.appendSlice(self.allocator, empathy_injection.suffix);
        }

        const elapsed_ms = timer.read() / std.time.ns_per_ms;

        // Build response
        var response = types.PersonaResponse{
            .content = try content_builder.toOwnedSlice(self.allocator),
            .persona = .abbey,
            .confidence = legacy_resp.confidence.score,
            .emotional_tone = emotional_response.primary_emotion,
            .generation_time_ms = elapsed_ms,
        };

        // Extract reasoning chain if available
        if (self.config.include_reasoning) {
            if (reasoning_chain) |rc| {
                const step_count = rc.stepCount();
                if (step_count > 0) {
                    var chain = try self.allocator.alloc(types.ReasoningStep, step_count);
                    for (rc.steps.items, 0..) |step, i| {
                        chain[i] = .{
                            .title = try self.allocator.dupe(u8, step.title),
                            .explanation = try self.allocator.dupe(u8, step.explanation),
                            .confidence = step.confidence,
                        };
                    }
                    response.reasoning_chain = chain;
                }
            } else if (legacy_resp.reasoning_summary.len > 0) {
                // Fall back to legacy reasoning
                var chain = try self.allocator.alloc(types.ReasoningStep, 1);
                chain[0] = .{
                    .title = try self.allocator.dupe(u8, "Analysis"),
                    .explanation = try self.allocator.dupe(u8, legacy_resp.reasoning_summary),
                    .confidence = legacy_resp.confidence.score,
                };
                response.reasoning_chain = chain;
            }
        }

        return response;
    }

    /// Process with explicit emotional context override.
    pub fn processWithEmotion(
        self: *Self,
        request: types.PersonaRequest,
        emotion_override: EmotionalResponse,
    ) !types.PersonaResponse {
        // Override current emotional state
        self.current_emotional_state.update(
            emotion_override.primary_emotion,
            emotion_override.intensity,
        );

        // Process with standard flow (will use updated state)
        return self.process(request);
    }

    /// Get the current emotional trajectory.
    pub fn getEmotionalTrajectory(self: *const Self) emotion_mod.EmotionTrajectory {
        return self.emotion_processor.getTrajectoryTrend();
    }

    /// Get suggested tone based on current emotional state.
    pub fn getSuggestedTone(self: *const Self) ToneStyle {
        return self.emotion_processor.suggestTone(self.current_emotional_state.current);
    }

    /// Get empathy level for current state.
    pub fn getCurrentEmpathyLevel(self: *const Self) f32 {
        return self.emotion_processor.calibrateEmpathy(
            self.current_emotional_state.current,
            self.current_emotional_state.intensity,
        );
    }

    /// Create the interface wrapper for this persona.
    pub fn interface(self: *Self) types.PersonaInterface {
        return .{
            .ptr = self,
            .vtable = &.{
                .process = @ptrCast(&process),
                .getName = @ptrCast(&getName),
                .getType = @ptrCast(&getType),
            },
        };
    }
};

// Tests

test "abbey persona initialization" {
    const cfg = config.AbbeyConfig{
        .emotion_adaptation = true,
        .include_reasoning = true,
        .max_reasoning_steps = 5,
    };

    // Note: This test would need mocked dependencies in practice
    // Just verify types compile correctly
    _ = cfg;
    _ = AbbeyPersona;
}

test "emotion module re-exports" {
    // Verify re-exports work
    _ = EmotionProcessor;
    _ = EmotionalResponse;
    _ = ToneStyle;
}

test "empathy module re-exports" {
    _ = EmpathyInjector;
    _ = EmpathyInjection;
}

test "reasoning module re-exports" {
    _ = ReasoningEngine;
    _ = ReasoningChain;
    _ = ReasoningStep;
}
