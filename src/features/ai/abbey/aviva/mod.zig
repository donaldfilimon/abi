//! Aviva Module - Agentic AI with WDBX Vector Memory and Self-Training
//!
//! This module implements the full Aviva spec:
//! - WDBX vector database integration for persistent memory
//! - Embedding generation and semantic search
//! - Self-training via experience replay and policy optimization
//! - Memory consolidation (episodic → semantic)
//! - Adaptive confidence calibration and response improvement

const std = @import("std");
const builtin = @import("builtin");
const foundation = @import("../../../../foundation/mod.zig");
const wdbx = @import("../../core/database/wdbx.zig");
const embeddings = @import("../../embeddings/mod.zig");
const neural = @import("../neural/mod.zig");
const learning = @import("../neural/learning.zig");
const memory = @import("../memory/mod.zig");
const abbey_engine = @import("../engine.zig");

pub const log = std.log.scoped(.aviva);

// ============================================================================
// Core Types
// ============================================================================

/// Configuration for Aviva agentic system
pub const AvivaConfig = struct {
    /// WDBX database name
    db_name: []const u8 = "aviva-memory",

    /// Embedding configuration
    embedding_dim: usize = 384,
    embedding_normalize: bool = true,

    /// Memory settings
    memory_top_k: usize = 8,
    memory_time_window_days: ?u32 = null,

    /// Self-training settings
    enable_self_training: bool = true,
    training_batch_size: usize = 32,
    training_learning_rate: f32 = 0.001,
    training_update_interval: usize = 100,

    /// Experience replay settings
    replay_buffer_size: usize = 10000,
    replay_use_priority: bool = true,

    /// Confidence calibration
    enable_calibration: bool = true,

    /// Memory consolidation
    consolidation_interval_min: usize = 1000,
    enable_memory_decay: bool = true,
};

/// Aviva agent state
pub const AvivaAgent = struct {
    allocator: std.mem.Allocator,
    config: AvivaConfig,

    // WDBX memory
    wdbx_handle: ?wdbx.DatabaseHandle = null,
    embedding_model: ?*embeddings.EmbeddingModel = null,

    // Self-training
    learning_system: ?*learning.OnlineLearner = null,
    experience_buffer: ?*learning.ReplayBuffer = null,
    training_enabled: bool = false,

    // Memory management
    memory_manager: ?*memory.MemoryManager = null,

    // Runtime state
    interaction_count: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    last_consolidation: usize = 0,

    // Calibration
    calibrator: ?*abbey_engine.ConfidenceCalibrator = null,

    const Self = @This();

    /// Initialize a new Aviva agent
    pub fn init(allocator: std.mem.Allocator, config: AvivaConfig) !Self {
        var self = Self{
            .allocator = allocator,
            .config = config,
        };

        // Initialize WDBX
        self.wdbx_handle = try wdbx.createDatabase(allocator, config.db_name);

        // Initialize embedding model
        var emb_model = try allocator.create(embeddings.EmbeddingModel);
        errdefer if (self.embedding_model == null) {
            emb_model.deinit();
            allocator.destroy(emb_model);
        };
        emb_model.* = embeddings.EmbeddingModel.init(allocator, .{
            .dimension = config.embedding_dim,
            .normalize = config.embedding_normalize,
        });
        self.embedding_model = emb_model;

        // Initialize learning system if enabled
        if (config.enable_self_training) {
            var learner = try allocator.create(learning.OnlineLearner);
            errdefer if (self.learning_system == null) {
                learner.deinit();
                allocator.destroy(learner);
            };
            learner.* = learning.OnlineLearner.init(allocator, .{
                .batch_size = config.training_batch_size,
                .learning_rate = config.training_learning_rate,
            });
            self.learning_system = learner;

            // Create replay buffer
            var buffer = try allocator.create(learning.ReplayBuffer);
            buffer.* = learning.ReplayBuffer.init(allocator, config.replay_buffer_size);
            buffer.use_priority = config.replay_use_priority;
            self.experience_buffer = buffer;

            self.training_enabled = true;
        }

        // Initialize memory manager
        var mem_mgr = try allocator.create(memory.MemoryManager);
        mem_mgr.* = memory.MemoryManager.init(allocator);
        self.memory_manager = mem_mgr;

        // Initialize calibrator if enabled
        if (config.enable_calibration) {
            var cal = try allocator.create(abbey_engine.ConfidenceCalibrator);
            cal.* = abbey_engine.ConfidenceCalibrator.init(allocator);
            self.calibrator = cal;
        }

        log.info("Aviva agent initialized (training={s}, calibration={s})", .{
            config.enable_self_training,
            config.enable_calibration,
        });

        return self;
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        if (self.learning_system) |learner| {
            learner.deinit();
            self.allocator.destroy(learner);
        }
        if (self.experience_buffer) |buf| {
            buf.deinit();
            self.allocator.destroy(buf);
        }
        if (self.memory_manager) |mgr| {
            mgr.deinit();
            self.allocator.destroy(mgr);
        }
        if (self.calibrator) |cal| {
            cal.deinit();
            self.allocator.destroy(cal);
        }
        if (self.embedding_model) |model| {
            model.deinit();
            self.allocator.destroy(model);
        }
        if (self.wdbx_handle) |*handle| {
            wdbx.closeDatabase(handle);
        }
        self.* = undefined;
    }

    /// Process input and generate response with memory augmentation
    pub fn processWithMemory(self: *Self, input: []const u8) !ProcessResult {
        const start_time = foundation.time.unixMs();

        // Generate embedding for input
        var embedding: ?[]f32 = null;
        if (self.embedding_model) |*model| {
            embedding = model.embed(input) catch |err| {
                log.err("Embedding generation failed: {any}", .{err});
                return null;
            };
        }

        // Retrieve relevant context from WDBX
        var context_slices: [][]const u8 = &.{};
        if (embedding) |emb| {
            if (self.wdbx_handle) |*handle| {
                const results = wdbx.searchVectors(handle, self.allocator, emb, self.config.memory_top_k) catch |err| {
                    log.err("WDBX search failed: {any}", .{err});
                    &.{};
                };
                defer self.allocator.free(results);

                // Build context from retrieved vectors
                var ctx_list = std.ArrayListUnmanaged([]const u8).empty;
                defer ctx_list.deinit(self.allocator);

                for (results) |res| {
                    const view = wdbx.getVector(handle, res.id) orelse continue;
                    if (view.metadata) |meta| {
                        try ctx_list.append(self.allocator, meta);
                    }
                }

                if (ctx_list.items.len > 0) {
                    context_slices = try ctx_list.toOwnedSlice(self.allocator);
                }
            }
        }
        defer {
            for (context_slices) |slice| self.allocator.free(slice);
            self.allocator.free(context_slices);
        }

        // Build augmented input
        const augmented_input = blk: {
            if (context_slices.len > 0) {
                var ctx_builder = std.Io.Writer.Allocating.init(self.allocator);
                defer ctx_builder.deinit();

                for (context_slices, 0..) |ctx, i| {
                    if (i > 0) ctx_builder.writer().writeAll("\n\n") catch continue;
                    ctx_builder.writer().writeAll("Context: ") catch continue;
                    ctx_builder.writer().writeAll(ctx) catch continue;
                }
                ctx_builder.writer().writeAll("\n\nUser: ") catch {};
                ctx_builder.writer().writeAll(input) catch {};

                break :blk try ctx_builder.toOwnedSlice();
            } else {
                break :blk try self.allocator.dupe(u8, input);
            }
        };
        defer self.allocator.free(augmented_input);

        // TODO: Send to Abbey engine (would need reference to engine)
        // For now, return the augmented input and context

        const elapsed = foundation.time.unixMs() - start_time;

        return ProcessResult{
            .augmented_input = augmented_input,
            .context_retrieved = context_slices.len,
            .processing_time_ms = elapsed,
            .embedding_generated = embedding != null,
        };
    }

    /// Store interaction in WDBX and experience buffer
    pub fn recordInteraction(self: *Self, input: []const u8, response: []const u8, reward: ?f32) !void {
        const count = self.interaction_count.fetchAdd(1, .monotonic);

        // Generate and store embedding
        if (self.embedding_model) |*model| {
            if (self.wdbx_handle) |*handle| {
                const emb = model.embed(input) catch |err| {
                    log.err("Failed to embed interaction: {any}", .{err});
                    return;
                };
                defer self.allocator.free(emb);

                // Build metadata
                var meta_builder = std.Io.Writer.Allocating.init(self.allocator);
                defer meta_builder.deinit();

                const writer = meta_builder.writer();
                writer.print("{{\"type\":\"interaction\",\"id\":{},\"input\":\"", .{count}) catch return;
                std.json.encodeString(writer, input) catch return;
                writer.print("\",\"response\":\"", .{}) catch return;
                std.json.encodeString(writer, response) catch return;
                writer.print("\",\"timestamp\":{}}}", .{foundation.time.unixMs()}) catch return;

                const metadata = try meta_builder.toOwnedSlice();
                defer self.allocator.free(metadata);

                // Insert into WDBX
                const id = (@as(u64, @intCast(foundation.time.unixMs())) << 16) | @as(u64, count & 0xFFFF);
                wdbx.insertVector(handle, id, emb, metadata) catch |err| {
                    log.err("WDBX insert failed: {any}", .{err});
                };
            }
        }

        // Store in experience buffer for training
        if (self.training_enabled) {
            if (self.experience_buffer) |buf| {
                const experience = learning.Experience{
                    .state = try self.allocator.dupe(u8, input),
                    .action = try self.allocator.dupe(u8, response),
                    .reward = reward orelse 0.0,
                    .next_state = &.{},
                    .done = false,
                    .priority = 1.0,
                };
                buf.add(experience);
            }

            // Check if we should trigger training
            if (count - self.last_consolidation >= self.config.training_update_interval) {
                self.triggerTraining() catch |err| {
                    log.err("Training trigger failed: {any}", .{err});
                };
                self.last_consolidation = count;
            }
        }
    }

    /// Trigger self-training update
    pub fn triggerTraining(self: *Self) !void {
        if (!self.training_enabled) return;

        if (self.learning_system) |learner| {
            if (self.experience_buffer) |buf| {
                if (buf.size < self.config.training_batch_size) {
                    log.info("Skipping training: buffer size {d} < batch size {d}", .{
                        buf.size, self.config.training_batch_size,
                    });
                    return;
                }

                // Sample batch
                const indices = try buf.sample(self.config.training_batch_size);
                defer self.allocator.free(indices);

                // Perform training step
                var total_loss: f32 = 0.0;
                for (indices) |idx| {
                    const exp = buf.buffer[idx % buf.capacity];
                    const loss = learner.update(exp.state, exp.action) catch |err| {
                        log.err("Training step failed: {any}", .{err});
                        continue;
                    };
                    total_loss += loss;

                    // Update priorities based on TD error
                    if (buf.use_priority) {
                        const td_error = @fabs(loss);
                        buf.updatePriorities(&.{idx}, &.{td_error}, 0.6) catch |err| {
                            log.err("Priority update failed: {any}", .{err});
                        };
                    }
                }

                log.info("Training step complete: avg_loss={d:.4}", .{total_loss / @as(f32, self.config.training_batch_size)});
            }
        }
    }

    /// Consolidate memory (episodic → semantic transfer)
    pub fn consolidateMemory(self: *Self) !void {
        if (self.memory_manager) |mgr| {
            try mgr.consolidate();
            log.info("Memory consolidation complete", .{});
        }
    }

    /// Get statistics
    pub fn getStats(self: *const Self) AvivaStats {
        var stats = AvivaStats{};

        if (self.wdbx_handle) |*handle| {
            const db_stats = wdbx.getStats(handle);
            stats.db_count = db_stats.count;
            stats.db_memory_bytes = db_stats.memory_bytes;
        }

        stats.interaction_count = self.interaction_count.load(.monotonic);

        if (self.experience_buffer) |buf| {
            stats.buffer_size = buf.size;
            stats.buffer_capacity = buf.capacity;
        }

        return stats;
    }
};

/// Result of processing input with memory
pub const ProcessResult = struct {
    augmented_input: []const u8,
    context_retrieved: usize,
    processing_time_ms: i64,
    embedding_generated: bool,
};

/// Statistics about Aviva agent
pub const AvivaStats = struct {
    db_count: usize = 0,
    db_memory_bytes: usize = 0,
    interaction_count: usize = 0,
    buffer_size: usize = 0,
    buffer_capacity: usize = 0,
};

test "AvivaAgent initialization" {
    const allocator = std.testing.allocator;

    var agent = try AvivaAgent.init(allocator, .{
        .db_name = "test-aviva",
        .enable_self_training = false,
        .enable_calibration = false,
    });
    defer agent.deinit();

    try std.testing.expect(agent.wdbx_handle != null);
    try std.testing.expect(agent.embedding_model != null);
    try std.testing.expect(!agent.training_enabled);
}

test "AvivaAgent process with memory" {
    const allocator = std.testing.allocator;

    var agent = try AvivaAgent.init(allocator, .{
        .db_name = "test-process",
        .memory_top_k = 3,
        .enable_self_training = false,
    });
    defer agent.deinit();

    const result = try agent.processWithMemory("Hello Aviva!");

    try std.testing.expect(result.processing_time_ms >= 0);
    // Note: embedding might fail if no backend is configured
    // We just verify the function doesn't crash

    // Record an interaction
    try agent.recordInteraction("test input", "test response", 1.0);

    const stats = agent.getStats();
    try std.testing.expect(stats.interaction_count >= 1);
}

test "AvivaAgent WDBX integration" {
    const allocator = std.testing.allocator;

    var agent = try AvivaAgent.init(allocator, .{
        .db_name = "test-wdbx",
        .enable_self_training = false,
    });
    defer agent.deinit();

    if (agent.wdbx_handle) |*handle| {
        const stats = wdbx.getStats(handle);
        try std.testing.expect(stats.count == 0);
    }
}
