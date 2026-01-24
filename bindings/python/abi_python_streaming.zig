//! Python FFI bindings for streaming inference and training.
//!
//! This module provides C-compatible exports for streaming LLM inference
//! and training pipeline operations, designed for ctypes integration.

const std = @import("std");
const root = @import("../../src/abi.zig");

// Import streaming and training modules
const streaming = root.ai.llm.generation.streaming;
const training = root.ai.training;

/// Global allocator for FFI operations
var global_allocator: std.mem.Allocator = std.heap.page_allocator;

/// Last error message buffer
var last_error: [256]u8 = undefined;
var last_error_len: usize = 0;

fn setLastError(msg: []const u8) void {
    const copy_len = @min(msg.len, last_error.len - 1);
    @memcpy(last_error[0..copy_len], msg[0..copy_len]);
    last_error[copy_len] = 0;
    last_error_len = copy_len;
}

/// Get last error message
pub export fn abi_get_last_error() [*:0]const u8 {
    if (last_error_len == 0) {
        return "No error";
    }
    return @ptrCast(&last_error);
}

// =============================================================================
// C-Compatible Structs for FFI
// =============================================================================

/// C-compatible token event struct
pub const CTokenEvent = extern struct {
    token_id: u32,
    text_ptr: [*c]const u8,
    text_len: u32,
    position: u32,
    is_final: bool,
    timestamp_ns: u64,
};

/// C-compatible streaming config
pub const CStreamingConfig = extern struct {
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    repetition_penalty: f32,
    seed: u64,
};

/// C-compatible training config
pub const CTrainingConfig = extern struct {
    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
    optimizer: u32, // 0=sgd, 1=adam, 2=adamw
    weight_decay: f32,
    gradient_clip_norm: f32,
    warmup_steps: u32,
    checkpoint_interval: u32,
};

/// C-compatible training metrics (per step)
pub const CTrainingMetrics = extern struct {
    step: u32,
    epoch: u32,
    loss: f32,
    accuracy: f32,
    learning_rate: f32,
    gradient_norm: f32,
};

/// C-compatible training report (final)
pub const CTrainingReport = extern struct {
    epochs: u32,
    batches: u32,
    final_loss: f32,
    final_accuracy: f32,
    best_loss: f32,
    gradient_updates: u64,
    checkpoints_saved: u32,
    early_stopped: bool,
    total_time_ms: u64,
};

// =============================================================================
// Handle Registries
// =============================================================================

/// Streaming handle entry
const StreamHandle = struct {
    config: streaming.StreamingConfig,
    prompt: []const u8,
    current_position: u32,
    is_cancelled: bool,
    text_buffer: [512]u8,
    text_len: usize,
};

/// Training handle entry
const TrainHandle = struct {
    config: training.TrainingConfig,
    current_step: u32,
    current_epoch: u32,
    is_running: bool,
    accumulated_loss: f32,
    step_count: u32,
};

/// Handle registries
var stream_handles: std.AutoHashMap(i32, *StreamHandle) = undefined;
var train_handles: std.AutoHashMap(i32, *TrainHandle) = undefined;
var next_handle_id: std.atomic.Value(i32) = std.atomic.Value(i32).init(1);
var registries_initialized: bool = false;

fn ensureRegistriesInitialized() void {
    if (!registries_initialized) {
        stream_handles = std.AutoHashMap(i32, *StreamHandle).init(global_allocator);
        train_handles = std.AutoHashMap(i32, *TrainHandle).init(global_allocator);
        registries_initialized = true;
    }
}

fn getNextHandleId() i32 {
    return next_handle_id.fetchAdd(1, .monotonic);
}

// =============================================================================
// Streaming API Exports
// =============================================================================

/// Create a new streaming session
pub export fn abi_llm_stream_create(
    prompt: [*:0]const u8,
    max_tokens: u32,
    temperature: f32,
) i32 {
    ensureRegistriesInitialized();

    const prompt_slice = std.mem.sliceTo(prompt, 0);

    // Allocate handle
    const handle = global_allocator.create(StreamHandle) catch {
        setLastError("Failed to allocate stream handle");
        return -1;
    };

    // Copy prompt
    const prompt_copy = global_allocator.dupe(u8, prompt_slice) catch {
        global_allocator.destroy(handle);
        setLastError("Failed to copy prompt");
        return -1;
    };

    handle.* = .{
        .config = .{
            .max_tokens = max_tokens,
            .temperature = temperature,
            .top_k = 40,
            .top_p = 0.9,
            .repetition_penalty = 1.1,
            .seed = 0,
        },
        .prompt = prompt_copy,
        .current_position = 0,
        .is_cancelled = false,
        .text_buffer = undefined,
        .text_len = 0,
    };

    const handle_id = getNextHandleId();
    stream_handles.put(handle_id, handle) catch {
        global_allocator.free(prompt_copy);
        global_allocator.destroy(handle);
        setLastError("Failed to register stream handle");
        return -1;
    };

    return handle_id;
}

/// Create streaming session with full config
pub export fn abi_llm_stream_create_ex(
    prompt: [*:0]const u8,
    config: *const CStreamingConfig,
) i32 {
    ensureRegistriesInitialized();

    const prompt_slice = std.mem.sliceTo(prompt, 0);

    const handle = global_allocator.create(StreamHandle) catch {
        setLastError("Failed to allocate stream handle");
        return -1;
    };

    const prompt_copy = global_allocator.dupe(u8, prompt_slice) catch {
        global_allocator.destroy(handle);
        setLastError("Failed to copy prompt");
        return -1;
    };

    handle.* = .{
        .config = .{
            .max_tokens = config.max_tokens,
            .temperature = config.temperature,
            .top_k = config.top_k,
            .top_p = config.top_p,
            .repetition_penalty = config.repetition_penalty,
            .seed = config.seed,
        },
        .prompt = prompt_copy,
        .current_position = 0,
        .is_cancelled = false,
        .text_buffer = undefined,
        .text_len = 0,
    };

    const handle_id = getNextHandleId();
    stream_handles.put(handle_id, handle) catch {
        global_allocator.free(prompt_copy);
        global_allocator.destroy(handle);
        setLastError("Failed to register stream handle");
        return -1;
    };

    return handle_id;
}

/// Get next token from streaming session (returns false when done)
pub export fn abi_llm_stream_next(stream_id: i32, event_out: *CTokenEvent) bool {
    const handle = stream_handles.get(stream_id) orelse {
        setLastError("Invalid stream handle");
        return false;
    };

    if (handle.is_cancelled) {
        return false;
    }

    // Check if we've reached max tokens
    if (handle.current_position >= handle.config.max_tokens) {
        event_out.* = .{
            .token_id = 0,
            .text_ptr = null,
            .text_len = 0,
            .position = handle.current_position,
            .is_final = true,
            .timestamp_ns = 0,
        };
        return false;
    }

    // Simulate token generation (in real implementation, this would call the model)
    // For now, generate tokens from the prompt words
    const words = [_][]const u8{ "The", " ", "AI", " ", "assistant", " ", "responds", ":", " " };
    const word_idx = handle.current_position % words.len;
    const word = words[word_idx];

    // Copy word to buffer
    const copy_len = @min(word.len, handle.text_buffer.len);
    @memcpy(handle.text_buffer[0..copy_len], word[0..copy_len]);
    handle.text_len = copy_len;

    const is_final = handle.current_position + 1 >= handle.config.max_tokens;

    event_out.* = .{
        .token_id = @intCast(handle.current_position),
        .text_ptr = &handle.text_buffer,
        .text_len = @intCast(handle.text_len),
        .position = handle.current_position,
        .is_final = is_final,
        .timestamp_ns = @intCast(std.time.nanoTimestamp()),
    };

    handle.current_position += 1;
    return !is_final;
}

/// Cancel an active streaming session
pub export fn abi_llm_stream_cancel(stream_id: i32) void {
    if (stream_handles.get(stream_id)) |handle| {
        handle.is_cancelled = true;
    }
}

/// Destroy a streaming session and free resources
pub export fn abi_llm_stream_destroy(stream_id: i32) void {
    if (stream_handles.fetchRemove(stream_id)) |kv| {
        const handle = kv.value;
        global_allocator.free(handle.prompt);
        global_allocator.destroy(handle);
    }
}

// =============================================================================
// Training API Exports
// =============================================================================

/// Create a new training session
pub export fn abi_train_create(config: *const CTrainingConfig) i32 {
    ensureRegistriesInitialized();

    const handle = global_allocator.create(TrainHandle) catch {
        setLastError("Failed to allocate training handle");
        return -1;
    };

    const optimizer: training.OptimizerType = switch (config.optimizer) {
        0 => .sgd,
        1 => .adam,
        else => .adamw,
    };

    handle.* = .{
        .config = .{
            .epochs = config.epochs,
            .batch_size = config.batch_size,
            .learning_rate = config.learning_rate,
            .optimizer = optimizer,
            .weight_decay = config.weight_decay,
            .gradient_clip_norm = config.gradient_clip_norm,
            .warmup_steps = config.warmup_steps,
            .checkpoint_interval = config.checkpoint_interval,
        },
        .current_step = 0,
        .current_epoch = 0,
        .is_running = true,
        .accumulated_loss = 0,
        .step_count = 0,
    };

    const handle_id = getNextHandleId();
    train_handles.put(handle_id, handle) catch {
        global_allocator.destroy(handle);
        setLastError("Failed to register training handle");
        return -1;
    };

    return handle_id;
}

/// Run one training step (returns false when training complete)
pub export fn abi_train_step(trainer_id: i32, metrics_out: *CTrainingMetrics) bool {
    const handle = train_handles.get(trainer_id) orelse {
        setLastError("Invalid trainer handle");
        return false;
    };

    if (!handle.is_running) {
        return false;
    }

    // Calculate total steps
    const steps_per_epoch: u32 = 100; // Simulated
    const total_steps = handle.config.epochs * steps_per_epoch;

    if (handle.current_step >= total_steps) {
        handle.is_running = false;
        return false;
    }

    // Simulate training step with decreasing loss
    const progress = @as(f32, @floatFromInt(handle.current_step)) / @as(f32, @floatFromInt(total_steps));
    const loss = 2.0 * (1.0 - progress) + 0.1 * @as(f32, @floatFromInt(handle.current_step % 10)) * 0.01;
    const accuracy = 0.5 + 0.4 * progress;

    // Calculate current learning rate with warmup
    var lr = handle.config.learning_rate;
    if (handle.current_step < handle.config.warmup_steps) {
        lr = handle.config.learning_rate * @as(f32, @floatFromInt(handle.current_step + 1)) / @as(f32, @floatFromInt(handle.config.warmup_steps));
    }

    handle.accumulated_loss += loss;
    handle.step_count += 1;

    metrics_out.* = .{
        .step = handle.current_step,
        .epoch = handle.current_step / steps_per_epoch,
        .loss = loss,
        .accuracy = accuracy,
        .learning_rate = lr,
        .gradient_norm = 0.5 + 0.3 * (1.0 - progress),
    };

    handle.current_step += 1;
    handle.current_epoch = handle.current_step / steps_per_epoch;

    return handle.current_step < total_steps;
}

/// Save training checkpoint
pub export fn abi_train_save_checkpoint(trainer_id: i32, path: [*:0]const u8) i32 {
    const handle = train_handles.get(trainer_id) orelse {
        setLastError("Invalid trainer handle");
        return -1;
    };

    _ = path; // Would write checkpoint to this path
    _ = handle;

    // In real implementation, serialize training state to file
    return 0;
}

/// Get final training report
pub export fn abi_train_get_report(trainer_id: i32, report_out: *CTrainingReport) void {
    const handle = train_handles.get(trainer_id) orelse {
        setLastError("Invalid trainer handle");
        return;
    };

    const avg_loss = if (handle.step_count > 0) handle.accumulated_loss / @as(f32, @floatFromInt(handle.step_count)) else 0;

    report_out.* = .{
        .epochs = handle.current_epoch,
        .batches = handle.current_step,
        .final_loss = avg_loss,
        .final_accuracy = 0.9, // Would be calculated
        .best_loss = avg_loss * 0.8,
        .gradient_updates = handle.current_step,
        .checkpoints_saved = handle.current_step / @max(1, handle.config.checkpoint_interval),
        .early_stopped = false,
        .total_time_ms = handle.current_step * 10, // Simulated
    };
}

/// Destroy training session
pub export fn abi_train_destroy(trainer_id: i32) void {
    if (train_handles.fetchRemove(trainer_id)) |kv| {
        global_allocator.destroy(kv.value);
    }
}

// =============================================================================
// Tests
// =============================================================================

test "streaming create and destroy" {
    ensureRegistriesInitialized();

    const handle_id = abi_llm_stream_create("Hello, world!", 10, 0.7);
    try std.testing.expect(handle_id > 0);

    abi_llm_stream_destroy(handle_id);
}

test "streaming iteration" {
    ensureRegistriesInitialized();

    const handle_id = abi_llm_stream_create("Test prompt", 5, 0.7);
    try std.testing.expect(handle_id > 0);
    defer abi_llm_stream_destroy(handle_id);

    var event: CTokenEvent = undefined;
    var count: u32 = 0;

    while (abi_llm_stream_next(handle_id, &event)) {
        count += 1;
        if (count > 100) break; // Safety limit
    }

    try std.testing.expect(count > 0);
}

test "training create and step" {
    ensureRegistriesInitialized();

    var config = CTrainingConfig{
        .epochs = 2,
        .batch_size = 32,
        .learning_rate = 0.001,
        .optimizer = 2, // adamw
        .weight_decay = 0.01,
        .gradient_clip_norm = 1.0,
        .warmup_steps = 10,
        .checkpoint_interval = 50,
    };

    const trainer_id = abi_train_create(&config);
    try std.testing.expect(trainer_id > 0);
    defer abi_train_destroy(trainer_id);

    var metrics: CTrainingMetrics = undefined;
    var step_count: u32 = 0;

    while (abi_train_step(trainer_id, &metrics)) {
        step_count += 1;
        if (step_count > 500) break; // Safety limit
    }

    try std.testing.expect(step_count > 0);

    var report: CTrainingReport = undefined;
    abi_train_get_report(trainer_id, &report);
    try std.testing.expect(report.batches > 0);
}
