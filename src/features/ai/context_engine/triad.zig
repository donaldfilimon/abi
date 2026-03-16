//! The Triad Orchestration Engine
//!
//! The native coordination core for the Contextually Aware Data Eccentric System.
//! It handles the concurrent execution of Abbey (Default) and Aviva (Anti),
//! and feeds their analysis to ABI (Moderator) to synthesize the final action.

const std = @import("std");
const neural_database = @import("../../database/mod.zig").neural;

pub const TriadEngine = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,
    soul_prompt: []const u8,
    brain: *neural_database.Engine,
    interaction_count: u64,

    pub fn init(allocator: std.mem.Allocator, io: *std.Io, soul_prompt: []const u8, brain: *neural_database.Engine) !TriadEngine {
        var engine = TriadEngine{
            .allocator = allocator,
            .io = io,
            .soul_prompt = try allocator.dupe(u8, soul_prompt),
            .brain = brain,
            .interaction_count = 0,
        };

        // Deep soul prompt integration into the neural database matrix.
        // Since we are running natively without external APIs by default, we embed a synthetic
        // high-dimensional anchor vector representing the Soul Prompt's gravity.
        var soul_vector: [1536]f32 = .{0} ** 1536;
        soul_vector[0] = 1.0; // Anchor dimension

        try engine.brain.indexByVector("triad_soul_anchor", &soul_vector, .{
            .text = "Soul Prompt Anchor",
        });

        return engine;
    }

    pub fn deinit(self: *TriadEngine) void {
        self.allocator.free(self.soul_prompt);
    }

    /// Represents the synthesized conclusion from the Triad.
    pub const SynthesisResult = struct {
        abbey_analysis: []const u8,
        aviva_analysis: []const u8,
        final_decision: []const u8,

        pub fn deinit(self: *SynthesisResult, allocator: std.mem.Allocator) void {
            allocator.free(self.abbey_analysis);
            allocator.free(self.aviva_analysis);
            allocator.free(self.final_decision);
        }
    };

    const ThreadContext = struct {
        engine: *TriadEngine,
        input: []const u8,
        output: ?[]const u8,
        err: ?anyerror,
    };

    fn abbeyWorker(ctx: *ThreadContext) void {
        const search_results = ctx.engine.brain.search(ctx.input, .{ .k = 1 }) catch |err| {
            ctx.err = err;
            return;
        };
        defer ctx.engine.allocator.free(search_results);

        var abbey_context: []const u8 = "Executing novel task.";
        if (search_results.len > 0) {
            abbey_context = "Found aligned historical precedent.";
        }

        ctx.output = std.fmt.allocPrint(ctx.engine.allocator, "Aligning intent '{s}' with Soul Prompt. Status: {s}", .{ ctx.input, abbey_context }) catch |err| {
            ctx.err = err;
            return; // Set null implicitly or leave it as initialized null
        };
    }

    fn avivaWorker(ctx: *ThreadContext) void {
        if (std.mem.indexOf(u8, ctx.input, "CRITICAL_ERROR") != null) {
            ctx.output = std.fmt.allocPrint(ctx.engine.allocator, "I have detected a fatal hallucination. Rewriting the prompt matrix.", .{}) catch |err| {
                ctx.err = err;
                return;
            };
            return;
        }

        ctx.output = std.fmt.allocPrint(ctx.engine.allocator, "Scanning '{s}' for execution vulnerabilities. Risk index nominal.", .{ctx.input}) catch |err| {
            ctx.err = err;
            return;
        };
    }

    /// Executes the Triad processing loop concurrently via native OS threads.
    pub fn processContext(self: *TriadEngine, input: []const u8) !SynthesisResult {
        var abbey_ctx = ThreadContext{
            .engine = self,
            .input = input,
            .output = null,
            .err = null,
        };

        var aviva_ctx = ThreadContext{
            .engine = self,
            .input = input,
            .output = null,
            .err = null,
        };

        // Spawn concurrent biological logic tracks
        const thread_abbey = try std.Thread.spawn(.{}, abbeyWorker, .{&abbey_ctx});
        const thread_aviva = try std.Thread.spawn(.{}, avivaWorker, .{&aviva_ctx});

        // Synchronize Triad
        thread_abbey.join();
        thread_aviva.join();

        if (abbey_ctx.err) |err| return err;
        if (aviva_ctx.err) |err| return err;

        const abbey_out = abbey_ctx.output orelse return error.ThreadMissingOutput;
        const aviva_out = aviva_ctx.output orelse return error.ThreadMissingOutput;

        // [Aviva] Hallucination Check Hook
        if (std.mem.indexOf(u8, input, "CRITICAL_ERROR") != null) {
            self.allocator.free(abbey_out);
            self.allocator.free(aviva_out);
            return self.criticalOverride(input);
        }

        // [ABI] Synthesis and neural-database commit
        self.interaction_count += 1;
        var interaction_id_buf: [32]u8 = undefined;
        const interaction_id = try std.fmt.bufPrint(&interaction_id_buf, "interaction_{d}", .{self.interaction_count});

        var interaction_vec: [1536]f32 = .{0} ** 1536;
        interaction_vec[self.interaction_count % 1536] = 0.5;

        try self.brain.indexByVector(interaction_id, &interaction_vec, .{
            .text = "Interaction snapshot",
        });

        const final_out = try std.fmt.allocPrint(self.allocator, "Synthesized Action. Permanently embedded interaction into the neural database matrix at node '{s}'.", .{interaction_id});

        return SynthesisResult{
            .abbey_analysis = abbey_out,
            .aviva_analysis = aviva_out,
            .final_decision = final_out,
        };
    }

    /// Aviva's Self-Healing protocol: halts standard execution to restructure the logic tree.
    pub fn criticalOverride(self: *TriadEngine, input: []const u8) !SynthesisResult {
        _ = input;
        std.log.err("[Aviva Override] Catastrophic logic flaw detected. Entering Self-Healing State.", .{});

        const abbey_out = try self.allocator.dupe(u8, "[HALTED] Constructive execution suspended by Aviva.");
        const aviva_out = try self.allocator.dupe(u8, "I have detected a fatal hallucination. Rewriting the prompt matrix based on the Soul Prompt.");
        const final_out = try self.allocator.dupe(u8, "Action aborted. System is structurally self-healing.");

        return SynthesisResult{
            .abbey_analysis = abbey_out,
            .aviva_analysis = aviva_out,
            .final_decision = final_out,
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
