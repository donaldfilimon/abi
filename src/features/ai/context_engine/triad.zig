//! The Triad Orchestration Engine
//!
//! The native coordination core for the Contextually Aware Data Eccentric System.
//! It handles the concurrent execution of Abbey (Default) and Aviva (Anti),
//! and feeds their analysis to ABI (Moderator) to synthesize the final action.

const std = @import("std");

pub const TriadEngine = struct {
    allocator: std.mem.Allocator,
    io: *std.Io,
    soul_prompt: []const u8,

    pub fn init(allocator: std.mem.Allocator, io: *std.Io, soul_prompt: []const u8) !TriadEngine {
        return .{
            .allocator = allocator,
            .io = io,
            .soul_prompt = try allocator.dupe(u8, soul_prompt),
        };
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

    /// Executes the Triad processing loop on a given input context.
    /// In a real system, Abbey and Aviva run concurrently via std.Io tasks.
    pub fn processContext(self: *TriadEngine, input: []const u8) !SynthesisResult {
        // [Abbey] Constructive Analysis
        const abbey_out = try std.fmt.allocPrint(
            self.allocator,
            "Executing task based on intent: '{s}'. Aligning with Soul Prompt: {s}",
            .{ input, self.soul_prompt[0..@min(self.soul_prompt.len, 30)] }
        );

        // [Aviva] Contrarian Analysis
        const aviva_out = try std.fmt.allocPrint(
            self.allocator,
            "Scanning '{s}' for vulnerabilities. Rejecting assumptions.",
            .{ input }
        );

        // [ABI] Synthesis
        const final_out = try std.fmt.allocPrint(
            self.allocator,
            "Synthesized Action: Extracted core intent while mitigating risks highlighted by Aviva.",
            .{}
        );

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
