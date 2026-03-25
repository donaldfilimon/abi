//! Shared types for the reasoning module.
//!
//! Used by mod.zig (via engine.zig) and stub.zig to prevent type drift
//! between enabled and disabled paths.
//!
//! Source of truth: engine.zig StepType definition.

/// Type of reasoning step
pub const StepType = enum {
    /// Initial assessment of the query
    assessment,
    /// Breaking down the problem
    decomposition,
    /// Retrieving relevant information
    retrieval,
    /// Analyzing retrieved information
    analysis,
    /// Synthesizing conclusions
    synthesis,
    /// Indicating research is needed
    research,
    /// Validating the response
    validation,
    /// Formulating the final response
    response,

    pub fn toString(self: StepType) []const u8 {
        return switch (self) {
            .assessment => "Assessment",
            .decomposition => "Decomposition",
            .retrieval => "Retrieval",
            .analysis => "Analysis",
            .synthesis => "Synthesis",
            .research => "Research Needed",
            .validation => "Validation",
            .response => "Response Formation",
        };
    }

    pub fn getEmoji(self: StepType) []const u8 {
        return switch (self) {
            .assessment => "[?]",
            .decomposition => "[/]",
            .retrieval => "[>]",
            .analysis => "[~]",
            .synthesis => "[+]",
            .research => "[!]",
            .validation => "[v]",
            .response => "[=]",
        };
    }
};
