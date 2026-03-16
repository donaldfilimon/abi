//! Abbey reasoning compatibility shim.
//!
//! The Abbey package historically imported `abbey/reasoning.zig` directly.
//! Keep that path stable by forwarding to the canonical reasoning module.

pub const mod = @import("../reasoning/mod.zig");

pub const Confidence = mod.engine.Confidence;
pub const ConfidenceLevel = mod.engine.ConfidenceLevel;
pub const ReasoningChain = mod.ReasoningChain;
pub const ReasoningStep = mod.ReasoningStep;
pub const StepType = mod.StepType;
pub const ReasoningConfig = mod.ReasoningConfig;
pub const Context = mod.Context;
pub const confidenceToDisplayString = mod.engine.confidenceToDisplayString;
