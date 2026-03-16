//! Enhanced Routing Logic Module
//!
//! Re-exports the enhanced persona routing framework which implements
//! multi-persona routing, mathematical blending, and WDBX integration.

const enhanced = @import("enhanced.zig");

// Re-export all public types and functions
pub const EnhancedRoutingDecision = enhanced.EnhancedRoutingDecision;
pub const PersonaScores = enhanced.PersonaScores;
pub const IntentClassifier = enhanced.IntentClassifier;
pub const IntentCategory = enhanced.IntentCategory;
pub const EnhancedRouter = enhanced.EnhancedRouter;

test {
    _ = enhanced;
}
