//! AI profile router hub — re-exports leaf modules so callers keep
//! `@import("router.zig")` stable after the strangler extract.
//!
//! Leaves:
//! - `router_weights.zig` — ProfileWeights + blendWeights
//! - `router_keywords.zig` — SentimentKeyword table
//! - `router_sentiment.zig` — keyword/explicit sentiment analysis
//! - `router_profiles.zig` — abbey / aviva / abi_profile processInput
//! - `router_modulator.zig` — AdaptiveModulator EMA persistence
//! - `router_route.zig` — routeInput / routeInputWithSoul / routeToProfile

const std = @import("std");

const weights = @import("router_weights.zig");
const keywords = @import("router_keywords.zig");
const sentiment = @import("router_sentiment.zig");
const profiles = @import("router_profiles.zig");
const modulator = @import("router_modulator.zig");
const route = @import("router_route.zig");

pub const ProfileWeights = weights.ProfileWeights;
pub const blendWeights = weights.blendWeights;

pub const SentimentKeyword = keywords.SentimentKeyword;
pub const SENTIMENT_KEYWORDS = keywords.SENTIMENT_KEYWORDS;

pub const explicitProfileSelector = sentiment.explicitProfileSelector;
pub const analyzeSentiment = sentiment.analyzeSentiment;
pub const selectBestProfile = sentiment.selectBestProfile;

pub const abbey = profiles.abbey;
pub const aviva = profiles.aviva;
pub const abi_profile = profiles.abi_profile;

pub const AdaptiveModulator = modulator.AdaptiveModulator;

pub const routeToProfile = route.routeToProfile;
pub const routeInput = route.routeInput;
pub const routeInputWithSoul = route.routeInputWithSoul;

test {
    _ = @import("router_weights.zig");
    _ = @import("router_keywords.zig");
    _ = @import("router_sentiment.zig");
    _ = @import("router_profiles.zig");
    _ = @import("router_modulator.zig");
    _ = @import("router_route.zig");
    std.testing.refAllDecls(@This());
}
