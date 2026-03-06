//! Defines behavioral policy overlays.

const std = @import("std");

pub const PersonaMode = enum {
    abbey,
    aviva,
    abi,
};

pub const PersonaRouter = struct {
    // TODO: routing from request features, tone/verbosity/retrieval bias policies, moderation or regulation hooks
};
