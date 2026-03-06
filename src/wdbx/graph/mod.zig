//! Owns explicit relationships.

const std = @import("std");

pub const EdgeKind = enum {
    derived_from,
    summarized_by,
    contradicts,
    supports,
    authored_by,
    belongs_to_project,
    relevant_to_persona,
    references_artifact,
    follows_conversation_turn,
    supersedes,
};

pub const GraphStore = struct {
    // TODO: adjacency lists, forward and reverse traversal, path scoring hooks
};
