//! Shared enums, tags, and foundational structs.

const std = @import("std");

pub const BlockKind = enum {
    text_summary,
    vector_embedding,
    relationship_list,
    code_snippet,
    metadata_structure,
    message_fragment,
    task_artifact,
};
