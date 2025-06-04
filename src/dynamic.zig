const std = @import("std");

pub const RouterError = error{
    NoPersonasAvailable,
    InvalidQuery,
    ScoringFailed,
};

/// Represents a single conversational persona with basic metrics.
pub const Persona = struct {
    name: []const u8,
    empathy_score: f32,
    glue_accuracy: f32,
    codegen_score: f32,

    pub fn validate(self: Persona) bool {
        return self.empathy_score >= 0 and self.empathy_score <= 1 and
            self.glue_accuracy >= 0 and self.glue_accuracy <= 1 and
            self.codegen_score >= 0 and self.codegen_score <= 1;
    }
};

/// Represents a user query with context information.
pub const Query = struct {
    text: []const u8,
    context: []const u8,

    pub fn validate(self: Query) RouterError!void {
        if (self.text.len == 0) return RouterError.InvalidQuery;
    }
};

/// Placeholder transformer model used to evaluate personas.
pub const TransformerModel = struct {
    /// Score a persona for the given query.
    pub fn scorePersona(self: TransformerModel, persona: Persona, query: Query) RouterError!f32 {
        _ = self;
        if (!persona.validate()) return RouterError.ScoringFailed;
        try query.validate();

        var score: f32 = 0.0;
        const text = query.text;
        const context_weight: f32 = 0.3;

        if (query.context.len > 0) {
            score += persona.glue_accuracy * context_weight;
        }

        if (std.mem.indexOf(u8, text, "code") != null) {
            score += persona.codegen_score * 0.4;
        }
        if (std.mem.indexOf(u8, text, "help") != null) {
            score += persona.empathy_score * 0.4;
        }

        score += persona.glue_accuracy * 0.2;
        return std.math.clamp(score, 0.0, 1.0);
    }
};

/// Router selects the best persona for a given query.
pub const DynamicPersonaRouter = struct {
    personas: []const Persona,
    model: TransformerModel,

    pub fn init(personas: []const Persona) RouterError!DynamicPersonaRouter {
        if (personas.len == 0) return RouterError.NoPersonasAvailable;
        return DynamicPersonaRouter{
            .personas = personas,
            .model = TransformerModel{},
        };
    }

    /// Select a persona based on query context and user needs.
    pub fn select(self: DynamicPersonaRouter, query: Query) RouterError!Persona {
        try query.validate();

        var best_index: usize = 0;
        var best_score: f32 = 0.0;

        for (self.personas, 0..) |persona, i| {
            const score = try self.evaluatePersona(persona, query);
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }

        return self.personas[best_index];
    }

    /// Evaluate persona suitability using the transformer model.
    fn evaluatePersona(self: DynamicPersonaRouter, persona: Persona, query: Query) RouterError!f32 {
        return self.model.scorePersona(persona, query);
    }
};

/// Example usage of the router.
pub fn main() !void {
    const personas = [_]Persona{
        .{ .name = "helper", .empathy_score = 0.9, .glue_accuracy = 0.8, .codegen_score = 0.4 },
        .{ .name = "coder", .empathy_score = 0.6, .glue_accuracy = 0.7, .codegen_score = 0.9 },
        .{ .name = "default", .empathy_score = 0.5, .glue_accuracy = 0.6, .codegen_score = 0.5 },
    };

    var router = DynamicPersonaRouter{
        .personas = personas[0..],
        .model = TransformerModel{},
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const args = try std.process.argsAlloc(gpa.allocator());
    defer std.process.argsFree(gpa.allocator(), args);

    const query_text = if (args.len > 1) args[1] else "I need help with zig";
    const query = Query{ .text = query_text, .context = "" };
    const persona = router.select(query);
    std.debug.print("Selected persona: {s}\n", .{persona.name});
}

test "router selects coder when query mentions code" {
    const personas = [_]Persona{
        .{ .name = "helper", .empathy_score = 0.9, .glue_accuracy = 0.8, .codegen_score = 0.4 },
        .{ .name = "coder", .empathy_score = 0.6, .glue_accuracy = 0.7, .codegen_score = 0.9 },
    };
    var router = DynamicPersonaRouter{
        .personas = personas[0..],
        .model = TransformerModel{},
    };
    const query = Query{ .text = "please show code", .context = "" };
    const persona = try router.select(query);
    try std.testing.expectEqualStrings("coder", persona.name);
}

test "router selects helper for help query" {
    const personas = [_]Persona{
        .{ .name = "helper", .empathy_score = 0.9, .glue_accuracy = 0.8, .codegen_score = 0.4 },
        .{ .name = "coder", .empathy_score = 0.6, .glue_accuracy = 0.7, .codegen_score = 0.9 },
    };
    var router = DynamicPersonaRouter{
        .personas = personas[0..],
        .model = TransformerModel{},
    };
    const query = Query{ .text = "I need help", .context = "" };
    const persona = try router.select(query);
    try std.testing.expectEqualStrings("helper", persona.name);
}
