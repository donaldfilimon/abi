//! AI Chatbot with Multi-Persona Support and Vector Context
//!
//! This example demonstrates a sophisticated AI chatbot that uses:
//! - Multiple AI personas with different characteristics
//! - Vector-based context search for relevant information
//! - Real-time conversation management
//! - Memory-efficient message handling

const std = @import("std");
const print = std.debug.print;

// Vector operations for context search
const VectorStore = struct {
    allocator: std.mem.Allocator,
    contexts: std.ArrayList(Context),
    dimension: usize,

    const Context = struct {
        id: []const u8,
        text: []const u8,
        embedding: []f32,
        category: []const u8,
        importance: f32,
    };

    const SearchResult = struct {
        context: *const Context,
        similarity: f32,
    };

    fn init(allocator: std.mem.Allocator, dim: usize) @This() {
        return @This(){
            .allocator = allocator,
            .contexts = std.ArrayList(Context).init(allocator),
            .dimension = dim,
        };
    }

    fn deinit(self: *@This()) void {
        for (self.contexts.items) |context| {
            self.allocator.free(context.id);
            self.allocator.free(context.text);
            self.allocator.free(context.embedding);
            self.allocator.free(context.category);
        }
        self.contexts.deinit();
    }

    fn addContext(self: *@This(), id: []const u8, text: []const u8, embedding: []const f32, category: []const u8, importance: f32) !void {
        const context = Context{
            .id = try self.allocator.dupe(u8, id),
            .text = try self.allocator.dupe(u8, text),
            .embedding = try self.allocator.dupe(f32, embedding),
            .category = try self.allocator.dupe(u8, category),
            .importance = importance,
        };
        try self.contexts.append(context);
    }

    fn searchSimilar(self: *@This(), query_embedding: []const f32, k: usize) ![]SearchResult {
        var results = try self.allocator.alloc(SearchResult, @min(k, self.contexts.items.len));
        var result_count: usize = 0;

        for (self.contexts.items) |*context| {
            const similarity = cosineSimilarity(query_embedding, context.embedding);

            if (result_count < k) {
                results[result_count] = SearchResult{
                    .context = context,
                    .similarity = similarity,
                };
                result_count += 1;
            } else {
                // Find lowest similarity and replace if current is higher
                var min_idx: usize = 0;
                for (1..result_count) |i| {
                    if (results[i].similarity < results[min_idx].similarity) {
                        min_idx = i;
                    }
                }
                if (similarity > results[min_idx].similarity) {
                    results[min_idx] = SearchResult{
                        .context = context,
                        .similarity = similarity,
                    };
                }
            }
        }

        // Sort by similarity (descending)
        std.sort.insertion(SearchResult, results[0..result_count], {}, struct {
            fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                return a.similarity > b.similarity;
            }
        }.lessThan);

        return results[0..result_count];
    }

    fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;

        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;

        for (0..a.len) |i| {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        const magnitude = @sqrt(norm_a * norm_b);
        return if (magnitude > 0) dot_product / magnitude else 0.0;
    }
};

// AI Persona definitions
const AIPersona = struct {
    name: []const u8,
    personality: []const u8,
    expertise: []const u8,
    response_style: []const u8,
    creativity: f32, // 0.0 to 1.0
    formality: f32, // 0.0 to 1.0
};

const Persona = enum {
    Assistant,
    Scientist,
    Creative,
    Analyst,
    Mentor,

    fn getPersona(self: Persona) AIPersona {
        return switch (self) {
            .Assistant => AIPersona{
                .name = "Ava",
                .personality = "Helpful, patient, and detail-oriented",
                .expertise = "General assistance and problem-solving",
                .response_style = "Clear, structured, and comprehensive",
                .creativity = 0.3,
                .formality = 0.7,
            },
            .Scientist => AIPersona{
                .name = "Dr. Sage",
                .personality = "Analytical, precise, and evidence-based",
                .expertise = "Scientific research and technical analysis",
                .response_style = "Methodical with citations and data",
                .creativity = 0.2,
                .formality = 0.9,
            },
            .Creative => AIPersona{
                .name = "Luna",
                .personality = "Imaginative, expressive, and intuitive",
                .expertise = "Creative writing and artistic endeavors",
                .response_style = "Colorful, metaphorical, and inspiring",
                .creativity = 0.9,
                .formality = 0.2,
            },
            .Analyst => AIPersona{
                .name = "Logic",
                .personality = "Systematic, thorough, and objective",
                .expertise = "Data analysis and strategic thinking",
                .response_style = "Structured with pros/cons and metrics",
                .creativity = 0.1,
                .formality = 0.8,
            },
            .Mentor => AIPersona{
                .name = "Wisdom",
                .personality = "Encouraging, experienced, and supportive",
                .expertise = "Guidance and personal development",
                .response_style = "Warm, encouraging with practical advice",
                .creativity = 0.5,
                .formality = 0.4,
            },
        };
    }
};

// Chat message structure
const Message = struct {
    id: u64,
    content: []const u8,
    author: []const u8,
    timestamp: i64,
    persona: ?Persona,
    context_used: [][]const u8,
};

// Main chatbot implementation
const AIChatbot = struct {
    allocator: std.mem.Allocator,
    vector_store: VectorStore,
    conversation_history: std.ArrayList(Message),
    current_persona: Persona,
    message_counter: u64,

    fn init(allocator: std.mem.Allocator) @This() {
        return @This(){
            .allocator = allocator,
            .vector_store = VectorStore.init(allocator, 384), // Standard sentence embedding size
            .conversation_history = std.ArrayList(Message).init(allocator),
            .current_persona = .Assistant,
            .message_counter = 0,
        };
    }

    fn deinit(self: *@This()) void {
        for (self.conversation_history.items) |message| {
            self.allocator.free(message.content);
            self.allocator.free(message.author);
            for (message.context_used) |context| {
                self.allocator.free(context);
            }
            self.allocator.free(message.context_used);
        }
        self.conversation_history.deinit();
        self.vector_store.deinit();
    }

    fn switchPersona(self: *@This(), persona: Persona) void {
        self.current_persona = persona;
        const p = persona.getPersona();
        print("🔄 Switched to {s}: {s}\n", .{ p.name, p.personality });
    }

    fn loadKnowledgeBase(self: *@This()) !void {
        // Simulate loading knowledge base with various topics
        const knowledge_entries = [_]struct {
            id: []const u8,
            text: []const u8,
            category: []const u8,
            importance: f32,
        }{
            .{
                .id = "ai_ethics_001",
                .text = "AI ethics involves ensuring artificial intelligence systems are developed and deployed responsibly, considering fairness, transparency, accountability, and human values.",
                .category = "AI Ethics",
                .importance = 0.9,
            },
            .{
                .id = "machine_learning_002",
                .text = "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
                .category = "Machine Learning",
                .importance = 0.8,
            },
            .{
                .id = "quantum_computing_003",
                .text = "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers.",
                .category = "Quantum Computing",
                .importance = 0.7,
            },
            .{
                .id = "climate_change_004",
                .text = "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities and greenhouse gas emissions.",
                .category = "Climate Science",
                .importance = 0.9,
            },
            .{
                .id = "productivity_005",
                .text = "Effective productivity involves prioritizing important tasks, minimizing distractions, and using systematic approaches to achieve goals efficiently.",
                .category = "Productivity",
                .importance = 0.6,
            },
        };

        print("📚 Loading knowledge base...\n", .{});
        for (knowledge_entries) |entry| {
            // Generate mock embedding (in real application, use actual embeddings)
            var embedding = try self.allocator.alloc(f32, 384);
            for (0..384) |i| {
                const hash = std.hash_map.hashString(entry.text);
                embedding[i] = @sin(@as(f32, @floatFromInt(hash + i)) / 1000.0) * 0.5;
            }

            try self.vector_store.addContext(entry.id, entry.text, embedding, entry.category, entry.importance);
        }
        print("✅ Loaded {} knowledge entries\n", .{knowledge_entries.len});
    }

    fn generateQueryEmbedding(self: *@This(), query: []const u8) ![]f32 {
        // Generate mock embedding for query (in real application, use actual embedding model)
        var embedding = try self.allocator.alloc(f32, 384);
        const hash = std.hash_map.hashString(query);
        for (0..384) |i| {
            embedding[i] = @sin(@as(f32, @floatFromInt(hash + i)) / 800.0) * 0.7;
        }
        return embedding;
    }

    fn processMessage(self: *@This(), user_input: []const u8) !void {
        const timestamp = std.time.milliTimestamp();
        self.message_counter += 1;

        // Add user message to history
        const user_message = Message{
            .id = self.message_counter,
            .content = try self.allocator.dupe(u8, user_input),
            .author = try self.allocator.dupe(u8, "User"),
            .timestamp = timestamp,
            .persona = null,
            .context_used = &[_][]const u8{},
        };
        try self.conversation_history.append(user_message);

        // Generate query embedding and search for relevant context
        const query_embedding = try self.generateQueryEmbedding(user_input);
        defer self.allocator.free(query_embedding);

        const context_results = try self.vector_store.searchSimilar(query_embedding, 3);
        defer self.allocator.free(context_results);

        // Generate response based on persona and context
        const response = try self.generateResponse(user_input, context_results);
        defer self.allocator.free(response.content);

        // Add AI response to history
        try self.conversation_history.append(response);

        // Display the conversation
        self.displayMessage(user_message);
        self.displayMessage(response);
    }

    fn generateResponse(self: *@This(), user_input: []const u8, context: []VectorStore.SearchResult) !Message {
        const persona = self.current_persona.getPersona();
        self.message_counter += 1;

        // Collect relevant context
        var context_used = try self.allocator.alloc([]const u8, context.len);
        for (context, 0..) |result, i| {
            if (result.similarity > 0.3) { // Only use relevant context
                context_used[i] = try self.allocator.dupe(u8, result.context.text);
            } else {
                context_used[i] = try self.allocator.dupe(u8, "");
            }
        }

        // Generate response based on persona characteristics
        var response_content = std.ArrayList(u8).init(self.allocator);
        defer response_content.deinit();

        // Add persona-specific greeting/style
        try response_content.appendSlice(try self.getPersonaGreeting());

        // Analyze user input and generate appropriate response
        if (std.mem.indexOf(u8, user_input, "help") != null or
            std.mem.indexOf(u8, user_input, "how") != null)
        {
            try response_content.appendSlice(try self.generateHelpResponse(user_input, context));
        } else if (std.mem.indexOf(u8, user_input, "explain") != null or
            std.mem.indexOf(u8, user_input, "what") != null)
        {
            try response_content.appendSlice(try self.generateExplanationResponse(context));
        } else if (std.mem.indexOf(u8, user_input, "create") != null or
            std.mem.indexOf(u8, user_input, "write") != null)
        {
            try response_content.appendSlice(try self.generateCreativeResponse(user_input));
        } else {
            try response_content.appendSlice(try self.generateGeneralResponse(user_input, context));
        }

        // Add persona-specific closing
        try response_content.appendSlice(try self.getPersonaClosing());

        return Message{
            .id = self.message_counter,
            .content = try response_content.toOwnedSlice(),
            .author = try std.fmt.allocPrint(self.allocator, "{s} ({s})", .{ persona.name, @tagName(self.current_persona) }),
            .timestamp = std.time.milliTimestamp(),
            .persona = self.current_persona,
            .context_used = context_used,
        };
    }

    fn getPersonaGreeting(self: *@This()) ![]const u8 {
        return switch (self.current_persona) {
            .Assistant => "I'm here to help! ",
            .Scientist => "Let me approach this systematically. ",
            .Creative => "Ooh, this sparks my imagination! ✨ ",
            .Analyst => "Let me analyze this for you. ",
            .Mentor => "I'm glad you asked! 😊 ",
        };
    }

    fn getPersonaClosing(self: *@This()) ![]const u8 {
        return switch (self.current_persona) {
            .Assistant => "\n\nIs there anything else I can help you with?",
            .Scientist => "\n\nWould you like me to elaborate on any specific aspect?",
            .Creative => "\n\nLet your creativity flow! 🌟",
            .Analyst => "\n\nShall we dive deeper into the data?",
            .Mentor => "\n\nRemember, you've got this! 💪",
        };
    }

    fn generateHelpResponse(self: *@This(), _: []const u8, context: []VectorStore.SearchResult) ![]const u8 {
        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        try response.appendSlice("Based on your question, here's what I can help with:\n\n");

        if (context.len > 0 and context[0].similarity > 0.4) {
            try response.writer().print("🔍 **Relevant Information:**\n{s}\n\n", .{context[0].context.text});
        }

        const persona = self.current_persona.getPersona();
        if (persona.creativity > 0.5) {
            try response.appendSlice("Let me paint you a picture of the solution! 🎨\n");
        } else {
            try response.appendSlice("Here's a step-by-step approach:\n");
        }

        try response.appendSlice("1. First, let's identify the core challenge\n");
        try response.appendSlice("2. Then, we'll explore potential solutions\n");
        try response.appendSlice("3. Finally, we'll create an action plan\n");

        return try response.toOwnedSlice();
    }

    fn generateExplanationResponse(self: *@This(), context: []VectorStore.SearchResult) ![]const u8 {
        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        if (context.len > 0 and context[0].similarity > 0.3) {
            try response.writer().print("📖 **Here's what I know about this topic:**\n\n{s}\n\n", .{context[0].context.text});

            if (context.len > 1 and context[1].similarity > 0.3) {
                try response.writer().print("🔗 **Related concept:**\n{s}\n\n", .{context[1].context.text});
            }
        } else {
            try response.appendSlice("I'd be happy to explain! While I don't have specific information about this in my knowledge base, ");
        }

        const persona = self.current_persona.getPersona();
        if (persona.formality > 0.7) {
            try response.appendSlice("This is a complex topic that warrants careful examination.");
        } else {
            try response.appendSlice("This is really interesting stuff!");
        }

        return try response.toOwnedSlice();
    }

    fn generateCreativeResponse(self: *@This(), _: []const u8) ![]const u8 {
        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        const persona = self.current_persona.getPersona();
        if (persona.creativity > 0.7) {
            try response.appendSlice("🎭 Oh, I love creative challenges! Let me weave something magical for you...\n\n");
            try response.appendSlice("*channels creative energy* ✨\n\n");
        } else {
            try response.appendSlice("For creative tasks, I recommend a structured approach:\n\n");
        }

        try response.appendSlice("Here's what I envision:\n");
        try response.appendSlice("• Start with brainstorming and free association\n");
        try response.appendSlice("• Gather inspiration from diverse sources\n");
        try response.appendSlice("• Create an initial draft or prototype\n");
        try response.appendSlice("• Iterate and refine based on feedback\n");

        if (persona.creativity > 0.5) {
            try response.appendSlice("\nRemember: creativity thrives on constraints and bold experimentation! 🚀");
        }

        return try response.toOwnedSlice();
    }

    fn generateGeneralResponse(self: *@This(), _: []const u8, context: []VectorStore.SearchResult) ![]const u8 {
        var response = std.ArrayList(u8).init(self.allocator);
        defer response.deinit();

        try response.appendSlice("I understand you're asking about this topic. ");

        if (context.len > 0 and context[0].similarity > 0.2) {
            try response.writer().print("Here's some relevant information:\n\n💡 {s}\n\n", .{context[0].context.text});
        }

        const persona = self.current_persona.getPersona();
        if (persona.formality > 0.6) {
            try response.appendSlice("I hope this information proves helpful for your inquiry.");
        } else {
            try response.appendSlice("Hope this helps! Feel free to ask if you want to explore this further.");
        }

        return try response.toOwnedSlice();
    }

    fn displayMessage(_: *@This(), message: Message) void {
        const persona_emoji = if (message.persona) |p| switch (p) {
            .Assistant => "🤖",
            .Scientist => "🔬",
            .Creative => "🎨",
            .Analyst => "📊",
            .Mentor => "🧙‍♂️",
        } else "👤";

        print("\n{s} **{s}**: {s}\n", .{ persona_emoji, message.author, message.content });

        // Show context usage if available
        var relevant_contexts: usize = 0;
        for (message.context_used) |context| {
            if (context.len > 0) relevant_contexts += 1;
        }
        if (relevant_contexts > 0) {
            print("   📚 Used {} knowledge base entries\n", .{relevant_contexts});
        }
    }

    fn showStats(self: *@This()) void {
        print("\n📊 **Chatbot Statistics**\n", .{});
        print("├─ Messages: {}\n", .{self.conversation_history.items.len});
        print("├─ Knowledge Base: {} entries\n", .{self.vector_store.contexts.items.len});
        print("├─ Current Persona: {s}\n", .{self.current_persona.getPersona().name});
        print("└─ Vector Dimension: {}\n", .{self.vector_store.dimension});
    }
};

// Interactive demo
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var chatbot = AIChatbot.init(allocator);
    defer chatbot.deinit();

    print("🤖 **Abi AI Multi-Persona Chatbot Demo**\n", .{});
    print("═══════════════════════════════════════\n", .{});

    // Load knowledge base
    try chatbot.loadKnowledgeBase();

    // Demo conversation with different personas
    const demo_conversations = [_]struct {
        persona: Persona,
        message: []const u8,
    }{
        .{ .persona = .Assistant, .message = "Hello! Can you help me understand AI ethics?" },
        .{ .persona = .Scientist, .message = "Explain quantum computing to me" },
        .{ .persona = .Creative, .message = "Help me write a story about machine learning" },
        .{ .persona = .Analyst, .message = "What are the pros and cons of different AI approaches?" },
        .{ .persona = .Mentor, .message = "I'm feeling overwhelmed by learning all this technology" },
    };

    print("\n🎭 **Multi-Persona Demonstration**\n", .{});
    print("───────────────────────────────────\n", .{});

    for (demo_conversations) |conv| {
        print("\n" ++ "─" ** 50 ++ "\n", .{});
        chatbot.switchPersona(conv.persona);
        try chatbot.processMessage(conv.message);
    }

    // Show final statistics
    print("\n" ++ "═" ** 50 ++ "\n", .{});
    chatbot.showStats();

    print("\n✨ **Demo Complete!**\n", .{});
    print("This chatbot demonstrates:\n", .{});
    print("├─ 🎭 Multiple AI personas with unique characteristics\n", .{});
    print("├─ 🔍 Vector-based context search and retrieval\n", .{});
    print("├─ 💬 Conversation history management\n", .{});
    print("├─ 📚 Knowledge base integration\n", .{});
    print("└─ 🚀 Real-time response generation\n", .{});
    print("\nPerfect for customer service, education, and creative applications! 🌟\n", .{});
}
