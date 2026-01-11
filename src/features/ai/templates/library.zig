//! Built-in prompt templates for common LLM interaction patterns.
//!
//! Provides ready-to-use templates for chat, RAG, tool calling,
//! code generation, and other common use cases.

const std = @import("std");

/// Available built-in templates.
pub const BuiltinTemplates = enum {
    /// Basic system message template.
    system_message,
    /// Chat completion with system, context, and user message.
    chat_completion,
    /// RAG context injection template.
    rag_context,
    /// Tool/function calling prompt.
    tool_prompt,
    /// Code generation prompt.
    code_generation,
    /// Code review prompt.
    code_review,
    /// Text summarization prompt.
    summarization,
    /// Question answering prompt.
    question_answer,
    /// Translation prompt.
    translation,
    /// JSON extraction prompt.
    json_extraction,
    /// Classification prompt.
    classification,
    /// Conversation continuation.
    conversation,
};

pub const TemplateInfo = struct {
    name: []const u8,
    description: []const u8,
    source: []const u8,
    variables: []const []const u8,
};

/// Get information about a built-in template.
pub fn getBuiltinTemplate(template: BuiltinTemplates) TemplateInfo {
    return switch (template) {
        .system_message => .{
            .name = "system_message",
            .description = "Basic system message for setting assistant behavior",
            .source = system_message_template,
            .variables = &[_][]const u8{ "role", "instructions" },
        },
        .chat_completion => .{
            .name = "chat_completion",
            .description = "Standard chat completion with system and user messages",
            .source = chat_completion_template,
            .variables = &[_][]const u8{ "system", "history", "user_message" },
        },
        .rag_context => .{
            .name = "rag_context",
            .description = "RAG prompt with retrieved context injection",
            .source = rag_context_template,
            .variables = &[_][]const u8{ "context", "question" },
        },
        .tool_prompt => .{
            .name = "tool_prompt",
            .description = "Tool/function calling prompt with available tools",
            .source = tool_prompt_template,
            .variables = &[_][]const u8{ "tools_json", "user_request" },
        },
        .code_generation => .{
            .name = "code_generation",
            .description = "Code generation with language and requirements",
            .source = code_generation_template,
            .variables = &[_][]const u8{ "language", "requirements", "context" },
        },
        .code_review => .{
            .name = "code_review",
            .description = "Code review prompt for analyzing code quality",
            .source = code_review_template,
            .variables = &[_][]const u8{ "code", "language", "focus_areas" },
        },
        .summarization => .{
            .name = "summarization",
            .description = "Text summarization with configurable length",
            .source = summarization_template,
            .variables = &[_][]const u8{ "text", "max_length", "style" },
        },
        .question_answer => .{
            .name = "question_answer",
            .description = "Question answering with optional context",
            .source = question_answer_template,
            .variables = &[_][]const u8{ "context", "question" },
        },
        .translation => .{
            .name = "translation",
            .description = "Language translation prompt",
            .source = translation_template,
            .variables = &[_][]const u8{ "source_lang", "target_lang", "text" },
        },
        .json_extraction => .{
            .name = "json_extraction",
            .description = "Extract structured JSON from unstructured text",
            .source = json_extraction_template,
            .variables = &[_][]const u8{ "schema", "text" },
        },
        .classification => .{
            .name = "classification",
            .description = "Text classification into categories",
            .source = classification_template,
            .variables = &[_][]const u8{ "categories", "text" },
        },
        .conversation => .{
            .name = "conversation",
            .description = "Multi-turn conversation continuation",
            .source = conversation_template,
            .variables = &[_][]const u8{ "persona", "history", "user_input" },
        },
    };
}

/// List all available built-in templates.
pub fn listBuiltinTemplates() []const BuiltinTemplates {
    return &[_]BuiltinTemplates{
        .system_message,
        .chat_completion,
        .rag_context,
        .tool_prompt,
        .code_generation,
        .code_review,
        .summarization,
        .question_answer,
        .translation,
        .json_extraction,
        .classification,
        .conversation,
    };
}

// Template source strings

const system_message_template =
    \\You are {{role|a helpful AI assistant}}.
    \\
    \\{{instructions|Please be helpful, accurate, and concise in your responses.}}
;

const chat_completion_template =
    \\<|system|>
    \\{{system|You are a helpful AI assistant.}}
    \\<|end|>
    \\
    \\{{history}}
    \\
    \\<|user|>
    \\{{user_message}}
    \\<|end|>
    \\
    \\<|assistant|>
;

const rag_context_template =
    \\Use the following context to answer the question. If the answer is not in the context, say so.
    \\
    \\Context:
    \\{{context}}
    \\
    \\Question: {{question}}
    \\
    \\Answer:
;

const tool_prompt_template =
    \\You have access to the following tools:
    \\
    \\{{tools_json}}
    \\
    \\To use a tool, respond with a JSON object in this format:
    \\{"tool": "tool_name", "arguments": {"arg1": "value1"}}
    \\
    \\If you don't need to use a tool, respond normally.
    \\
    \\User request: {{user_request}}
;

const code_generation_template =
    \\Generate {{language}} code that meets the following requirements:
    \\
    \\Requirements:
    \\{{requirements}}
    \\
    \\{{context|}}
    \\
    \\Provide clean, well-documented code with appropriate error handling.
    \\
    \\```{{language}}
;

const code_review_template =
    \\Review the following {{language|}} code for:
    \\{{focus_areas|bugs, security issues, performance, and code quality}}
    \\
    \\Code:
    \\```
    \\{{code}}
    \\```
    \\
    \\Provide specific feedback with line references where applicable.
;

const summarization_template =
    \\Summarize the following text in {{max_length|a few sentences}}.
    \\Style: {{style|concise and informative}}
    \\
    \\Text:
    \\{{text}}
    \\
    \\Summary:
;

const question_answer_template =
    \\{{context|}}
    \\
    \\Question: {{question}}
    \\
    \\Provide a clear and accurate answer based on the information available.
    \\
    \\Answer:
;

const translation_template =
    \\Translate the following text from {{source_lang}} to {{target_lang}}.
    \\Preserve the original meaning and tone.
    \\
    \\Original ({{source_lang}}):
    \\{{text}}
    \\
    \\Translation ({{target_lang}}):
;

const json_extraction_template =
    \\Extract information from the following text and return it as JSON matching this schema:
    \\
    \\{{schema}}
    \\
    \\Text:
    \\{{text}}
    \\
    \\JSON output:
;

const classification_template =
    \\Classify the following text into one of these categories:
    \\{{categories}}
    \\
    \\Text: {{text}}
    \\
    \\Respond with only the category name.
    \\
    \\Category:
;

const conversation_template =
    \\{{persona|You are a helpful assistant engaged in a conversation.}}
    \\
    \\Conversation history:
    \\{{history}}
    \\
    \\User: {{user_input}}
    \\
    \\Assistant:
;

test "get builtin template" {
    const template = getBuiltinTemplate(.rag_context);
    try std.testing.expectEqualStrings("rag_context", template.name);
    try std.testing.expect(template.source.len > 0);
    try std.testing.expectEqual(@as(usize, 2), template.variables.len);
}

test "list builtin templates" {
    const templates = listBuiltinTemplates();
    try std.testing.expect(templates.len >= 10);
}
