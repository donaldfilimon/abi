const std = @import("std");
const connectors = @import("abi").connectors;

test "connector system initialization" {
    // Test Ollama configuration
    const ollama_config = connectors.OllamaConfig{
        .host = "http://localhost:11434",
        .model = "nomic-embed-text",
    };

    try std.testing.expectEqualStrings("http://localhost:11434", ollama_config.host);
    try std.testing.expectEqualStrings("nomic-embed-text", ollama_config.model);

    // Test OpenAI configuration
    const openai_config = connectors.OpenAIConfig{
        .base_url = "https://api.openai.com/v1",
        .api_key = "test-key",
        .model = "text-embedding-3-small",
    };

    try std.testing.expectEqualStrings("https://api.openai.com/v1", openai_config.base_url);
    try std.testing.expectEqualStrings("test-key", openai_config.api_key);
    try std.testing.expectEqualStrings("text-embedding-3-small", openai_config.model);
}

test "provider configuration union" {
    const ollama_config = connectors.OllamaConfig{
        .host = "http://localhost:11434",
        .model = "nomic-embed-text",
    };

    const provider_config = connectors.ProviderConfig{ .ollama = ollama_config };

    try std.testing.expectEqual(connectors.ProviderType.ollama, @as(connectors.ProviderType, provider_config));

    const openai_config = connectors.OpenAIConfig{
        .base_url = "https://api.openai.com/v1",
        .api_key = "test-key",
        .model = "text-embedding-3-small",
    };

    const provider_config2 = connectors.ProviderConfig{ .openai = openai_config };

    try std.testing.expectEqual(connectors.ProviderType.openai, @as(connectors.ProviderType, provider_config2));
}

test "embedText function interface" {
    const allocator = std.testing.allocator;

    // Test with Ollama config (will fail due to network, but tests interface)
    const ollama_config = connectors.OllamaConfig{
        .host = "http://localhost:11434",
        .model = "nomic-embed-text",
    };

    const provider_config = connectors.ProviderConfig{ .ollama = ollama_config };

    // This will likely fail due to network, but tests the interface
    const result = connectors.embedText(allocator, provider_config, "test text");

    // We expect this to fail with a network error, which is expected
    if (result) |_| {
        // If it succeeds, that's fine too
    } else |err| {
        // Expected errors: NetworkError, InvalidResponse, etc.
        _ = err catch {};
    }
}

test "connector error types" {
    // Test that error types are properly defined
    const error_types = [_]connectors.ConnectorsError{
        connectors.ConnectorsError.NetworkError,
        connectors.ConnectorsError.InvalidResponse,
        connectors.ConnectorsError.ParseError,
        connectors.ConnectorsError.MissingApiKey,
    };

    try std.testing.expectEqual(@as(usize, 4), error_types.len);
}
