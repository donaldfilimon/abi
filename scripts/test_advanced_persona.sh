#!/bin/bash
# Script to test the advanced persona system when Zig is available

echo "Advanced Persona System Test Script"
echo "==================================="

# Check if Zig is installed
if ! command -v zig &> /dev/null; then
    echo "Error: Zig is not installed. Please install Zig first."
    echo "Visit: https://ziglang.org/download/"
    exit 1
fi

echo "Zig version: $(zig version)"
echo ""

# Run tests for the advanced persona system
echo "Running advanced persona system tests..."
zig test src/advanced_persona_system_test.zig --test-filter "ActiveAgentPool" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "AgentRegistry" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "Agent processing" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "EmbeddingVector" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "AgentCoordinationSystem" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "Error handling" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "AgentId" || exit 1
zig test src/advanced_persona_system_test.zig --test-filter "KDTree" || exit 1

echo ""
echo "All tests completed successfully!"

# Optional: Build the main project
echo ""
echo "Building main project..."
zig build || echo "Note: Main build failed (expected if Zig is not set up)"