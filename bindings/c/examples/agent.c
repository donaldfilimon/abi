// agent.c - Example: AI Agent usage from C
// SPDX-License-Identifier: MIT
//
// Demonstrates creating an AI agent and having a conversation.

#include <stdio.h>
#include <stdlib.h>
#include "../include/abi.h"

int main(void) {
    abi_framework_t framework = NULL;
    abi_agent_t agent = NULL;
    abi_error_t err;
    char* response = NULL;

    printf("=== ABI Agent Example ===\n\n");

    // Initialize framework
    err = abi_init(&framework);
    if (err != ABI_OK) {
        fprintf(stderr, "Failed to initialize: %s\n", abi_error_string(err));
        return 1;
    }

    // Create an agent
    abi_agent_config_t agent_config = {
        .name = "assistant",
        .persona = NULL,
        .temperature = 0.7f,
        .enable_history = true
    };

    err = abi_agent_create(framework, &agent_config, &agent);
    if (err != ABI_OK) {
        fprintf(stderr, "Failed to create agent: %s\n", abi_error_string(err));
        abi_shutdown(framework);
        return 1;
    }
    printf("Agent '%s' created\n\n", agent_config.name);

    // Have a conversation
    const char* messages[] = {
        "Hello! How are you?",
        "What can you help me with?",
        "Tell me about Zig programming."
    };

    for (size_t i = 0; i < sizeof(messages) / sizeof(messages[0]); i++) {
        printf("User: %s\n", messages[i]);

        err = abi_agent_chat(agent, messages[i], &response);
        if (err != ABI_OK) {
            fprintf(stderr, "Chat error: %s\n", abi_error_string(err));
            continue;
        }

        printf("Agent: %s\n\n", response);
        abi_free_string(response);
        response = NULL;
    }

    // Clear history and continue
    printf("--- Clearing history ---\n\n");
    abi_agent_clear_history(agent);

    err = abi_agent_chat(agent, "What did we just talk about?", &response);
    if (err == ABI_OK) {
        printf("User: What did we just talk about?\n");
        printf("Agent: %s\n\n", response);
        abi_free_string(response);
    }

    // Cleanup
    abi_agent_destroy(agent);
    abi_shutdown(framework);

    printf("=== Example complete ===\n");
    return 0;
}
