// hello.c - Example: Basic ABI framework usage from C
// SPDX-License-Identifier: MIT
//
// Compile with:
//   gcc -I../include -L../zig-out/lib -labi hello.c -o hello
//   LD_LIBRARY_PATH=../zig-out/lib ./hello
//
// Or with static library:
//   gcc -I../include hello.c ../zig-out/lib/libabi_static.a -o hello -lm
//   ./hello

#include <stdio.h>
#include <stdlib.h>
#include "../include/abi.h"

int main(void) {
    abi_framework_t framework = NULL;
    abi_error_t err;

    printf("=== ABI C Bindings Example ===\n\n");

    // -------------------------------------------------------------------------
    // Framework Initialization
    // -------------------------------------------------------------------------
    printf("1. Initializing ABI framework...\n");
    err = abi_init(&framework);
    if (err != ABI_OK) {
        fprintf(stderr, "   ERROR: Failed to initialize: %s\n", abi_error_string(err));
        return 1;
    }
    printf("   OK: Framework initialized\n");

    // Print version
    printf("   Version: %s\n", abi_version());

    abi_version_t ver;
    abi_version_info(&ver);
    printf("   Version info: %d.%d.%d\n", ver.major, ver.minor, ver.patch);

    // -------------------------------------------------------------------------
    // SIMD Capabilities
    // -------------------------------------------------------------------------
    printf("\n2. Checking SIMD capabilities...\n");
    abi_simd_caps_t caps;
    abi_simd_get_caps(&caps);
    printf("   SSE:     %s\n", caps.sse ? "yes" : "no");
    printf("   SSE2:    %s\n", caps.sse2 ? "yes" : "no");
    printf("   AVX:     %s\n", caps.avx ? "yes" : "no");
    printf("   AVX2:    %s\n", caps.avx2 ? "yes" : "no");
    printf("   AVX-512: %s\n", caps.avx512f ? "yes" : "no");
    printf("   NEON:    %s\n", caps.neon ? "yes" : "no");

    // -------------------------------------------------------------------------
    // SIMD Vector Operations
    // -------------------------------------------------------------------------
    printf("\n3. Testing SIMD vector operations...\n");
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {4.0f, 3.0f, 2.0f, 1.0f};
    float result[4];

    abi_simd_vector_add(a, b, result, 4);
    printf("   Vector add: [%.1f, %.1f, %.1f, %.1f] + [%.1f, %.1f, %.1f, %.1f]\n",
           a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
    printf("            = [%.1f, %.1f, %.1f, %.1f]\n",
           result[0], result[1], result[2], result[3]);

    float dot = abi_simd_vector_dot(a, b, 4);
    printf("   Dot product: %.1f\n", dot);

    float norm = abi_simd_vector_l2_norm(a, 4);
    printf("   L2 norm of a: %.3f\n", norm);

    float sim = abi_simd_cosine_similarity(a, b, 4);
    printf("   Cosine similarity: %.3f\n", sim);

    // -------------------------------------------------------------------------
    // Feature Checks
    // -------------------------------------------------------------------------
    printf("\n4. Checking enabled features...\n");
    printf("   AI:       %s\n", abi_is_feature_enabled(framework, "ai") ? "enabled" : "disabled");
    printf("   GPU:      %s\n", abi_is_feature_enabled(framework, "gpu") ? "enabled" : "disabled");
    printf("   Database: %s\n", abi_is_feature_enabled(framework, "database") ? "enabled" : "disabled");
    printf("   Network:  %s\n", abi_is_feature_enabled(framework, "network") ? "enabled" : "disabled");
    printf("   Web:      %s\n", abi_is_feature_enabled(framework, "web") ? "enabled" : "disabled");

    // -------------------------------------------------------------------------
    // GPU Backend
    // -------------------------------------------------------------------------
    printf("\n5. Checking GPU backend...\n");
    printf("   GPU available: %s\n", abi_gpu_is_available() ? "yes" : "no");

    abi_gpu_t gpu = NULL;
    abi_gpu_config_t gpu_config = ABI_GPU_CONFIG_DEFAULT;
    err = abi_gpu_init(&gpu_config, &gpu);
    if (err == ABI_OK) {
        printf("   Active backend: %s\n", abi_gpu_backend_name(gpu));
        abi_gpu_shutdown(gpu);
    } else {
        printf("   GPU init failed: %s\n", abi_error_string(err));
    }

    // -------------------------------------------------------------------------
    // Vector Database
    // -------------------------------------------------------------------------
    printf("\n6. Testing vector database...\n");
    abi_database_t db = NULL;
    abi_database_config_t db_config = {
        .name = "example_db",
        .dimension = 4,
        .initial_capacity = 100
    };

    err = abi_database_create(&db_config, &db);
    if (err != ABI_OK) {
        fprintf(stderr, "   ERROR: Failed to create database: %s\n", abi_error_string(err));
    } else {
        printf("   OK: Database created (dim=%zu)\n", db_config.dimension);

        // Insert vectors
        float v1[] = {1.0f, 0.0f, 0.0f, 0.0f};
        float v2[] = {0.9f, 0.1f, 0.0f, 0.0f};
        float v3[] = {0.0f, 1.0f, 0.0f, 0.0f};

        abi_database_insert(db, 1, v1, 4);
        abi_database_insert(db, 2, v2, 4);
        abi_database_insert(db, 3, v3, 4);

        size_t count = 0;
        abi_database_count(db, &count);
        printf("   Inserted %zu vectors\n", count);

        // Search
        abi_search_result_t results[3];
        size_t result_count = 0;
        abi_database_search(db, v1, 4, 3, results, &result_count);
        printf("   Search results for v1:\n");
        for (size_t i = 0; i < result_count; i++) {
            printf("     - ID: %llu, Score: %.3f\n",
                   (unsigned long long)results[i].id, results[i].score);
        }

        abi_database_close(db);
        printf("   OK: Database closed\n");
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    printf("\n7. Shutting down...\n");
    abi_shutdown(framework);
    printf("   OK: Framework shutdown complete\n");

    printf("\n=== Example complete ===\n");
    return 0;
}
