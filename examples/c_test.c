/**
 * ABI Framework C Binding Example
 *
 * Demonstrates the C API for framework initialization,
 * version querying, and database operations.
 *
 * Build: Link against the ABI static library with -labi
 */

#include <stdio.h>
#include <stdlib.h>
#include "bindings/c/include/abi.h"

int main() {
    printf("Initializing ABI framework...\n");

    abi_framework_t *fw = NULL;
    int err = abi_init(&fw);
    if (err != ABI_OK) {
        fprintf(stderr, "Failed to initialize ABI framework: %d\n", err);
        return 1;
    }
    printf("Framework initialized. Version: %s\n", abi_version());
    printf("Framework state: %s\n", abi_get_state(fw));
    printf("Enabled features: %d\n", abi_enabled_feature_count(fw));

    /* Create a vector database (128 dimensions) */
    printf("Creating vector database...\n");
    abi_database_config_t db_cfg;
    abi_database_config_init(&db_cfg);
    db_cfg.dimension = 128;

    abi_database_t *db = NULL;
    err = abi_database_create(&db_cfg, &db);
    if (err != ABI_OK) {
        fprintf(stderr, "Failed to create database: %d\n", err);
        abi_shutdown(fw);
        return 1;
    }
    printf("Database created.\n");

    /* Insert a test vector */
    float vector[128];
    for (int i = 0; i < 128; i++) {
        vector[i] = (float)i / 128.0f;
    }
    err = abi_database_insert(db, 1, vector, 128, "{\"source\": \"c_test\"}");
    if (err == ABI_OK) {
        printf("Inserted vector with ID 1.\n");
    }

    /* Query count */
    size_t count = 0;
    err = abi_database_count(db, &count);
    if (err == ABI_OK) {
        printf("Database contains %zu vectors.\n", count);
    }

    printf("Cleaning up...\n");
    abi_database_close(db);
    abi_shutdown(fw);
    printf("Done.\n");
    return 0;
}
