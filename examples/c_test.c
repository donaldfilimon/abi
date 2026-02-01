#include <stdio.h>
#include <stdlib.h>
#include "src/bindings/c/abi.h"

int main() {
    printf("Initializing ABI framework...\n");
    AbiHandle handle = abi_init();
    if (handle == NULL) {
        fprintf(stderr, "Failed to initialize ABI framework\n");
        return 1;
    }
    printf("Framework initialized. Version: %s\n", abi_version());

    printf("Creating vector database...\n");
    AbiHandle db;
    AbiStatus status = abi_db_create(handle, 128, &db);
    if (status != ABI_SUCCESS) {
        fprintf(stderr, "Failed to create database: %d\n", status);
        abi_shutdown(handle);
        return 1;
    }
    printf("Database created.\n");

    printf("Cleaning up...\n");
    abi_db_destroy(db);
    abi_shutdown(handle);
    printf("Done.\n");
    return 0;
}
