#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define REPO "https://github.com/ziglang/zig-bootstrap.git"
#define DIR "zig-bootstrap-emergency"

int run(const char *cmd) {
    printf("Executing: %s\n", cmd);
    return system(cmd);
}

int main() {
    printf("--- Emergency Zig Bootstrapper ---\n");
    
    if (access(DIR, F_OK) == 0) {
        printf("Directory %s already exists. Cleaning up...\n", DIR);
        run("rm -rf " DIR);
    }

    if (run("git clone --depth 1 " REPO " " DIR) != 0) {
        fprintf(stderr, "Failed to clone repository\n");
        return 1;
    }

    if (chdir(DIR) != 0) {
        perror("chdir");
        return 1;
    }

    const char *target = "aarch64-macos-none";
    const char *mcpu = "native";
    
    char build_cmd[256];
    snprintf(build_cmd, sizeof(build_cmd), "./build %s %s", target, mcpu);
    
    printf("Starting build (this will take a LONG time)...\n");
    if (run(build_cmd) != 0) {
        fprintf(stderr, "Build failed\n");
        return 1;
    }

    printf("\n--- SUCCESS ---\n");
    printf("Bootstrapped Zig is available in: %s/out/bin/zig\n", DIR);
    printf("You can use it by running:\n");
    printf("  export PATH=\"$(pwd)/out/bin:$PATH\"\n");

    return 0;
}
