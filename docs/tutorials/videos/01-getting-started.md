# Video Walkthrough: Getting Started with ABI

**Duration:** 20 minutes
**Difficulty:** Beginner
**Code:** `docs/tutorials/code/getting-started/`

---

## Video Metadata

**Title:** ABI Framework Tutorial: Getting Started

**Description:**
Learn to install, configure, and use the ABI framework. This tutorial covers
framework initialization, feature detection, error handling, and CLI usage.

**Tags:** #zig #abi #framework #tutorial #getting-started

**Chapters/Timestamps:**
- 0:00 Introduction
- 1:30 Installation & Build
- 4:00 First Program
- 7:30 Feature Detection
- 10:00 Error Handling
- 13:00 CLI Overview
- 16:00 Configuration
- 18:30 Wrap-up

---

## Script

### [0:00] Introduction

**[Title Slide: "Getting Started with ABI"]**

> "Welcome to the ABI framework tutorial series. In this first video, I'll
> show you how to install, configure, and run your first ABI program.
>
> ABI is a modular framework written in Zig that provides AI, GPU acceleration,
> vector database, and distributed computing capabilities. By the end of this
> video, you'll have a working ABI installation and understand its core concepts."

**[Transition to screen capture]**

---

### [1:30] Installation & Build

**[Terminal visible]**

> "Let's start by cloning the repository. Open your terminal and run:"

```bash
git clone https://github.com/donaldfilimon/abi.git
cd abi
```

> "Now let's build the project with Zig. Make sure you have Zig 0.16 or later."

```bash
zig version
# Should show 0.16.x

zig build
```

> "ABI uses feature gating, which means you only compile the features you need.
> Let's see the available options."

```bash
zig build --help
```

**[Highlight the enable flags]**

> "For example, to build with only AI and database support, you'd run:"

```bash
zig build -Denable-ai=true -Denable-database=true -Denable-gpu=false
```

> "This keeps your binary small and compilation fast."

---

### [4:00] First Program

**[Open editor with hello_abi.zig]**

> "Now let's write our first ABI program. Create a new file called hello_abi.zig."

```zig
const std = @import("std");
const abi = @import("src/abi.zig");

pub fn main() !void {
    // Get an allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
```

> "Every ABI program starts with an allocator. We're using the general purpose
> allocator here, but you can use any Zig allocator."

```zig
    // Initialize ABI
    std.debug.print("Initializing ABI framework...\n", .{});
    try abi.init(allocator);
    defer abi.shutdown();
```

> "The key pattern here is `init` followed by `defer shutdown`. This ensures
> proper cleanup regardless of how the function exits."

```zig
    // Print version
    const version = abi.version();
    std.debug.print("ABI Version: {d}.{d}.{d}\n", .{
        version.major,
        version.minor,
        version.patch,
    });
}
```

**[Run the program]**

```bash
zig run hello_abi.zig
```

**[Show output]**

> "There we go - ABI is initialized and we can see the version. The framework
> is now ready to use."

---

### [7:30] Feature Detection

**[Open feature_check.zig]**

> "ABI lets you check which features are enabled at runtime. This is useful for
> graceful degradation when a feature isn't available."

```zig
const features = [_]struct { name: []const u8, enabled: bool }{
    .{ .name = "AI/LLM", .enabled = abi.isFeatureEnabled(.ai) },
    .{ .name = "GPU", .enabled = abi.isFeatureEnabled(.gpu) },
    .{ .name = "Database", .enabled = abi.isFeatureEnabled(.database) },
};

for (features) |f| {
    const status = if (f.enabled) "[ENABLED]" else "[DISABLED]";
    std.debug.print("  {s:<12} {s}\n", .{ f.name, status });
}
```

**[Run and show output]**

> "As you can see, all features are enabled in this build. If we rebuild with
> different flags, we'd see different results."

---

### [10:00] Error Handling

**[Show error handling code]**

> "When you try to use a disabled feature, ABI returns specific error types.
> Let's see how to handle this gracefully."

```zig
if (abi.database.openOrCreate(allocator, "test_db")) |*db| {
    defer abi.database.close(db);
    std.debug.print("Database opened successfully\n", .{});
} else |err| {
    switch (err) {
        error.DatabaseDisabled => {
            std.debug.print("Database feature is disabled.\n", .{});
            std.debug.print("Rebuild with: zig build -Denable-database=true\n", .{});
        },
        else => std.debug.print("Error: {t}\n", .{err}),
    }
}
```

> "This pattern lets your code adapt to different build configurations
> without crashing."

---

### [13:00] CLI Overview

**[Terminal visible]**

> "ABI includes a command-line interface for common operations."

```bash
zig build run -- --help
```

**[Show help output, highlight key commands]**

> "Let's try a few useful commands."

```bash
# System information
zig build run -- system-info

# GPU backends
zig build run -- gpu backends

# Search the codebase
zig build run -- explore "fn init" --level quick
```

**[Show each command's output]**

---

### [16:00] Configuration

**[Show environment variables]**

> "ABI reads configuration from environment variables. Here are the key ones
> for AI integration."

```bash
export ABI_OLLAMA_HOST="http://localhost:11434"
export ABI_OLLAMA_MODEL="gpt-oss"
```

> "If you're using OpenAI:"

```bash
export ABI_OPENAI_API_KEY="sk-..."
```

> "These let you configure connectors without changing code."

---

### [18:30] Wrap-up

**[Show resources slide]**

> "That's it for getting started! You now know how to:
> - Install and build ABI
> - Initialize the framework
> - Check feature availability
> - Handle errors gracefully
> - Use the CLI
>
> In the next tutorial, we'll dive into the vector database for storing
> and searching high-dimensional data.
>
> All code is available in docs/tutorials/code/getting-started.
> Thanks for watching!"

**[End screen with links]**

---

## Production Checklist

**Before Recording:**
- [ ] Verify all code examples compile and run
- [ ] Test on clean Zig 0.16.x installation
- [ ] Prepare title slides and diagrams
- [ ] Set up clean terminal environment (no personal info visible)

**Recording Setup:**
- [ ] Screen: 1920x1080, 60fps
- [ ] Audio: Clear microphone, no background noise
- [ ] Editor: VS Code with Zig syntax highlighting
- [ ] Terminal: Font size 16+, high contrast

**Post-Production:**
- [ ] Add chapter markers at timestamps above
- [ ] Add code overlays for key snippets
- [ ] Add captions/subtitles
- [ ] Color-code output (green=success, red=error)
- [ ] Add end screen with next tutorial link

**YouTube Description:**
```
Learn to install, configure, and use the ABI framework in Zig.

What You'll Learn:
- Framework installation and build options
- Initialization and shutdown patterns
- Feature detection and error handling
- CLI commands for common operations
- Environment configuration for AI connectors

Resources:
- Written Tutorial: [link]
- Code Examples: [link]
- API Reference: [link]
- ABI Documentation: [link]

Chapters:
0:00 Introduction
1:30 Installation & Build
4:00 First Program
7:30 Feature Detection
10:00 Error Handling
13:00 CLI Overview
16:00 Configuration
18:30 Wrap-up

Prerequisites:
- Zig 0.16.x: https://ziglang.org/download/
- Basic terminal knowledge

Next Tutorial: Vector Database Operations

#zig #programming #tutorial #abi #framework
```
