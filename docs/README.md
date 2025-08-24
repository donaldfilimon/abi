# Abi AI Framework Documentation

## Overview

Abi is a high-performance AI framework written in Zig 0.14.1 that provides multiple AI personas, machine learning capabilities, and platform-specific optimizations. The framework is designed to be fast, memory-safe, and cross-platform.

## Features

### 🤖 **AI Personas**
- **EmpatheticAnalyst**: Caring and understanding personality
- **DirectExpert**: Direct and authoritative personality  
- **AdaptiveModerator**: Balanced and adaptive personality
- **CreativeWriter**: Creative and imaginative personality
- **TechnicalAdvisor**: Technical and analytical personality
- **ProblemSolver**: Logical and systematic personality
- **Educator**: Educational and encouraging personality
- **Counselor**: Supportive and understanding personality

### 🚀 **Performance Features**
- GPU acceleration support (OpenGL, Metal, DirectX)
- SIMD optimizations for text processing
- Lock-free concurrency patterns
- Platform-specific optimizations
- Memory-safe resource management

### 🌐 **Web Server**
- Built-in HTTP server with REST API
- CORS support
- JSON request/response handling
- Health check endpoints

### 🧠 **Machine Learning**
- Local logistic regression training
- Cross-platform model persistence
- Real-time prediction capabilities

## Quick Start

### Installation

1. Install Zig 0.14.1 from [ziglang.org](https://ziglang.org/download/)
2. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/abi.git
   cd abi
   ```

### Basic Usage

```bash
# Build the project
zig build

# Run the main application
zig build run

# Run with specific features
zig build run -- --gpu --simd

# Run different modes
zig build run -- tui      # Terminal UI
zig build run -- agent    # AI agent client
zig build run -- ml       # Machine learning example
zig build run -- bench    # Performance benchmarks
```

### Web Server

```bash
# Start the web server
zig build web

# The server will be available at http://localhost:3000
```

## API Reference

### REST API Endpoints

#### POST /api/chat
Send a message to the AI agent.

**Request:**
```json
{
  "message": "Hello, how are you?",
  "persona": "EmpatheticAnalyst",
  "max_tokens": 100
}
```

**Response:**
```json
{
  "response": "I'm doing well, thank you for asking! How can I help you today?",
  "persona": "EmpatheticAnalyst"
}
```

#### GET /api/health
Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "1703123456789"
}
```

#### GET /api/personas
Get available AI personas.

**Response:**
```json
{
  "personas": [
    {
      "name": "EmpatheticAnalyst",
      "description": "Caring and understanding personality"
    },
    {
      "name": "DirectExpert", 
      "description": "Direct and authoritative personality"
    }
  ]
}
```

### Command Line Interface

#### Agent Client
```bash
# Start agent with specific persona
zig run agent_client.zig -- --persona EmpatheticAnalyst

# Available personas: EmpatheticAnalyst, DirectExpert, AdaptiveModerator,
# CreativeWriter, TechnicalAdvisor, ProblemSolver, Educator, Counselor
```

#### Machine Learning
```bash
# Train a model
zig run localml.zig -- train data.csv model.txt

# Make predictions
zig run localml.zig -- predict model.txt 1.2 3.4
```

## Architecture

### Core Components

1. **Agent System** (`src/agent.zig`)
   - Manages AI personas and conversation history
   - Handles OpenAI API integration
   - Provides content filtering and risk assessment

2. **Platform Layer** (`src/platform.zig`)
   - Platform-specific optimizations
   - System information gathering
   - Memory management optimizations

3. **Web Server** (`src/web_server.zig`)
   - HTTP server implementation
   - REST API endpoints
   - Request/response handling

4. **Machine Learning** (`src/localml.zig`)
   - Local model training
   - Prediction capabilities
   - Model persistence

### Build System

The project uses Zig's modern build system with the following features:

- **Cross-compilation** support for multiple platforms
- **Feature flags** for conditional compilation
- **Dependency management** with build.zig.zon
- **Multiple executables** (main, agent, web server)
- **Test and benchmark** integration

## Platform Support

### Supported Platforms
- **Windows**: x86_64, ARM64
- **Linux**: x86_64, ARM64 (GNU and musl)
- **macOS**: x86_64, ARM64
- **iOS**: ARM64

### Platform-Specific Features

#### Windows
- ANSI color support
- System information gathering
- DirectX integration

#### Linux
- io_uring support (future)
- /proc filesystem integration
- System call optimizations

#### macOS
- Metal framework support
- Unified memory management
- Purgeable memory allocation

#### iOS
- Memory pressure handling
- Sandbox restrictions
- Limited resource usage

## Development

### Building

```bash
# Build all targets
zig build

# Build specific target
zig build -Dtarget=x86_64-linux-gnu

# Build with optimizations
zig build -Doptimize=ReleaseFast

# Build with specific features
zig build -Dgpu=true -Dsimd=true
```

### Testing

```bash
# Run all tests
zig build test

# Run specific test
zig test src/agent.zig

# Run tests with coverage
zig test src/main.zig -femit-bin=test-coverage
```

### Formatting

```bash
# Format all source files
zig build fmt

# Format specific files
zig fmt src/
```

### Documentation

```bash
# Generate documentation
zig build docs
```

## Examples

### Basic AI Interaction

```zig
const agent = @import("agent.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var ai_agent = try agent.Agent.init(allocator, .{});
    defer ai_agent.deinit();

    ai_agent.setPersona(.EmpatheticAnalyst);
    const response = try ai_agent.generateResponse("Hello, how are you?");
    defer allocator.free(response);

    std.debug.print("AI Response: {s}\n", .{response});
}
```

### Web Server Integration

```zig
const web = @import("web_server.zig");

pub fn main() !void {
    const config = web.WebConfig{
        .port = 8080,
        .host = "127.0.0.1",
    };

    var server = try web.WebServer.init(allocator, config);
    defer server.deinit();

    try server.start();
}
```

### Platform Information

```zig
const platform = @import("platform.zig");

pub fn main() !void {
    const info = platform.PlatformLayer.getSystemInfo();
    std.debug.print("Platform: {s}\n", .{info.platform});
    std.debug.print("Memory: {d} bytes\n", .{info.memory});
    std.debug.print("CPU Count: {d}\n", .{info.cpu_count});
}
```

## Performance

### Benchmarks

The framework includes built-in benchmarks:

```bash
# Run performance benchmarks
zig build bench

# Custom benchmark parameters
zig build run -- bench --iterations=10000
```

### Optimization Tips

1. **Use Release Mode**: Build with `-Doptimize=ReleaseFast` for maximum performance
2. **Enable SIMD**: Use `-Dsimd=true` for vectorized operations
3. **GPU Acceleration**: Enable with `-Dgpu=true` where supported
4. **Platform Optimizations**: The framework automatically detects and uses platform-specific optimizations

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `zig build test`
6. Submit a pull request

### Code Style

- Follow Zig's official style guide
- Use `zig fmt` to format code
- Add comprehensive documentation
- Include tests for new features

### Testing

- Unit tests for all modules
- Integration tests for API endpoints
- Performance benchmarks
- Cross-platform testing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Zig](https://ziglang.org/) programming language
- Inspired by modern AI frameworks
- Community contributions welcome

## Support

- **Issues**: Report bugs and feature requests on GitHub
- **Discussions**: Join community discussions
- **Documentation**: Check the docs folder for detailed guides
- **Examples**: See the examples directory for usage patterns 