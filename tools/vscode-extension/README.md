# ABI Development Tools for VS Code

A VS Code extension for developing with the ABI framework, providing integrated build commands, testing support, and code snippets for Zig development.

## Features

- **Build Commands**: Quick access to build, test, and run commands
- **Status Bar Integration**: Shows ABI status and provides quick build access
- **Task Provider**: Integrates with VS Code's task system
- **Code Snippets**: Common ABI patterns and boilerplate
- **Zig Language Support**: Works alongside existing Zig extensions

## Installation

### From Source (Development)

1. Clone the ABI repository:
   ```bash
   git clone https://github.com/abi-project/abi.git
   cd abi/tools/vscode-extension
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Compile the extension:
   ```bash
   npm run compile
   ```

4. Open VS Code in the extension directory:
   ```bash
   code .
   ```

5. Press `F5` to launch the Extension Development Host

### From VSIX Package

1. Build the VSIX package:
   ```bash
   npm run package
   ```

2. Install in VS Code:
   - Open VS Code
   - Go to Extensions view (Ctrl+Shift+X)
   - Click the "..." menu
   - Select "Install from VSIX..."
   - Choose the generated `.vsix` file

## Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| `ABI: Build Project` | `Ctrl+Shift+B` | Build the ABI project with configured feature flags |
| `ABI: Run Tests` | - | Run all tests with summary output |
| `ABI: Test Current File` | `Ctrl+Shift+T` | Test the currently open Zig file |
| `ABI: Run Application` | - | Run the application |
| `ABI: Run with Arguments` | - | Run with custom CLI arguments |
| `ABI: Format Code` | `Ctrl+Alt+F` | Format Zig code using zig fmt |
| `ABI: List Features` | - | List available ABI features |
| `ABI: Show Documentation` | - | Open project documentation |
| `ABI: Generate API Docs` | - | Generate API documentation |

## Configuration

Configure the extension in VS Code settings:

```json
{
  "abi.zigPath": "zig",
  "abi.enableGpu": true,
  "abi.enableAi": true,
  "abi.enableDatabase": true,
  "abi.gpuBackend": "auto",
  "abi.optimize": "Debug",
  "abi.showStatusBar": true
}
```

### Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `abi.zigPath` | `"zig"` | Path to the Zig compiler |
| `abi.enableGpu` | `true` | Enable GPU feature flag |
| `abi.enableAi` | `true` | Enable AI feature flag |
| `abi.enableDatabase` | `true` | Enable Database feature flag |
| `abi.gpuBackend` | `"auto"` | GPU backend (auto, none, cuda, vulkan, metal, webgpu, opengl) |
| `abi.optimize` | `"Debug"` | Optimization level (Debug, ReleaseSafe, ReleaseFast, ReleaseSmall) |
| `abi.showStatusBar` | `true` | Show ABI status bar item |

## Code Snippets

The extension provides snippets for common ABI patterns:

| Prefix | Description |
|--------|-------------|
| `abi-init` | Full ABI initialization with I/O backend |
| `abi-config` | Configuration builder pattern |
| `abi-framework` | Framework initialization |
| `abi-gpu-config` | GPU configuration |
| `abi-ai-config` | AI configuration |
| `abi-db-config` | Database configuration |
| `abi-test` | Test setup |
| `abi-error` | Error handling pattern |
| `abi-stub` | Stub function pattern |
| `abi-io` | I/O backend initialization (Zig 0.16) |
| `abi-file-read` | File read with I/O |
| `abi-file-write` | File write with I/O |
| `abi-timer` | Timing measurement |
| `abi-agent` | Agent handler struct |
| `abi-vector-search` | Vector database search |
| `abi-leak-check` | Memory leak detection |
| `abi-comptime-config` | Comptime configuration |
| `abi-module` | Module with public API |

### Example: ABI Initialization

Type `abi-init` and press Tab to insert:

```zig
const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize I/O backend (required for Zig 0.16)
    var io_backend = std.Io.Threaded.init(allocator, .{
        .environ = std.process.Environ.init(),
    });
    defer io_backend.deinit();
    const io = io_backend.io();

    // Initialize ABI framework
    const config = abi.Config.init()
        .withAI(true)
        .withGPU(true)
        .withDatabase(true);

    var framework = try abi.Framework.init(allocator, config);
    defer framework.deinit();

    // Your code here
}
```

## Tasks

The extension provides task definitions that integrate with VS Code's task system. Access tasks via:

- `Terminal > Run Task...`
- `Ctrl+Shift+P` > "Tasks: Run Task"

Available tasks:
- **ABI: Build** - Build the project
- **ABI: Test** - Run all tests
- **ABI: Run** - Run the application
- **ABI: Format** - Format all Zig files
- **ABI: Generate Docs** - Generate API documentation

## Requirements

- **VS Code** 1.85.0 or higher
- **Zig** 0.16.0 or higher
- **ABI project** cloned locally

### Recommended Extensions

- [Zig Language](https://marketplace.visualstudio.com/items?itemName=ziglang.vscode-zig) - Official Zig language support
- [Error Lens](https://marketplace.visualstudio.com/items?itemName=usernamehw.errorlens) - Inline error display

## Development

### Building

```bash
npm install
npm run compile
```

### Watching for Changes

```bash
npm run watch
```

### Running Tests

```bash
npm test
```

### Packaging

```bash
npm run package
```

### Publishing

```bash
npm run publish
```

## Troubleshooting

### "zig: command not found"

Ensure Zig is installed and in your PATH, or set the `abi.zigPath` setting to the full path of the Zig executable.

### Build fails with feature errors

Check your feature flag settings. Some features may require specific dependencies or backends:

```json
{
  "abi.enableGpu": false
}
```

### Status bar not showing

Ensure `abi.showStatusBar` is set to `true` and the workspace contains an ABI project (has `src/abi.zig`).

## Contributing

Contributions are welcome. Please follow the ABI project's contribution guidelines.

## License

This extension is part of the ABI project and is licensed under the same terms.
