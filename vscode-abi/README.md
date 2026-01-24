# ABI Framework VS Code Extension

VS Code extension for the ABI (Adaptive Binary Intelligence) Framework, providing build integration, AI chat, and GPU monitoring.

## Features

### Build & Test Integration
- **Build**: Run `zig build` with configurable flags
- **Test**: Run full test suite or filtered tests
- **Format**: Format code with `zig fmt`
- **Lint**: Check formatting compliance

### AI Chat Sidebar
- Interactive chat interface with the ABI agent
- Streaming responses
- Conversation history persistence
- Model selection via settings

### GPU Status Panel
- Real-time GPU backend information
- Device enumeration with memory usage
- Configurable refresh interval

### Diagnostics
- Inline error highlighting for Zig files
- Auto-refresh on file save
- Integration with VS Code Problems panel

### Status Bar
- Build status indicator (idle/building/success/error)
- Quick action menu (click status bar item)
- Last build timestamp

### Code Snippets
15 ABI-specific Zig snippets including:
- `abiimport` - Import the ABI framework
- `abiinit` - Initialize framework with config
- `abivectordb` - Create vector database
- `abillm` - LLM engine usage
- `abiagent` - Create AI agent
- `abigpu` - GPU context
- `abiio` - Zig 0.16 I/O backend

## Requirements

- VS Code 1.85.0 or higher
- ABI Framework binary (build with `zig build` in the workspace)
- Zig 0.16+ toolchain

## Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `abi.binaryPath` | Auto-detect | Path to ABI binary |
| `abi.buildFlags` | `[]` | Additional flags for `zig build` |
| `abi.gpu.refreshInterval` | `5000` | GPU status refresh interval (ms) |
| `abi.chat.model` | `gpt-oss` | Model for chat responses |

## Commands

| Command | Description |
|---------|-------------|
| `ABI: Build` | Build the project |
| `ABI: Test` | Run all tests |
| `ABI: Test (Filtered)` | Run tests with filter |
| `ABI: Format` | Format code |
| `ABI: Lint` | Check formatting |
| `ABI: Refresh GPU Status` | Refresh GPU panel |
| `ABI: Show Quick Actions` | Open status bar quick pick menu |
| `ABI: Run Diagnostics` | Manually trigger diagnostics |
| `ABI: Clear Diagnostics` | Clear all diagnostics |

## Development

```bash
# Install dependencies
npm install

# Compile
npm run compile

# Watch mode
npm run watch

# Package extension
npm run package
```

## Testing

```bash
# Install dependencies
npm install

# Compile extension and tests
npm run pretest

# Run tests (requires VS Code)
npm test
```

### Test Suites

| Suite | File | Coverage |
|-------|------|----------|
| Extension | `extension.test.ts` | Commands, configuration, initialization |
| Commands | `commands.test.ts` | Build, test, format, lint, GPU refresh |
| Chat | `chat.test.ts` | Chat configuration, model settings |
| GPU | `gpu.test.ts` | GPU configuration, refresh interval |
| Status Bar | `statusBar.test.ts` | Quick pick, diagnostics commands |

## Building from Source

1. Clone the ABI repository
2. Navigate to `vscode-abi/`
3. Run `npm install`
4. Run `npm run compile`
5. Press F5 to launch Extension Development Host

## License

MIT - See LICENSE file in the root repository.
