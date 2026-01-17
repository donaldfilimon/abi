# ZLS Plugin for Claude Code

Zig Language Server (ZLS) integration for real-time code intelligence.

## Features

- Diagnostics (errors and warnings)
- Go to definition / find references
- Code completion and hover info
- Semantic highlighting

## Prerequisites

ZLS must be installed and in PATH:
```bash
# Check installation
zls --version
```

## Installation

```bash
claude plugin install zls-lsp --scope project --plugin-dir ./zls-plugin
```

## Configuration

Edit `.lsp.json` to customize:
- `warn_style`: Show style warnings
- `enable_build_on_save`: Auto-build on save
- `semantic_tokens`: Semantic token highlighting
