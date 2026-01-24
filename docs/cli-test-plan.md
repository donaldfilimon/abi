# CLI Test Plan for ABI Framework (Zig 0.16)

This document provides a comprehensive test plan for all CLI commands and their subcommands.

## Environment Requirements

- **Zig Version**: 0.16.x (requires kernel support for `O_TMPFILE` atomic file operations)
- **Build Command**: `zig build`
- **Run Command**: `zig build run -- <command>`

> **Note**: Zig 0.16 dev builds require `O_TMPFILE` support (Linux kernel 3.11+). On environments without this support (errno 95 EOPNOTSUPP), use a native Linux system or Docker with a compatible kernel.

---

## Test Categories

### 1. Top-Level Commands

| Command | Expected Behavior | Test Command |
|---------|-------------------|--------------|
| `--help` | Show main help text | `abi --help` |
| `-h` | Show main help text | `abi -h` |
| `help` | Show main help text | `abi help` |
| `version` | Show version string | `abi version` |
| `--list-features` | List all features and status | `abi --list-features` |
| `--enable-<feature>` | Enable feature for run | `abi --enable-gpu system-info` |
| `--disable-<feature>` | Disable feature for run | `abi --disable-ai system-info` |

---

### 2. Database Commands (`db`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| `help` | | `abi db help` |
| `stats` | | `abi db stats` |
| `add` | `--id <n> --embed <text>` | `abi db add --id 1 --embed "Hello world"` |
| `search` | `--embed <text> --top <n>` | `abi db search --embed "similar text" --top 5` |
| `backup` | `--path <file>` | `abi db backup --path backup.db` |
| `restore` | `--path <file>` | `abi db restore --path backup.db` |

**Common Options:**
- `--path <path>` - Database file path (default: wdbx_data)
- `--dimensions <n>` - Vector dimensions (default: 384)
- `--metric <type>` - Distance metric: cosine, euclidean, dot

---

### 3. Agent Commands (`agent`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| (default) | Interactive session | `abi agent` |
| `help` | | `abi agent help` |
| `ralph` | `--task <text> --iterations <n>` | `abi agent ralph --task "Analyze code"` |

**Options:**
| Option | Description | Example |
|--------|-------------|---------|
| `--name <name>` | Agent name | `abi agent --name test-agent` |
| `-m, --message <msg>` | One-shot message | `abi agent -m "Hello"` |
| `-s, --session <n>` | Session name | `abi agent -s project-x` |
| `--load <id>` | Load session by ID | `abi agent --load session_123` |
| `--persona <type>` | Persona type | `abi agent --persona coder` |
| `--show-prompt` | Display full prompt | `abi agent --show-prompt -m "Hi"` |
| `--list-sessions` | List saved sessions | `abi agent --list-sessions` |
| `--list-personas` | List persona types | `abi agent --list-personas` |

**Personas to Test:**
- `assistant` (default), `coder`, `writer`, `analyst`, `companion`, `docs`, `reviewer`, `minimal`, `abbey`, `ralph`

**Interactive Commands:**
- `/help`, `/save [name]`, `/load <id>`, `/sessions`, `/clear`, `/info`, `/history`, `/prompt`, `/persona`, `/personas`, `exit`, `quit`

---

### 4. GPU Commands (`gpu`)

| Subcommand | Description | Test Command |
|------------|-------------|--------------|
| (default) | Show backends + devices | `abi gpu` |
| `help` | Show help | `abi gpu help` |
| `backends` | List GPU backends | `abi gpu backends` |
| `summary` | Show GPU summary | `abi gpu summary` |
| `devices` / `list` | List detected devices | `abi gpu devices` |
| `default` | Show default device | `abi gpu default` |
| `status` | Show native/fallback status | `abi gpu status` |

---

### 5. LLM Commands (`llm`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| `help` | | `abi llm help` |
| `info <model>` | | `abi llm info ./model.gguf` |
| `generate <model>` | See options below | `abi llm generate ./model.gguf -p "Hello"` |
| `chat <model>` | | `abi llm chat ./model.gguf` |
| `bench <model>` | `--prompt-tokens`, `--gen-tokens` | `abi llm bench ./model.gguf` |
| `list` | | `abi llm list` |
| `list-local [dir]` | | `abi llm list-local ./models` |
| `download <url>` | `-o, --output` | `abi llm download https://example.com/model.gguf` |
| `demo` | `-p, -n` | `abi llm demo -p "Test prompt"` |

**Generate Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <path>` | Model file path | - |
| `-p, --prompt <text>` | Input prompt | - |
| `-n, --max-tokens <n>` | Max tokens | 256 |
| `-t, --temperature <f>` | Temperature | 0.7 |
| `--top-p <f>` | Nucleus sampling | 0.9 |
| `--top-k <n>` | Top-k filtering | 40 |
| `--repeat-penalty <f>` | Repetition penalty | 1.1 |
| `--seed <n>` | Random seed | - |
| `--stop <text>` | Stop sequence | - |
| `--stream` | Streaming output | false |
| `--tfs <f>` | Tail-free sampling | 1.0 |
| `--mirostat <n>` | Mirostat mode (0/1/2) | 0 |
| `--mirostat-tau <f>` | Mirostat tau | 5.0 |
| `--mirostat-eta <f>` | Mirostat eta | 0.1 |

**Chat Commands:**
- `/quit`, `/exit`, `/clear`, `/system`, `/help`, `/stats`

---

### 6. Training Commands (`train`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| `help` | | `abi train help` |
| `run` | See options below | `abi train run --epochs 10` |
| `llm <model>` | See LLM options | `abi train llm model.gguf --epochs 1` |
| `resume <ckpt>` | | `abi train resume ./checkpoint.ckpt` |
| `monitor [id]` | `--log-dir`, `--no-tui` | `abi train monitor` |
| `info` | | `abi train info` |

**Basic Training Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `-e, --epochs <n>` | Number of epochs | 10 |
| `-b, --batch-size <n>` | Batch size | 32 |
| `--model-size <n>` | Model parameters | 512 |
| `--sample-count <n>` | Training samples | 1024 |
| `--lr, --learning-rate <f>` | Learning rate | 0.001 |
| `--optimizer <type>` | sgd, adam, adamw | adamw |
| `--lr-schedule <type>` | LR schedule | warmup_cosine |
| `--warmup-steps <n>` | Warmup steps | 100 |
| `--weight-decay <f>` | Weight decay | 0.01 |
| `--gradient-clip <f>` | Gradient clip norm | 1.0 |
| `--checkpoint-interval <n>` | Checkpoint interval | 0 |
| `--mixed-precision` | Enable mixed precision | false |

**LLM Training Additional Options:**
| Option | Description |
|--------|-------------|
| `--max-seq-len <n>` | Max sequence length |
| `--grad-accum <n>` | Gradient accumulation |
| `--use-gpu` | Enable GPU training |
| `--dataset-url <url>` | Download dataset |
| `--dataset-path <path>` | Local dataset |
| `--dataset-format <type>` | tokenbin, text, jsonl |
| `--export-gguf <path>` | Export trained model |

---

### 7. Task Commands (`task`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| `help` | | `abi task help` |
| `add <title>` | `-p`, `-c`, `-d`, `--due` | `abi task add "Fix bug" -p high` |
| `list` / `ls` | `-s`, `-p`, `-c`, `--sort` | `abi task ls --status pending` |
| `show <id>` | | `abi task show 1` |
| `edit <id>` | `--title`, `--desc`, `-p`, `-c` | `abi task edit 1 --priority critical` |
| `done <id>` | | `abi task done 1` |
| `start <id>` | | `abi task start 1` |
| `cancel <id>` | | `abi task cancel 1` |
| `delete <id>` / `rm` | | `abi task rm 1` |
| `due <id> <date>` | | `abi task due 1 +7d` |
| `block <id> <by>` | | `abi task block 2 1` |
| `unblock <id>` | | `abi task unblock 2` |
| `stats` | | `abi task stats` |
| `import-roadmap` | | `abi task import-roadmap` |

**Status Values:** pending, in_progress, completed, cancelled, blocked
**Priority Values:** low, normal, high, critical
**Category Values:** personal, roadmap, compute, bug, feature
**Sort Fields:** created, updated, priority, due_date, status
**Due Date Formats:** +Nd (days), +Nh (hours), +Nm (minutes), clear, Unix timestamp

---

### 8. Network Commands (`network`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| (default) | Show status | `abi network` |
| `help` | | `abi network help` |
| `status` | | `abi network status` |
| `list` / `nodes` | | `abi network list` |
| `register <id> <addr>` | | `abi network register node-1 127.0.0.1:8080` |
| `unregister <id>` | | `abi network unregister node-1` |
| `touch <id>` | | `abi network touch node-1` |
| `set-status <id> <s>` | | `abi network set-status node-1 healthy` |

**Status Values:** healthy, degraded, offline

---

### 9. System Info (`system-info`)

| Test | Command |
|------|---------|
| Show full info | `abi system-info` |
| Show help | `abi system-info --help` |

---

### 10. Benchmark Commands (`bench`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| `help` | | `abi bench help` |
| `list` | | `abi bench list` |
| `all` | `--json`, `--output` | `abi bench all` |
| `<suite>` | simd, memory, concurrency, database, network, crypto, ai | `abi bench simd` |
| `quick` | | `abi bench quick` |
| `micro <op>` | hash, alloc, parse | `abi bench micro hash` |

---

### 11. Config Commands (`config`)

| Subcommand | Options | Test Command |
|------------|---------|--------------|
| `help` | | `abi config help` |
| `init` | `-o, --output` | `abi config init -o abi.json` |
| `show` | `--format`, path | `abi config show` |
| `validate` | path | `abi config validate abi.json` |
| `env` | | `abi config env` |

---

### 12. Embed Commands (`embed`)

| Option | Description | Test Command |
|--------|-------------|--------------|
| `help` | | `abi embed help` |
| `--text <t>` | Generate from text | `abi embed --text "Hello"` |
| `--file <path>` | Generate from file | `abi embed --file doc.txt` |
| `--provider <p>` | Provider: openai, ollama, mistral, cohere | `abi embed --text "Hi" --provider ollama` |
| `--model <m>` | Model name | `abi embed --text "Hi" --model text-embedding-3-small` |
| `--output <path>` | Save to file | `abi embed --text "Hi" -o out.json` |
| `--format <f>` | json, csv, raw | `abi embed --text "Hi" --format csv` |

---

### 13. SIMD Commands (`simd`)

| Option | Description | Test Command |
|--------|-------------|--------------|
| `help` | | `abi simd help` |
| (default) | Run SIMD benchmark | `abi simd` |
| `-s, --size <n>` | Vector size | `abi simd --size 1000000` |

---

### 14. TUI Commands (`tui`)

| Test | Command |
|------|---------|
| Launch TUI | `abi tui` |
| Show help | `abi tui --help` |

---

### 15. Other Commands

| Command | Description | Test Command |
|---------|-------------|--------------|
| `explore` | Code exploration | `abi explore --help` |
| `discord` | Discord integration | `abi discord --help` |
| `multi-agent` | Multi-agent orchestration | `abi multi-agent --help` |
| `convert` | Format conversion | `abi convert --help` |
| `completions` | Shell completions | `abi completions --help` |

---

## Test Script Template

```bash
#!/bin/bash
# CLI Test Script for ABI Framework

set -e

echo "=== ABI CLI Test Suite ==="

# Build
echo "[1/N] Building..."
zig build

# Test top-level help
echo "[2/N] Testing help..."
zig build run -- --help
zig build run -- help
zig build run -- -h

# Test version
echo "[3/N] Testing version..."
zig build run -- version

# Test system-info
echo "[4/N] Testing system-info..."
zig build run -- system-info

# Test feature flags
echo "[5/N] Testing feature flags..."
zig build run -- --list-features

# Test db help
echo "[6/N] Testing db commands..."
zig build run -- db help

# Test gpu commands
echo "[7/N] Testing gpu commands..."
zig build run -- gpu
zig build run -- gpu backends
zig build run -- gpu devices
zig build run -- gpu summary

# Test llm commands
echo "[8/N] Testing llm commands..."
zig build run -- llm help
zig build run -- llm list
zig build run -- llm demo

# Test train commands
echo "[9/N] Testing train commands..."
zig build run -- train help
zig build run -- train info

# Test task commands
echo "[10/N] Testing task commands..."
zig build run -- task help
zig build run -- task stats

# Test network commands
echo "[11/N] Testing network commands..."
zig build run -- network help

# Test bench commands
echo "[12/N] Testing bench commands..."
zig build run -- bench help
zig build run -- bench list

# Test config commands
echo "[13/N] Testing config commands..."
zig build run -- config help

# Test embed commands
echo "[14/N] Testing embed commands..."
zig build run -- embed help

# Test simd commands
echo "[15/N] Testing simd commands..."
zig build run -- simd help

# Test agent commands
echo "[16/N] Testing agent commands..."
zig build run -- agent help
zig build run -- agent --list-personas

echo "=== All tests passed ==="
```

---

## Error Handling Tests

| Scenario | Expected | Test Command |
|----------|----------|--------------|
| Unknown command | Error + help suggestion | `abi unknown` |
| Missing required arg | Error message | `abi db add` |
| Invalid option value | Error + valid values | `abi task add "X" -p invalid` |
| Feature disabled | Error + rebuild hint | `abi db stats` (with `-Denable-database=false`) |
| File not found | Error message | `abi llm info nonexistent.gguf` |

---

## Feature Flag Combinations

Test with different build flags:

```bash
# Minimal build
zig build -Denable-ai=false -Denable-gpu=false -Denable-database=false

# Full build
zig build -Denable-ai=true -Denable-gpu=true -Denable-database=true

# GPU backend combinations
zig build -Dgpu-backend=vulkan
zig build -Dgpu-backend=cuda
zig build -Dgpu-backend=auto
```

---

## Exit Code Verification

| Scenario | Expected Exit Code |
|----------|-------------------|
| Successful command | 0 |
| Help displayed | 0 |
| Command error | Non-zero |
| Invalid arguments | Non-zero |

---

## Notes

1. **Interactive commands** (`agent`, `llm chat`, `tui`, `train monitor`) require manual testing with actual terminal input
2. **External dependencies** (OpenAI, Ollama, etc.) require environment variables to be set
3. **GPU commands** require actual GPU hardware or will show emulated/fallback devices
4. **Training commands** require GGUF model files for full testing
5. **Network commands** require network feature to be enabled at build time
