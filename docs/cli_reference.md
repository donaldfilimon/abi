# CLI Reference

This document provides comprehensive documentation for the Abi AI Framework command-line interface, including detailed usage patterns, examples, and advanced configuration options.

## Table of Contents

- [Core Commands](#core-commands)
- [Configuration Options](#configuration-options)
- [Memory Management](#memory-management)
- [Performance Monitoring](#performance-monitoring)
- [Development Tools](#development-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Core Commands

### `abi help`

Display help information and available commands.

```bash
abi help                    # Show all commands
abi help chat              # Show help for specific command
abi help --verbose         # Show detailed help with examples
```

### `abi version`

Display version information and build details.

```bash
abi version                 # Show basic version info
abi version --verbose       # Show detailed build information
abi version --system        # Show system information
```

### `abi chat`

Start an interactive AI chat session with configurable persona and backend.

```bash
# Basic usage
abi chat

# With specific persona
abi chat --persona creative
abi chat --persona analytical
abi chat --persona technical

# With different backends
abi chat --backend openai --api-key your-key
abi chat --backend anthropic --model claude-3-sonnet

# Advanced configuration
abi chat --persona adaptive --backend local --verbose --log-level info
```

**Options:**
- `--persona <type>`: AI persona (adaptive, creative, analytical, technical)
- `--backend <type>`: Backend (local, openai, anthropic)
- `--api-key <key>`: API key for cloud backends
- `--model <model>`: Model name for cloud backends
- `--stream`: Enable streaming responses
- `--max-tokens <n>`: Maximum response length
- `--temperature <0.0-1.0>`: Response creativity

### `abi train`

Train a neural network model from input data.

```bash
# Basic training
abi train data.txt --output model.bin

# Advanced training with GPU acceleration
abi train dataset/ --output trained_model.bin --gpu --threads 8

# Training with memory monitoring
abi train large_dataset.txt --memory-track --memory-warn 1GB

# Training with performance profiling
abi train data.txt --profile --profile-output training_profile.json

# Custom training parameters
abi train data.txt \
  --learning-rate 0.001 \
  --batch-size 32 \
  --epochs 100 \
  --validation-split 0.2 \
  --early-stopping 10
```

**Options:**
- `--output, -o <file>`: Output model file
- `--gpu`: Enable GPU acceleration
- `--threads, -t <n>`: Number of training threads
- `--learning-rate <rate>`: Learning rate (default: 0.001)
- `--batch-size <size>`: Training batch size (default: 32)
- `--epochs <n>`: Number of training epochs
- `--validation-split <ratio>`: Validation data ratio
- `--early-stopping <n>`: Early stopping patience
- `--momentum <value>`: SGD momentum
- `--weight-decay <value>`: L2 regularization

### `abi serve`

Start a model serving server for inference requests.

```bash
# Basic model serving
abi serve model.bin

# Advanced configuration
abi serve model.bin \
  --port 8080 \
  --host 0.0.0.0 \
  --max-connections 1000 \
  --timeout 30 \
  --gpu

# With monitoring
abi serve model.bin --memory-track --profile

# Production configuration
abi serve model.bin \
  --wdbx-production \
  --sharding 8 \
  --replication 3 \
  --compression lz4
```

**Options:**
- `--port <port>`: Server port (default: 8080)
- `--host <address>`: Bind address (default: 127.0.0.1)
- `--max-connections <n>`: Maximum concurrent connections
- `--timeout <seconds>`: Request timeout
- `--model-path <path>`: Path to model file
- `--workers <n>`: Number of worker processes

### `abi benchmark`

Run comprehensive performance benchmarks.

```bash
# Run all benchmarks
abi benchmark

# Run specific benchmark categories
abi benchmark --category neural
abi benchmark --category simd
abi benchmark --category memory
abi benchmark --category database

# With performance monitoring
abi benchmark --memory-track --profile

# Custom benchmark parameters
abi benchmark \
  --iterations 10000 \
  --warmup 1000 \
  --confidence 0.99 \
  --min-time 10 \
  --max-time 300

# Continuous benchmarking
abi benchmark --continuous --interval 3600 --threshold 5.0

# Compare against baseline
abi benchmark --compare baseline.json --output comparison.html
```

**Options:**
- `--category <name>`: Benchmark category to run
- `--iterations <n>`: Number of iterations
- `--warmup <n>`: Warm-up iterations
- `--confidence <0.0-1.0>`: Statistical confidence level
- `--min-time <seconds>`: Minimum benchmark time
- `--max-time <seconds>`: Maximum benchmark time
- `--continuous`: Enable continuous benchmarking
- `--interval <seconds>`: Continuous mode interval
- `--threshold <percent>`: Performance regression threshold
- `--compare <file>`: Compare against baseline file

### `abi analyze`

Analyze text files and generate reports.

```bash
# Basic text analysis
abi analyze document.txt

# Multiple files
abi analyze *.txt --recursive

# With output formatting
abi analyze data/ --format json --output analysis.json
abi analyze document.txt --format csv --output stats.csv

# Advanced analysis
abi analyze large_file.txt \
  --memory-track \
  --chunk-size 1MB \
  --parallel 4
```

**Options:**
- `--format <type>`: Output format (text, json, yaml, csv)
- `--output, -o <file>`: Output file
- `--recursive, -r`: Process directories recursively
- `--chunk-size <size>`: Processing chunk size
- `--parallel <n>`: Number of parallel workers
- `--include-pattern <pattern>`: File inclusion pattern
- `--exclude-pattern <pattern>`: File exclusion pattern

### `abi convert`

Convert between different model formats.

```bash
# Convert ONNX to internal format
abi convert model.onnx --output model.bin

# Convert with specific format
abi convert model.bin --output model.onnx --format onnx
abi convert model.bin --output model.json --format json

# Convert with validation
abi convert model.onnx --output model.bin --validate

# Batch conversion
abi convert input_dir/ --output output_dir/ --format binary
```

**Supported Formats:**
- `binary`: Internal Abi format (default)
- `onnx`: Open Neural Network Exchange
- `tflite`: TensorFlow Lite
- `json`: JSON representation
- `yaml`: YAML representation
- `csv`: CSV data format

## Configuration Options

### Memory Management

```bash
# Memory tracking and monitoring
abi --memory-track command                    # Enable memory tracking
abi --memory-profile command                  # Enable detailed profiling
abi --memory-warn 100MB command              # Memory usage warning threshold
abi --memory-critical 500MB command          # Memory usage critical threshold
abi --leak-threshold 1000000000 command      # Leak detection threshold (ns)

# Memory allocation settings
abi --memory-pool-size 1GB command           # Memory pool size
abi --memory-max-alloc 2GB command           # Maximum allocation size
abi --memory-alignment 64 command            # Memory alignment
```

### Performance Monitoring

```bash
# Performance profiling
abi --profile command                        # Enable performance profiling
abi --profile-output profile.json command     # Profile output file
abi --profile-functions command               # Profile function calls
abi --profile-memory command                  # Profile memory operations

# Performance counters
abi --performance-counters command            # Enable hardware counters
abi --counter-frequency command               # CPU frequency counter
abi --counter-cache command                   # Cache performance counters
abi --counter-branch command                  # Branch prediction counters
```

### Development and Debugging

```bash
# Debugging options
abi --verbose command                        # Enable verbose output
abi --debug command                          # Enable debug mode
abi --quiet command                          # Suppress non-error output
abi --log-level <level> command              # Set log level (trace, debug, info, warn, err)

# Development features
abi --hot-reload command                     # Enable hot code reloading
abi --dev-mode command                       # Enable development features
abi --test-mode command                      # Enable test mode features
```

### GPU and Acceleration

```bash
# GPU configuration
abi --gpu command                            # Enable GPU acceleration
abi --no-gpu command                         # Disable GPU acceleration
abi --gpu-backend <type> command             # GPU backend (auto, vulkan, metal, d3d12)
abi --gpu-device <id> command                # Specific GPU device ID

# SIMD configuration
abi --simd command                           # Enable SIMD optimizations (default)
abi --no-simd command                        # Disable SIMD optimizations
abi --simd-level <level> command             # SIMD optimization level (basic, optimized, aggressive)
```

### Threading and Concurrency

```bash
# Threading options
abi --threads 8 command                      # Number of worker threads
abi --threads auto command                   # Auto-detect thread count
abi --thread-affinity command                # Enable CPU affinity
abi --thread-pinning command                 # Pin threads to CPU cores

# Concurrency settings
abi --max-concurrent 100 command             # Maximum concurrent operations
abi --queue-size 1000 command                # Internal queue size
abi --timeout 30 command                     # Operation timeout (seconds)
```

## Memory Management

The CLI includes comprehensive memory management features:

### Memory Tracking

```bash
# Enable memory tracking for any command
abi --memory-track benchmark
abi --memory-track serve model.bin

# Detailed memory profiling
abi --memory-profile --memory-report memory.json benchmark

# Memory leak detection
abi --leak-detection --leak-threshold 5000000000 serve model.bin
```

### Memory Thresholds

```bash
# Set memory usage thresholds
abi --memory-warn 100MB --memory-critical 500MB benchmark

# Memory pool configuration
abi --memory-pool-size 1GB --memory-max-alloc 2GB train data.txt
```

### Memory Reports

```bash
# Generate memory usage reports
abi --memory-report memory.json benchmark
abi --memory-report-html memory.html benchmark

# Continuous memory monitoring
abi --memory-monitor --memory-interval 1000 benchmark
```

## Performance Monitoring

### Performance Profiling

```bash
# Basic performance profiling
abi --profile benchmark

# Detailed profiling with output
abi --profile --profile-output profile.json benchmark

# Function-level profiling
abi --profile-functions --profile-call-graph benchmark
```

### Performance Counters

```bash
# Hardware performance counters
abi --performance-counters benchmark

# Specific counter categories
abi --counter-cpu --counter-memory --counter-cache benchmark
```

### Benchmarking Features

```bash
# Statistical benchmarking
abi benchmark --confidence 0.99 --iterations 10000

# Performance regression detection
abi benchmark --baseline baseline.json --threshold 5.0

# Continuous performance monitoring
abi benchmark --continuous --interval 3600
```

## Development Tools

### Logging and Debugging

```bash
# Logging configuration
abi --log-level debug --log-file abi.log benchmark

# Structured logging
abi --structured-log --log-format json benchmark

# Debug mode
abi --debug --verbose --trace benchmark
```

### Testing and Validation

```bash
# Run tests with CLI
abi test --memory-check --performance-check

# Validation mode
abi validate --comprehensive model.bin

# Health checks
abi health --detailed --report health.json
```

### Hot Reloading

```bash
# Enable hot code reloading
abi --hot-reload serve model.bin

# Development mode with auto-reload
abi --dev-mode --watch src/ serve model.bin
```

## Examples

### Complete Workflow

```bash
# 1. Train a model with monitoring
abi --memory-track --profile train data.txt \
  --output my_model.bin \
  --gpu \
  --threads 8 \
  --epochs 50

# 2. Validate the trained model
abi validate my_model.bin --comprehensive

# 3. Benchmark the model performance
abi benchmark --category neural --iterations 1000 \
  --memory-track \
  --profile-output benchmark_profile.json

# 4. Serve the model with monitoring
abi serve my_model.bin \
  --port 8080 \
  --memory-track \
  --profile \
  --memory-warn 1GB \
  --performance-counters
```

### Development Workflow

```bash
# Development with full monitoring
abi --dev-mode --verbose --debug --memory-track chat

# Testing with memory leak detection
abi --memory-profile --leak-detection test

# Performance optimization workflow
abi --profile --performance-counters benchmark
abi --profile --profile-compare previous_profile.json benchmark
```

### Production Deployment

```bash
# Production server with monitoring
abi serve model.bin \
  --host 0.0.0.0 \
  --port 80 \
  --max-connections 10000 \
  --timeout 60 \
  --wdbx-production \
  --sharding 16 \
  --replication 3 \
  --compression lz4 \
  --memory-monitor \
  --performance-counters \
  --log-level warn \
  --log-file /var/log/abi.log
```

## Troubleshooting

### Common Issues

**Memory Issues:**
```bash
# Diagnose memory problems
abi --memory-profile --memory-report memory.json command
abi --leak-detection --verbose command

# Fix memory issues
abi --memory-pool-size 2GB --memory-max-alloc 4GB command
```

**Performance Issues:**
```bash
# Profile performance problems
abi --profile --profile-output profile.json command
abi --performance-counters --verbose command

# Optimize performance
abi --gpu --threads 16 --simd-level aggressive command
```

**Connection Issues:**
```bash
# Debug connection problems
abi --verbose --log-level debug --trace serve model.bin

# Fix connection issues
abi --max-connections 500 --timeout 120 serve model.bin
```

### Error Codes

- `1001`: Memory allocation failed
- `1002`: Memory leak detected
- `2001`: GPU initialization failed
- `2002`: GPU device not found
- `3001`: Model file corrupted
- `3002`: Model format not supported
- `4001`: Network connection failed
- `4002`: Timeout exceeded
- `5001`: Invalid configuration
- `5002`: Missing required parameter

### Getting Help

```bash
# Get help for specific commands
abi help command-name
abi help --verbose

# Report issues with diagnostic information
abi health --detailed --report diagnostic.json

# Get system information
abi version --system --verbose
```

## Advanced Configuration

### Configuration Files

```bash
# Use configuration file
abi --config config.json command

# Generate default configuration
abi config --generate default.json

# Validate configuration
abi config --validate my_config.json
```

### Environment Variables

- `ABI_MEMORY_WARN`: Memory warning threshold
- `ABI_MEMORY_CRITICAL`: Memory critical threshold
- `ABI_GPU_BACKEND`: Preferred GPU backend
- `ABI_LOG_LEVEL`: Default log level
- `ABI_API_KEY`: Default API key
- `ABI_MODEL_PATH`: Default model path

### Docker Integration

```bash
# Run with Docker
docker run abi:latest --memory-track benchmark

# With custom configuration
docker run -v config.json:/app/config.json abi:latest --config /app/config.json serve model.bin
```

This CLI reference covers the most important commands and options. For the latest updates and additional features, run `abi help --verbose` or check the online documentation.
