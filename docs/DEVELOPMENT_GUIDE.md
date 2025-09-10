# WDBX-AI Development Guide

## 🚀 **Getting Started**

### **Prerequisites**
- Zig 0.15.0 or later (tested with 0.16-dev)
- Git
- Windows 10/11, Linux, or macOS

### **Quick Setup**
```bash
# Clone the repository
git clone <repository-url>
cd abi

# Verify Zig installation
zig version

# Run initial build
zig build

# Run tests
zig build test
```

## 🏗️ **Project Architecture**

### **Directory Structure**
```
src/
├── ai/                     # AI and machine learning components
│   ├── enhanced_agent.zig  # Advanced AI agent implementation
│   └── mod.zig            # AI module exports
├── database/               # Database and storage
│   └── enhanced_db.zig     # High-performance database engine
├── plugins/                # Plugin system
├── simd/                   # SIMD-optimized operations
├── wdbx/                   # WDBX vector database core
└── root.zig               # Main module exports

benchmarks/                 # Performance benchmarking
├── main.zig               # Unified benchmark entry point
├── database_benchmark.zig # Database performance tests
└── performance_suite.zig  # General performance suite

tests/                     # Test suite
├── test_*.zig            # Unit and integration tests

docs/                      # Documentation
tools/                     # Development and analysis tools
```

### **Core Modules**

#### **Vector Database (WDBX)**
- High-performance vector operations with SIMD optimization
- Multiple API interfaces (CLI, HTTP, TCP, WebSocket)
- Production-ready with authentication and monitoring

#### **AI System**
- Enhanced AI agents with persona-based routing
- Local ML capabilities with training and inference
- Memory management and performance tracking

#### **Database Engine**
- Lock-free data structures for concurrent access
- HNSW (Hierarchical Navigable Small World) indexing
- Efficient storage and retrieval with caching

## 🛠️ **Development Workflow**

### **Build System**
```bash
# Standard build
zig build

# Run all tests
zig build test

# Run benchmarks (unified)
zig build benchmark

# Run specific benchmark type
zig run benchmarks/main.zig -- neural
zig run benchmarks/main.zig -- database
zig run benchmarks/main.zig -- performance
zig run benchmarks/main.zig -- all

# Static analysis
zig build analyze

# Legacy database benchmarks
zig build benchmark-db
```

### **Cross-Platform Compilation (Zig 0.16-dev)**
```bash
# Linux targets
zig build -Dtarget=x86_64-linux-gnu
zig build -Dtarget=aarch64-linux-gnu

# macOS targets
zig build -Dtarget=x86_64-macos
zig build -Dtarget=aarch64-macos

# WebAssembly (WASI)
zig build -Dtarget=wasm32-wasi

# Cross-verify multiple targets
zig build cross-platform
```

### **Conditional Compilation**
```zig
const builtin = @import("builtin");

fn platformInit() void {
    if (comptime builtin.os.tag == .windows) {
        // Windows-specific setup
    } else if (comptime builtin.os.tag == .linux) {
        // Linux-specific setup
    } else if (comptime builtin.os.tag == .macos) {
        // macOS-specific setup
    }
}
```

### **Windows Networking Guidance**
- Use Winsock (`recv`/`send`) on Windows to avoid `ReadFile` edge cases
- The diagnostic `zig build test-network` validates Windows socket behavior
- PowerShell helper: `fix_windows_networking.ps1`
- Server code uses conditional paths to ensure reliability on Windows

### **Code Quality Standards**

#### **Naming Conventions**
- **Functions**: `camelCase` (e.g., `processData`, `initVector`)
- **Types**: `PascalCase` (e.g., `Vector`, `Database`, `Agent`)
- **Constants**: `SCREAMING_SNAKE_CASE` (e.g., `MAX_BUFFER_SIZE`)
- **Variables**: `snake_case` (e.g., `user_input`, `file_path`)

#### **Documentation Requirements**
- All public functions must have `///` documentation
- Include parameter descriptions and return values
- Provide usage examples for complex APIs
- Document error conditions and handling

#### **Error Handling**
- Use specific error types (avoid generic `error` unions)
- Provide meaningful error messages
- Handle resource cleanup in defer blocks
- Test error conditions thoroughly

### **Testing Strategy**

#### **Test Types**
1. **Unit Tests**: Individual function/module testing
2. **Integration Tests**: Cross-module interaction testing
3. **Performance Tests**: Benchmark and regression testing
4. **Memory Tests**: Leak detection and safety validation

#### **Running Tests**
```bash
# All tests
zig build test

# Specific test file
zig test tests/test_database.zig

# Memory leak detection
zig test --test-filter "memory" tests/test_memory_management.zig

# Performance regression tests
zig test tests/test_performance_regression.zig
```

### **Performance Guidelines**

#### **SIMD Optimization**
- Use `@Vector` types for parallel operations
- Leverage platform-specific SIMD instructions
- Benchmark before and after optimizations
- Profile hot paths with appropriate tools

#### **Memory Management**
- Prefer arena allocators for bulk operations
- Use `defer` for resource cleanup
- Minimize allocations in hot paths
- Test for memory leaks regularly

#### **Lock-Free Programming**
- Use atomic operations for concurrent access
- Implement compare-and-swap patterns correctly
- Test thoroughly under high concurrency
- Document memory ordering requirements

## 🔧 **Development Tools**

### **Static Analysis**
```bash
# Run static analysis
zig build analyze

# Check for common issues
zig fmt --check src/

# Memory safety analysis
zig build test --sanitize-memory
```

### **Debugging**
```bash
# Debug build
zig build -Doptimize=Debug

# Enable logging
zig build -Dlog-level=debug

# Runtime safety checks
zig build -Doptimize=Debug -Dsafety=true
```

### **Profiling**
```bash
# Performance profiling
zig build benchmark -- performance

# Memory profiling
zig build test --track-memory

# GPU profiling (if available)
zig run src/gpu_examples.zig
```

## 📈 **Performance Monitoring**

### **Benchmark Results Tracking**
- Track performance trends over time
- Set regression thresholds (±5% tolerance)
- Monitor memory usage patterns
- Profile critical paths regularly

### **Key Metrics**
- **Database Operations**: Insert/query latency, throughput
- **Vector Operations**: SIMD efficiency, cache utilization
- **Memory Usage**: Peak RSS, allocation patterns, leak detection
- **Concurrency**: Lock contention, throughput scaling

## 🚀 **Deployment**

### **Production Builds**
```bash
# Optimized release build
zig build -Doptimize=ReleaseFast

# Small binary size
zig build -Doptimize=ReleaseSmall

# Production deployment
zig build -Doptimize=ReleaseFast -Dstrip=true
```

### **Environment Configuration**
- Set appropriate log levels for production
- Configure authentication and security settings
- Monitor system resources and performance
- Implement graceful shutdown handling

## 🤝 **Contributing**

### **Code Review Process**
1. Create feature branch from `main`
2. Implement changes with tests
3. Run full test suite locally
4. Submit pull request with detailed description
5. Address review feedback
6. Merge after approval

### **Commit Guidelines**
- Use conventional commit format
- Include relevant issue references
- Provide clear, descriptive messages
- Keep commits focused and atomic

### **Documentation Updates**
- Update relevant documentation for API changes
- Add examples for new features
- Update benchmarks for performance changes
- Maintain changelog for releases

## 📞 **Support**

### **Getting Help**
- Check existing documentation first
- Search issue tracker for similar problems
- Provide minimal reproduction cases
- Include system information and logs

### **Reporting Issues**
- Use appropriate issue templates
- Include steps to reproduce
- Provide system and version information
- Attach relevant logs and error messages

---

**Last Updated**: $(date)
**Version**: 1.0.0
