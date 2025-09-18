# Abi Framework Module Organization

Organized module structure with clear separation of concerns for the Abi AI Framework.

## 📁 Module Architecture

```
src/
├── root.zig          # Main module interface
├── core/             # Foundation utilities
├── ai/               # AI/ML functionality
├── database/         # Vector database
├── net/              # Networking
├── perf/             # Performance monitoring
├── gpu/              # GPU computing
├── ml/               # ML algorithms
├── api/              # C API bindings
├── simd/             # SIMD operations
├── plugins/          # Plugin system
└── wdbx/             # CLI interface
```

## 🔧 Module Details

### Core Module (`core/`)
- **Purpose**: Framework foundation
- **Components**: Memory management, error handling, cross-platform utilities
- **Dependencies**: None

### AI Module (`ai/`)
- **Purpose**: AI capabilities
- **Components**: Neural networks, agents, embeddings
- **Dependencies**: `core/`, `simd/`

### Database Module (`database/`)
- **Purpose**: Vector database operations
- **Components**: HNSW indexing, vector storage, query optimization
- **Dependencies**: `core/`, `simd/`

### Networking Module (`net/`)
- **Purpose**: HTTP client and communication
- **Components**: HTTP client, curl wrapper, weather API
- **Dependencies**: `core/`

### Performance Module (`perf/`)
- **Purpose**: Performance monitoring
- **Components**: Metrics, profiling, memory tracking
- **Dependencies**: `core/`

### GPU Module (`gpu/`)
- **Purpose**: GPU acceleration
- **Components**: Buffer management, shaders, matrix ops
- **Dependencies**: `core/`, `simd/`

### ML Module (`ml/`)
- **Purpose**: Machine learning algorithms
- **Components**: Training, inference, optimization
- **Dependencies**: `core/`, `simd/`, `perf/`

### API Module (`api/`)
- **Purpose**: C bindings
- **Components**: C API, foreign interfaces
- **Dependencies**: `core/`, `database/`

### SIMD Module (`simd/`)
- **Purpose**: Vector operations
- **Components**: Vector math, SIMD instructions
- **Dependencies**: None

### Plugins Module (`plugins/`)
- **Purpose**: Extensibility
- **Components**: Plugin loading, registry
- **Dependencies**: `core/`

### WDBX Module (`wdbx/`)
- **Purpose**: CLI interface
- **Components**: CLI processing, servers, configuration
- **Dependencies**: All modules

## 🔗 Dependencies

```
core/ ←─┬─ ai/ ←─┬─ ml/
        │       │
        ├─ net/ │
        ├─ perf/┼─┬─ gpu/
        ├─ api/─┼─┼─┬─ plugins/
        ├─ simd/┼─┼─┼─┬─ wdbx/
        └─ database/ ┼─┼─┼─┘
```

## 🏗️ Build Integration

- Module definitions in each subdirectory
- Dependency resolution in build.zig
- Platform-specific compilation
- Comprehensive test coverage

## 📚 Usage

```zig
// Import modules
const abi = @import("abi");
const db = abi.database.Db.init(allocator);

// Cross-module communication
const perf = abi.perf.PerformanceMonitor.init();
perf.startOperation("query");
const results = try db.search(query_vector, 10);
perf.endOperation();
```

## 🔍 Module Discovery

1. Check `mod.zig` files for documentation
2. Run tests: `zig build test-[module]`
3. View examples in `examples/`
4. Generated API docs in `docs/api/`

## 🎯 Benefits

- Clear separation of concerns
- Minimal coupling between modules
- Easy maintenance and testing
- Scalable architecture
- Intuitive navigation
