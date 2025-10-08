# ABI Framework Repository Redesign Plan

## Current Issues

1. **Redundant modules**: Multiple similar files (runtime.zig vs runtime_modern.zig)
2. **Mixed concerns**: CLI, features, and framework code scattered across locations
3. **Documentation bloat**: Many overlapping documentation files
4. **Script proliferation**: Scripts scattered without clear organization
5. **Inconsistent naming**: Mix of camelCase and snake_case
6. **Build complexity**: Multiple entry points and unclear dependencies

## Proposed New Architecture

```
abi/
├── lib/                    # Core library code
│   ├── core/              # Fundamental types and utilities
│   │   ├── collections.zig
│   │   ├── allocators.zig
│   │   ├── errors.zig
│   │   └── types.zig
│   ├── features/          # Feature modules
│   │   ├── ai/
│   │   ├── database/
│   │   ├── gpu/
│   │   ├── web/
│   │   ├── monitoring/
│   │   └── connectors/
│   ├── framework/         # Framework orchestration
│   │   ├── runtime.zig
│   │   ├── config.zig
│   │   └── lifecycle.zig
│   ├── shared/           # Shared utilities
│   │   ├── utils/
│   │   ├── logging/
│   │   ├── platform/
│   │   └── simd/
│   └── mod.zig           # Main library entry point
├── bin/                  # Executable entry points
│   ├── abi-cli.zig
│   └── abi-server.zig
├── examples/             # Usage examples
│   ├── basic-usage.zig
│   └── advanced-features.zig
├── tests/               # Test suite
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── tools/               # Development and build tools
│   ├── build/
│   ├── dev/
│   └── deploy/
├── docs/               # Documentation
│   ├── api/
│   ├── guides/
│   └── reference/
├── config/             # Configuration files
│   ├── default.zig
│   ├── production.zig
│   └── development.zig
├── build.zig           # Main build script
├── build.zig.zon       # Package metadata
└── README.md           # Project overview
```

## Key Improvements

### 1. Clear Separation of Concerns
- **lib/**: Pure library code with no executable dependencies
- **bin/**: Executable entry points that use the library
- **examples/**: Demonstrations of library usage
- **tools/**: Development and deployment utilities

### 2. Simplified Module Structure
- Single `mod.zig` entry point per major component
- Eliminate redundant files (e.g., merge modern/legacy versions)
- Consistent naming conventions (snake_case for files)

### 3. Streamlined Documentation
- **api/**: Generated API documentation
- **guides/**: User guides and tutorials
- **reference/**: Technical reference materials

### 4. Organized Configuration
- Environment-specific configuration files
- Clear separation between development and production settings

### 5. Consolidated Tools
- **build/**: Build and packaging tools
- **dev/**: Development utilities
- **deploy/**: Deployment scripts

## Migration Strategy

1. Create new directory structure
2. Move and consolidate existing files
3. Update import paths and build configuration
4. Remove redundant and deprecated files
5. Update documentation
6. Validate build and test suite

## Benefits

- **Maintainability**: Clear structure makes code easier to navigate
- **Modularity**: Clean separation enables independent development
- **Scalability**: Organized structure supports future growth
- **Developer Experience**: Intuitive organization improves onboarding
- **Build Performance**: Simplified dependencies reduce compilation time