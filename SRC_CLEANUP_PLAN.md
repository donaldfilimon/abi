# src/ Directory Cleanup Plan

## Current State

The `src/` directory currently contains both library code (which is now duplicated in `lib/`) and application code. After the consolidation to `lib/`, we need to clean up `src/`.

## What to Keep in src/

### Application Code (Keep)
- ✅ `comprehensive_cli.zig` - Main CLI application (referenced by build.zig)
- ✅ `cli/` - CLI-specific modules and commands
- ✅ `tools/` - Development and utility tools
- ✅ `examples/` - Example programs and demos
- ✅ `tests/` - Application-specific tests
- ✅ `bootstrap/main.zig` - Bootstrap entry point (if needed)

### Compatibility/Legacy (Keep for now)
- ⚠️ `compat.zig` - Compatibility shim (review if still needed)
- ⚠️ `root.zig` - Legacy root (review if still needed)

## What to Remove from src/

### Library Code (Now in lib/ - Can Remove)
- ❌ `core/` → Use `lib/core/`
- ❌ `features/` → Use `lib/features/`
- ❌ `framework/` → Use `lib/framework/`
- ❌ `shared/` → Use `lib/shared/`
- ❌ `mod.zig` → Use `lib/mod.zig`

### Standalone Modules (Move or Remove)
- ❓ `agent/` - Agent-specific code (may belong in lib/features/ai/)
- ❓ `connectors/` - Connector code (may belong in lib/features/connectors/)
- ❓ `ml/` - ML code (may belong in lib/features/ai/)
- ❓ `metrics.zig` - Metrics code (may belong in lib/monitoring/)
- ❓ `simd.zig` - SIMD code (may belong in lib/shared/)

## Recommended Actions

### Phase 1: Verify Duplication
Before removing, ensure lib/ has equivalent or better versions:

```bash
# Compare key files
diff -r src/core lib/core
diff -r src/features lib/features
diff -r src/framework lib/framework
diff -r src/shared lib/shared
```

### Phase 2: Check Dependencies
Identify files in src/ that import from the duplicated modules:

```bash
# Find imports of src/core, src/features, etc. in non-lib files
grep -r "@import(\"core" src/ --include="*.zig"
grep -r "@import(\"features" src/ --include="*.zig"
grep -r "@import(\"framework" src/ --include="*.zig"
```

### Phase 3: Move Standalone Modules
Determine where standalone modules belong:

1. **agent/** → Likely belongs in `lib/features/ai/agent/`
2. **connectors/** → Already in `lib/features/connectors/`
3. **ml/** → Likely belongs in `lib/features/ai/ml/`
4. **metrics.zig** → Likely belongs in `lib/features/monitoring/`
5. **simd.zig** → Already in `lib/shared/simd.zig`

### Phase 4: Remove Duplicates
Once verified, remove duplicated directories:

```bash
# After backing up and verifying
rm -rf src/core
rm -rf src/features
rm -rf src/framework
rm -rf src/shared
rm src/mod.zig  # If not needed
```

### Phase 5: Update Imports
Update any remaining files in src/ to import from lib:

**Before:**
```zig
const core = @import("../core/mod.zig");
const features = @import("../features/mod.zig");
```

**After:**
```zig
const abi = @import("abi");
const core = abi.core;
const features = abi.features;
```

## Expected Final Structure

```
src/
├── comprehensive_cli.zig    # Main CLI entry
├── cli/                     # CLI modules
│   ├── commands/
│   ├── main.zig
│   └── ...
├── tools/                   # Development tools
│   ├── docs_generator.zig
│   ├── performance_profiler.zig
│   └── ...
├── examples/                # Example programs
│   ├── ai_demo.zig
│   ├── gpu_demo.zig
│   └── ...
├── tests/                   # Application tests
│   ├── integration/
│   └── ...
├── bootstrap/               # Bootstrap code
│   └── main.zig
├── compat.zig              # Compatibility (if needed)
└── root.zig                # Legacy root (if needed)
```

## Verification Steps

1. ✅ Ensure lib/ is complete and has all modules
2. ✅ Update build.zig to use lib/mod.zig
3. ⏳ Check all imports in src/ reference lib or abi
4. ⏳ Remove duplicate directories
5. ⏳ Run `zig build test-all` to verify
6. ⏳ Update documentation

## Migration Checklist

- [x] lib/core is complete with all modules
- [x] lib/features is synced with src/features
- [x] lib/framework is synced with src/framework
- [x] lib/shared is synced with src/shared
- [x] build.zig uses lib/mod.zig
- [ ] Check agent/ - move to lib/features/ai/ if library code
- [ ] Check connectors/ - already in lib/features/connectors/
- [ ] Check ml/ - move to lib/features/ai/ if library code
- [ ] Check metrics.zig - move to lib/features/monitoring/
- [ ] Check simd.zig - already in lib/shared/
- [ ] Update imports in src/comprehensive_cli.zig
- [ ] Update imports in src/tools/
- [ ] Update imports in src/examples/
- [ ] Remove src/core, src/features, src/framework, src/shared
- [ ] Remove src/mod.zig
- [ ] Run tests
- [ ] Update CONTRIBUTING.md

---

*Plan created: 2025-10-16*
*Status: Phase 1 - Verification*
