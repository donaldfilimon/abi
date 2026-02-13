# Split Large Files Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the 10 largest source files (1,374-2,398 lines each) into focused modules with tests extracted to `*_test.zig` files.

**Architecture:** Each file is split along natural struct/type boundaries. The original file becomes a hub that re-exports from new siblings. All new files use relative imports within the same directory. No public API changes — `mod.zig`/`stub.zig` signatures stay identical.

**Tech Stack:** Zig 0.16.0-dev.2535+b5bd49460, `zig fmt`, `zig build test --summary all` (baseline: 980 pass, 5 skip)

**Rules:**
- Never change `mod.zig` or `stub.zig` public signatures
- Every new file gets `const std = @import("std");` and only the imports it needs
- Tests move to `*_test.zig` and import the source via relative path
- Run `./zigw build test --summary all` after each task — must stay at 980/5
- Run `./zigw fmt .` before each commit
- Commit after each task

---

## Task 1: Split `trainable_model.zig` (2,398 lines → 5 files)

**Directory:** `src/features/ai/training/`

**Files:**
- Modify: `src/features/ai/training/trainable_model.zig` (keep lines 431-1851 + re-exports)
- Create: `src/features/ai/training/weights.zig` (lines 25-303: TrainableLayerWeights, TrainableWeights, ActivationCache)
- Create: `src/features/ai/training/checkpoint.zig` (lines 1852-2035: LoadError, GradientCheckpointer, ModelCheckpoint, helpers)
- Create: `src/features/ai/training/trainable_model_test.zig` (lines 2037-2398: all 11 tests)

**Step 1: Create `weights.zig`**

Cut lines 25-428 from trainable_model.zig (TrainableLayerWeights, TrainableWeights, ActivationCache) into new file `weights.zig`. Add the imports this code needs:
```zig
const std = @import("std");
const ops = @import("../llm/ops/mod.zig");
const gguf = @import("../llm/io/gguf.zig");
const quantized = @import("../llm/tensor/quantized.zig");
```

Keep the doc comments on each struct.

**Step 2: Create `checkpoint.zig`**

Cut lines 1852-2035 (LoadError, GradientCheckpointer, ModelCheckpoint, initializeXavier, dequantizeQ4_0, dequantizeQ8_0) into `checkpoint.zig`. Add imports:
```zig
const std = @import("std");
const model_config = @import("model/config.zig");
pub const CheckpointingStrategy = model_config.CheckpointingStrategy;
```

**Step 3: Create `trainable_model_test.zig`**

Cut lines 2037-2398 (all `test "..."` blocks) into `trainable_model_test.zig`. Import:
```zig
const std = @import("std");
const trainable_model = @import("trainable_model.zig");
const TrainableModel = trainable_model.TrainableModel;
const TrainableWeights = trainable_model.TrainableWeights;
// ... other types used in tests
```

**Step 4: Update `trainable_model.zig`**

Replace extracted code with re-exports:
```zig
const weights_mod = @import("weights.zig");
const checkpoint_mod = @import("checkpoint.zig");

pub const TrainableLayerWeights = weights_mod.TrainableLayerWeights;
pub const TrainableWeights = weights_mod.TrainableWeights;
pub const ActivationCache = weights_mod.ActivationCache;
pub const LoadError = checkpoint_mod.LoadError;
pub const GradientCheckpointer = checkpoint_mod.GradientCheckpointer;
pub const ModelCheckpoint = checkpoint_mod.ModelCheckpoint;
```

Keep `TrainableModel` struct in this file — it's the core implementation (~900 lines).

**Step 5: Run tests**

```bash
./zigw fmt . && ./zigw build test --summary all
```

Expected: 980 pass, 5 skip. If any test fails, the test file likely needs an additional import.

**Step 6: Commit**

```bash
git add src/features/ai/training/weights.zig src/features/ai/training/checkpoint.zig src/features/ai/training/trainable_model_test.zig src/features/ai/training/trainable_model.zig
git commit -m "refactor: split trainable_model.zig into weights, checkpoint, and test files"
```

---

## Task 2: Split `simd.zig` (1,993 lines → subdirectory)

**Directory:** `src/services/shared/` → `src/services/shared/simd/`

**Files:**
- Remove: `src/services/shared/simd.zig`
- Create: `src/services/shared/simd/mod.zig` (re-exports + SimdCapabilities)
- Create: `src/services/shared/simd/vector_ops.zig` (lines 21-263)
- Create: `src/services/shared/simd/activations.zig` (lines 264-696: activations + normalization)
- Create: `src/services/shared/simd/distances.zig` (lines 697-853: distances + matrix + capabilities)
- Create: `src/services/shared/simd/integer_ops.zig` (lines 854-1058: integer + FMA + scalar)
- Create: `src/services/shared/simd/extras.zig` (lines 1059-1399: element-wise + memory + v2 kernels)
- Create: `src/services/shared/simd/simd_test.zig` (lines 1402-1993: all tests)

**IMPORTANT:** This file is imported by 24+ files via `@import("simd.zig")` from `shared/mod.zig`. The `shared/mod.zig` re-export must change from:
```zig
pub const simd = @import("simd.zig");
```
to:
```zig
pub const simd = @import("simd/mod.zig");
```

**Step 1: Create directory and `mod.zig`**

Create `src/services/shared/simd/` directory. Create `mod.zig` that re-exports everything from submodules:
```zig
const vector_ops = @import("vector_ops.zig");
const activations = @import("activations.zig");
const distances = @import("distances.zig");
const integer_ops = @import("integer_ops.zig");
const extras = @import("extras.zig");

// Re-export all public functions
pub const vectorAdd = vector_ops.vectorAdd;
pub const vectorDot = vector_ops.vectorDot;
pub const vectorL2Norm = vector_ops.vectorL2Norm;
pub const cosineSimilarity = vector_ops.cosineSimilarity;
// ... every pub fn and const from simd.zig
```

Every public symbol from the original `simd.zig` must appear as a `pub const` re-export in `mod.zig`. Use the original file as a checklist.

**Step 2: Create submodule files**

Each file gets:
```zig
const std = @import("std");
const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;
```

Split functions by the section headers already in the file:
- `vector_ops.zig`: vectorAdd through hasSimdSupport (lines 21-263)
- `activations.zig`: siluInPlace through layerNormInPlace (lines 264-696)
- `distances.zig`: l2DistanceSquared through matrixMultiply + SimdCapabilities (lines 697-853)
- `integer_ops.zig`: vectorAddI32 through fmaScalar + scalar ops (lines 854-1058)
- `extras.zig`: hadamard through scale (lines 1059-1399)

Some functions call others across boundaries (e.g., `softmaxInPlace` calls `maxValue`, `expSubtractMax`, `sum`, `divideByScalar`). These must either:
- Stay together (preferred — keep softmax with its helpers in activations.zig), OR
- Import from sibling: `const vector_ops = @import("vector_ops.zig");`

**Step 3: Create `simd_test.zig`**

Move all test blocks (lines 1402-1993) into `simd_test.zig`. Import via:
```zig
const simd = @import("mod.zig");
```

**Step 4: Update `shared/mod.zig`**

Change the simd import path from `simd.zig` to `simd/mod.zig`.

**Step 5: Delete old `simd.zig`**

```bash
git rm src/services/shared/simd.zig
```

**Step 6: Run tests**

```bash
./zigw fmt . && ./zigw build test --summary all
```

Expected: 980 pass, 5 skip.

**Step 7: Commit**

```bash
git add src/services/shared/simd/ src/services/shared/mod.zig
git commit -m "refactor: split simd.zig into subdirectory with focused modules"
```

---

## Task 3: Split `rest.zig` (1,680 lines → 3 files)

**Directory:** `src/services/connectors/discord/`

**Files:**
- Modify: `src/services/connectors/discord/rest.zig` (keep Config + Client core methods, lines 1-190)
- Create: `src/services/connectors/discord/endpoints.zig` (Client endpoint methods, lines ~211-1189)
- Create: `src/services/connectors/discord/rest_test.zig` (tests if any)

**Step 1: Extract endpoint methods**

The `Client` struct is 1,611 lines. The core (init, deinit, makeRequest, executeRequest) is lines 69-210. Everything after is endpoint-specific methods grouped by API resource.

Create `endpoints.zig` with endpoint functions as standalone functions that take `*Client` as first parameter. OR, since Zig doesn't support partial struct definitions, keep `Client` in `rest.zig` but extract the endpoint method bodies into helper functions in `endpoints.zig`:

```zig
// endpoints.zig
const std = @import("std");
const types = @import("types.zig");
const Client = @import("rest.zig").Client;

pub fn getCurrentUser(client: *Client) !types.User { ... }
pub fn getGuild(client: *Client, guild_id: []const u8) !types.Guild { ... }
// ... all 60+ endpoint functions
```

Then in `rest.zig`, the Client methods delegate:
```zig
pub fn getCurrentUser(self: *Client) !types.User {
    return endpoints.getCurrentUser(self);
}
```

**Alternative (simpler):** Since the Client struct methods all follow the same pattern (build URL, call makeRequest/makeRequestWithBody, parse response), and tests for this file are minimal, just extract tests and leave the file as-is. It's a 1:1 mapping of Discord API endpoints — splitting it adds indirection without much benefit.

**Recommended: Extract tests only.** This file is a thin API client; splitting the struct across files would add complexity. Extract any tests to `rest_test.zig` and move on.

**Step 2: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/services/connectors/discord/
git commit -m "refactor: extract discord rest tests to separate file"
```

---

## Task 4: Split `self_learning.zig` (1,643 lines → 6 files)

**Directory:** `src/features/ai/training/`

**Files:**
- Modify: `src/features/ai/training/self_learning.zig` (keep SelfLearningSystem, ~320 lines)
- Create: `src/features/ai/training/learning_types.zig` (lines 51-182: SelfLearningConfig, ExperienceType, FeedbackType, LearningExperience)
- Create: `src/features/ai/training/experience_buffer.zig` (lines 183-352: ExperienceBuffer, SampledBatch)
- Create: `src/features/ai/training/reward_policy.zig` (lines 353-679: RewardModel, PolicyNetwork)
- Create: `src/features/ai/training/dpo_optimizer.zig` (lines 680-881: DPOOptimizer)
- Create: `src/features/ai/training/self_learning_test.zig` (lines 1589-1643: 3 tests)

**Step 1: Create type files**

Each file contains one or two related structs. `learning_types.zig` has the config and enum types. `experience_buffer.zig` has the buffer. `reward_policy.zig` has RewardModel and PolicyNetwork. `dpo_optimizer.zig` has DPOOptimizer.

Note: FeedbackIntegrator (lines 882-1018), VisionTrainer (lines 1019-1111), and DocumentTrainer (lines 1112-1265) can stay in `self_learning.zig` alongside `SelfLearningSystem` since they're tightly coupled to it.

**Step 2: Update `self_learning.zig`**

Add re-exports at the top:
```zig
pub const learning_types = @import("learning_types.zig");
pub const SelfLearningConfig = learning_types.SelfLearningConfig;
pub const ExperienceType = learning_types.ExperienceType;
// ... etc
```

**Step 3: Create test file, run tests, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/ai/training/learning_types.zig src/features/ai/training/experience_buffer.zig src/features/ai/training/reward_policy.zig src/features/ai/training/dpo_optimizer.zig src/features/ai/training/self_learning_test.zig src/features/ai/training/self_learning.zig
git commit -m "refactor: split self_learning.zig into focused modules"
```

---

## Task 5: Split `hnsw.zig` (1,599 lines → 4 files)

**Directory:** `src/features/database/`

**Files:**
- Modify: `src/features/database/hnsw.zig` (keep HnswIndex, ~1,100 lines)
- Create: `src/features/database/search_state.zig` (lines 17-151: SearchState, SearchStatePool)
- Create: `src/features/database/distance_cache.zig` (lines 152-259: DistanceCache)
- Create: `src/features/database/hnsw_test.zig` (lines 1365-1599: 7 tests)

**Step 1: Create `search_state.zig`**

Extract SearchState and SearchStatePool. Imports needed:
```zig
const std = @import("std");
```

**Step 2: Create `distance_cache.zig`**

Extract DistanceCache. Imports needed:
```zig
const std = @import("std");
```

**Step 3: Create `hnsw_test.zig`**

Move all test blocks. Import:
```zig
const std = @import("std");
const hnsw = @import("hnsw.zig");
const HnswIndex = hnsw.HnswIndex;
const SearchStatePool = hnsw.SearchStatePool;
const DistanceCache = hnsw.DistanceCache;
```

**Step 4: Update `hnsw.zig`**

Add re-exports:
```zig
pub const SearchState = @import("search_state.zig").SearchState;
pub const SearchStatePool = @import("search_state.zig").SearchStatePool;
pub const DistanceCache = @import("distance_cache.zig").DistanceCache;
```

HnswIndex uses SearchState and DistanceCache internally — update its field types to reference the re-exported types.

**Step 5: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/database/search_state.zig src/features/database/distance_cache.zig src/features/database/hnsw_test.zig src/features/database/hnsw.zig
git commit -m "refactor: extract search_state and distance_cache from hnsw.zig"
```

---

## Task 6: Split `dispatcher.zig` (1,424 lines → 4 files)

**Directory:** `src/features/gpu/`

**Files:**
- Modify: `src/features/gpu/dispatcher.zig` (keep KernelDispatcher, ~940 lines)
- Create: `src/features/gpu/dispatch_types.zig` (lines 67-192: errors, configs, handles)
- Create: `src/features/gpu/batched_dispatch.zig` (lines 1132-1366: BatchedOp, BatchedDispatcher)
- Create: `src/features/gpu/dispatcher_test.zig` (lines 1367-1424: 4 tests)

**Step 1-4:** Same pattern as above. Extract types, extract batch operations, extract tests, add re-exports.

**Step 5: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/gpu/dispatch_types.zig src/features/gpu/batched_dispatch.zig src/features/gpu/dispatcher_test.zig src/features/gpu/dispatcher.zig
git commit -m "refactor: split dispatcher.zig into types, batched, and test files"
```

---

## Task 7: Split `vulkan.zig` (1,423 lines → 3 files)

**Directory:** `src/features/gpu/backends/`

**Files:**
- Modify: `src/features/gpu/backends/vulkan.zig` (keep VulkanBackend + vtable, lines 919-1390)
- Create: `src/features/gpu/backends/vulkan_types.zig` (lines 11-632: errors, types, handles, extern structs, function pointers)
- Create: `src/features/gpu/backends/vulkan_test.zig` (lines 1391-1423: 3 tests)

**Step 1-3:** Extract Vulkan type definitions (they're pure type definitions with no runtime code). Extract tests. Add re-exports.

**Step 4: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/gpu/backends/vulkan_types.zig src/features/gpu/backends/vulkan_test.zig src/features/gpu/backends/vulkan.zig
git commit -m "refactor: extract vulkan types and tests to separate files"
```

---

## Task 8: Split `metal.zig` (1,374 lines → 3 files)

**Directory:** `src/features/gpu/backends/`

**Files:**
- Modify: `src/features/gpu/backends/metal.zig` (keep init through synchronize, lines 218-1157)
- Create: `src/features/gpu/backends/metal_types.zig` (lines 25-216: errors, types, global state, ObjC runtime)
- Create: `src/features/gpu/backends/metal_test.zig` (lines 1335-1374: 4 tests)

**Step 1-3:** Extract Metal type definitions and ObjC runtime setup. Extract tests. Add re-exports.

Note: Global state variables (metal_lib, objc_lib, etc.) must stay accessible from both files. Either keep them in `metal.zig` and have `metal_types.zig` be pure types, or put globals in `metal_types.zig` and import from `metal.zig`.

**Recommended:** Keep globals in `metal.zig` (they're implementation state). Extract only pure type definitions and error types to `metal_types.zig`.

**Step 4: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/gpu/backends/metal_types.zig src/features/gpu/backends/metal_test.zig src/features/gpu/backends/metal.zig
git commit -m "refactor: extract metal types and tests to separate files"
```

---

## Task 9: Split `multi_device.zig` (1,409 lines → 5 files)

**Directory:** `src/features/gpu/`

**Files:**
- Modify: `src/features/gpu/multi_device.zig` (keep as re-export hub, ~50 lines)
- Create: `src/features/gpu/device_group.zig` (lines 122-486: DeviceGroup, WorkDistribution, PeerTransfer, DeviceBarrier)
- Create: `src/features/gpu/gpu_cluster.zig` (lines 488-1062: GPUCluster, ReduceOp, ParallelismStrategy, etc.)
- Create: `src/features/gpu/gradient_sync.zig` (lines 1064-1213: GradientBucket, GradientBucketManager)
- Create: `src/features/gpu/multi_device_test.zig` (lines 1215-1409: 11 tests)

**Step 1:** Create `device_group.zig` with DeviceGroup and related types (DeviceId, DeviceType, DeviceCapabilities, DeviceInfo, LoadBalanceStrategy, MultiDeviceConfig stay here since DeviceGroup uses them).

**Step 2:** Create `gpu_cluster.zig` with GPUCluster and advanced clustering types.

**Step 3:** Create `gradient_sync.zig` with gradient bucket management.

**Step 4:** Create `multi_device_test.zig` with all 11 tests.

**Step 5:** Update `multi_device.zig` to re-export from all subfiles.

**IMPORTANT:** This file is imported by 13 files including `gpu/mod.zig` and `mega/coordinator.zig`. All public types must remain accessible via `@import("multi_device.zig")`.

**Step 6: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/gpu/device_group.zig src/features/gpu/gpu_cluster.zig src/features/gpu/gradient_sync.zig src/features/gpu/multi_device_test.zig src/features/gpu/multi_device.zig
git commit -m "refactor: split multi_device.zig into device_group, cluster, and gradient modules"
```

---

## Task 10: Split `server.zig` (1,487 lines → 3 files)

**Directory:** `src/features/ai/streaming/`

**Files:**
- Modify: `src/features/ai/streaming/server.zig` (keep StreamingServer core, ~800 lines)
- Create: `src/features/ai/streaming/request_types.zig` (lines 1232-1368: AbiStreamRequest)
- Create: `src/features/ai/streaming/server_test.zig` (lines 1370-1487: 13 tests)

**Step 1:** Extract AbiStreamRequest to `request_types.zig`.

**Step 2:** Extract all 13 tests to `server_test.zig`.

**Step 3:** Add re-export in `server.zig`:
```zig
pub const AbiStreamRequest = @import("request_types.zig").AbiStreamRequest;
```

**Step 4: Run tests, format, commit**

```bash
./zigw fmt . && ./zigw build test --summary all
git add src/features/ai/streaming/request_types.zig src/features/ai/streaming/server_test.zig src/features/ai/streaming/server.zig
git commit -m "refactor: extract streaming request types and tests from server.zig"
```

---

## Final Verification

After all 10 tasks:

```bash
./zigw build test --summary all    # Must be 980 pass, 5 skip
./zigw build validate-flags        # Must pass all 16 flag combos
./zigw fmt .                       # Must be clean
```

If validate-flags fails on any combo, the issue is likely a stub.zig that needs updating or a re-export that's missing.
