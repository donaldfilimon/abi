# Database Domain Refactoring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `src/core/database/diskann.zig` into a focused `src/core/database/diskann/` directory structure.

**Architecture:** Use the "thin re-export facade" pattern to maintain API compatibility while improving modularity.

**Tech Stack:** Zig 0.16

---

### Task 1: Initialize Directory Structure

**Files:**
- Create: `src/core/database/diskann/`
- Create: `src/core/database/diskann/codebook.zig`
- Create: `src/core/database/diskann/graph.zig`
- Create: `src/core/database/diskann/index.zig`

- [ ] **Step 1: Create directory**
```bash
mkdir -p src/core/database/diskann/
```

- [ ] **Step 2: Create files (empty placeholders for now)**
```bash
touch src/core/database/diskann/codebook.zig src/core/database/diskann/graph.zig src/core/database/diskann/index.zig
```

- [ ] **Step 3: Commit**
```bash
git add src/core/database/diskann/
git commit -m "refactor: initialize diskann directory structure"
```

### Task 2: Decompose and Refactor diskann.zig

**Files:**
- Modify: `src/core/database/diskann.zig`
- Modify: `src/core/database/diskann/codebook.zig`
- Modify: `src/core/database/diskann/graph.zig`
- Modify: `src/core/database/diskann/index.zig`

- [ ] **Step 1: Identify and extract code**
(Manually read `src/core/database/diskann.zig`, partition into logical units, and write to new files)

- [ ] **Step 2: Create Facade**
Replace contents of `src/core/database/diskann.zig` with:
```zig
//! DiskANN Index — re-export facade
pub const PQCodebook = @import("diskann/codebook.zig").PQCodebook;
pub const VamanaGraph = @import("diskann/graph.zig").VamanaGraph;
pub const DiskANNIndex = @import("diskann/index.zig").DiskANNIndex;
// ... (include other original public declarations)
```

- [ ] **Step 3: Verify and Commit**
```bash
./build.sh test --summary all
git commit -m "refactor: decompose diskann.zig into focused sub-modules"
```
