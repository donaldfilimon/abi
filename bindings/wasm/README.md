---
title: "WASM Bindings"
tags: [wasm, bindings, javascript, typescript]
---
# @anthropic/abi-wasm
> **Codebase Status:** Synced with repository as of 2026-01-24.

<p align="center">
  <img src="https://img.shields.io/badge/Platform-WebAssembly-654FF0?style=for-the-badge&logo=webassembly&logoColor=white" alt="WASM"/>
  <img src="https://img.shields.io/badge/npm-@anthropic%2Fabi--wasm-CB3837?style=for-the-badge&logo=npm&logoColor=white" alt="npm"/>
  <img src="https://img.shields.io/badge/TypeScript-Ready-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript"/>
</p>

WebAssembly bindings for the ABI high-performance AI and vector database framework.

## Installation

```bash
npm install @anthropic/abi-wasm
```

## Quick Start

```typescript
import { init, shutdown, version, cosineSimilarity, VectorDatabase } from '@anthropic/abi-wasm';

// Initialize the WASM module
await init();

console.log(`ABI version: ${version()}`);

// Vector operations
const similarity = cosineSimilarity([1, 2, 3], [4, 5, 6]);
console.log(`Similarity: ${similarity}`);

// Vector database
const db = new VectorDatabase({ name: 'embeddings' });
db.add([1.0, 0.0, 0.0], { label: 'x-axis' });
db.add([0.0, 1.0, 0.0], { label: 'y-axis' });
db.add([0.0, 0.0, 1.0], { label: 'z-axis' });

const results = db.search([0.9, 0.1, 0.0], 2);
console.log(results);

// Cleanup
shutdown();
```

## Features

- **High-performance WASM core** - Native-speed vector operations
- **TypeScript support** - Full type definitions included
- **Vector operations** - Cosine similarity, dot product, normalization, and more
- **In-memory vector database** - Fast similarity search with metadata support
- **Browser and Node.js** - Works in both environments
- **Zero dependencies** - Lightweight and self-contained

## API Reference

### Initialization

```typescript
// Basic initialization (auto-detects WASM path)
await init();

// With custom WASM path
await init({ wasmPath: '/path/to/abi.wasm' });

// With pre-loaded binary (useful in Node.js)
import { readFile } from 'fs/promises';
const binary = await readFile('./abi.wasm');
await init({ wasmBinary: binary });

// Check initialization status
if (isReady()) {
  console.log(`Version: ${version()}`);
}

// Cleanup when done
shutdown();
```

### Vector Operations

```typescript
import {
  cosineSimilarity,
  dotProduct,
  l2Norm,
  normalize,
  vectorAdd,
  vectorSub,
  vectorScale,
} from '@anthropic/abi-wasm';

const a = [1, 2, 3, 4];
const b = [4, 3, 2, 1];

// Similarity measures
cosineSimilarity(a, b);  // Range: -1 to 1
dotProduct(a, b);        // Sum of element-wise products

// Vector math
l2Norm(a);               // Euclidean length
normalize(a);            // Unit vector
vectorAdd(a, b);         // Element-wise addition
vectorSub(a, b);         // Element-wise subtraction
vectorScale(a, 2.0);     // Scalar multiplication
```

### Vector Database

```typescript
import { VectorDatabase } from '@anthropic/abi-wasm';

// Create a database
const db = new VectorDatabase({ name: 'my_vectors' });

// Add vectors with optional metadata
const id1 = db.add([1.0, 0.0, 0.0], { label: 'x-axis', color: 'red' });
const id2 = db.add([0.0, 1.0, 0.0], { label: 'y-axis', color: 'green' });

// Search for similar vectors
const results = db.search([0.9, 0.1, 0.0], 5); // top 5 results
// Returns: [{ id: number, score: number, metadata?: object }, ...]

// Get database info
db.getName();        // 'my_vectors'
db.size();           // 2
db.getDimensions();  // 3

// Get/remove vectors
db.get(id1);         // { id, vector, metadata }
db.remove(id1);      // true if found

// Persistence
const json = db.toJSON();
const restored = VectorDatabase.fromJSON(json);

// Clear all vectors
db.clear();
```

### Memory Management

For advanced use cases requiring direct WASM memory access:

```typescript
import { alloc, free, getMemory, getExports } from '@anthropic/abi-wasm';

// Allocate memory in WASM linear memory
const ptr = alloc(1024);  // 1KB

// Access memory directly
const memory = getMemory();
const view = new Uint8Array(memory.buffer);
view.set(data, ptr);

// Free when done
free(ptr, 1024);

// Access raw exports for custom operations
const exports = getExports();
```

## Browser Usage

```html
<script type="module">
  import { init, VectorDatabase } from 'https://unpkg.com/@anthropic/abi-wasm';

  async function main() {
    await init({ wasmPath: '/abi.wasm' });

    const db = new VectorDatabase({ name: 'demo' });
    db.add([1, 0, 0], { label: 'x' });

    const results = db.search([0.9, 0.1, 0], 1);
    console.log(results);
  }

  main();
</script>
```

## Node.js Usage

```javascript
import { readFile } from 'node:fs/promises';
import { init, VectorDatabase } from '@anthropic/abi-wasm';

async function main() {
  // Load WASM binary from file system
  const wasmBinary = await readFile('./node_modules/@anthropic/abi-wasm/wasm/abi.wasm');
  await init({ wasmBinary });

  const db = new VectorDatabase({ name: 'demo' });
  // ... use the database
}

main();
```

## Building from Source

### Prerequisites

- Node.js 18+
- Zig 0.16.x (for WASM compilation)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/donaldfilimon/abi.git
cd abi/bindings/wasm

# Install dependencies
npm install

# Build WASM module
npm run build:wasm

# Build TypeScript
npm run build:ts

# Or build everything
npm run build
```

### Development

```bash
# Run tests
npm test

# Watch mode for tests
npm run test:watch

# Lint
npm run lint

# Format
npm run format
```

## TypeScript Types

Full TypeScript definitions are included:

```typescript
import type {
  AbiWasmExports,
  AbiInitOptions,
  SearchResult,
  VectorDatabaseConfig,
} from '@anthropic/abi-wasm';
```

## Performance Notes

- Vector operations in the TypeScript wrapper use pure JavaScript for portability
- For maximum performance, ensure the WASM module is loaded (operations will use WASM when available)
- The `VectorDatabase` class uses a linear search; for large datasets (>10,000 vectors), consider using the native database backend via the full ABI framework

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## See Also

- [ABI Framework](https://github.com/donaldfilimon/abi) - Main repository
- [Python Bindings](../python/README.md) - Python interface
- [C Bindings](../c/README.md) - C header files
- [API Reference](../../API_REFERENCE.md) - Full API documentation
