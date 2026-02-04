# ABI JavaScript/WASM Bindings

JavaScript and TypeScript bindings for the ABI Framework, with WASM SIMD support for high-performance vector operations in the browser.

## Features

- **SIMD Vector Operations**: Fast vector math with WASM SIMD
- **Vector Database**: In-memory similarity search
- **WebGPU Acceleration**: GPU compute when available
- **TypeScript Support**: Full type definitions

## Installation

```bash
npm install @abi/core
```

Or link locally for development:

```bash
cd bindings/js
npm install
npm link
```

## Usage

### Basic Setup

```typescript
import { Abi, Simd, VectorDatabase } from '@abi/core';

// Initialize ABI (loads WASM module)
const abi = await Abi.init();
console.log('ABI version:', abi.version());
```

### SIMD Operations

```typescript
import { Simd } from '@abi/core';

// Check SIMD availability
console.log('SIMD available:', Simd.isAvailable());
console.log('Capabilities:', Simd.capabilities());

// Vector operations
const a = [1, 2, 3, 4];
const b = [5, 6, 7, 8];

const sum = Simd.add(a, b);
const dot = Simd.dotProduct(a, b);
const norm = Simd.l2Norm(a);
const similarity = Simd.cosineSimilarity(a, b);
const distance = Simd.euclideanDistance(a, b);
const normalized = Simd.normalize(a);
```

### Vector Database

```typescript
import { VectorDatabase } from '@abi/core';

// Create a 128-dimensional database
const db = new VectorDatabase('embeddings', 128);

// Insert vectors
await db.insert(1, new Float32Array(128).fill(0.1));
await db.insert(2, new Float32Array(128).fill(0.2));

// Search
const query = new Float32Array(128).fill(0.15);
const results = await db.search(query, 10);

for (const result of results) {
  console.log(`ID: ${result.id}, Score: ${result.score}`);
}

// Export/Import for persistence
const json = db.toJSON();
localStorage.setItem('embeddings', JSON.stringify(json));

const loaded = VectorDatabase.fromJSON(JSON.parse(localStorage.getItem('embeddings')!));
```

### GPU Acceleration (WebGPU)

```typescript
import { Gpu, Backend } from '@abi/core';

if (await Gpu.isAvailable()) {
  const gpu = await Gpu.init({
    backend: Backend.Auto,
    powerPreference: 'high-performance',
  });

  console.log('Backend:', gpu.backendName);

  const info = await gpu.getAdapterInfo();
  console.log('GPU:', info?.device);

  gpu.destroy();
}
```

## Building

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Build WASM (requires Zig)
npm run build:wasm

# Run tests
npm test
```

## Browser Support

- **Chrome 94+**: Full WebGPU and WASM SIMD support
- **Firefox 118+**: WASM SIMD, WebGPU behind flag
- **Safari 16.4+**: WASM SIMD, WebGPU partial
- **Edge 94+**: Same as Chrome

## API Reference

### `Abi`

Main framework class.

- `static init(options?): Promise<Abi>` - Initialize the framework
- `version(): string` - Get version string
- `isFeatureEnabled(feature): boolean` - Check feature status
- `shutdown(): void` - Cleanup resources

### `Simd`

SIMD vector operations.

- `static isAvailable(): boolean` - Check SIMD support
- `static capabilities(): SimdCaps` - Get capability flags
- `static add(a, b): Float32Array` - Element-wise addition
- `static dotProduct(a, b): number` - Dot product
- `static l2Norm(v): number` - L2 norm
- `static cosineSimilarity(a, b): number` - Cosine similarity (-1 to 1)
- `static normalize(v): Float32Array` - Normalize to unit length
- `static euclideanDistance(a, b): number` - Euclidean distance

### `VectorDatabase`

In-memory vector database.

- `constructor(name, dimension, config?)` - Create database
- `insert(id, vector): Promise<void>` - Insert vector
- `search(query, k): Promise<SearchResult[]>` - Search similar
- `delete(id): Promise<void>` - Delete vector
- `get(id): Promise<Float32Array | null>` - Get by ID
- `clear(): Promise<void>` - Clear all vectors
- `toJSON()` / `fromJSON(data)` - Serialization

### `Gpu`

WebGPU acceleration.

- `static isAvailable(): Promise<boolean>` - Check WebGPU support
- `static init(config?): Promise<Gpu>` - Initialize GPU
- `backendName: string` - Active backend name
- `getAdapterInfo(): Promise<AdapterInfo>` - GPU info
- `destroy(): void` - Cleanup

## License

MIT License - see [LICENSE](../../LICENSE) for details.
