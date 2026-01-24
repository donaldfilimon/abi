/**
 * Node.js usage example for @anthropic/abi-wasm
 *
 * This example demonstrates using ABI WASM in a Node.js environment.
 *
 * Run with:
 *   node examples/node-usage.mjs
 */

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

// In production: import * as abi from '@anthropic/abi-wasm';
// For local development, we'll import directly:
import {
  init,
  shutdown,
  version,
  isReady,
  cosineSimilarity,
  VectorDatabase,
  normalize,
} from "../dist/index.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
  console.log("=== ABI WASM Node.js Example ===\n");

  try {
    // Load WASM binary from file system
    const wasmPath = join(__dirname, "..", "..", "..", "zig-out", "wasm", "abi.wasm");
    console.log(`Loading WASM from: ${wasmPath}`);

    const wasmBinary = await readFile(wasmPath);

    // Initialize with the pre-loaded binary
    await init({ wasmBinary });

    console.log(`ABI initialized: ${isReady()}`);
    console.log(`Version: ${version()}\n`);
  } catch (err) {
    // If WASM isn't built, demonstrate JS-only features
    console.log("Note: WASM module not available, using JS-only features\n");
    console.log(`(Build WASM with: zig build wasm)\n`);
  }

  // Vector operations work without WASM
  console.log("=== Vector Operations (JS Implementation) ===\n");

  const embeddings = {
    cat: normalize([0.8, 0.2, 0.1, 0.9, 0.3]),
    dog: normalize([0.7, 0.3, 0.2, 0.8, 0.4]),
    car: normalize([0.1, 0.9, 0.8, 0.1, 0.2]),
    truck: normalize([0.2, 0.8, 0.9, 0.2, 0.3]),
    apple: normalize([0.5, 0.1, 0.1, 0.5, 0.9]),
  };

  console.log("Computing pairwise similarities:\n");

  const labels = Object.keys(embeddings);
  for (let i = 0; i < labels.length; i++) {
    for (let j = i + 1; j < labels.length; j++) {
      const a = labels[i];
      const b = labels[j];
      const sim = cosineSimilarity(embeddings[a], embeddings[b]);
      console.log(`  ${a.padEnd(6)} <-> ${b.padEnd(6)}: ${sim.toFixed(4)}`);
    }
  }

  // Vector database example
  console.log("\n=== Vector Database ===\n");

  const db = new VectorDatabase({ name: "word_embeddings" });

  // Add all embeddings to database
  for (const [word, vec] of Object.entries(embeddings)) {
    db.add(vec, { word, category: getCategoryForWord(word) });
  }

  console.log(`Database "${db.getName()}" contains ${db.size()} vectors\n`);

  // Semantic search
  const queries = [
    { name: "animal-like", vector: normalize([0.75, 0.25, 0.15, 0.85, 0.35]) },
    { name: "vehicle-like", vector: normalize([0.15, 0.85, 0.85, 0.15, 0.25]) },
  ];

  for (const query of queries) {
    console.log(`Query: "${query.name}"`);
    const results = db.search(query.vector, 3);

    for (const result of results) {
      const meta = result.metadata;
      console.log(
        `  - ${meta.word} (${meta.category}): ${result.score.toFixed(4)}`
      );
    }
    console.log();
  }

  // Export/Import demonstration
  console.log("=== Database Persistence ===\n");

  const exported = db.toJSON();
  console.log(`Exported database: ${JSON.stringify(exported).length} bytes`);

  const restored = VectorDatabase.fromJSON(exported);
  console.log(`Restored database: ${restored.size()} vectors`);
  console.log(`Databases match: ${restored.size() === db.size()}\n`);

  // Cleanup
  shutdown();
  console.log("=== Done ===");
}

function getCategoryForWord(word) {
  const categories = {
    cat: "animal",
    dog: "animal",
    car: "vehicle",
    truck: "vehicle",
    apple: "food",
  };
  return categories[word] || "unknown";
}

main().catch(console.error);
