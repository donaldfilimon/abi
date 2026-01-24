/**
 * Basic usage example for @anthropic/abi-wasm
 *
 * This example demonstrates the core functionality of the ABI WASM bindings.
 *
 * Run with:
 *   npx tsx examples/basic-usage.ts
 */

import {
  init,
  shutdown,
  version,
  isReady,
  cosineSimilarity,
  dotProduct,
  l2Norm,
  normalize,
  VectorDatabase,
} from "../src/index";

async function main() {
  console.log("=== ABI WASM Basic Usage Example ===\n");

  // Initialize the WASM module
  console.log("Initializing ABI WASM...");
  await init({
    // For local development, point to the built WASM file
    wasmPath: "../../zig-out/wasm/abi.wasm",
  });

  console.log(`Initialized: ${isReady()}`);
  console.log(`Version: ${version()}\n`);

  // Vector operations
  console.log("=== Vector Operations ===\n");

  const vectorA = [1.0, 2.0, 3.0, 4.0];
  const vectorB = [4.0, 3.0, 2.0, 1.0];

  console.log(`Vector A: [${vectorA.join(", ")}]`);
  console.log(`Vector B: [${vectorB.join(", ")}]`);
  console.log(`Cosine Similarity: ${cosineSimilarity(vectorA, vectorB).toFixed(4)}`);
  console.log(`Dot Product: ${dotProduct(vectorA, vectorB)}`);
  console.log(`L2 Norm of A: ${l2Norm(vectorA).toFixed(4)}`);
  console.log(`Normalized A: [${normalize(vectorA).map((v) => v.toFixed(4)).join(", ")}]\n`);

  // Vector database
  console.log("=== Vector Database ===\n");

  const db = new VectorDatabase({ name: "embeddings" });

  // Add some vectors representing 3D axes
  const idX = db.add([1.0, 0.0, 0.0], { label: "x-axis", color: "red" });
  const idY = db.add([0.0, 1.0, 0.0], { label: "y-axis", color: "green" });
  const idZ = db.add([0.0, 0.0, 1.0], { label: "z-axis", color: "blue" });

  // Add a diagonal vector
  const idDiag = db.add(normalize([1.0, 1.0, 1.0]), { label: "diagonal" });

  console.log(`Database: ${db.getName()}`);
  console.log(`Size: ${db.size()} vectors`);
  console.log(`Dimensions: ${db.getDimensions()}\n`);

  // Search for vectors similar to a query
  const query = [0.9, 0.1, 0.0]; // Mostly pointing towards X
  console.log(`Query: [${query.join(", ")}]`);
  console.log("Search results (top 3):");

  const results = db.search(query, 3);
  for (const result of results) {
    console.log(
      `  ID: ${result.id}, Score: ${result.score.toFixed(4)}, ` +
        `Label: ${(result.metadata as Record<string, string>)?.label ?? "N/A"}`
    );
  }

  console.log("\n=== Done ===");

  // Cleanup
  shutdown();
}

main().catch(console.error);
