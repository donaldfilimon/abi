/**
 * Vector database for similarity search.
 */

import { AbiError, ErrorCode } from './error';
import { Simd } from './simd';

/**
 * Database configuration.
 */
export interface DatabaseConfig {
  /** Database name */
  name: string;
  /** Vector dimension */
  dimension: number;
  /** Initial capacity hint */
  initialCapacity?: number;
  /** Distance metric */
  metric?: 'cosine' | 'euclidean' | 'dot';
}

/**
 * Search result.
 */
export interface SearchResult {
  /** Vector ID */
  id: number;
  /** Similarity score */
  score: number;
}

/**
 * Internal vector storage.
 */
interface StoredVector {
  id: number;
  vector: Float32Array;
}

/**
 * In-memory vector database for similarity search.
 *
 * Note: This is a pure JavaScript implementation for browser use.
 * For production use with large datasets, consider using the native
 * ABI database through the C/WASM bindings.
 *
 * @example
 * ```typescript
 * const db = new VectorDatabase('embeddings', 128);
 *
 * // Insert vectors
 * await db.insert(1, new Float32Array(128));
 * await db.insert(2, new Float32Array(128));
 *
 * // Search
 * const results = await db.search(queryVector, 10);
 * ```
 */
export class VectorDatabase {
  private config: DatabaseConfig;
  private vectors: Map<number, StoredVector> = new Map();

  constructor(name: string, dimension: number, config?: Partial<DatabaseConfig>) {
    this.config = {
      name,
      dimension,
      initialCapacity: config?.initialCapacity ?? 1000,
      metric: config?.metric ?? 'cosine',
    };
  }

  /**
   * Get the database name.
   */
  get name(): string {
    return this.config.name;
  }

  /**
   * Get the vector dimension.
   */
  get dimension(): number {
    return this.config.dimension;
  }

  /**
   * Get the number of vectors.
   */
  get count(): number {
    return this.vectors.size;
  }

  /**
   * Insert a vector.
   *
   * @param id - Unique vector ID
   * @param vector - Vector data (must match database dimension)
   */
  async insert(id: number, vector: Float32Array | number[]): Promise<void> {
    const data = vector instanceof Float32Array ? vector : new Float32Array(vector);

    if (data.length !== this.config.dimension) {
      throw new AbiError(
        ErrorCode.InvalidArgument,
        `Vector dimension ${data.length} doesn't match database dimension ${this.config.dimension}`
      );
    }

    this.vectors.set(id, {
      id,
      vector: new Float32Array(data), // Copy to avoid external mutation
    });
  }

  /**
   * Search for similar vectors.
   *
   * @param query - Query vector
   * @param k - Maximum number of results
   * @returns Search results sorted by similarity (highest first)
   */
  async search(query: Float32Array | number[], k: number): Promise<SearchResult[]> {
    const queryData = query instanceof Float32Array ? query : new Float32Array(query);

    if (queryData.length !== this.config.dimension) {
      throw new AbiError(
        ErrorCode.InvalidArgument,
        `Query dimension ${queryData.length} doesn't match database dimension ${this.config.dimension}`
      );
    }

    // Compute similarities
    const scores: Array<{ id: number; score: number }> = [];

    for (const [id, stored] of this.vectors) {
      const score = this.computeScore(queryData, stored.vector);
      scores.push({ id, score });
    }

    // Sort by score (descending for similarity)
    scores.sort((a, b) => b.score - a.score);

    // Return top k
    return scores.slice(0, k).map(({ id, score }) => ({ id, score }));
  }

  /**
   * Delete a vector.
   *
   * @param id - Vector ID to delete
   */
  async delete(id: number): Promise<void> {
    this.vectors.delete(id);
  }

  /**
   * Get a vector by ID.
   *
   * @param id - Vector ID
   * @returns The vector data or null if not found
   */
  async get(id: number): Promise<Float32Array | null> {
    const stored = this.vectors.get(id);
    return stored ? new Float32Array(stored.vector) : null;
  }

  /**
   * Clear all vectors.
   */
  async clear(): Promise<void> {
    this.vectors.clear();
  }

  /**
   * Compute similarity/distance score.
   */
  private computeScore(a: Float32Array, b: Float32Array): number {
    switch (this.config.metric) {
      case 'cosine':
        return Simd.cosineSimilarity(a, b);
      case 'euclidean':
        // Negative distance so higher is better
        return -Simd.euclideanDistance(a, b);
      case 'dot':
        return Simd.dotProduct(a, b);
      default:
        return Simd.cosineSimilarity(a, b);
    }
  }

  /**
   * Export database to JSON.
   */
  toJSON(): { config: DatabaseConfig; vectors: Array<{ id: number; vector: number[] }> } {
    const vectors: Array<{ id: number; vector: number[] }> = [];
    for (const [, stored] of this.vectors) {
      vectors.push({
        id: stored.id,
        vector: Array.from(stored.vector),
      });
    }
    return { config: this.config, vectors };
  }

  /**
   * Import database from JSON.
   */
  static fromJSON(data: {
    config: DatabaseConfig;
    vectors: Array<{ id: number; vector: number[] }>;
  }): VectorDatabase {
    const db = new VectorDatabase(data.config.name, data.config.dimension, data.config);
    for (const { id, vector } of data.vectors) {
      db.vectors.set(id, { id, vector: new Float32Array(vector) });
    }
    return db;
  }
}
