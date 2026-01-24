/**
 * Tests for @anthropic/abi-wasm
 */

import { describe, it, expect, beforeEach } from "vitest";
import {
  cosineSimilarity,
  dotProduct,
  l2Norm,
  normalize,
  vectorAdd,
  vectorSub,
  vectorScale,
  VectorDatabase,
} from "../index";

describe("Vector Operations", () => {
  describe("cosineSimilarity", () => {
    it("should return 1 for identical vectors", () => {
      const vec = [1, 2, 3, 4];
      expect(cosineSimilarity(vec, vec)).toBeCloseTo(1);
    });

    it("should return 0 for orthogonal vectors", () => {
      const a = [1, 0, 0];
      const b = [0, 1, 0];
      expect(cosineSimilarity(a, b)).toBeCloseTo(0);
    });

    it("should return -1 for opposite vectors", () => {
      const a = [1, 0, 0];
      const b = [-1, 0, 0];
      expect(cosineSimilarity(a, b)).toBeCloseTo(-1);
    });

    it("should throw for mismatched lengths", () => {
      expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow();
    });

    it("should handle zero vectors", () => {
      const zero = [0, 0, 0];
      const vec = [1, 2, 3];
      expect(cosineSimilarity(zero, vec)).toBe(0);
    });
  });

  describe("dotProduct", () => {
    it("should compute correct dot product", () => {
      const a = [1, 2, 3];
      const b = [4, 5, 6];
      expect(dotProduct(a, b)).toBe(1 * 4 + 2 * 5 + 3 * 6); // 32
    });

    it("should return 0 for orthogonal vectors", () => {
      const a = [1, 0];
      const b = [0, 1];
      expect(dotProduct(a, b)).toBe(0);
    });

    it("should throw for mismatched lengths", () => {
      expect(() => dotProduct([1, 2], [1])).toThrow();
    });
  });

  describe("l2Norm", () => {
    it("should compute correct L2 norm", () => {
      const vec = [3, 4]; // 3-4-5 triangle
      expect(l2Norm(vec)).toBe(5);
    });

    it("should return 0 for zero vector", () => {
      expect(l2Norm([0, 0, 0])).toBe(0);
    });

    it("should return 1 for unit vectors", () => {
      expect(l2Norm([1, 0, 0])).toBe(1);
      expect(l2Norm([0, 1, 0])).toBe(1);
    });
  });

  describe("normalize", () => {
    it("should produce unit vectors", () => {
      const vec = [3, 4];
      const normalized = normalize(vec);
      expect(l2Norm(normalized)).toBeCloseTo(1);
    });

    it("should preserve direction", () => {
      const vec = [2, 0, 0];
      const normalized = normalize(vec);
      expect(normalized).toEqual([1, 0, 0]);
    });

    it("should handle zero vector", () => {
      const zero = [0, 0, 0];
      const normalized = normalize(zero);
      expect(normalized).toEqual([0, 0, 0]);
    });
  });

  describe("vectorAdd", () => {
    it("should add vectors element-wise", () => {
      const a = [1, 2, 3];
      const b = [4, 5, 6];
      expect(vectorAdd(a, b)).toEqual([5, 7, 9]);
    });

    it("should throw for mismatched lengths", () => {
      expect(() => vectorAdd([1, 2], [1])).toThrow();
    });
  });

  describe("vectorSub", () => {
    it("should subtract vectors element-wise", () => {
      const a = [5, 7, 9];
      const b = [1, 2, 3];
      expect(vectorSub(a, b)).toEqual([4, 5, 6]);
    });
  });

  describe("vectorScale", () => {
    it("should scale vectors", () => {
      const vec = [1, 2, 3];
      expect(vectorScale(vec, 2)).toEqual([2, 4, 6]);
    });

    it("should handle zero scalar", () => {
      const vec = [1, 2, 3];
      expect(vectorScale(vec, 0)).toEqual([0, 0, 0]);
    });

    it("should handle negative scalar", () => {
      const vec = [1, 2, 3];
      expect(vectorScale(vec, -1)).toEqual([-1, -2, -3]);
    });
  });
});

describe("VectorDatabase", () => {
  let db: VectorDatabase;

  beforeEach(() => {
    db = new VectorDatabase({ name: "test_db" });
  });

  describe("basic operations", () => {
    it("should create with correct name", () => {
      expect(db.getName()).toBe("test_db");
    });

    it("should start empty", () => {
      expect(db.size()).toBe(0);
      expect(db.getDimensions()).toBeNull();
    });

    it("should add vectors and return IDs", () => {
      const id1 = db.add([1, 0, 0]);
      const id2 = db.add([0, 1, 0]);

      expect(id1).toBe(0);
      expect(id2).toBe(1);
      expect(db.size()).toBe(2);
    });

    it("should auto-detect dimensions from first vector", () => {
      db.add([1, 2, 3]);
      expect(db.getDimensions()).toBe(3);
    });

    it("should reject vectors with wrong dimensions", () => {
      db.add([1, 2, 3]);
      expect(() => db.add([1, 2])).toThrow(/dimension mismatch/);
    });

    it("should allow specifying dimensions in config", () => {
      const db2 = new VectorDatabase({ name: "test", dimensions: 5 });
      expect(db2.getDimensions()).toBe(5);
      expect(() => db2.add([1, 2, 3])).toThrow(/dimension mismatch/);
    });
  });

  describe("get and remove", () => {
    it("should get vector by ID", () => {
      const id = db.add([1, 2, 3], { label: "test" });
      const entry = db.get(id);

      expect(entry).not.toBeNull();
      expect(entry!.id).toBe(id);
      expect(entry!.vector).toEqual([1, 2, 3]);
      expect(entry!.metadata).toEqual({ label: "test" });
    });

    it("should return null for non-existent ID", () => {
      expect(db.get(999)).toBeNull();
    });

    it("should remove vector by ID", () => {
      const id = db.add([1, 2, 3]);
      expect(db.size()).toBe(1);

      const removed = db.remove(id);
      expect(removed).toBe(true);
      expect(db.size()).toBe(0);
      expect(db.get(id)).toBeNull();
    });

    it("should return false for non-existent remove", () => {
      expect(db.remove(999)).toBe(false);
    });
  });

  describe("search", () => {
    beforeEach(() => {
      // Add axis-aligned unit vectors
      db.add([1, 0, 0], { axis: "x" });
      db.add([0, 1, 0], { axis: "y" });
      db.add([0, 0, 1], { axis: "z" });
    });

    it("should find most similar vector", () => {
      const results = db.search([0.9, 0.1, 0], 1);

      expect(results).toHaveLength(1);
      expect(results[0].metadata).toEqual({ axis: "x" });
      expect(results[0].score).toBeGreaterThan(0.9);
    });

    it("should return results sorted by similarity", () => {
      const results = db.search([1, 1, 0], 3);

      // X and Y should be equally similar, Z should be least similar
      expect(results).toHaveLength(3);
      expect(results[0].score).toBeGreaterThanOrEqual(results[1].score);
      expect(results[1].score).toBeGreaterThanOrEqual(results[2].score);
    });

    it("should respect topK limit", () => {
      const results = db.search([1, 0, 0], 2);
      expect(results).toHaveLength(2);
    });

    it("should throw for wrong query dimensions", () => {
      expect(() => db.search([1, 0], 1)).toThrow(/dimension mismatch/);
    });
  });

  describe("persistence", () => {
    it("should export to JSON", () => {
      db.add([1, 2, 3], { label: "test" });
      const json = db.toJSON();

      expect(json).toHaveProperty("name", "test_db");
      expect(json).toHaveProperty("dimensions", 3);
      expect(json).toHaveProperty("vectors");
      expect(json).toHaveProperty("nextId");
    });

    it("should import from JSON", () => {
      db.add([1, 2, 3], { label: "a" });
      db.add([4, 5, 6], { label: "b" });

      const json = db.toJSON() as Parameters<typeof VectorDatabase.fromJSON>[0];
      const restored = VectorDatabase.fromJSON(json);

      expect(restored.getName()).toBe(db.getName());
      expect(restored.size()).toBe(db.size());
      expect(restored.getDimensions()).toBe(db.getDimensions());
    });

    it("should clear database", () => {
      db.add([1, 2, 3]);
      db.add([4, 5, 6]);

      db.clear();

      expect(db.size()).toBe(0);
      expect(db.getDimensions()).toBeNull();
    });
  });
});
