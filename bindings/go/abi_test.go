package abi

import (
	"math"
	"testing"
)

func TestVersion(t *testing.T) {
	version := Version()
	if version != "0.4.0" {
		t.Errorf("expected version 0.4.0, got %s", version)
	}
}

func TestFrameworkLifecycle(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Fatalf("failed to initialize framework: %v", err)
	}

	// Shutdown should not panic
	framework.Shutdown()

	// Double shutdown should be safe
	framework.Shutdown()
}

func TestDatabaseLifecycle(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Fatalf("failed to initialize framework: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(128)
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}

	if db.Dimension() != 128 {
		t.Errorf("expected dimension 128, got %d", db.Dimension())
	}

	// Destroy should not panic
	db.Destroy()

	// Double destroy should be safe
	db.Destroy()
}

func TestVectorOperations(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Fatalf("failed to initialize framework: %v", err)
	}
	defer framework.Shutdown()

	dim := uint32(4)
	db, err := framework.CreateDB(dim)
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}
	defer db.Destroy()

	// Insert vectors
	vec1 := []float32{1.0, 0.0, 0.0, 0.0}
	vec2 := []float32{0.0, 1.0, 0.0, 0.0}
	vec3 := []float32{0.707, 0.707, 0.0, 0.0}

	if err := db.Insert(1, vec1); err != nil {
		t.Fatalf("failed to insert vector 1: %v", err)
	}
	if err := db.Insert(2, vec2); err != nil {
		t.Fatalf("failed to insert vector 2: %v", err)
	}
	if err := db.Insert(3, vec3); err != nil {
		t.Fatalf("failed to insert vector 3: %v", err)
	}

	// Search for vector similar to vec1
	results, err := db.Search(vec1, 3)
	if err != nil {
		t.Fatalf("failed to search: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	// First result should be ID 1 (exact match)
	if results[0].ID != 1 {
		t.Errorf("expected first result ID to be 1, got %d", results[0].ID)
	}

	// Score should be 1.0 for exact match
	if math.Abs(float64(results[0].Score-1.0)) > 0.0001 {
		t.Errorf("expected score 1.0 for exact match, got %f", results[0].Score)
	}

	t.Logf("Search results for vec1:")
	for i, r := range results {
		t.Logf("  %d: ID=%d, Score=%f", i+1, r.ID, r.Score)
	}
}

func TestDimensionMismatch(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Fatalf("failed to initialize framework: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}
	defer db.Destroy()

	// Try to insert vector with wrong dimension
	wrongVec := []float32{1.0, 0.0} // 2 elements instead of 4
	err = db.Insert(1, wrongVec)
	if err != ErrInvalidArgument {
		t.Errorf("expected ErrInvalidArgument for dimension mismatch, got %v", err)
	}

	// Insert correct vector first
	correctVec := []float32{1.0, 0.0, 0.0, 0.0}
	if err := db.Insert(1, correctVec); err != nil {
		t.Fatalf("failed to insert correct vector: %v", err)
	}

	// Try to search with wrong dimension
	_, err = db.Search(wrongVec, 1)
	if err != ErrInvalidArgument {
		t.Errorf("expected ErrInvalidArgument for search dimension mismatch, got %v", err)
	}
}

func TestSearchWithZeroK(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Fatalf("failed to initialize framework: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("failed to create database: %v", err)
	}
	defer db.Destroy()

	vec := []float32{1.0, 0.0, 0.0, 0.0}
	if err := db.Insert(1, vec); err != nil {
		t.Fatalf("failed to insert vector: %v", err)
	}

	// Search with k=0 should fail
	_, err = db.Search(vec, 0)
	if err != ErrInvalidArgument {
		t.Errorf("expected ErrInvalidArgument for k=0, got %v", err)
	}
}

func BenchmarkInsert(b *testing.B) {
	framework, err := Init()
	if err != nil {
		b.Fatalf("failed to initialize framework: %v", err)
	}
	defer framework.Shutdown()

	dim := uint32(128)
	db, err := framework.CreateDB(dim)
	if err != nil {
		b.Fatalf("failed to create database: %v", err)
	}
	defer db.Destroy()

	// Create a sample vector
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = float32(i) / float32(dim)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Insert(uint64(i), vec)
	}
}

func BenchmarkSearch(b *testing.B) {
	framework, err := Init()
	if err != nil {
		b.Fatalf("failed to initialize framework: %v", err)
	}
	defer framework.Shutdown()

	dim := uint32(128)
	db, err := framework.CreateDB(dim)
	if err != nil {
		b.Fatalf("failed to create database: %v", err)
	}
	defer db.Destroy()

	// Insert some vectors
	vec := make([]float32, dim)
	for i := 0; i < 1000; i++ {
		for j := range vec {
			vec[j] = float32(i*int(dim)+j) / float32(dim*1000)
		}
		db.Insert(uint64(i), vec)
	}

	// Query vector
	query := make([]float32, dim)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.Search(query, 10)
	}
}
