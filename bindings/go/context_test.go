package abi

import (
	"context"
	"testing"
	"time"
)

func TestSearchWithContext(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Insert a vector
	err = db.Insert(1, []float32{1.0, 0.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Test with valid context
	ctx := context.Background()
	results, err := db.SearchWithContext(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Errorf("SearchWithContext failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
	if results[0].ID != 1 {
		t.Errorf("Expected result ID 1, got %d", results[0].ID)
	}
}

func TestSearchWithCancelledContext(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Insert a vector for the search
	err = db.Insert(1, []float32{1.0, 0.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err = db.SearchWithContext(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestSearchWithTimeout(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Insert a vector
	err = db.Insert(1, []float32{1.0, 0.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Test with very short timeout on already-expired context
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()
	time.Sleep(2 * time.Millisecond) // Ensure context expires

	_, err = db.SearchWithContext(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != context.DeadlineExceeded {
		t.Errorf("Expected context.DeadlineExceeded, got %v", err)
	}
}

func TestSearchWithContextPreservesErrors(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Test that underlying errors are preserved when context is valid
	ctx := context.Background()

	// Wrong dimension should return ErrInvalidArgument
	_, err = db.SearchWithContext(ctx, []float32{1.0, 0.0}, 1)
	if err != ErrInvalidArgument {
		t.Errorf("Expected ErrInvalidArgument for dimension mismatch, got %v", err)
	}

	// k=0 should return ErrInvalidArgument
	_, err = db.SearchWithContext(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 0)
	if err != ErrInvalidArgument {
		t.Errorf("Expected ErrInvalidArgument for k=0, got %v", err)
	}
}

func TestInsertWithContext(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Test with valid context
	ctx := context.Background()
	err = db.InsertWithContext(ctx, 1, []float32{1.0, 0.0, 0.0, 0.0})
	if err != nil {
		t.Errorf("InsertWithContext failed: %v", err)
	}

	// Verify the insert worked by searching
	results, err := db.Search([]float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Errorf("Search failed: %v", err)
	}
	if len(results) != 1 || results[0].ID != 1 {
		t.Errorf("Insert verification failed: expected ID 1, got results %v", results)
	}
}

func TestInsertWithCancelledContext(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	err = db.InsertWithContext(ctx, 1, []float32{1.0, 0.0, 0.0, 0.0})
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
}

func TestInsertWithContextPreservesErrors(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Test that underlying errors are preserved when context is valid
	ctx := context.Background()

	// Wrong dimension should return ErrInvalidArgument
	err = db.InsertWithContext(ctx, 1, []float32{1.0, 0.0})
	if err != ErrInvalidArgument {
		t.Errorf("Expected ErrInvalidArgument for dimension mismatch, got %v", err)
	}
}

func TestInsertBatchWithContext(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	ctx := context.Background()
	ids := []uint64{1, 2, 3}
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
	}

	inserted, err := db.InsertBatchWithContext(ctx, ids, vectors)
	if err != nil {
		t.Errorf("InsertBatchWithContext failed: %v", err)
	}
	if inserted != 3 {
		t.Errorf("Expected 3 inserted, got %d", inserted)
	}

	// Verify all vectors were inserted
	results, err := db.Search([]float32{1.0, 0.0, 0.0, 0.0}, 3)
	if err != nil {
		t.Errorf("Search failed: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
}

func TestInsertBatchWithCancelledContext(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	ids := []uint64{1, 2, 3}
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
		{0.0, 0.0, 1.0, 0.0},
	}

	inserted, err := db.InsertBatchWithContext(ctx, ids, vectors)
	if err != context.Canceled {
		t.Errorf("Expected context.Canceled, got %v", err)
	}
	if inserted != 0 {
		t.Errorf("Expected 0 inserted before cancellation, got %d", inserted)
	}
}

func TestInsertBatchWithContextMismatchedLengths(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	ctx := context.Background()
	ids := []uint64{1, 2}
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
	}

	inserted, err := db.InsertBatchWithContext(ctx, ids, vectors)
	if err != ErrInvalidArgument {
		t.Errorf("Expected ErrInvalidArgument for mismatched lengths, got %v", err)
	}
	if inserted != 0 {
		t.Errorf("Expected 0 inserted for mismatched lengths, got %d", inserted)
	}
}

func TestInsertBatchWithContextPartialFailure(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	ctx := context.Background()
	ids := []uint64{1, 2, 3}
	vectors := [][]float32{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0}, // Wrong dimension - should fail
		{0.0, 0.0, 1.0, 0.0},
	}

	inserted, err := db.InsertBatchWithContext(ctx, ids, vectors)
	if err != ErrInvalidArgument {
		t.Errorf("Expected ErrInvalidArgument for wrong dimension, got %v", err)
	}
	if inserted != 1 {
		t.Errorf("Expected 1 inserted before failure, got %d", inserted)
	}
}

func TestContextWithValue(t *testing.T) {
	framework, err := Init()
	if err != nil {
		t.Skipf("Skipping test: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(4)
	if err != nil {
		t.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Insert a vector
	err = db.Insert(1, []float32{1.0, 0.0, 0.0, 0.0})
	if err != nil {
		t.Fatalf("Insert failed: %v", err)
	}

	// Test that context.WithValue works correctly (doesn't interfere with operations)
	type key string
	ctx := context.WithValue(context.Background(), key("request_id"), "test-123")

	results, err := db.SearchWithContext(ctx, []float32{1.0, 0.0, 0.0, 0.0}, 1)
	if err != nil {
		t.Errorf("SearchWithContext with value failed: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("Expected 1 result, got %d", len(results))
	}
}

// BenchmarkSearchWithContext benchmarks the overhead of context checking.
func BenchmarkSearchWithContext(b *testing.B) {
	framework, err := Init()
	if err != nil {
		b.Skipf("Skipping benchmark: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(128)
	if err != nil {
		b.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Insert some vectors
	vec := make([]float32, 128)
	for i := 0; i < 100; i++ {
		for j := range vec {
			vec[j] = float32(i*128+j) / float32(128*100)
		}
		db.Insert(uint64(i), vec)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 0.5
	}

	ctx := context.Background()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		db.SearchWithContext(ctx, query, 10)
	}
}

// BenchmarkSearchWithContextVsSearch compares context vs non-context search.
func BenchmarkSearchWithContextVsSearch(b *testing.B) {
	framework, err := Init()
	if err != nil {
		b.Skipf("Skipping benchmark: %v", err)
	}
	defer framework.Shutdown()

	db, err := framework.CreateDB(128)
	if err != nil {
		b.Fatalf("CreateDB failed: %v", err)
	}
	defer db.Destroy()

	// Insert some vectors
	vec := make([]float32, 128)
	for i := 0; i < 100; i++ {
		for j := range vec {
			vec[j] = float32(i*128+j) / float32(128*100)
		}
		db.Insert(uint64(i), vec)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 0.5
	}

	b.Run("WithContext", func(b *testing.B) {
		ctx := context.Background()
		for i := 0; i < b.N; i++ {
			db.SearchWithContext(ctx, query, 10)
		}
	})

	b.Run("WithoutContext", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			db.Search(query, 10)
		}
	})
}
