// Context-aware methods for ABI Go bindings.
//
// This file provides context-aware versions of VectorDatabase methods that support
// Go's standard context.Context for cancellation and timeout handling.
//
// Note: CGO calls cannot be truly interrupted mid-execution. Context checking
// happens before and after the CGO call, which is the standard pattern for
// CGO-based bindings. For very long-running operations, consider breaking
// them into smaller batches with context checks between each batch.

package abi

import (
	"context"
)

// SearchWithContext performs a k-NN search with context support for cancellation/timeout.
// If the context is cancelled or times out, returns context.Canceled or context.DeadlineExceeded.
//
// Example usage with timeout:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
//	defer cancel()
//	results, err := db.SearchWithContext(ctx, query, 10)
//	if err == context.DeadlineExceeded {
//	    // Handle timeout
//	}
func (db *VectorDatabase) SearchWithContext(ctx context.Context, query []float32, k uint32) ([]SearchResult, error) {
	// Check context before starting the operation
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	// Perform the actual search
	// Note: CGO calls are not interruptible, but we check context before and after
	results, err := db.Search(query, k)

	// Check context after search completes
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	return results, err
}

// InsertWithContext adds a vector with context support for cancellation/timeout.
// If the context is cancelled or times out, returns context.Canceled or context.DeadlineExceeded.
//
// Example usage with cancellation:
//
//	ctx, cancel := context.WithCancel(context.Background())
//	go func() {
//	    // Cancel after some condition
//	    cancel()
//	}()
//	err := db.InsertWithContext(ctx, id, vector)
//	if err == context.Canceled {
//	    // Handle cancellation
//	}
func (db *VectorDatabase) InsertWithContext(ctx context.Context, id uint64, vector []float32) error {
	// Check context before starting the operation
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Perform the actual insert
	err := db.Insert(id, vector)

	// Check context after insert completes
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	return err
}

// InsertBatchWithContext inserts multiple vectors with context checking between each insert.
// This provides better cancellation granularity for batch operations.
// Returns the number of vectors successfully inserted and any error that occurred.
//
// Example usage:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
//	defer cancel()
//	ids := []uint64{1, 2, 3, 4, 5}
//	vectors := [][]float32{...}
//	inserted, err := db.InsertBatchWithContext(ctx, ids, vectors)
//	if err != nil {
//	    fmt.Printf("Inserted %d vectors before error: %v\n", inserted, err)
//	}
func (db *VectorDatabase) InsertBatchWithContext(ctx context.Context, ids []uint64, vectors [][]float32) (int, error) {
	if len(ids) != len(vectors) {
		return 0, ErrInvalidArgument
	}

	for i := range ids {
		// Check context before each insert
		select {
		case <-ctx.Done():
			return i, ctx.Err()
		default:
		}

		if err := db.Insert(ids[i], vectors[i]); err != nil {
			return i, err
		}
	}

	// Final context check
	select {
	case <-ctx.Done():
		return len(ids), ctx.Err()
	default:
	}

	return len(ids), nil
}
