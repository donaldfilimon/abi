// Example: Vector similarity search with ABI Framework Go bindings.
//
// Run from the ABI repository root after building:
//
//	zig build lib
//	export DYLD_LIBRARY_PATH=$PWD/zig-out/lib:$DYLD_LIBRARY_PATH  # macOS
//	export LD_LIBRARY_PATH=$PWD/zig-out/lib:$LD_LIBRARY_PATH      # Linux
//	go run bindings/go/examples/vector_search/main.go
package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	abi "github.com/donaldfilimon/abi/bindings/go"
)

func normalize(vec []float32) []float32 {
	var sum float32
	for _, v := range vec {
		sum += v * v
	}
	mag := float32(math.Sqrt(float64(sum)))
	if mag == 0 {
		return vec
	}
	result := make([]float32, len(vec))
	for i, v := range vec {
		result[i] = v / mag
	}
	return result
}

func randomVector(dim int) []float32 {
	vec := make([]float32, dim)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // [-1, 1]
	}
	return normalize(vec)
}

func main() {
	fmt.Println("============================================================")
	fmt.Println("ABI Framework - Go Vector Similarity Search Example")
	fmt.Println("============================================================")

	// Initialize ABI
	framework, err := abi.Init()
	if err != nil {
		fmt.Printf("\nError: %v\n", err)
		fmt.Println("Make sure to:")
		fmt.Println("  1. Build the shared library: zig build lib")
		fmt.Println("  2. Set library path:")
		fmt.Println("     macOS: export DYLD_LIBRARY_PATH=$PWD/zig-out/lib:$DYLD_LIBRARY_PATH")
		fmt.Println("     Linux: export LD_LIBRARY_PATH=$PWD/zig-out/lib:$LD_LIBRARY_PATH")
		return
	}
	defer framework.Shutdown()

	fmt.Printf("\nABI Framework version: %s\n", abi.Version())

	// Configuration
	const (
		dimension  = 128
		numVectors = 1000
		k          = 10
	)

	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Vector dimension: %d\n", dimension)
	fmt.Printf("  Number of vectors: %d\n", numVectors)
	fmt.Printf("  Top-K results: %d\n", k)

	// Create database
	fmt.Printf("\nCreating vector database...\n")
	db, err := framework.CreateDB(dimension)
	if err != nil {
		fmt.Printf("Error creating database: %v\n", err)
		return
	}
	defer db.Destroy()
	fmt.Printf("  Database created successfully\n")

	// Generate and insert random vectors
	fmt.Printf("\nInserting %d random vectors...\n", numVectors)
	vectors := make(map[uint64][]float32)
	startTime := time.Now()

	for i := 0; i < numVectors; i++ {
		vec := randomVector(dimension)
		vectors[uint64(i)] = vec
		if err := db.Insert(uint64(i), vec); err != nil {
			fmt.Printf("Error inserting vector %d: %v\n", i, err)
			return
		}
	}

	insertTime := time.Since(startTime)
	fmt.Printf("  Inserted %d vectors in %v\n", numVectors, insertTime)
	fmt.Printf("  Rate: %.0f vectors/sec\n", float64(numVectors)/insertTime.Seconds())

	// Perform searches
	fmt.Printf("\nPerforming similarity search...\n")

	// Search 1: Use an existing vector (should return itself as top match)
	queryID := uint64(rand.Intn(numVectors))
	query := vectors[queryID]

	startTime = time.Now()
	results, err := db.Search(query, k)
	searchTime := time.Since(startTime)

	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		return
	}

	fmt.Printf("\n  Query: vector ID %d\n", queryID)
	fmt.Printf("  Search time: %v\n", searchTime)
	fmt.Printf("\n  Top %d results:\n", k)
	fmt.Printf("  %-6s %-8s %-12s\n", "Rank", "ID", "Score")
	fmt.Printf("  %s\n", "-------------------------------")

	for rank, result := range results {
		fmt.Printf("  %-6d %-8d %-12.6f\n", rank+1, result.ID, result.Score)
	}

	// Search 2: Random query vector
	fmt.Printf("\n  Query: random vector (not in database)\n")
	randomQuery := randomVector(dimension)

	startTime = time.Now()
	results, err = db.Search(randomQuery, k)
	searchTime = time.Since(startTime)

	if err != nil {
		fmt.Printf("Error searching: %v\n", err)
		return
	}

	fmt.Printf("  Search time: %v\n", searchTime)
	fmt.Printf("\n  Top %d results:\n", k)
	fmt.Printf("  %-6s %-8s %-12s\n", "Rank", "ID", "Score")
	fmt.Printf("  %s\n", "-------------------------------")

	for rank, result := range results {
		fmt.Printf("  %-6d %-8d %-12.6f\n", rank+1, result.ID, result.Score)
	}

	// Benchmark: Multiple searches
	fmt.Println()
	fmt.Println("============================================================")
	fmt.Println("Benchmark: 100 random searches")
	fmt.Println("============================================================")

	const numSearches = 100
	startTime = time.Now()

	for i := 0; i < numSearches; i++ {
		q := randomVector(dimension)
		db.Search(q, k)
	}

	totalTime := time.Since(startTime)
	avgTime := totalTime / numSearches

	fmt.Printf("\n  Total time: %v\n", totalTime)
	fmt.Printf("  Average per search: %v\n", avgTime)
	fmt.Printf("  Queries per second: %.0f\n", float64(numSearches)/totalTime.Seconds())

	fmt.Println()
	fmt.Println("============================================================")
	fmt.Println("Example completed successfully!")
	fmt.Println("============================================================")
}
