// Example demonstrating ABI Framework Go bindings
package main

import (
	"fmt"
	"log"

	abi "github.com/donaldfilimon/abi-go"
)

func main() {
	fmt.Println("ABI Framework Go Example")
	fmt.Println("========================")
	fmt.Println()

	// Check version
	fmt.Printf("Version: %s\n", abi.Version())

	// Check SIMD capabilities
	caps := abi.GetSIMDCapabilities()
	fmt.Printf("SIMD available: %v\n", caps.HasSIMD)
	fmt.Printf("Architecture: %s\n", caps.Arch)
	fmt.Printf("Vector size: %d bytes\n", caps.VectorSize)
	fmt.Println()

	// Vector operations
	fmt.Println("Vector Operations:")
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{4.0, 3.0, 2.0, 1.0}

	// Dot product
	dot := abi.DotProduct(a, b)
	fmt.Printf("  Dot product: %.1f\n", dot)

	// Cosine similarity
	similarity := abi.CosineSimilarity(a, b)
	fmt.Printf("  Cosine similarity: %.4f\n", similarity)

	// L2 norm
	norm := abi.L2Norm(a)
	fmt.Printf("  L2 norm of a: %.4f\n", norm)

	// Vector addition
	sum := abi.Add(a, b)
	fmt.Printf("  a + b = %v\n", sum)

	// Normalize
	normalized := abi.Normalize(a)
	fmt.Printf("  Normalized a: %v\n", normalized)

	// Euclidean distance
	distance := abi.EuclideanDistance(a, b)
	fmt.Printf("  Euclidean distance: %.4f\n", distance)
	fmt.Println()

	// Matrix multiplication
	fmt.Println("Matrix Operations:")
	matA := []float32{1.0, 2.0, 3.0, 4.0} // 2x2
	matB := []float32{5.0, 6.0, 7.0, 8.0} // 2x2
	result := abi.MatrixMultiply(matA, matB, 2, 2, 2)
	fmt.Println("  A = [1 2; 3 4]")
	fmt.Println("  B = [5 6; 7 8]")
	fmt.Printf("  A * B = [%.0f %.0f; %.0f %.0f]\n", result[0], result[1], result[2], result[3])
	fmt.Println()

	// Framework initialization (requires native library)
	fmt.Println("Framework Initialization:")
	fw, err := abi.NewFramework(abi.MinimalConfig())
	if err != nil {
		fmt.Printf("  Framework not available: %v (this is expected if native library isn't linked)\n", err)
	} else {
		defer fw.Close()
		fmt.Println("  Framework initialized successfully")
		fmt.Printf("  AI enabled: %v\n", fw.IsFeatureEnabled("ai"))
		fmt.Printf("  GPU enabled: %v\n", fw.IsFeatureEnabled("gpu"))
		fmt.Printf("  Database enabled: %v\n", fw.IsFeatureEnabled("database"))
	}

	fmt.Println()
	fmt.Println("Example complete!")
}

func vectorDatabaseExample() {
	fmt.Println("\n--- Vector Database Example ---\n")

	// Create database
	db, err := abi.NewVectorDatabase("example_db", 4)
	if err != nil {
		log.Printf("Failed to create database: %v\n", err)
		return
	}
	defer db.Close()

	fmt.Printf("Created database with dimension: %d\n", db.Dimension())

	// Sample vectors
	vectors := []struct {
		id     uint64
		vector []float32
		name   string
	}{
		{1, []float32{1.0, 0.0, 0.0, 0.0}, "North"},
		{2, []float32{0.0, 1.0, 0.0, 0.0}, "East"},
		{3, []float32{0.0, 0.0, 1.0, 0.0}, "Up"},
		{4, []float32{0.707, 0.707, 0.0, 0.0}, "Northeast"},
	}

	// Insert vectors
	for _, v := range vectors {
		if err := db.Insert(v.id, v.vector); err != nil {
			log.Printf("Failed to insert %s: %v\n", v.name, err)
			continue
		}
		fmt.Printf("  Inserted %s (ID=%d)\n", v.name, v.id)
	}

	fmt.Printf("\nDatabase now contains %d vectors\n", db.Len())

	// Search
	query := []float32{0.9, 0.1, 0.0, 0.0}
	fmt.Printf("\nSearching for similar vectors to %v:\n", query)

	results, err := db.Search(query, 3)
	if err != nil {
		log.Printf("Search failed: %v\n", err)
		return
	}

	for i, r := range results {
		name := "Unknown"
		for _, v := range vectors {
			if v.id == r.ID {
				name = v.name
				break
			}
		}
		fmt.Printf("  %d. %s (ID=%d, score=%.4f)\n", i+1, name, r.ID, r.Score)
	}
}

func gpuExample() {
	fmt.Println("\n--- GPU Example ---\n")

	// Check availability
	if !abi.GPUAvailable() {
		fmt.Println("No GPU available")
		return
	}

	// List devices
	devices, err := abi.ListGPUDevices()
	if err != nil {
		log.Printf("Failed to list devices: %v\n", err)
		return
	}

	fmt.Printf("Found %d GPU device(s):\n", len(devices))
	for i, d := range devices {
		fmt.Printf("  %d. %s (%s) - %d MB total, %d compute units\n",
			i+1, d.Name, d.Backend,
			d.TotalMemory/1024/1024, d.ComputeUnits)
	}

	// Create GPU context
	gpu, err := abi.NewGPUContext(abi.BackendAuto)
	if err != nil {
		log.Printf("Failed to create GPU context: %v\n", err)
		return
	}
	defer gpu.Close()

	// GPU matrix multiplication
	a := []float32{1.0, 2.0, 3.0, 4.0}
	b := []float32{5.0, 6.0, 7.0, 8.0}

	result, err := gpu.MatrixMultiply(a, b, 2, 2, 2)
	if err != nil {
		log.Printf("GPU matrix multiply failed: %v\n", err)
		return
	}

	fmt.Printf("\nGPU Matrix multiply: [%.0f %.0f; %.0f %.0f]\n",
		result[0], result[1], result[2], result[3])
}
