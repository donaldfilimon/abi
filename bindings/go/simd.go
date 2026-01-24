package abi

/*
#include <stdlib.h>
#include <stdbool.h>

extern bool abi_simd_available();
extern void abi_simd_vector_add(const float* a, const float* b, float* result, size_t len);
extern float abi_simd_vector_dot(const float* a, const float* b, size_t len);
extern float abi_simd_vector_l2_norm(const float* v, size_t len);
extern float abi_simd_cosine_similarity(const float* a, const float* b, size_t len);
extern void abi_simd_matrix_multiply(const float* a, const float* b, float* result,
                                     size_t m, size_t n, size_t k);
*/
import "C"
import (
	"math"
	"unsafe"
)

// Add performs element-wise vector addition: result[i] = a[i] + b[i]
// Panics if vectors have different lengths.
func Add(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vectors must have same length")
	}
	if len(a) == 0 {
		return []float32{}
	}

	result := make([]float32, len(a))

	if SIMDAvailable() && len(a) >= 4 {
		C.abi_simd_vector_add(
			(*C.float)(unsafe.Pointer(&a[0])),
			(*C.float)(unsafe.Pointer(&b[0])),
			(*C.float)(unsafe.Pointer(&result[0])),
			C.size_t(len(a)),
		)
	} else {
		// Scalar fallback
		for i := range a {
			result[i] = a[i] + b[i]
		}
	}

	return result
}

// Subtract performs element-wise vector subtraction: result[i] = a[i] - b[i]
// Panics if vectors have different lengths.
func Subtract(a, b []float32) []float32 {
	if len(a) != len(b) {
		panic("vectors must have same length")
	}

	result := make([]float32, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

// Scale multiplies each element by a scalar: result[i] = v[i] * scalar
func Scale(v []float32, scalar float32) []float32 {
	result := make([]float32, len(v))
	for i := range v {
		result[i] = v[i] * scalar
	}
	return result
}

// DotProduct computes the dot product of two vectors: sum(a[i] * b[i])
// Panics if vectors have different lengths.
func DotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same length")
	}
	if len(a) == 0 {
		return 0
	}

	if SIMDAvailable() && len(a) >= 4 {
		return float32(C.abi_simd_vector_dot(
			(*C.float)(unsafe.Pointer(&a[0])),
			(*C.float)(unsafe.Pointer(&b[0])),
			C.size_t(len(a)),
		))
	}

	// Scalar fallback
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// L2Norm computes the L2 (Euclidean) norm: sqrt(sum(v[i]^2))
func L2Norm(v []float32) float32 {
	if len(v) == 0 {
		return 0
	}

	if SIMDAvailable() && len(v) >= 4 {
		return float32(C.abi_simd_vector_l2_norm(
			(*C.float)(unsafe.Pointer(&v[0])),
			C.size_t(len(v)),
		))
	}

	// Scalar fallback
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	return float32(math.Sqrt(float64(sum)))
}

// Normalize normalizes a vector to unit length.
// Returns a zero vector if the input has zero length.
func Normalize(v []float32) []float32 {
	norm := L2Norm(v)
	if norm == 0 {
		return make([]float32, len(v))
	}
	return Scale(v, 1.0/norm)
}

// CosineSimilarity computes the cosine similarity between two vectors.
// Returns a value in [-1, 1], where 1 means identical direction.
// Panics if vectors have different lengths.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("vectors must have same length")
	}
	if len(a) == 0 {
		return 0
	}

	if SIMDAvailable() && len(a) >= 4 {
		return float32(C.abi_simd_cosine_similarity(
			(*C.float)(unsafe.Pointer(&a[0])),
			(*C.float)(unsafe.Pointer(&b[0])),
			C.size_t(len(a)),
		))
	}

	// Scalar fallback
	dot := DotProduct(a, b)
	normA := L2Norm(a)
	normB := L2Norm(b)
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (normA * normB)
}

// EuclideanDistance computes the Euclidean distance between two vectors.
// Panics if vectors have different lengths.
func EuclideanDistance(a, b []float32) float32 {
	return L2Norm(Subtract(a, b))
}

// MatrixMultiply performs matrix multiplication: C = A * B
// A is m x k, B is k x n, result is m x n (row-major order)
// Panics if matrices have incompatible dimensions.
func MatrixMultiply(a, b []float32, m, n, k int) []float32 {
	if len(a) != m*k {
		panic("matrix A has wrong size")
	}
	if len(b) != k*n {
		panic("matrix B has wrong size")
	}

	result := make([]float32, m*n)

	if SIMDAvailable() && m >= 4 && n >= 4 {
		C.abi_simd_matrix_multiply(
			(*C.float)(unsafe.Pointer(&a[0])),
			(*C.float)(unsafe.Pointer(&b[0])),
			(*C.float)(unsafe.Pointer(&result[0])),
			C.size_t(m),
			C.size_t(n),
			C.size_t(k),
		)
	} else {
		// Scalar fallback
		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var sum float32
				for l := 0; l < k; l++ {
					sum += a[i*k+l] * b[l*n+j]
				}
				result[i*n+j] = sum
			}
		}
	}

	return result
}
