//! SIMD-accelerated vector operations
//!
//! This module provides high-performance vector math operations that use
//! SIMD instructions when available, with automatic fallback to scalar code.
//!
//! # Example
//!
//! ```rust
//! use abi::simd;
//!
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![4.0, 3.0, 2.0, 1.0];
//!
//! // Compute cosine similarity
//! let similarity = simd::cosine_similarity(&a, &b);
//! println!("Similarity: {:.4}", similarity);
//!
//! // Compute dot product
//! let dot = simd::dot_product(&a, &b);
//! println!("Dot product: {}", dot);
//!
//! // Vector addition
//! let sum = simd::add(&a, &b);
//! println!("Sum: {:?}", sum);
//! ```

use crate::ffi;

/// SIMD capabilities for the current platform
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    /// Size of SIMD vector in bytes
    pub vector_size: usize,
    /// Whether SIMD is available
    pub has_simd: bool,
    /// Architecture identifier
    pub arch: Architecture,
}

/// CPU architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Generic/unknown architecture
    Generic,
    /// x86-64 with SSE/AVX
    X86_64,
    /// ARM64 with NEON
    Aarch64,
    /// WebAssembly SIMD
    Wasm,
}

impl From<i32> for Architecture {
    fn from(value: i32) -> Self {
        match value {
            1 => Architecture::X86_64,
            2 => Architecture::Aarch64,
            3 => Architecture::Wasm,
            _ => Architecture::Generic,
        }
    }
}

/// Get SIMD capabilities for the current platform
pub fn capabilities() -> Capabilities {
    let mut caps = ffi::abi_simd_caps_t {
        vector_size: 0,
        has_simd: false,
        arch: 0,
    };

    unsafe {
        ffi::abi_simd_get_caps(&mut caps);
    }

    Capabilities {
        vector_size: caps.vector_size,
        has_simd: caps.has_simd,
        arch: Architecture::from(caps.arch),
    }
}

/// Check if SIMD operations are available
pub fn is_available() -> bool {
    unsafe { ffi::abi_simd_available() }
}

/// Element-wise vector addition: result[i] = a[i] + b[i]
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Example
///
/// ```rust
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let sum = abi::simd::add(&a, &b);
/// assert_eq!(sum, vec![5.0, 7.0, 9.0]);
/// ```
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    let mut result = vec![0.0f32; a.len()];

    if is_available() && a.len() >= 4 {
        unsafe {
            ffi::abi_simd_vector_add(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
        }
    } else {
        // Fallback to scalar
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    result
}

/// Element-wise vector subtraction: result[i] = a[i] - b[i]
///
/// # Panics
///
/// Panics if vectors have different lengths.
pub fn subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    let mut result = vec![0.0f32; a.len()];
    for i in 0..a.len() {
        result[i] = a[i] - b[i];
    }
    result
}

/// Scalar multiplication: result[i] = v[i] * scalar
pub fn scale(v: &[f32], scalar: f32) -> Vec<f32> {
    v.iter().map(|x| x * scalar).collect()
}

/// Dot product of two vectors: sum(a[i] * b[i])
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Example
///
/// ```rust
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let dot = abi::simd::dot_product(&a, &b);
/// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
/// ```
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    if is_available() && a.len() >= 4 {
        unsafe { ffi::abi_simd_vector_dot(a.as_ptr(), b.as_ptr(), a.len()) }
    } else {
        // Fallback to scalar
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// L2 (Euclidean) norm of a vector: sqrt(sum(v[i]^2))
///
/// # Example
///
/// ```rust
/// let v = vec![3.0, 4.0];
/// let norm = abi::simd::l2_norm(&v);
/// assert!((norm - 5.0).abs() < 1e-6); // 3-4-5 triangle
/// ```
pub fn l2_norm(v: &[f32]) -> f32 {
    if is_available() && v.len() >= 4 {
        unsafe { ffi::abi_simd_vector_l2_norm(v.as_ptr(), v.len()) }
    } else {
        // Fallback to scalar
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Normalize a vector to unit length
///
/// Returns a zero vector if the input has zero length.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let norm = l2_norm(v);
    if norm == 0.0 {
        vec![0.0; v.len()]
    } else {
        scale(v, 1.0 / norm)
    }
}

/// Cosine similarity between two vectors
///
/// Returns a value in the range [-1, 1], where 1 means identical direction,
/// 0 means orthogonal, and -1 means opposite direction.
///
/// # Panics
///
/// Panics if vectors have different lengths.
///
/// # Example
///
/// ```rust
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![1.0, 0.0, 0.0];
/// let sim = abi::simd::cosine_similarity(&a, &b);
/// assert!((sim - 1.0).abs() < 1e-6); // Identical vectors
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");

    if is_available() && a.len() >= 4 {
        unsafe { ffi::abi_simd_cosine_similarity(a.as_ptr(), b.as_ptr(), a.len()) }
    } else {
        // Fallback to scalar
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

/// Euclidean distance between two vectors
///
/// # Panics
///
/// Panics if vectors have different lengths.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    l2_norm(&subtract(a, b))
}

/// Matrix multiplication: C = A * B
///
/// A is m x k, B is k x n, C is m x n (row-major order)
///
/// # Panics
///
/// Panics if matrices have incompatible dimensions.
pub fn matrix_multiply(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "matrix A has wrong size");
    assert_eq!(b.len(), k * n, "matrix B has wrong size");

    let mut result = vec![0.0f32; m * n];

    if is_available() && m >= 4 && n >= 4 {
        unsafe {
            ffi::abi_simd_matrix_multiply(
                a.as_ptr(),
                b.as_ptr(),
                result.as_mut_ptr(),
                m,
                n,
                k,
            );
        }
    } else {
        // Fallback to scalar
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = add(&a, &b);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_subtract() {
        let a = vec![5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let result = subtract(&a, &b);
        assert_eq!(result, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scale() {
        let v = vec![1.0, 2.0, 3.0];
        let result = scale(&v, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let result = dot_product(&a, &b);
        assert!((result - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let v = vec![3.0, 4.0];
        let result = l2_norm(&v);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let result = normalize(&v);
        assert!((result[0] - 0.6).abs() < 1e-6);
        assert!((result[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero() {
        let v = vec![0.0, 0.0];
        let result = normalize(&v);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_multiply() {
        // 2x2 identity matrix times a vector
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let b = vec![3.0, 4.0]; // 2x1
        let result = matrix_multiply(&a, &b, 2, 1, 2);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    #[should_panic(expected = "vectors must have same length")]
    fn test_add_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        add(&a, &b);
    }
}
