//! SIMD vector operations.

use crate::ffi::{self, AbiSimdCaps};

/// SIMD-accelerated vector operations.
///
/// Provides high-performance vectorized operations using SIMD instructions
/// when available (SSE, AVX, NEON).
///
/// # Example
///
/// ```no_run
/// use abi::Simd;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![5.0, 6.0, 7.0, 8.0];
///
/// // Check SIMD availability
/// if Simd::is_available() {
///     let dot = Simd::dot_product(&a, &b);
///     println!("Dot product: {}", dot);
/// }
/// ```
pub struct Simd;

impl Simd {
    /// Check if any SIMD instruction set is available.
    pub fn is_available() -> bool {
        unsafe { ffi::abi_simd_available() }
    }

    /// Get detailed SIMD capability information.
    pub fn capabilities() -> SimdCaps {
        let mut caps = AbiSimdCaps::default();
        unsafe {
            ffi::abi_simd_get_caps(&mut caps);
        }
        SimdCaps {
            sse: caps.sse,
            sse2: caps.sse2,
            sse3: caps.sse3,
            ssse3: caps.ssse3,
            sse4_1: caps.sse4_1,
            sse4_2: caps.sse4_2,
            avx: caps.avx,
            avx2: caps.avx2,
            avx512f: caps.avx512f,
            neon: caps.neon,
        }
    }

    /// Vector element-wise addition: result[i] = a[i] + b[i]
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths.
    pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        let mut result = vec![0.0; a.len()];
        unsafe {
            ffi::abi_simd_vector_add(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
        }
        result
    }

    /// Vector element-wise addition into existing buffer.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths.
    pub fn add_into(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        assert_eq!(a.len(), result.len(), "Result must have the same length");
        unsafe {
            ffi::abi_simd_vector_add(a.as_ptr(), b.as_ptr(), result.as_mut_ptr(), a.len());
        }
    }

    /// Vector dot product: sum(a[i] * b[i])
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths.
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        unsafe { ffi::abi_simd_vector_dot(a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// Vector L2 norm: sqrt(sum(v[i]^2))
    pub fn l2_norm(v: &[f32]) -> f32 {
        unsafe { ffi::abi_simd_vector_l2_norm(v.as_ptr(), v.len()) }
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns a value between -1.0 and 1.0:
    /// - 1.0: vectors are identical
    /// - 0.0: vectors are orthogonal
    /// - -1.0: vectors are opposite
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        unsafe { ffi::abi_simd_cosine_similarity(a.as_ptr(), b.as_ptr(), a.len()) }
    }

    /// Normalize a vector to unit length.
    pub fn normalize(v: &[f32]) -> Vec<f32> {
        let norm = Self::l2_norm(v);
        if norm == 0.0 {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }

    /// Euclidean distance between two vectors.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths.
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        let diff: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        Self::l2_norm(&diff)
    }
}

/// SIMD capability flags.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdCaps {
    /// SSE support.
    pub sse: bool,
    /// SSE2 support.
    pub sse2: bool,
    /// SSE3 support.
    pub sse3: bool,
    /// SSSE3 support.
    pub ssse3: bool,
    /// SSE4.1 support.
    pub sse4_1: bool,
    /// SSE4.2 support.
    pub sse4_2: bool,
    /// AVX support.
    pub avx: bool,
    /// AVX2 support.
    pub avx2: bool,
    /// AVX-512F support.
    pub avx512f: bool,
    /// ARM NEON support.
    pub neon: bool,
}

impl SimdCaps {
    /// Check if any x86 SIMD is available.
    pub fn has_x86_simd(&self) -> bool {
        self.sse || self.sse2 || self.avx || self.avx2 || self.avx512f
    }

    /// Check if ARM SIMD is available.
    pub fn has_arm_simd(&self) -> bool {
        self.neon
    }

    /// Get the best available SIMD level as a string.
    pub fn best_level(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
        } else if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse4_2 {
            "SSE4.2"
        } else if self.sse4_1 {
            "SSE4.1"
        } else if self.ssse3 {
            "SSSE3"
        } else if self.sse3 {
            "SSE3"
        } else if self.sse2 {
            "SSE2"
        } else if self.sse {
            "SSE"
        } else if self.neon {
            "NEON"
        } else {
            "None"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_available() {
        // This should work even without the library
        let _ = Simd::is_available();
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let normalized = Simd::normalize(&v);
        // Length should be ~1.0
        let len: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }
}
