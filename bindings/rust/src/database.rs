//! Vector database operations.

use crate::error::{check_error, Error, Result};
use crate::ffi::{self, AbiDatabase, AbiDatabaseConfig, AbiSearchResult};
use std::ffi::CString;
use std::ptr;

/// Vector database for similarity search.
///
/// Provides efficient nearest-neighbor search using HNSW index.
///
/// # Example
///
/// ```no_run
/// use abi::VectorDatabase;
///
/// // Create a 128-dimensional database
/// let db = VectorDatabase::new("embeddings", 128).expect("Failed to create database");
///
/// // Insert a vector
/// let vector: Vec<f32> = vec![0.0; 128];
/// db.insert(1, &vector).expect("Failed to insert");
///
/// // Search for similar vectors
/// let results = db.search(&vector, 10).expect("Failed to search");
/// for result in results {
///     println!("ID: {}, Score: {}", result.id, result.score);
/// }
/// ```
pub struct VectorDatabase {
    handle: AbiDatabase,
    dimension: usize,
}

impl VectorDatabase {
    /// Create a new vector database.
    ///
    /// # Arguments
    ///
    /// * `name` - Database name (used for persistence)
    /// * `dimension` - Vector dimension (e.g., 128, 384, 768, 1536)
    pub fn new(name: &str, dimension: usize) -> Result<Self> {
        Self::with_capacity(name, dimension, 1000)
    }

    /// Create a new vector database with initial capacity hint.
    ///
    /// # Arguments
    ///
    /// * `name` - Database name
    /// * `dimension` - Vector dimension
    /// * `initial_capacity` - Expected number of vectors (optimization hint)
    pub fn with_capacity(name: &str, dimension: usize, initial_capacity: usize) -> Result<Self> {
        let c_name = CString::new(name).map_err(|_| Error::InvalidArgument("Invalid name".into()))?;

        let config = AbiDatabaseConfig {
            name: c_name.as_ptr(),
            dimension,
            initial_capacity,
        };

        let mut handle: AbiDatabase = ptr::null_mut();
        let err = unsafe { ffi::abi_database_create(&config, &mut handle) };
        check_error(err)?;

        if handle.is_null() {
            return Err(Error::NullPointer);
        }

        Ok(Self { handle, dimension })
    }

    /// Get the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Insert a vector into the database.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique vector ID
    /// * `vector` - Vector data (must match database dimension)
    ///
    /// # Errors
    ///
    /// Returns an error if the vector dimension doesn't match.
    pub fn insert(&self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::InvalidArgument(format!(
                "Vector dimension {} doesn't match database dimension {}",
                vector.len(),
                self.dimension
            )));
        }

        let err = unsafe { ffi::abi_database_insert(self.handle, id, vector.as_ptr(), vector.len()) };
        check_error(err)
    }

    /// Search for similar vectors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query vector (must match database dimension)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A vector of search results, sorted by similarity (highest first).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidArgument(format!(
                "Query dimension {} doesn't match database dimension {}",
                query.len(),
                self.dimension
            )));
        }

        let mut results: Vec<AbiSearchResult> = vec![
            AbiSearchResult {
                id: 0,
                score: 0.0,
                vector: ptr::null(),
                vector_len: 0,
            };
            k
        ];
        let mut count: usize = 0;

        let err = unsafe {
            ffi::abi_database_search(
                self.handle,
                query.as_ptr(),
                query.len(),
                k,
                results.as_mut_ptr(),
                &mut count,
            )
        };
        check_error(err)?;

        results.truncate(count);
        Ok(results.into_iter().map(|r| SearchResult {
            id: r.id,
            score: r.score,
        }).collect())
    }

    /// Delete a vector from the database.
    pub fn delete(&self, id: u64) -> Result<()> {
        let err = unsafe { ffi::abi_database_delete(self.handle, id) };
        check_error(err)
    }

    /// Get the number of vectors in the database.
    pub fn count(&self) -> Result<usize> {
        let mut count: usize = 0;
        let err = unsafe { ffi::abi_database_count(self.handle, &mut count) };
        check_error(err)?;
        Ok(count)
    }
}

impl Drop for VectorDatabase {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                ffi::abi_database_close(self.handle);
            }
        }
    }
}

unsafe impl Send for VectorDatabase {}

/// Search result from a vector query.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector ID.
    pub id: u64,
    /// Similarity score (higher = more similar).
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch() {
        // This test can run without the library
        let db_result = VectorDatabase::new("test", 128);
        if db_result.is_err() {
            // Library not available, skip
            return;
        }

        let db = db_result.unwrap();
        let wrong_dim = vec![0.0; 64];
        assert!(db.insert(1, &wrong_dim).is_err());
    }
}
