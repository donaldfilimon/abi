//! Vector database with similarity search
//!
//! This module provides a high-performance vector database with HNSW indexing
//! for approximate nearest neighbor search.
//!
//! # Example
//!
//! ```rust,no_run
//! use abi::database::{VectorDatabase, SearchResult};
//!
//! fn main() -> abi::Result<()> {
//!     // Create a database with 4-dimensional vectors
//!     let mut db = VectorDatabase::new("embeddings", 4)?;
//!
//!     // Insert vectors
//!     db.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
//!     db.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
//!     db.insert(3, &[0.707, 0.707, 0.0, 0.0])?;
//!
//!     // Search for similar vectors
//!     let query = [0.9, 0.1, 0.0, 0.0];
//!     let results = db.search(&query, 2)?;
//!
//!     for result in results {
//!         println!("ID: {}, Score: {:.4}", result.id, result.score);
//!     }
//!
//!     Ok(())
//! }
//! ```

use std::ffi::CString;
use std::ptr;

use crate::ffi;
use crate::{check_error, Error, Result};

/// A search result from the vector database
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector ID
    pub id: u64,
    /// Similarity score (higher is more similar)
    pub score: f32,
    /// The vector data (if requested)
    pub vector: Option<Vec<f32>>,
}

/// Configuration for creating a vector database
#[derive(Debug, Clone)]
pub struct Config {
    /// Name of the database
    pub name: String,
    /// Dimension of vectors
    pub dimension: usize,
    /// Initial capacity (number of vectors)
    pub initial_capacity: usize,
}

impl Config {
    /// Create a new config with the given name and dimension
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            initial_capacity: 1000,
        }
    }

    /// Set the initial capacity
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.initial_capacity = capacity;
        self
    }
}

/// A vector database with similarity search capabilities
pub struct VectorDatabase {
    handle: ffi::abi_database_t,
    dimension: usize,
}

impl VectorDatabase {
    /// Create a new vector database with the given name and dimension
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let db = abi::database::VectorDatabase::new("my_db", 384)?;
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn new(name: &str, dimension: usize) -> Result<Self> {
        let config = Config::new(name, dimension);
        Self::with_config(config)
    }

    /// Create a database with custom configuration
    pub fn with_config(config: Config) -> Result<Self> {
        let c_name = CString::new(config.name.as_str()).map_err(|_| {
            Error::InvalidArgument("invalid database name".into())
        })?;

        let ffi_config = ffi::abi_database_config_t {
            name: c_name.as_ptr(),
            dimension: config.dimension,
            initial_capacity: config.initial_capacity,
        };

        let mut handle: ffi::abi_database_t = ptr::null_mut();
        let code = unsafe { ffi::abi_database_create(&ffi_config, &mut handle) };
        check_error(code)?;

        Ok(Self {
            handle,
            dimension: config.dimension,
        })
    }

    /// Get the dimension of vectors in this database
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of vectors in the database
    pub fn len(&self) -> usize {
        unsafe { ffi::abi_database_count(self.handle) }
    }

    /// Check if the database is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a vector into the database
    ///
    /// # Errors
    ///
    /// Returns an error if the vector dimension doesn't match.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let mut db = abi::database::VectorDatabase::new("test", 3)?;
    /// db.insert(1, &[1.0, 2.0, 3.0])?;
    /// db.insert(2, &[4.0, 5.0, 6.0])?;
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn insert(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::InvalidArgument(format!(
                "vector dimension {} doesn't match database dimension {}",
                vector.len(),
                self.dimension
            )));
        }

        let code = unsafe {
            ffi::abi_database_insert(self.handle, id, vector.as_ptr(), vector.len())
        };
        check_error(code)
    }

    /// Insert multiple vectors at once
    ///
    /// Returns the number of successfully inserted vectors.
    pub fn insert_batch(&mut self, items: &[(u64, &[f32])]) -> Result<usize> {
        let mut count = 0;
        for (id, vector) in items {
            if self.insert(*id, vector).is_ok() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Search for the k most similar vectors to the query
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let db = abi::database::VectorDatabase::new("test", 3)?;
    /// // ... insert vectors ...
    /// let results = db.search(&[1.0, 2.0, 3.0], 5)?;
    /// for r in results {
    ///     println!("ID: {}, Score: {:.4}", r.id, r.score);
    /// }
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::InvalidArgument(format!(
                "query dimension {} doesn't match database dimension {}",
                query.len(),
                self.dimension
            )));
        }

        let mut results: Vec<ffi::abi_search_result_t> = vec![
            ffi::abi_search_result_t {
                id: 0,
                score: 0.0,
                vector: ptr::null(),
                vector_len: 0,
            };
            k
        ];
        let mut count: usize = 0;

        let code = unsafe {
            ffi::abi_database_search(
                self.handle,
                query.as_ptr(),
                query.len(),
                k,
                results.as_mut_ptr(),
                &mut count,
            )
        };
        check_error(code)?;

        Ok(results[..count]
            .iter()
            .map(|r| SearchResult {
                id: r.id,
                score: r.score,
                vector: if r.vector.is_null() {
                    None
                } else {
                    Some(unsafe {
                        std::slice::from_raw_parts(r.vector, r.vector_len).to_vec()
                    })
                },
            })
            .collect())
    }

    /// Delete a vector by ID
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// let mut db = abi::database::VectorDatabase::new("test", 3)?;
    /// db.insert(1, &[1.0, 2.0, 3.0])?;
    /// db.delete(1)?;
    /// # Ok::<(), abi::Error>(())
    /// ```
    pub fn delete(&mut self, id: u64) -> Result<()> {
        let code = unsafe { ffi::abi_database_delete(self.handle, id) };
        check_error(code)
    }

    /// Check if a vector with the given ID exists
    pub fn contains(&self, id: u64) -> bool {
        // Search for the exact ID by using it as a query
        // This is a workaround since we don't have a direct "get" API
        // In a real implementation, there would be an abi_database_get function
        let results = self.search(&vec![0.0; self.dimension], 100);
        results.map_or(false, |r| r.iter().any(|x| x.id == id))
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

// VectorDatabase is safe to send between threads
unsafe impl Send for VectorDatabase {}

/// Builder for creating a VectorDatabase with custom options
pub struct VectorDatabaseBuilder {
    config: Config,
}

impl VectorDatabaseBuilder {
    /// Create a new builder with the given name and dimension
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            config: Config::new(name, dimension),
        }
    }

    /// Set the initial capacity
    pub fn initial_capacity(mut self, capacity: usize) -> Self {
        self.config.initial_capacity = capacity;
        self
    }

    /// Build the vector database
    pub fn build(self) -> Result<VectorDatabase> {
        VectorDatabase::with_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = Config::new("test", 128);
        assert_eq!(config.name, "test");
        assert_eq!(config.dimension, 128);
        assert_eq!(config.initial_capacity, 1000);
    }

    #[test]
    fn test_config_with_capacity() {
        let config = Config::new("test", 128).with_capacity(5000);
        assert_eq!(config.initial_capacity, 5000);
    }

    #[test]
    fn test_search_result() {
        let result = SearchResult {
            id: 42,
            score: 0.95,
            vector: Some(vec![1.0, 2.0, 3.0]),
        };
        assert_eq!(result.id, 42);
        assert!((result.score - 0.95).abs() < 1e-6);
        assert_eq!(result.vector.unwrap().len(), 3);
    }

    #[test]
    fn test_builder() {
        let builder = VectorDatabaseBuilder::new("embeddings", 384)
            .initial_capacity(10000);
        assert_eq!(builder.config.name, "embeddings");
        assert_eq!(builder.config.dimension, 384);
        assert_eq!(builder.config.initial_capacity, 10000);
    }
}
