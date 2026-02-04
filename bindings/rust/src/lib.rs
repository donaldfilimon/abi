//! ABI Framework Rust Bindings
//!
//! Safe Rust bindings for the ABI Framework, providing:
//! - SIMD-accelerated vector operations
//! - Vector database for similarity search
//! - GPU acceleration
//! - AI agent capabilities
//!
//! # Quick Start
//!
//! ```no_run
//! use abi::{Framework, VectorDatabase, Simd};
//!
//! // Initialize the framework
//! let framework = Framework::new().expect("Failed to initialize ABI");
//!
//! // Use SIMD operations
//! let a = vec![1.0, 2.0, 3.0, 4.0];
//! let b = vec![5.0, 6.0, 7.0, 8.0];
//! let dot = Simd::dot_product(&a, &b);
//! println!("Dot product: {}", dot);
//!
//! // Create a vector database
//! let db = VectorDatabase::new("test_db", 128).expect("Failed to create database");
//! db.insert(1, &vec![0.0; 128]).expect("Failed to insert");
//! ```

mod ffi;
mod error;
mod framework;
mod simd;
mod database;
mod gpu;
mod agent;

pub use error::{Error, Result};
pub use framework::Framework;
pub use simd::Simd;
pub use database::VectorDatabase;
pub use gpu::Gpu;
pub use agent::Agent;

/// Re-export FFI types for advanced usage.
pub mod raw {
    pub use crate::ffi::*;
}
