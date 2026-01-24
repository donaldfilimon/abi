//! Basic example demonstrating ABI Framework usage
//!
//! Run with: cargo run --example basic

use abi::{simd, Framework, Config};

fn main() -> abi::Result<()> {
    println!("ABI Framework Rust Example");
    println!("==========================\n");

    // Check version
    println!("Version: {}", abi::version());

    // Check SIMD availability
    let caps = simd::capabilities();
    println!("SIMD available: {}", caps.has_simd);
    println!("Architecture: {:?}", caps.arch);
    println!("Vector size: {} bytes\n", caps.vector_size);

    // Vector operations
    println!("Vector Operations:");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![4.0, 3.0, 2.0, 1.0];

    // Dot product
    let dot = simd::dot_product(&a, &b);
    println!("  Dot product: {}", dot);

    // Cosine similarity
    let similarity = simd::cosine_similarity(&a, &b);
    println!("  Cosine similarity: {:.4}", similarity);

    // L2 norm
    let norm = simd::l2_norm(&a);
    println!("  L2 norm of a: {:.4}", norm);

    // Vector addition
    let sum = simd::add(&a, &b);
    println!("  a + b = {:?}", sum);

    // Normalize
    let normalized = simd::normalize(&a);
    println!("  Normalized a: {:?}", normalized);

    // Euclidean distance
    let distance = simd::euclidean_distance(&a, &b);
    println!("  Euclidean distance: {:.4}\n", distance);

    // Matrix multiplication
    println!("Matrix Operations:");
    // 2x2 matrices
    let mat_a = vec![1.0, 2.0, 3.0, 4.0];
    let mat_b = vec![5.0, 6.0, 7.0, 8.0];
    let result = simd::matrix_multiply(&mat_a, &mat_b, 2, 2, 2);
    println!("  A = [1 2; 3 4]");
    println!("  B = [5 6; 7 8]");
    println!("  A * B = [{} {}; {} {}]\n", result[0], result[1], result[2], result[3]);

    // Framework initialization (requires native library)
    println!("Framework Initialization:");
    match Framework::new(Config::minimal()) {
        Ok(framework) => {
            println!("  Framework initialized successfully");
            println!("  AI enabled: {}", framework.is_feature_enabled("ai"));
            println!("  GPU enabled: {}", framework.is_feature_enabled("gpu"));
            println!("  Database enabled: {}", framework.is_feature_enabled("database"));
        }
        Err(e) => {
            println!("  Framework not available: {} (this is expected if native library isn't linked)", e);
        }
    }

    println!("\nExample complete!");
    Ok(())
}
