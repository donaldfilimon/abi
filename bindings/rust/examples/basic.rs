//! Basic usage example for ABI Rust bindings.

use abi::{Framework, Simd, VectorDatabase};

fn main() -> Result<(), abi::Error> {
    println!("ABI Rust Bindings Example");
    println!("=========================\n");

    // Initialize the framework
    println!("Initializing framework...");
    let framework = Framework::new()?;
    println!("ABI version: {}", framework.version());

    let info = framework.version_info();
    println!("Version info: {}.{}.{}", info.major, info.minor, info.patch);

    // Check features
    println!("\nFeature status:");
    for feature in &["ai", "gpu", "database", "network", "web"] {
        let enabled = framework.is_feature_enabled(feature);
        println!("  {}: {}", feature, if enabled { "enabled" } else { "disabled" });
    }

    // SIMD operations
    println!("\nSIMD Operations:");
    println!("  Available: {}", Simd::is_available());

    let caps = Simd::capabilities();
    println!("  Best level: {}", caps.best_level());

    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    let dot = Simd::dot_product(&a, &b);
    println!("  Dot product: {}", dot);

    let similarity = Simd::cosine_similarity(&a, &b);
    println!("  Cosine similarity: {:.4}", similarity);

    let norm_a = Simd::l2_norm(&a);
    println!("  L2 norm of a: {:.4}", norm_a);

    // Vector database
    println!("\nVector Database:");
    let db = VectorDatabase::new("example_db", 128)?;
    println!("  Created database with dimension {}", db.dimension());

    // Insert some vectors
    for i in 0..5 {
        let vector: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 / 1000.0).collect();
        db.insert(i as u64, &vector)?;
    }
    println!("  Inserted 5 vectors");

    let count = db.count()?;
    println!("  Total vectors: {}", count);

    // Search
    let query: Vec<f32> = (0..128).map(|j| j as f32 / 1000.0).collect();
    let results = db.search(&query, 3)?;
    println!("  Search results:");
    for result in results {
        println!("    ID: {}, Score: {:.4}", result.id, result.score);
    }

    println!("\nExample completed successfully!");
    Ok(())
}
