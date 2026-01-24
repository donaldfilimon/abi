//! Vector database example demonstrating similarity search
//!
//! Run with: cargo run --example vector_db --features database

use abi::database::{VectorDatabase, VectorDatabaseBuilder};
use abi::simd;

fn main() -> abi::Result<()> {
    println!("ABI Vector Database Example");
    println!("============================\n");

    // Create a database with 4-dimensional vectors
    let dimension = 4;
    let mut db = VectorDatabaseBuilder::new("example_db", dimension)
        .initial_capacity(100)
        .build()?;

    println!("Created database with dimension: {}", db.dimension());

    // Sample vectors representing different concepts
    // These could be embeddings from an AI model in real usage
    let vectors = [
        (1, vec![1.0, 0.0, 0.0, 0.0], "North"),
        (2, vec![0.0, 1.0, 0.0, 0.0], "East"),
        (3, vec![0.0, 0.0, 1.0, 0.0], "Up"),
        (4, vec![0.0, 0.0, 0.0, 1.0], "Time"),
        (5, vec![0.707, 0.707, 0.0, 0.0], "Northeast"),
        (6, vec![0.707, 0.0, 0.707, 0.0], "North-Up"),
        (7, vec![0.577, 0.577, 0.577, 0.0], "Diagonal-3D"),
        (8, vec![-1.0, 0.0, 0.0, 0.0], "South"),
        (9, vec![0.0, -1.0, 0.0, 0.0], "West"),
        (10, vec![0.0, 0.0, -1.0, 0.0], "Down"),
    ];

    // Insert vectors
    println!("\nInserting {} vectors...", vectors.len());
    for (id, vector, name) in &vectors {
        db.insert(*id, vector)?;
        println!("  Inserted {}: {} (ID={})", name, format_vec(vector), id);
    }

    println!("\nDatabase now contains {} vectors", db.len());

    // Demonstrate similarity search
    println!("\n--- Similarity Search Examples ---\n");

    // Query 1: Find vectors similar to "North"
    let query1 = vec![0.9, 0.1, 0.0, 0.0];
    println!("Query 1: {} (slightly off North)", format_vec(&query1));
    search_and_display(&db, &query1, 3, &vectors)?;

    // Query 2: Find vectors similar to diagonal
    let query2 = vec![0.5, 0.5, 0.5, 0.0];
    println!("\nQuery 2: {} (roughly diagonal)", format_vec(&query2));
    search_and_display(&db, &query2, 3, &vectors)?;

    // Query 3: Find opposite vectors (South-ish)
    let query3 = vec![-0.8, -0.2, 0.0, 0.0];
    println!("\nQuery 3: {} (South-West-ish)", format_vec(&query3));
    search_and_display(&db, &query3, 3, &vectors)?;

    // Demonstrate vector operations alongside database
    println!("\n--- Vector Analysis ---\n");

    // Calculate similarity between stored vectors
    let north = &vectors[0].1;
    let northeast = &vectors[4].1;
    let south = &vectors[7].1;

    println!("Cosine similarity:");
    println!(
        "  North <-> Northeast: {:.4}",
        simd::cosine_similarity(north, northeast)
    );
    println!(
        "  North <-> South: {:.4}",
        simd::cosine_similarity(north, south)
    );

    // Calculate distances
    println!("\nEuclidean distance:");
    println!(
        "  North <-> Northeast: {:.4}",
        simd::euclidean_distance(north, northeast)
    );
    println!(
        "  North <-> South: {:.4}",
        simd::euclidean_distance(north, south)
    );

    // Demonstrate deletion
    println!("\n--- Deletion ---\n");
    println!("Deleting vector ID=5 (Northeast)...");
    db.delete(5)?;
    println!("Database now contains {} vectors", db.len());

    // Search again to verify deletion
    println!("\nSearching for North again (Northeast should not appear):");
    let query4 = vec![0.9, 0.1, 0.0, 0.0];
    search_and_display(&db, &query4, 3, &vectors)?;

    println!("\nExample complete!");
    Ok(())
}

fn search_and_display(
    db: &VectorDatabase,
    query: &[f32],
    k: usize,
    vectors: &[(u64, Vec<f32>, &str)],
) -> abi::Result<()> {
    let results = db.search(query, k)?;
    println!("  Top {} results:", results.len());
    for (i, result) in results.iter().enumerate() {
        let name = vectors
            .iter()
            .find(|(id, _, _)| *id == result.id)
            .map(|(_, _, n)| *n)
            .unwrap_or("Unknown");
        println!(
            "    {}. {} (ID={}, score={:.4})",
            i + 1,
            name,
            result.id,
            result.score
        );
    }
    Ok(())
}

fn format_vec(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{:.2}", x)).collect();
    format!("[{}]", parts.join(", "))
}
