// High-performance matrix multiplication compute shader for AI operations
// Optimized for both desktop and WebAssembly WebGPU implementations

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // [M, N, K]

// Workgroup size optimized for most GPUs
const TILE_SIZE: u32 = 16u;

// Shared memory for tiling optimization
var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;
    
    let row = global_id.x;
    let col = global_id.y;
    
    // Check bounds
    if (row >= M || col >= N) {
        return;
    }
    
    var sum = 0.0;
    
    // Tiled matrix multiplication for better cache performance
    let num_tiles = (K + TILE_SIZE - 1u) / TILE_SIZE;
    
    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tile of matrix A into shared memory
        let a_col = tile * TILE_SIZE + local_id.y;
        let a_row = workgroup_id.x * TILE_SIZE + local_id.x;
        
        if (a_row < M && a_col < K) {
            tile_a[local_id.x][local_id.y] = matrix_a[a_row * K + a_col];
        } else {
            tile_a[local_id.x][local_id.y] = 0.0;
        }
        
        // Load tile of matrix B into shared memory
        let b_row = tile * TILE_SIZE + local_id.x;
        let b_col = workgroup_id.y * TILE_SIZE + local_id.y;
        
        if (b_row < K && b_col < N) {
            tile_b[local_id.x][local_id.y] = matrix_b[b_row * N + b_col];
        } else {
            tile_b[local_id.x][local_id.y] = 0.0;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial dot product using shared memory
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_id.x][k] * tile_b[k][local_id.y];
        }
        
        // Synchronize before next tile
        workgroupBarrier();
    }
    
    // Write result
    result[row * N + col] = sum;
}

// Alternative simpler version for small matrices or debugging
@compute @workgroup_size(8, 8, 1)
fn simple_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;
    
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= M || col >= N) {
        return;
    }
    
    var sum = 0.0;
    for (var k = 0u; k < K; k++) {
        sum += matrix_a[row * K + k] * matrix_b[k * N + col];
    }
    
    result[row * N + col] = sum;
}

// Vectorized version for better SIMD utilization
@compute @workgroup_size(64, 1, 1)
fn vectorized_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = dimensions.x;
    let N = dimensions.y;
    let K = dimensions.z;
    
    let index = global_id.x;
    let total_elements = M * N;
    
    if (index >= total_elements) {
        return;
    }
    
    let row = index / N;
    let col = index % N;
    
    var sum = 0.0;
    
    // Process 4 elements at a time for better vectorization
    let k_vec4 = K / 4u;
    let k_remainder = K % 4u;
    
    for (var k = 0u; k < k_vec4; k++) {
        let k_base = k * 4u;
        let a_offset = row * K + k_base;
        let b_offset = k_base * N + col;
        
        // Load 4 elements from A (consecutive in memory)
        let a_vec = vec4<f32>(
            matrix_a[a_offset],
            matrix_a[a_offset + 1u],
            matrix_a[a_offset + 2u],
            matrix_a[a_offset + 3u]
        );
        
        // Load 4 elements from B (strided by N)
        let b_vec = vec4<f32>(
            matrix_b[b_offset],
            matrix_b[b_offset + N],
            matrix_b[b_offset + 2u * N],
            matrix_b[b_offset + 3u * N]
        );
        
        // Compute dot product
        sum += dot(a_vec, b_vec);
    }
    
    // Handle remainder elements
    for (var k = k_vec4 * 4u; k < K; k++) {
        sum += matrix_a[row * K + k] * matrix_b[k * N + col];
    }
    
    result[index] = sum;
} 