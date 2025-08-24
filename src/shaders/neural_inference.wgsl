// Neural network inference compute shader for AI operations
// Supports fully connected layers, activation functions, and batch processing
// Optimized for both desktop and WebAssembly WebGPU implementations

// Input/output buffers
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;

// Network configuration
@group(0) @binding(4) var<uniform> config: NetworkConfig;

struct NetworkConfig {
    batch_size: u32,
    input_size: u32,
    output_size: u32,
    activation_type: u32, // 0=ReLU, 1=Sigmoid, 2=Tanh, 3=Softmax, 4=GELU
    use_bias: u32,
    layer_type: u32, // 0=Dense, 1=Conv1D, 2=Attention
}

// Activation functions
fn relu(x: f32) -> f32 {
    return max(0.0, x);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn tanh_activation(x: f32) -> f32 {
    return tanh(x);
}

fn gelu(x: f32) -> f32 {
    // Gaussian Error Linear Unit: x * Φ(x) where Φ is the CDF of standard normal
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608;
    let a = 0.044715;
    let inner = sqrt_2_over_pi * (x + a * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

fn apply_activation(x: f32, activation_type: u32) -> f32 {
    switch (activation_type) {
        case 0u: { return relu(x); }
        case 1u: { return sigmoid(x); }
        case 2u: { return tanh_activation(x); }
        case 4u: { return gelu(x); }
        default: { return x; } // Linear activation
    }
}

// Dense (fully connected) layer computation
@compute @workgroup_size(64, 1, 1)
fn dense_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x / config.output_size;
    let output_idx = global_id.x % config.output_size;
    
    if (batch_idx >= config.batch_size || output_idx >= config.output_size) {
        return;
    }
    
    let input_offset = batch_idx * config.input_size;
    let weight_offset = output_idx * config.input_size;
    let output_offset = batch_idx * config.output_size + output_idx;
    
    var sum = 0.0;
    
    // Vectorized computation for better performance
    let vec4_count = config.input_size / 4u;
    let remainder = config.input_size % 4u;
    
    // Process 4 elements at a time
    for (var i = 0u; i < vec4_count; i++) {
        let base_idx = i * 4u;
        
        let input_vec = vec4<f32>(
            input_data[input_offset + base_idx],
            input_data[input_offset + base_idx + 1u],
            input_data[input_offset + base_idx + 2u],
            input_data[input_offset + base_idx + 3u]
        );
        
        let weight_vec = vec4<f32>(
            weights[weight_offset + base_idx],
            weights[weight_offset + base_idx + 1u],
            weights[weight_offset + base_idx + 2u],
            weights[weight_offset + base_idx + 3u]
        );
        
        sum += dot(input_vec, weight_vec);
    }
    
    // Handle remaining elements
    for (var i = vec4_count * 4u; i < config.input_size; i++) {
        sum += input_data[input_offset + i] * weights[weight_offset + i];
    }
    
    // Add bias if enabled
    if (config.use_bias != 0u) {
        sum += biases[output_idx];
    }
    
    // Apply activation function
    let activated = apply_activation(sum, config.activation_type);
    output_data[output_offset] = activated;
}

// Softmax activation (requires separate pass for normalization)
@compute @workgroup_size(256, 1, 1)
fn softmax_activation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= config.batch_size) {
        return;
    }
    
    let batch_offset = batch_idx * config.output_size;
    
    // Find maximum value for numerical stability
    var max_val = output_data[batch_offset];
    for (var i = 1u; i < config.output_size; i++) {
        max_val = max(max_val, output_data[batch_offset + i]);
    }
    
    // Compute exp(x - max) and sum
    var sum_exp = 0.0;
    for (var i = 0u; i < config.output_size; i++) {
        let exp_val = exp(output_data[batch_offset + i] - max_val);
        output_data[batch_offset + i] = exp_val;
        sum_exp += exp_val;
    }
    
    // Normalize
    let inv_sum = 1.0 / sum_exp;
    for (var i = 0u; i < config.output_size; i++) {
        output_data[batch_offset + i] *= inv_sum;
    }
}

// Batch normalization
@compute @workgroup_size(64, 1, 1)
fn batch_normalization(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @group(1) @binding(0) var<storage, read> bn_mean: array<f32>,
                       @group(1) @binding(1) var<storage, read> bn_variance: array<f32>,
                       @group(1) @binding(2) var<storage, read> bn_gamma: array<f32>,
                       @group(1) @binding(3) var<storage, read> bn_beta: array<f32>) {
    let batch_idx = global_id.x / config.output_size;
    let feature_idx = global_id.x % config.output_size;
    
    if (batch_idx >= config.batch_size || feature_idx >= config.output_size) {
        return;
    }
    
    let idx = batch_idx * config.output_size + feature_idx;
    let x = output_data[idx];
    
    // Normalize: (x - mean) / sqrt(variance + epsilon)
    let epsilon = 1e-5;
    let normalized = (x - bn_mean[feature_idx]) / sqrt(bn_variance[feature_idx] + epsilon);
    
    // Scale and shift: gamma * normalized + beta
    output_data[idx] = bn_gamma[feature_idx] * normalized + bn_beta[feature_idx];
}

// Layer normalization (alternative to batch norm)
@compute @workgroup_size(256, 1, 1)
fn layer_normalization(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @group(1) @binding(0) var<storage, read> ln_gamma: array<f32>,
                       @group(1) @binding(1) var<storage, read> ln_beta: array<f32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx >= config.batch_size) {
        return;
    }
    
    let batch_offset = batch_idx * config.output_size;
    
    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < config.output_size; i++) {
        sum += output_data[batch_offset + i];
    }
    let mean = sum / f32(config.output_size);
    
    // Compute variance
    var variance_sum = 0.0;
    for (var i = 0u; i < config.output_size; i++) {
        let diff = output_data[batch_offset + i] - mean;
        variance_sum += diff * diff;
    }
    let variance = variance_sum / f32(config.output_size);
    
    // Normalize and apply learned parameters
    let epsilon = 1e-5;
    let inv_std = 1.0 / sqrt(variance + epsilon);
    
    for (var i = 0u; i < config.output_size; i++) {
        let normalized = (output_data[batch_offset + i] - mean) * inv_std;
        output_data[batch_offset + i] = ln_gamma[i] * normalized + ln_beta[i];
    }
}

// Dropout (for training - sets random elements to 0)
@compute @workgroup_size(64, 1, 1)
fn dropout(@builtin(global_invocation_id) global_id: vec3<u32>,
           @group(1) @binding(0) var<storage, read> random_mask: array<f32>) {
    let idx = global_id.x;
    
    if (idx >= config.batch_size * config.output_size) {
        return;
    }
    
    // Apply dropout mask (0 or 1 values)
    output_data[idx] *= random_mask[idx];
}

// Element-wise operations (for residual connections, etc.)
@compute @workgroup_size(64, 1, 1)
fn element_add(@builtin(global_invocation_id) global_id: vec3<u32>,
               @group(1) @binding(0) var<storage, read> add_tensor: array<f32>) {
    let idx = global_id.x;
    
    if (idx >= config.batch_size * config.output_size) {
        return;
    }
    
    output_data[idx] += add_tensor[idx];
}

// Matrix transpose utility (useful for weight transformations)
@compute @workgroup_size(16, 16, 1)
fn transpose_matrix(@builtin(global_invocation_id) global_id: vec3<u32>,
                    @group(1) @binding(0) var<storage, read> input_matrix: array<f32>,
                    @group(1) @binding(1) var<storage, read_write> output_matrix: array<f32>,
                    @group(1) @binding(2) var<uniform> matrix_dims: vec2<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    if (row >= matrix_dims.x || col >= matrix_dims.y) {
        return;
    }
    
    let input_idx = row * matrix_dims.y + col;
    let output_idx = col * matrix_dims.x + row;
    
    output_matrix[output_idx] = input_matrix[input_idx];
} 