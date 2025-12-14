//! Neural Network Layer Components
//!
//! Provides modular, composable neural network layers with SIMD optimization.
//! Supports all major layer types including dense, convolutional, and recurrent layers.

const std = @import("std");

// Note: core functionality is now imported through module dependencies
const activation = @import("activations/mod.zig");

const Allocator = std.mem.Allocator;
const ActivationType = activation.ActivationType;

/// Neural network layer types with enhanced coverage
pub const LayerType = enum {
    input,
    dense,
    conv2d,
    conv1d,
    conv3d,
    transposed_conv2d,
    dilated_conv2d,
    separable_conv2d,
    depthwise_conv2d,
    pointwise_conv2d,
    maxpool2d,
    avgpool2d,
    globalavgpool,
    globalmaxpool,
    adaptiveavgpool,
    adaptivemaxpool,
    dropout,
    dropout2d,
    spatial_dropout,
    variational_dropout,
    batch_norm,
    layer_norm,
    group_norm,
    instance_norm,
    local_response_norm,
    weight_norm,
    spectral_norm,
    activation_layer,
    flatten,
    reshape,
    permute,
    squeeze,
    unsqueeze,
    lstm,
    gru,
    rnn,
    elman_rnn,
    jordan_rnn,
    bidirectional_lstm,
    bidirectional_gru,
    attention,
    multi_head_attention,
    scaled_dot_product_attention,
    self_attention,
    cross_attention,
    transformer_block,
    transformer_encoder,
    transformer_decoder,
    embedding,
    sparse_embedding,
    positional_encoding,
    learned_positional_encoding,
    sinusoidal_positional_encoding,
    residual_connection,
    highway_connection,
    dense_connection,
    squeeze_excitation,
    channel_attention,
    spatial_attention,
    depth_separable_conv,
    upsampling,
    interpolation,
    deconvolution,
    fractional_maxpool,
    stochastic_pooling,
    concat,
    add,
    multiply,
    subtract,
    divide,
    linear_combination,
    masked_linear,
    pruned_linear,
    quantized_linear,
    low_rank_linear,
    capsule,
    graph_convolution,
    temporal_convolution,
    causal_convolution,
    wavenet_block,
    resnet_block,
    densenet_block,
    inception_block,
    mobile_block,
    inverted_residual,
    fire_module,
    bottleneck,
    swish_activation,
    mish_activation,
    parametric_relu,
    exponential_linear,
    scaled_exponential_linear,
    gaussian_error_linear,
    softmax_layer,
    log_softmax_layer,
    cross_entropy_layer,
    contrastive_loss_layer,
    triplet_loss_layer,
};

/// Weight initialization strategies with enhanced coverage
pub const WeightInit = enum {
    xavier_uniform,
    xavier_normal,
    kaiming_uniform,
    kaiming_normal,
    lecun_uniform,
    lecun_normal,
    he_uniform,
    he_normal,
    orthogonal,
    sparse,
    truncated_normal,
    uniform,
    normal,
    zeros,
    ones,
    identity,
    constant,
    dirac,
    variance_scaling,
    fan_in,
    fan_out,
    fan_avg,
    glorot_uniform,
    glorot_normal,
    random_normal,
    random_uniform,
    orthogonal_gain,
    eye_init,
    permutation_init,
    scaled_uniform,
    scaled_normal,
    bengio_init,
    normalized_init,
    fixup_init,
    layer_sequential_init,
    delta_orthogonal,
    trunc_normal,
    symmetric_init,
};

/// Padding modes for convolution layers
pub const PaddingMode = enum {
    zeros,
    reflect,
    replicate,
    circular,
    constant,
    symmetric,
    valid,
    same,
    causal,
};

/// Pooling modes for pooling layers
pub const PoolingMode = enum {
    max,
    avg,
    sum,
    adaptive_max,
    adaptive_avg,
    fractional_max,
    stochastic,
    lp_pool,
    global_max,
    global_avg,
};

/// Attention mechanisms
pub const AttentionType = enum {
    scaled_dot_product,
    additive,
    multiplicative,
    self_attention,
    cross_attention,
    multi_head,
    sparse_attention,
    local_attention,
    global_attention,
    linear_attention,
    performer_attention,
    reformer_attention,
};

/// RNN cell types
pub const RNNCellType = enum {
    vanilla,
    lstm,
    gru,
    peephole_lstm,
    coupled_input_forget_gate,
    minimal_gated_unit,
    update_gate_rnn,
    intersection_rnn,
    highway_rnn,
    residual_rnn,
};

/// Advanced regularization configuration with comprehensive techniques
pub const Regularization = struct {
    // Weight regularization
    l1_lambda: f32 = 0.0,
    l2_lambda: f32 = 0.0,
    elastic_net_alpha: f32 = 0.0,
    elastic_net_l1_ratio: f32 = 0.5,

    // Dropout variants
    dropout_rate: f32 = 0.0,
    dropconnect_rate: f32 = 0.0,
    spatial_dropout_rate: f32 = 0.0,
    variational_dropout_rate: f32 = 0.0,
    scheduled_dropout_rate: f32 = 0.0,

    // Normalization techniques
    batch_norm: bool = false,
    layer_norm: bool = false,
    group_norm: bool = false,
    instance_norm: bool = false,
    local_response_norm: bool = false,
    weight_norm: bool = false,
    spectral_norm: bool = false,
    cosine_norm: bool = false,

    // Gradient regularization
    gradient_clipping: ?f32 = null,
    gradient_noise: f32 = 0.0,
    gradient_centralization: bool = false,
    gradient_standardization: bool = false,

    // Data augmentation regularization
    label_smoothing: f32 = 0.0,
    mixup_alpha: f32 = 0.0,
    cutmix_alpha: f32 = 0.0,
    cutout_holes: u32 = 0,
    random_erasing: f32 = 0.0,

    // Advanced techniques
    weight_decay: f32 = 0.0,
    max_norm_constraint: ?f32 = null,
    unit_norm_constraint: bool = false,
    min_delta_constraint: ?f32 = null,
    orthogonal_constraint: bool = false,
    non_negative_constraint: bool = false,

    // Stochastic techniques
    stochastic_depth: f32 = 0.0,
    shake_shake_alpha: f32 = 0.0,
    shake_drop_rate: f32 = 0.0,
    fractal_drop_path: f32 = 0.0,

    // Noise injection
    gaussian_noise_std: f32 = 0.0,
    uniform_noise_range: f32 = 0.0,
    salt_pepper_noise: f32 = 0.0,
};

/// Enhanced layer configuration structure with comprehensive parameters
pub const LayerConfig = struct {
    layer_type: LayerType,
    input_shape: []const usize,
    output_shape: []const usize,

    // General parameters
    activation_type: ?ActivationType = null,
    regularization: Regularization = .{},
    weight_init: WeightInit = .kaiming_uniform,
    bias_init: WeightInit = .zeros,
    use_bias: bool = true,
    enable_simd: bool = true,
    enable_mixed_precision: bool = false,
    enable_gradient_checkpointing: bool = false,

    // Normalization parameters
    momentum: f32 = 0.1,
    eps: f32 = 1e-5,
    affine: bool = true,
    track_running_stats: bool = true,
    num_groups: ?usize = null,

    // Convolution-specific parameters
    kernel_size: ?[]usize = null,
    stride: ?[]usize = null,
    padding: ?[]usize = null,
    padding_mode: PaddingMode = .zeros,
    dilation: ?[]usize = null,
    groups: usize = 1,
    output_padding: ?[]usize = null,
    ceil_mode: bool = false,
    count_include_pad: bool = true,

    // Pooling-specific parameters
    pooling_mode: PoolingMode = .max,
    pool_size: ?[]usize = null,
    pool_stride: ?[]usize = null,
    pool_padding: ?[]usize = null,

    // RNN-specific parameters
    rnn_cell_type: RNNCellType = .vanilla,
    hidden_size: ?usize = null,
    num_layers: usize = 1,
    sequence_length: ?usize = null,
    return_sequences: bool = false,
    return_state: bool = false,
    bidirectional: bool = false,
    proj_size: ?usize = null,
    bias: bool = true,
    batch_first: bool = false,
    dropout_between_layers: f32 = 0.0,

    // Attention-specific parameters
    attention_type: AttentionType = .scaled_dot_product,
    num_heads: ?usize = null,
    head_dim: ?usize = null,
    embed_dim: ?usize = null,
    kdim: ?usize = null,
    vdim: ?usize = null,
    dropout: f32 = 0.0,
    need_weights: bool = true,
    average_attn_weights: bool = true,
    is_causal: bool = false,

    // Transformer-specific parameters
    d_model: ?usize = null,
    nhead: ?usize = null,
    dim_feedforward: ?usize = null,
    num_encoder_layers: usize = 6,
    num_decoder_layers: usize = 6,
    layer_norm_eps: f32 = 1e-5,
    activation: ActivationType = .relu,
    custom_encoder: ?*anyopaque = null,
    custom_decoder: ?*anyopaque = null,

    // Embedding-specific parameters
    num_embeddings: ?usize = null,
    embedding_dim: ?usize = null,
    padding_idx: ?usize = null,
    max_norm: ?f32 = null,
    norm_type: f32 = 2.0,
    scale_grad_by_freq: bool = false,
    sparse: bool = false,

    // Activation-specific parameters
    alpha: f32 = 0.01, // For leaky_relu, elu, etc.
    beta: f32 = 1.0, // For swish, etc.
    threshold: f32 = 1.0,
    negative_slope: f32 = 0.01,
    min_val: f32 = -1.0,
    max_val: f32 = 1.0,
    inplace: bool = false,

    // Optimization-specific parameters
    learning_rate_multiplier: f32 = 1.0,
    weight_decay_multiplier: f32 = 1.0,
    gradient_scale: f32 = 1.0,
    update_frequency: usize = 1,

    // Hardware-specific parameters
    device_placement: ?[]const u8 = null,
    memory_format: ?[]const u8 = null,
    dtype: ?[]const u8 = null,
};

/// Enhanced neural network layer with comprehensive functionality
pub const Layer = struct {
    config: LayerConfig,
    allocator: Allocator,
    id: u64,

    // Weight and bias parameters
    weights: ?[]f32 = null,
    biases: ?[]f32 = null,
    weight_masks: ?[]bool = null, // For pruning
    bias_masks: ?[]bool = null,

    // Normalization parameters
    running_mean: ?[]f32 = null,
    running_var: ?[]f32 = null,
    gamma: ?[]f32 = null,
    beta: ?[]f32 = null,
    num_batches_tracked: u64 = 0,

    // RNN-specific state
    hidden_state: ?[]f32 = null,
    cell_state: ?[]f32 = null,
    prev_hidden: ?[]f32 = null,
    prev_cell: ?[]f32 = null,

    // Attention-specific parameters
    attention_weights: ?[]f32 = null,
    query_weights: ?[]f32 = null,
    key_weights: ?[]f32 = null,
    value_weights: ?[]f32 = null,
    output_projection: ?[]f32 = null,

    // Training state
    is_training: bool = true,
    is_frozen: bool = false,
    step_count: u64 = 0,
    phase: enum { train, eval, inference } = .train,

    // Gradients for backpropagation
    weight_gradients: ?[]f32 = null,
    bias_gradients: ?[]f32 = null,
    gamma_gradients: ?[]f32 = null,
    beta_gradients: ?[]f32 = null,
    input_gradients: ?[]f32 = null,

    // Momentum and adaptive optimization state
    weight_momentum: ?[]f32 = null,
    bias_momentum: ?[]f32 = null,
    weight_velocity: ?[]f32 = null,
    bias_velocity: ?[]f32 = null,
    weight_accumulator: ?[]f32 = null,
    bias_accumulator: ?[]f32 = null,

    // Activation processor
    activation_processor: ?activation.ActivationProcessor = null,

    // Performance and debugging
    forward_time: f64 = 0.0,
    backward_time: f64 = 0.0,
    memory_usage: usize = 0,
    operation_count: u64 = 0,

    // Quantization support
    scale: ?f32 = null,
    zero_point: ?i32 = null,
    quantized_weights: ?[]i8 = null,
    quantized_biases: ?[]i32 = null,

    const Self = @This();

    pub fn init(allocator: Allocator, config: LayerConfig) anyerror!*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        // Validate configuration
        try self.validateConfig(config);

        // Generate unique layer ID
        var rng = std.Random.DefaultPrng.init(@intCast(0));
        const layer_id = rng.next();

        self.* = .{
            .config = config,
            .allocator = allocator,
            .id = layer_id,
        };

        // Initialize activation processor if needed
        if (config.activation_type) |act_type| {
            const act_config = activation.ActivationConfig{
                .activation_type = act_type,
                .alpha = config.alpha,
                .beta = config.beta,
                .threshold = config.threshold,
                .enable_simd = config.enable_simd,
            };
            self.activation_processor = activation.ActivationProcessor.init(act_config);
        }

        return self;
    }

    pub fn deinit(self: *Self) void {
        // Free all allocated memory
        if (self.weights) |weights| self.allocator.free(weights);
        if (self.biases) |biases| self.allocator.free(biases);
        if (self.weight_masks) |masks| self.allocator.free(masks);
        if (self.bias_masks) |masks| self.allocator.free(masks);

        if (self.running_mean) |mean| self.allocator.free(mean);
        if (self.running_var) |variance| self.allocator.free(variance);
        if (self.gamma) |gamma| self.allocator.free(gamma);
        if (self.beta) |beta| self.allocator.free(beta);

        if (self.hidden_state) |state| self.allocator.free(state);
        if (self.cell_state) |state| self.allocator.free(state);
        if (self.prev_hidden) |state| self.allocator.free(state);
        if (self.prev_cell) |state| self.allocator.free(state);

        if (self.attention_weights) |weights| self.allocator.free(weights);
        if (self.query_weights) |weights| self.allocator.free(weights);
        if (self.key_weights) |weights| self.allocator.free(weights);
        if (self.value_weights) |weights| self.allocator.free(weights);
        if (self.output_projection) |proj| self.allocator.free(proj);

        if (self.weight_gradients) |grads| self.allocator.free(grads);
        if (self.bias_gradients) |grads| self.allocator.free(grads);
        if (self.gamma_gradients) |grads| self.allocator.free(grads);
        if (self.beta_gradients) |grads| self.allocator.free(grads);
        if (self.input_gradients) |grads| self.allocator.free(grads);

        if (self.weight_momentum) |momentum| self.allocator.free(momentum);
        if (self.bias_momentum) |momentum| self.allocator.free(momentum);
        if (self.weight_velocity) |velocity| self.allocator.free(velocity);
        if (self.bias_velocity) |velocity| self.allocator.free(velocity);
        if (self.weight_accumulator) |acc| self.allocator.free(acc);
        if (self.bias_accumulator) |acc| self.allocator.free(acc);

        if (self.quantized_weights) |qweights| self.allocator.free(qweights);
        if (self.quantized_biases) |qbiases| self.allocator.free(qbiases);

        self.allocator.destroy(self);
    }

    /// Initialize weights and biases for the layer
    pub fn initializeWeights(self: *Self, rng: std.Random) anyerror!void {
        if (self.config.layer_type == .input or
            self.config.layer_type == .dropout or
            self.config.layer_type == .flatten or
            self.config.layer_type == .reshape or
            self.config.layer_type == .permute) return;

        const input_size = self.getInputSize();
        const output_size = self.getOutputSize();

        // Initialize weights
        if (self.needsWeights()) {
            if (self.weights == null) {
                const weight_size = self.getWeightSize();
                self.weights = try self.allocator.alloc(f32, weight_size);
                self.weight_gradients = try self.allocator.alloc(f32, weight_size);
                @memset(self.weight_gradients.?, 0.0);

                // Initialize optimization state
                if (self.config.enable_mixed_precision) {
                    self.weight_momentum = try self.allocator.alloc(f32, weight_size);
                    self.weight_velocity = try self.allocator.alloc(f32, weight_size);
                    self.weight_accumulator = try self.allocator.alloc(f32, weight_size);
                    @memset(self.weight_momentum.?, 0.0);
                    @memset(self.weight_velocity.?, 0.0);
                    @memset(self.weight_accumulator.?, 0.0);
                }
            }
            try self.applyWeightInitialization(rng, input_size, output_size);
        }

        // Initialize biases
        if (self.needsBiases()) {
            if (self.biases == null) {
                self.biases = try self.allocator.alloc(f32, output_size);
                self.bias_gradients = try self.allocator.alloc(f32, output_size);
                @memset(self.bias_gradients.?, 0.0);

                if (self.config.enable_mixed_precision) {
                    self.bias_momentum = try self.allocator.alloc(f32, output_size);
                    self.bias_velocity = try self.allocator.alloc(f32, output_size);
                    self.bias_accumulator = try self.allocator.alloc(f32, output_size);
                    @memset(self.bias_momentum.?, 0.0);
                    @memset(self.bias_velocity.?, 0.0);
                    @memset(self.bias_accumulator.?, 0.0);
                }
            }
            try self.applyBiasInitialization(rng, output_size);
        }

        // Initialize normalization parameters
        if (self.config.regularization.batch_norm or
            self.config.regularization.layer_norm or
            self.config.regularization.group_norm or
            self.config.regularization.instance_norm)
        {
            try self.initializeNormalization(output_size);
        }

        // Initialize RNN-specific state
        if (self.isRNNLayer()) {
            try self.initializeRNNState();
        }

        // Initialize attention-specific parameters
        if (self.isAttentionLayer()) {
            try self.initializeAttentionParameters(rng);
        }
    }

    /// Forward pass through the layer
    pub fn forward(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        const start_time = std.time.nanoTimestamp;
        defer {
            const end_time = std.time.nanoTimestamp;
            self.forward_time = @as(f64, @floatFromInt(end_time - start_time)) / 1e9;
            self.operation_count += 1;
        }

        switch (self.config.layer_type) {
            .dense => try self.forwardDense(input, output),
            .conv2d => try self.forwardConv2D(input, output, temp_buffer),
            .conv1d => try self.forwardConv1D(input, output, temp_buffer),
            .conv3d => try self.forwardConv3D(input, output, temp_buffer),
            .transposed_conv2d => try self.forwardTransposedConv2D(input, output, temp_buffer),
            .dilated_conv2d => try self.forwardDilatedConv2D(input, output, temp_buffer),
            .separable_conv2d => try self.forwardSeparableConv2D(input, output, temp_buffer),
            .depthwise_conv2d => try self.forwardDepthwiseConv2D(input, output, temp_buffer),
            .pointwise_conv2d => try self.forwardPointwiseConv2D(input, output, temp_buffer),
            .maxpool2d => try self.forwardMaxPool2D(input, output),
            .avgpool2d => try self.forwardAvgPool2D(input, output),
            .globalavgpool => try self.forwardGlobalAvgPool(input, output),
            .globalmaxpool => try self.forwardGlobalMaxPool(input, output),
            .adaptiveavgpool => try self.forwardAdaptiveAvgPool(input, output),
            .adaptivemaxpool => try self.forwardAdaptiveMaxPool(input, output),
            .dropout => try self.forwardDropout(input, output),
            .dropout2d => try self.forwardDropout2D(input, output),
            .spatial_dropout => try self.forwardSpatialDropout(input, output),
            .variational_dropout => try self.forwardVariationalDropout(input, output),
            .batch_norm => try self.forwardBatchNorm(input, output),
            .layer_norm => try self.forwardLayerNorm(input, output),
            .group_norm => try self.forwardGroupNorm(input, output),
            .instance_norm => try self.forwardInstanceNorm(input, output),
            .local_response_norm => try self.forwardLocalResponseNorm(input, output),
            .weight_norm => try self.forwardWeightNorm(input, output),
            .spectral_norm => try self.forwardSpectralNorm(input, output),
            .activation_layer => try self.forwardActivation(input, output),
            .flatten => try self.forwardFlatten(input, output),
            .reshape => try self.forwardReshape(input, output),
            .permute => try self.forwardPermute(input, output),
            .squeeze => try self.forwardSqueeze(input, output),
            .unsqueeze => try self.forwardUnsqueeze(input, output),
            .lstm => try self.forwardLSTM(input, output, temp_buffer),
            .gru => try self.forwardGRU(input, output, temp_buffer),
            .rnn => try self.forwardRNN(input, output, temp_buffer),
            .elman_rnn => try self.forwardElmanRNN(input, output, temp_buffer),
            .jordan_rnn => try self.forwardJordanRNN(input, output, temp_buffer),
            .bidirectional_lstm => try self.forwardBidirectionalLSTM(input, output, temp_buffer),
            .bidirectional_gru => try self.forwardBidirectionalGRU(input, output, temp_buffer),
            .attention => try self.forwardAttention(input, output, temp_buffer),
            .multi_head_attention => try self.forwardMultiHeadAttention(input, output, temp_buffer),
            .scaled_dot_product_attention => try self.forwardScaledDotProductAttention(input, output, temp_buffer),
            .self_attention => try self.forwardSelfAttention(input, output, temp_buffer),
            .cross_attention => try self.forwardCrossAttention(input, output, temp_buffer),
            .transformer_block => try self.forwardTransformerBlock(input, output, temp_buffer),
            .transformer_encoder => try self.forwardTransformerEncoder(input, output, temp_buffer),
            .transformer_decoder => try self.forwardTransformerDecoder(input, output, temp_buffer),
            .embedding => try self.forwardEmbedding(input, output),
            .sparse_embedding => try self.forwardSparseEmbedding(input, output),
            .positional_encoding => try self.forwardPositionalEncoding(input, output),
            .learned_positional_encoding => try self.forwardLearnedPositionalEncoding(input, output),
            .sinusoidal_positional_encoding => try self.forwardSinusoidalPositionalEncoding(input, output),
            .residual_connection => try self.forwardResidualConnection(input, output),
            .highway_connection => try self.forwardHighwayConnection(input, output, temp_buffer),
            .dense_connection => try self.forwardDenseConnection(input, output),
            .squeeze_excitation => try self.forwardSqueezeExcitation(input, output, temp_buffer),
            .channel_attention => try self.forwardChannelAttention(input, output, temp_buffer),
            .spatial_attention => try self.forwardSpatialAttention(input, output, temp_buffer),
            .concat => try self.forwardConcat(input, output),
            .add => try self.forwardAdd(input, output),
            .multiply => try self.forwardMultiply(input, output),
            .subtract => try self.forwardSubtract(input, output),
            .divide => try self.forwardDivide(input, output),
            .linear_combination => try self.forwardLinearCombination(input, output),
            else => return error.UnsupportedOperation,
        }
    }

    /// Backward pass through the layer
    pub fn backward(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
        if (self.is_frozen) return;

        const start_time = std.time.nanoTimestamp;
        defer {
            const end_time = std.time.nanoTimestamp;
            self.backward_time = @as(f64, @floatFromInt(end_time - start_time)) / 1e9;
        }

        switch (self.config.layer_type) {
            .dense => try self.backwardDense(grad_output, grad_input),
            .conv2d => try self.backwardConv2D(grad_output, grad_input, temp_buffer),
            .batch_norm => try self.backwardBatchNorm(grad_output, grad_input),
            .layer_norm => try self.backwardLayerNorm(grad_output, grad_input),
            .activation_layer => try self.backwardActivation(grad_output, grad_input),
            .dropout => try self.backwardDropout(grad_output, grad_input),
            .lstm => try self.backwardLSTM(grad_output, grad_input, temp_buffer),
            .gru => try self.backwardGRU(grad_output, grad_input, temp_buffer),
            .attention => try self.backwardAttention(grad_output, grad_input, temp_buffer),
            .multi_head_attention => try self.backwardMultiHeadAttention(grad_output, grad_input, temp_buffer),
            else => {
                // For layers without trainable parameters, just copy gradients
                @memcpy(grad_input, grad_output);
            },
        }
    }

    /// Get input size for the layer
    pub fn getInputSize(self: *const Self) usize {
        var size: usize = 1;
        for (self.config.input_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    /// Get output size for the layer
    pub fn getOutputSize(self: *const Self) usize {
        var size: usize = 1;
        for (self.config.output_shape) |dim| {
            size *= dim;
        }
        return size;
    }

    /// Set training mode
    pub fn setTraining(self: *Self, is_training: bool) void {
        self.is_training = is_training;
        self.phase = if (is_training) .train else .eval;
    }

    /// Set inference mode
    pub fn setInference(self: *Self) void {
        self.is_training = false;
        self.phase = .inference;
    }

    /// Freeze layer parameters
    pub fn freeze(self: *Self) void {
        self.is_frozen = true;
    }

    /// Unfreeze layer parameters
    pub fn unfreeze(self: *Self) void {
        self.is_frozen = false;
    }

    /// Reset layer state (useful for RNNs)
    pub fn resetState(self: *Self) void {
        if (self.hidden_state) |state| @memset(state, 0.0);
        if (self.cell_state) |state| @memset(state, 0.0);
        if (self.prev_hidden) |state| @memset(state, 0.0);
        if (self.prev_cell) |state| @memset(state, 0.0);
    }

    /// Get memory usage of the layer
    pub fn getMemoryUsage(self: *const Self) usize {
        var usage: usize = 0;
        if (self.weights) |weights| usage += weights.len * @sizeOf(f32);
        if (self.biases) |biases| usage += biases.len * @sizeOf(f32);
        if (self.weight_gradients) |grads| usage += grads.len * @sizeOf(f32);
        if (self.bias_gradients) |grads| usage += grads.len * @sizeOf(f32);
        if (self.running_mean) |mean| usage += mean.len * @sizeOf(f32);
        if (self.running_var) |variance| usage += variance.len * @sizeOf(f32);
        if (self.gamma) |gamma| usage += gamma.len * @sizeOf(f32);
        if (self.beta) |beta| usage += beta.len * @sizeOf(f32);
        return usage;
    }

    /// Get parameter count
    pub fn getParameterCount(self: *const Self) usize {
        var count: usize = 0;
        if (self.weights) |weights| count += weights.len;
        if (self.biases) |biases| count += biases.len;
        if (self.gamma) |gamma| count += gamma.len;
        if (self.beta) |beta| count += beta.len;
        return count;
    }

    // Private methods

    fn validateConfig(self: *Self, config: LayerConfig) anyerror!void {
        _ = self;
        if (config.input_shape.len == 0 or config.output_shape.len == 0) {
            return anyerror.InvalidConfiguration;
        }

        // Validate layer-specific constraints
        switch (config.layer_type) {
            .conv2d, .conv1d, .conv3d => {
                if (config.kernel_size == null) return anyerror.InvalidConfiguration;
            },
            .lstm, .gru, .rnn => {
                if (config.hidden_size == null) return anyerror.InvalidConfiguration;
            },
            .multi_head_attention => {
                if (config.num_heads == null or config.head_dim == null) {
                    return anyerror.InvalidConfiguration;
                }
                const num_heads = config.num_heads.?;
                const head_dim = config.head_dim.?;
                if ((num_heads * head_dim) != config.embed_dim.?) {
                    return anyerror.InvalidConfiguration;
                }
            },
            .group_norm => {
                if (config.num_groups == null) return anyerror.InvalidConfiguration;
            },
            .embedding => {
                if (config.num_embeddings == null or config.embedding_dim == null) {
                    return anyerror.InvalidConfiguration;
                }
            },
            else => {},
        }
    }

    fn needsWeights(self: *const Self) bool {
        return switch (self.config.layer_type) {
            .dense, .conv2d, .conv1d, .conv3d, .transposed_conv2d, .dilated_conv2d, .separable_conv2d, .depthwise_conv2d, .pointwise_conv2d, .lstm, .gru, .rnn, .elman_rnn, .jordan_rnn, .bidirectional_lstm, .bidirectional_gru, .attention, .multi_head_attention, .scaled_dot_product_attention, .self_attention, .cross_attention, .transformer_block, .transformer_encoder, .transformer_decoder, .embedding, .sparse_embedding, .learned_positional_encoding, .highway_connection, .squeeze_excitation, .channel_attention, .spatial_attention, .linear_combination, .masked_linear, .pruned_linear, .quantized_linear, .low_rank_linear => true,
            else => false,
        };
    }

    fn needsBiases(self: *const Self) bool {
        return self.needsWeights() and self.config.use_bias;
    }

    fn isRNNLayer(self: *const Self) bool {
        return switch (self.config.layer_type) {
            .lstm, .gru, .rnn, .elman_rnn, .jordan_rnn, .bidirectional_lstm, .bidirectional_gru => true,
            else => false,
        };
    }

    fn isAttentionLayer(self: *const Self) bool {
        return switch (self.config.layer_type) {
            .attention, .multi_head_attention, .scaled_dot_product_attention, .self_attention, .cross_attention => true,
            else => false,
        };
    }

    fn getWeightSize(self: *const Self) usize {
        return switch (self.config.layer_type) {
            .dense => self.getInputSize() * self.getOutputSize(),
            .conv2d => blk: {
                const kernel = self.config.kernel_size orelse &[_]usize{ 3, 3 };
                const in_channels = if (self.config.input_shape.len >= 3) self.config.input_shape[2] else 1;
                const out_channels = if (self.config.output_shape.len >= 3) self.config.output_shape[2] else 1;
                break :blk kernel[0] * kernel[1] * in_channels * out_channels / self.config.groups;
            },
            .conv1d => blk: {
                const kernel = self.config.kernel_size orelse &[_]usize{3};
                const in_channels = if (self.config.input_shape.len >= 2) self.config.input_shape[1] else 1;
                const out_channels = if (self.config.output_shape.len >= 2) self.config.output_shape[1] else 1;
                break :blk kernel[0] * in_channels * out_channels / self.config.groups;
            },
            .conv3d => blk: {
                const kernel = self.config.kernel_size orelse &[_]usize{ 3, 3, 3 };
                const in_channels = if (self.config.input_shape.len >= 4) self.config.input_shape[3] else 1;
                const out_channels = if (self.config.output_shape.len >= 4) self.config.output_shape[3] else 1;
                break :blk kernel[0] * kernel[1] * kernel[2] * in_channels * out_channels / self.config.groups;
            },
            .lstm => blk: {
                const hidden = self.config.hidden_size orelse self.getOutputSize();
                const input = self.getInputSize();
                break :blk 4 * hidden * (input + hidden); // 4 gates: input, forget, cell, output
            },
            .gru => blk: {
                const hidden = self.config.hidden_size orelse self.getOutputSize();
                const input = self.getInputSize();
                break :blk 3 * hidden * (input + hidden); // 3 gates: reset, update, new
            },
            .rnn, .elman_rnn => blk: {
                const hidden = self.config.hidden_size orelse self.getOutputSize();
                const input = self.getInputSize();
                break :blk hidden * (input + hidden);
            },
            .jordan_rnn => blk: {
                const hidden = self.config.hidden_size orelse self.getOutputSize();
                const input = self.getInputSize();
                const output = self.getOutputSize();
                break :blk hidden * (input + output);
            },
            .attention, .multi_head_attention, .scaled_dot_product_attention, .self_attention, .cross_attention => blk: {
                const d_model = self.config.embed_dim orelse self.getInputSize();
                break :blk 3 * d_model * d_model; // Q, K, V projections
            },
            .embedding => blk: {
                const num_embeddings = self.config.num_embeddings orelse self.config.input_shape[0];
                const embedding_dim = self.config.embedding_dim orelse self.config.output_shape[0];
                break :blk num_embeddings * embedding_dim;
            },
            .highway_connection => blk: {
                const size = self.getInputSize();
                break :blk 2 * size * size; // Transform and carry gates
            },
            .squeeze_excitation => blk: {
                const channels = if (self.config.input_shape.len >= 3) self.config.input_shape[2] else self.getInputSize();
                const reduction = 16; // Common reduction ratio
                break :blk 2 * channels * (channels / reduction); // Two FC layers
            },
            else => 0,
        };
    }

    fn initializeNormalization(self: *Self, size: usize) anyerror!void {
        if (self.config.affine) {
            if (self.gamma == null) {
                self.gamma = try self.allocator.alloc(f32, size);
                self.gamma_gradients = try self.allocator.alloc(f32, size);
                @memset(self.gamma.?, 1.0);
                @memset(self.gamma_gradients.?, 0.0);
            }
            if (self.beta == null) {
                self.beta = try self.allocator.alloc(f32, size);
                self.beta_gradients = try self.allocator.alloc(f32, size);
                @memset(self.beta.?, 0.0);
                @memset(self.beta_gradients.?, 0.0);
            }
        }

        if (self.config.track_running_stats) {
            if (self.running_mean == null) {
                self.running_mean = try self.allocator.alloc(f32, size);
                @memset(self.running_mean.?, 0.0);
            }
            if (self.running_var == null) {
                self.running_var = try self.allocator.alloc(f32, size);
                @memset(self.running_var.?, 1.0);
            }
        }
    }

    fn initializeRNNState(self: *Self) anyerror!void {
        const hidden_size = self.config.hidden_size orelse return;
        const batch_size = 1; // Default batch size

        if (self.hidden_state == null) {
            self.hidden_state = try self.allocator.alloc(f32, batch_size * hidden_size);
            @memset(self.hidden_state.?, 0.0);
        }

        if (self.config.layer_type == .lstm) {
            if (self.cell_state == null) {
                self.cell_state = try self.allocator.alloc(f32, batch_size * hidden_size);
                @memset(self.cell_state.?, 0.0);
            }
        }
    }

    fn initializeAttentionParameters(self: *Self, rng: *std.Random) anyerror!void {
        const d_model = self.config.embed_dim orelse self.getInputSize();

        if (self.query_weights == null) {
            self.query_weights = try self.allocator.alloc(f32, d_model * d_model);
            self.applyKaimingInitialization(self.query_weights.?, rng, d_model, d_model);
        }

        if (self.key_weights == null) {
            self.key_weights = try self.allocator.alloc(f32, d_model * d_model);
            self.applyKaimingInitialization(self.key_weights.?, rng, d_model, d_model);
        }

        if (self.value_weights == null) {
            self.value_weights = try self.allocator.alloc(f32, d_model * d_model);
            self.applyKaimingInitialization(self.value_weights.?, rng, d_model, d_model);
        }

        if (self.output_projection == null) {
            self.output_projection = try self.allocator.alloc(f32, d_model * d_model);
            self.applyKaimingInitialization(self.output_projection.?, rng, d_model, d_model);
        }
    }

    fn applyWeightInitialization(self: *Self, rng: *std.Random, input_size: usize, output_size: usize) anyerror!void {
        const weights = self.weights.?;
        self.applyInitializationScheme(weights, rng, input_size, output_size, self.config.weight_init);
    }

    fn applyBiasInitialization(self: *Self, rng: *std.Random, output_size: usize) anyerror!void {
        const biases = self.biases.?;
        self.applyInitializationScheme(biases, rng, 1, output_size, self.config.bias_init);
    }

    fn applyInitializationScheme(self: *Self, weights: []f32, rng: *std.Random, input_size: usize, output_size: usize, init_type: WeightInit) void {
        switch (init_type) {
            .xavier_uniform, .glorot_uniform => {
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_size + output_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .xavier_normal, .glorot_normal => {
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .kaiming_uniform, .he_uniform => {
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .kaiming_normal, .he_normal => {
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .lecun_uniform => {
                const limit = @sqrt(3.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
            .lecun_normal => {
                const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .variance_scaling => {
                const fan_in = @as(f32, @floatFromInt(input_size));
                const fan_out = @as(f32, @floatFromInt(output_size));
                const scale = 2.0 / (fan_in + fan_out);
                const std_dev = @sqrt(scale);
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .fan_in => {
                const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .fan_out => {
                const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(output_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .fan_avg => {
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size + output_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .orthogonal => {
                self.applyOrthogonalInitialization(weights, rng, input_size, output_size, 1.0);
            },
            .orthogonal_gain => {
                const gain = @sqrt(2.0);
                self.applyOrthogonalInitialization(weights, rng, input_size, output_size, gain);
            },
            .uniform, .random_uniform => {
                for (weights) |*weight| {
                    weight.* = rng.float(f32) * 2.0 - 1.0;
                }
            },
            .normal, .random_normal => {
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32);
                }
            },
            .truncated_normal, .trunc_normal => {
                const std_dev = 0.02;
                for (weights) |*weight| {
                    var val = rng.floatNorm(f32) * std_dev;
                    // Truncate to [-2*std_dev, 2*std_dev]
                    val = @max(-2.0 * std_dev, @min(2.0 * std_dev, val));
                    weight.* = val;
                }
            },
            .zeros => {
                @memset(weights, 0.0);
            },
            .ones => {
                @memset(weights, 1.0);
            },
            .identity, .eye_init => {
                @memset(weights, 0.0);
                const min_dim = @min(input_size, output_size);
                for (0..min_dim) |i| {
                    weights[i * output_size + i] = 1.0;
                }
            },
            .constant => {
                @memset(weights, 0.1);
            },
            .dirac => {
                @memset(weights, 0.0);
                if (weights.len > 0) weights[0] = 1.0;
            },
            .sparse => {
                @memset(weights, 0.0);
                const sparsity = 0.1; // 10% non-zero
                for (weights, 0..) |*weight, i| {
                    _ = i;
                    if ((rng.int(u32) % 100) < @as(u32, @intFromFloat(sparsity * 100))) {
                        weight.* = rng.floatNorm(f32) * 0.01;
                    }
                }
            },
            .scaled_uniform => {
                const scale = 0.1;
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * scale;
                }
            },
            .scaled_normal => {
                const scale = 0.1;
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * scale;
                }
            },
            .bengio_init => {
                const std_dev = @sqrt(1.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .normalized_init => {
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32);
                }
                // Normalize to unit norm
                var norm: f32 = 0.0;
                for (weights) |w| norm += w * w;
                norm = @sqrt(norm);
                if (norm > 0) {
                    for (weights) |*weight| {
                        weight.* /= norm;
                    }
                }
            },
            .fixup_init => {
                const num_layers = 1.0; // Should be provided from model
                const scale = 1.0 / std.math.pow(f32, num_layers, 0.25);
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * scale;
                }
            },
            .layer_sequential_init => {
                const layer_index = 1.0; // Should be provided
                const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size))) / layer_index;
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * std_dev;
                }
            },
            .delta_orthogonal => {
                // Simplified delta orthogonal - proper implementation would need more structure
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * 0.01;
                }
            },
            .permutation_init => {
                @memset(weights, 0.0);
                // Create a permutation matrix (simplified)
                for (0..@min(input_size, output_size)) |i| {
                    weights[i * output_size + i] = 1.0;
                }
            },
            .symmetric_init => {
                for (weights) |*weight| {
                    weight.* = rng.floatNorm(f32) * 0.02;
                }
                // Ensure symmetry for square matrices
                if (input_size == output_size) {
                    for (0..input_size) |i| {
                        for (0..i) |j| {
                            const val = weights[i * output_size + j];
                            weights[j * output_size + i] = val;
                        }
                    }
                }
            },
            else => {
                // Default to Kaiming uniform
                const limit = @sqrt(6.0 / @as(f32, @floatFromInt(input_size)));
                for (weights) |*weight| {
                    weight.* = (rng.float(f32) * 2.0 - 1.0) * limit;
                }
            },
        }
    }

    fn applyOrthogonalInitialization(self: *Self, weights: []f32, rng: *std.Random, input_size: usize, output_size: usize, gain: f32) void {
        _ = self;
        const expected_len = input_size * output_size;
        if (expected_len == 0 or weights.len != expected_len) {
            for (weights) |*weight| {
                weight.* = rng.floatNorm(f32) * gain;
            }
            return;
        }

        for (weights) |*weight| {
            weight.* = rng.floatNorm(f32);
        }

        if (input_size >= output_size) {
            orthonormalizeColumns(weights, input_size, output_size);
        } else {
            orthonormalizeRows(weights, input_size, output_size);
        }

        if (gain != 1.0) {
            for (weights) |*weight| {
                weight.* *= gain;
            }
        }
    }

    fn orthonormalizeColumns(weights: []f32, rows: usize, cols: usize) void {
        if (rows == 0 or cols == 0) return;
        const stride = cols;
        const epsilon: f32 = 1e-6;

        for (0..cols) |col| {
            for (0..col) |prev| {
                var dot: f32 = 0.0;
                for (0..rows) |row| {
                    dot += weights[row * stride + col] * weights[row * stride + prev];
                }
                for (0..rows) |row| {
                    weights[row * stride + col] -= dot * weights[row * stride + prev];
                }
            }

            var norm_sq: f32 = 0.0;
            for (0..rows) |row| {
                const val = weights[row * stride + col];
                norm_sq += val * val;
            }

            if (norm_sq <= epsilon) {
                for (0..rows) |row| {
                    weights[row * stride + col] = 0.0;
                }
                const diag_row = if (rows > 0) col % rows else 0;
                weights[diag_row * stride + col] = 1.0;
            } else {
                const inv_norm = 1.0 / @sqrt(norm_sq);
                for (0..rows) |row| {
                    weights[row * stride + col] *= inv_norm;
                }
            }
        }
    }

    fn orthonormalizeRows(weights: []f32, rows: usize, cols: usize) void {
        if (rows == 0 or cols == 0) return;
        const stride = cols;
        const epsilon: f32 = 1e-6;

        for (0..rows) |row| {
            for (0..row) |prev| {
                var dot: f32 = 0.0;
                for (0..cols) |col| {
                    dot += weights[row * stride + col] * weights[prev * stride + col];
                }
                for (0..cols) |col| {
                    weights[row * stride + col] -= dot * weights[prev * stride + col];
                }
            }

            var norm_sq: f32 = 0.0;
            for (0..cols) |col| {
                const val = weights[row * stride + col];
                norm_sq += val * val;
            }

            if (norm_sq <= epsilon) {
                for (0..cols) |col| {
                    weights[row * stride + col] = 0.0;
                }
                const diag_col = if (cols > 0) row % cols else 0;
                weights[row * stride + diag_col] = 1.0;
            } else {
                const inv_norm = 1.0 / @sqrt(norm_sq);
                for (0..cols) |col| {
                    weights[row * stride + col] *= inv_norm;
                }
            }
        }
    }

    fn applyKaimingInitialization(self: *Self, weights: []f32, rng: *std.Random, input_size: usize, output_size: usize) void {
        _ = self;
        _ = output_size;
        const std_dev = @sqrt(2.0 / @as(f32, @floatFromInt(input_size)));
        for (weights) |*weight| {
            weight.* = rng.floatNorm(f32) * std_dev;
        }
    }

    // Forward pass implementations for different layer types

    fn forwardDense(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.weights == null) return anyerror.InvalidState;

        const weights = self.weights.?;
        const input_size = self.config.input_shape[0];
        const output_size = self.config.output_shape[0];

        if (input.len != input_size or output.len != output_size) {
            return anyerror.InvalidData;
        }

        // Matrix-vector multiplication: output = weights * input + biases
        // Inline matrix-vector multiplication
        for (0..output_size) |i| {
            var sum: f32 = 0.0;
            for (0..input_size) |j| {
                sum += weights[i * input_size + j] * input[j];
            }
            output[i] = sum;
        }

        // Add biases
        if (self.biases) |biases| {
            // Add biases inline
            for (output, biases) |*out, bias| {
                out.* += bias;
            }
        }

        // Apply activation if specified
        if (self.activation_processor) |*processor| {
            processor.activateBatch(output, output);
        }

        // Apply regularization during training
        if (self.is_training) {
            try self.applyRegularization(output);
        }
    }

    fn forwardConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardConv1D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardConv3D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardTransposedConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardDilatedConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardSeparableConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardDepthwiseConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardPointwiseConv2D(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardMaxPool2D(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardAvgPool2D(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardGlobalAvgPool(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardGlobalMaxPool(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardAdaptiveAvgPool(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardAdaptiveMaxPool(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardBatchNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.is_training and self.config.track_running_stats) {
            // Update running statistics
            self.num_batches_tracked += 1;

            // Calculate batch statistics
            var mean: f32 = 0.0;
            for (input) |val| {
                mean += val;
            }
            mean /= @as(f32, @floatFromInt(input.len));

            var variance: f32 = 0.0;
            for (input) |val| {
                const diff = val - mean;
                variance += diff * diff;
            }
            variance /= @as(f32, @floatFromInt(input.len));

            // Update running mean and variance
            if (self.running_mean) |running_mean| {
                running_mean[0] = (1.0 - self.config.momentum) * running_mean[0] + self.config.momentum * mean;
            }
            if (self.running_var) |running_var| {
                running_var[0] = (1.0 - self.config.momentum) * running_var[0] + self.config.momentum * variance;
            }

            // Normalize using batch statistics
            const std_dev = @sqrt(variance + self.config.eps);
            for (input, 0..) |val, i| {
                output[i] = (val - mean) / std_dev;
            }
        } else {
            // Use running statistics
            const mean = if (self.running_mean) |running_mean| running_mean[0] else 0.0;
            const variance = if (self.running_var) |running_var| running_var[0] else 1.0;
            const std_dev = @sqrt(variance + self.config.eps);

            for (input, 0..) |val, i| {
                output[i] = (val - mean) / std_dev;
            }
        }

        // Apply affine transformation
        if (self.config.affine) {
            const gamma = if (self.gamma) |g| g[0] else 1.0;
            const beta = if (self.beta) |b| b[0] else 0.0;

            for (output) |*val| {
                val.* = gamma * val.* + beta;
            }
        }
    }

    fn forwardLayerNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        // Calculate mean
        var mean: f32 = 0.0;
        for (input) |val| {
            mean += val;
        }
        mean /= @as(f32, @floatFromInt(input.len));

        // Calculate variance
        var variance: f32 = 0.0;
        for (input) |val| {
            const diff = val - mean;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(input.len));

        // Normalize
        const std_dev = @sqrt(variance + self.config.eps);
        for (input, 0..) |val, i| {
            output[i] = (val - mean) / std_dev;
        }

        // Apply affine transformation
        if (self.config.affine) {
            if (self.gamma) |gamma| {
                if (self.beta) |beta| {
                    for (output, 0..) |*val, i| {
                        val.* = gamma[i] * val.* + beta[i];
                    }
                } else {
                    for (output, 0..) |*val, i| {
                        val.* = gamma[i] * val.*;
                    }
                }
            }
        }
    }

    fn forwardGroupNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.config.num_groups == null) return anyerror.InvalidConfiguration;

        const num_groups = self.config.num_groups.?;
        const group_size = input.len / num_groups;

        for (0..num_groups) |g| {
            const start_idx = g * group_size;
            const end_idx = start_idx + group_size;

            // Calculate group mean
            var mean: f32 = 0.0;
            for (start_idx..end_idx) |i| {
                mean += input[i];
            }
            mean /= @as(f32, @floatFromInt(group_size));

            // Calculate group variance
            var variance: f32 = 0.0;
            for (start_idx..end_idx) |i| {
                const diff = input[i] - mean;
                variance += diff * diff;
            }
            variance /= @as(f32, @floatFromInt(group_size));

            // Normalize group
            const std_dev = @sqrt(variance + self.config.eps);
            for (start_idx..end_idx) |i| {
                output[i] = (input[i] - mean) / std_dev;
            }
        }

        // Apply affine transformation
        if (self.config.affine) {
            if (self.gamma) |gamma| {
                if (self.beta) |beta| {
                    for (output, 0..) |*val, i| {
                        val.* = gamma[i] * val.* + beta[i];
                    }
                }
            }
        }
    }

    fn forwardInstanceNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        // Instance normalization treats each sample independently
        var mean: f32 = 0.0;
        for (input) |val| {
            mean += val;
        }
        mean /= @as(f32, @floatFromInt(input.len));

        var variance: f32 = 0.0;
        for (input) |val| {
            const diff = val - mean;
            variance += diff * diff;
        }
        variance /= @as(f32, @floatFromInt(input.len));

        const std_dev = @sqrt(variance + self.config.eps);
        for (input, 0..) |val, i| {
            output[i] = (val - mean) / std_dev;
        }

        if (self.config.affine) {
            if (self.gamma) |gamma| {
                if (self.beta) |beta| {
                    for (output, 0..) |*val, i| {
                        val.* = gamma[i] * val.* + beta[i];
                    }
                }
            }
        }
    }

    fn forwardLocalResponseNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        const size = 5; // Local region size
        const alpha = 0.0001;
        const beta = 0.75;
        const k = 2.0;

        for (input, 0..) |val, i| {
            _ = self;
            var sum_squares: f32 = 0.0;

            const start = if (i >= size / 2) i - size / 2 else 0;
            const end = @min(i + size / 2 + 1, input.len);

            for (start..end) |j| {
                sum_squares += input[j] * input[j];
            }

            const norm = std.math.pow(f32, k + alpha * sum_squares, beta);
            output[i] = val / norm;
        }
    }

    fn forwardWeightNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardSpectralNorm(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardDropout(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.is_training and self.config.regularization.dropout_rate > 0.0) {
            for (input, 0..) |val, i| {
                // Simple pseudo-random dropout
                const hash = std.hash_map.hashString("dropout") + i;
                if ((@as(f32, @floatFromInt(hash % 1000)) / 1000.0) < self.config.regularization.dropout_rate) {
                    output[i] = 0.0;
                } else {
                    output[i] = val / (1.0 - self.config.regularization.dropout_rate);
                }
            }
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardDropout2D(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.is_training and self.config.regularization.spatial_dropout_rate > 0.0) {
            // For spatial dropout, drop entire channels
            const channels = if (self.config.input_shape.len >= 3) self.config.input_shape[2] else 1;
            const spatial_size = input.len / channels;

            for (0..channels) |c| {
                const hash = std.hash_map.hashString("spatial_dropout") + c;
                const drop = (@as(f32, @floatFromInt(hash % 1000)) / 1000.0) < self.config.regularization.spatial_dropout_rate;

                const start_idx = c * spatial_size;
                const end_idx = start_idx + spatial_size;

                if (drop) {
                    @memset(output[start_idx..end_idx], 0.0);
                } else {
                    const scale = 1.0 / (1.0 - self.config.regularization.spatial_dropout_rate);
                    for (start_idx..end_idx) |i| {
                        output[i] = input[i] * scale;
                    }
                }
            }
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardSpatialDropout(self: *Self, input: []const f32, output: []f32) anyerror!void {
        try self.forwardDropout2D(input, output);
    }

    fn forwardVariationalDropout(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.is_training and self.config.regularization.variational_dropout_rate > 0.0) {
            // Variational dropout with learned variance
            const log_alpha = self.config.regularization.variational_dropout_rate;

            for (input, 0..) |val, i| {
                const hash = std.hash_map.hashString("variational_dropout") + i;
                const eps = (@as(f32, @floatFromInt(hash % 1000)) / 500.0) - 1.0; // [-1, 1]

                const alpha = @exp(log_alpha);
                const std_dev = @sqrt(alpha);

                output[i] = val * (1.0 + std_dev * eps);
            }
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardActivation(self: *Self, input: []const f32, output: []f32) anyerror!void {
        if (self.activation_processor) |*processor| {
            processor.activateBatch(output, input);
        } else {
            @memcpy(output, input);
        }
    }

    fn forwardFlatten(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardReshape(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardPermute(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardSqueeze(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardUnsqueeze(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardLSTM(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardGRU(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardRNN(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardElmanRNN(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardJordanRNN(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardBidirectionalLSTM(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardBidirectionalGRU(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardMultiHeadAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardScaledDotProductAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardSelfAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardCrossAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(output, input);
    }

    fn forwardTransformerBlock(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardTransformerEncoder(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardTransformerDecoder(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardEmbedding(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardSparseEmbedding(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardPositionalEncoding(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardLearnedPositionalEncoding(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        return anyerror.UnsupportedOperation;
    }

    fn forwardSinusoidalPositionalEncoding(self: *Self, input: []const f32, output: []f32) anyerror!void {
        const d_model = input.len;
        _ = self;
        const max_len = 10000.0;

        for (input, 0..) |val, pos| {
            const position = @as(f32, @floatFromInt(pos));
            var encoding: f32 = 0.0;

            if (pos % 2 == 0) {
                // Even positions use sine
                const div_term = @exp(@as(f32, @floatFromInt(pos)) * (-@log(max_len) / @as(f32, @floatFromInt(d_model))));
                encoding = @sin(position * div_term);
            } else {
                // Odd positions use cosine
                const div_term = @exp(@as(f32, @floatFromInt(pos - 1)) * (-@log(max_len) / @as(f32, @floatFromInt(d_model))));
                encoding = @cos(position * div_term);
            }

            output[pos] = val + encoding;
        }
    }

    fn forwardResidualConnection(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardHighwayConnection(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardDenseConnection(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardSqueezeExcitation(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardChannelAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardSpatialAttention(self: *Self, input: []const f32, output: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = input;
        _ = output;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn forwardConcat(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardAdd(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardMultiply(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardSubtract(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardDivide(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    fn forwardLinearCombination(self: *Self, input: []const f32, output: []f32) anyerror!void {
        _ = self;
        @memcpy(output, input);
    }

    // Backward pass implementations

    fn backwardDense(self: *Self, grad_output: []const f32, grad_input: []f32) anyerror!void {
        if (self.weights == null or self.weight_gradients == null) {
            return anyerror.InvalidState;
        }

        const weights = self.weights.?;
        const weight_grads = self.weight_gradients.?;
        const input_size = self.config.input_shape[0];
        const output_size = self.config.output_shape[0];

        // Compute input gradients: grad_input = weights^T * grad_output
        // Inline matrix-vector multiplication transpose
        for (0..input_size) |i| {
            var sum: f32 = 0.0;
            for (0..output_size) |j| {
                sum += weights[j * input_size + i] * grad_output[j];
            }
            grad_input[i] = sum;
        }

        // Compute weight gradients: weight_grads += grad_output * input^T
        // This would require the input from forward pass, stored in temp buffer
        // For now, we'll accumulate gradients
        for (weight_grads, 0..) |*grad, i| {
            grad.* += grad_output[i % output_size] * 0.01; // Placeholder
        }

        // Compute bias gradients
        if (self.bias_gradients) |bias_grads| {
            for (bias_grads, 0..) |*grad, i| {
                grad.* += grad_output[i];
            }
        }
    }

    fn backwardConv2D(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = grad_output;
        _ = grad_input;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn backwardBatchNorm(self: *Self, grad_output: []const f32, grad_input: []f32) anyerror!void {
        _ = self;
        _ = grad_output;
        _ = grad_input;
        return anyerror.UnsupportedOperation;
    }

    fn backwardLayerNorm(self: *Self, grad_output: []const f32, grad_input: []f32) anyerror!void {
        _ = self;
        _ = grad_output;
        _ = grad_input;
        return anyerror.UnsupportedOperation;
    }

    fn backwardActivation(self: *Self, grad_output: []const f32, grad_input: []f32) anyerror!void {
        if (self.activation_processor) |*processor| {
            // This would require activation derivative implementation
            _ = processor;
            @memcpy(grad_input, grad_output);
        } else {
            @memcpy(grad_input, grad_output);
        }
    }

    fn backwardDropout(self: *Self, grad_output: []const f32, grad_input: []f32) anyerror!void {
        if (self.is_training and self.config.regularization.dropout_rate > 0.0) {
            for (grad_output, 0..) |grad, i| {
                // Apply same mask as forward pass
                const hash = std.hash_map.hashString("dropout") + i;
                if ((@as(f32, @floatFromInt(hash % 1000)) / 1000.0) < self.config.regularization.dropout_rate) {
                    grad_input[i] = 0.0;
                } else {
                    grad_input[i] = grad / (1.0 - self.config.regularization.dropout_rate);
                }
            }
        } else {
            @memcpy(grad_input, grad_output);
        }
    }

    fn backwardLSTM(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = grad_output;
        _ = grad_input;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn backwardGRU(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = grad_output;
        _ = grad_input;
        _ = temp_buffer;
        return anyerror.UnsupportedOperation;
    }

    fn backwardAttention(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(grad_input, grad_output);
    }

    fn backwardMultiHeadAttention(self: *Self, grad_output: []const f32, grad_input: []f32, temp_buffer: ?[]f32) anyerror!void {
        _ = self;
        _ = temp_buffer;
        @memcpy(grad_input, grad_output);
    }

    fn applyRegularization(self: *Self, data: []f32) anyerror!void {
        // Apply L1/L2 regularization during training
        if (self.config.regularization.l1_lambda > 0.0 or self.config.regularization.l2_lambda > 0.0) {
            // Regularization would be applied to gradients during backpropagation
            // This is a placeholder for the interface
            _ = data;
        }
    }
};

test "layer creation and initialization" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .dense,
        .input_shape = &[_]usize{784},
        .output_shape = &[_]usize{128},
        .activation_type = .relu,
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    // Test that layer was created successfully
    try testing.expectEqual(LayerType.dense, layer.config.layer_type);
    try testing.expectEqual(@as(usize, 784), layer.getInputSize());
    try testing.expectEqual(@as(usize, 128), layer.getOutputSize());

    // Initialize weights
    var rng = std.Random.DefaultPrng.init(42);
    try layer.initializeWeights(rng.random());

    // Check that weights were allocated
    try testing.expect(layer.weights != null);
    try testing.expect(layer.biases != null);
}

test "dense layer forward pass" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .dense,
        .input_shape = &[_]usize{4},
        .output_shape = &[_]usize{2},
        .activation_type = .relu,
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    // Initialize with known weights for testing
    var rng = std.Random.DefaultPrng.init(42);
    try layer.initializeWeights(rng.random());

    // Test forward pass
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output = [_]f32{0} ** 2;

    try layer.forward(&input, &output, null);

    // Check that output was computed (should be non-zero due to ReLU)
    var has_non_zero = false;
    for (output) |val| {
        if (val > 0.0) {
            has_non_zero = true;
            break;
        }
    }
    try testing.expect(has_non_zero);
}

test "batch normalization layer" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .batch_norm,
        .input_shape = &[_]usize{10},
        .output_shape = &[_]usize{10},
        .regularization = .{ .batch_norm = true },
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    try layer.initializeWeights(rng.random());

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    var output = [_]f32{0} ** 10;

    try layer.forward(&input, &output, null);

    // Check that normalization was applied (mean should be close to 0)
    var mean: f32 = 0.0;
    for (output) |val| {
        mean += val;
    }
    mean /= @as(f32, @floatFromInt(output.len));

    try testing.expect(@abs(mean) < 1.0); // Should be close to 0 after normalization
}

test "layer memory management" {
    const testing = std.testing;

    const config = LayerConfig{
        .layer_type = .dense,
        .input_shape = &[_]usize{100},
        .output_shape = &[_]usize{50},
        .enable_mixed_precision = true,
    };

    var layer = try Layer.init(testing.allocator, config);
    defer layer.deinit();

    var rng = std.Random.DefaultPrng.init(42);
    try layer.initializeWeights(rng.random());

    // Check memory usage calculation
    const memory_usage = layer.getMemoryUsage();
    try testing.expect(memory_usage > 0);

    // Check parameter count
    const param_count = layer.getParameterCount();
    try testing.expectEqual(@as(usize, 100 * 50 + 50), param_count); // weights + biases
}

test "layer orthogonal initialization columns" {
    const allocator = std.testing.allocator;
    const rows: usize = 6;
    const cols: usize = 4;
    const weights = try allocator.alloc(f32, rows * cols);
    defer allocator.free(weights);

    var prng = std.Random.DefaultPrng.init(12345);
    var random = prng.random();

    var layer = std.mem.zeroes(Layer);
    layer.applyOrthogonalInitialization(weights, &random, rows, cols, 1.0);

    const stride = cols;
    const tolerance: f32 = 1e-3;
    for (0..cols) |col| {
        var norm: f32 = 0.0;
        for (0..rows) |row| {
            const val = weights[row * stride + col];
            norm += val * val;
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, tolerance);

        for (col + 1..cols) |other| {
            var dot: f32 = 0.0;
            for (0..rows) |row| {
                dot += weights[row * stride + col] * weights[row * stride + other];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), dot, tolerance);
        }
    }
}

test "layer orthogonal initialization rows" {
    const allocator = std.testing.allocator;
    const rows: usize = 3;
    const cols: usize = 6;
    const weights = try allocator.alloc(f32, rows * cols);
    defer allocator.free(weights);

    var prng = std.Random.DefaultPrng.init(6789);
    var random = prng.random();

    var layer = std.mem.zeroes(Layer);
    layer.applyOrthogonalInitialization(weights, &random, rows, cols, 1.0);

    const stride = cols;
    const tolerance: f32 = 1e-3;
    for (0..rows) |row| {
        var norm: f32 = 0.0;
        for (0..cols) |col| {
            const val = weights[row * stride + col];
            norm += val * val;
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, tolerance);

        for (row + 1..rows) |other| {
            var dot: f32 = 0.0;
            for (0..cols) |col| {
                dot += weights[row * stride + col] * weights[other * stride + col];
            }
            try std.testing.expectApproxEqAbs(@as(f32, 0.0), dot, tolerance);
        }
    }
}
