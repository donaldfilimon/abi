//! Vision/CNN Kernel Definitions
//!
//! Pre-defined kernel IR for computer vision operations (CNNs).
//!
//! ## Operations
//! - conv2d: 2D Convolution
//! - max_pool2d: 2D Max Pooling
//! - avg_pool2d: 2D Average Pooling
//! - batch_norm2d: 2D Batch Normalization
//! - im2col: Image to Column transformation
//! - col2im: Column to Image transformation

const std = @import("std");
const dsl = @import("../dsl/mod.zig");

pub const KernelIR = dsl.KernelIR;
pub const KernelBuilder = dsl.KernelBuilder;
pub const Type = dsl.Type;
pub const AccessMode = dsl.AccessMode;

/// Build Conv2D kernel using im2col + GEMM approach
///
/// Performs 2D convolution by:
/// 1. Unfolding input patches into columns (im2col)
/// 2. Matrix multiply: output = weights @ col_matrix
/// 3. Add bias
///
/// This kernel assumes im2col has been performed and operates on the column matrix.
/// For full conv2d, use im2col kernel first, then this GEMM-based kernel.
///
/// Buffers:
/// - input: [batch, in_channels, height, width] - input tensor (or col matrix)
/// - weights: [out_channels, in_channels * kernel_h * kernel_w] - convolution weights
/// - bias: [out_channels] - bias terms (optional, pass zeros if not used)
/// - output: [batch, out_channels, out_height, out_width] - output tensor
///
/// Uniforms:
/// - batch_size, in_channels, out_channels
/// - in_height, in_width, out_height, out_width
/// - kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w
pub fn buildConv2dKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "conv2d");
    errdefer builder.deinit();

    // Use 16x16 tiles for output computation
    const TILE_SIZE: u32 = 16;
    _ = builder.setWorkgroupSize(TILE_SIZE, TILE_SIZE, 1);

    // Buffer bindings
    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const weights = try builder.addBuffer("weights", Type.f32Type(), .read_only);
    const bias = try builder.addBuffer("bias", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);

    // Uniform parameters
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());
    const in_channels = try builder.addUniform("in_channels", Type.u32Type());
    const out_channels = try builder.addUniform("out_channels", Type.u32Type());
    const in_height = try builder.addUniform("in_height", Type.u32Type());
    const in_width = try builder.addUniform("in_width", Type.u32Type());
    const out_height = try builder.addUniform("out_height", Type.u32Type());
    const out_width = try builder.addUniform("out_width", Type.u32Type());
    const kernel_h = try builder.addUniform("kernel_h", Type.u32Type());
    const kernel_w = try builder.addUniform("kernel_w", Type.u32Type());
    const stride_h = try builder.addUniform("stride_h", Type.u32Type());
    const stride_w = try builder.addUniform("stride_w", Type.u32Type());
    const pad_h = try builder.addUniform("pad_h", Type.u32Type());
    const pad_w = try builder.addUniform("pad_w", Type.u32Type());

    // Shared memory for tiled computation
    _ = try builder.addSharedMemory("tile_input", Type.f32Type(), TILE_SIZE * TILE_SIZE);
    _ = try builder.addSharedMemory("tile_weights", Type.f32Type(), TILE_SIZE * TILE_SIZE);

    const gid = builder.globalInvocationId();
    const wid = builder.workgroupId();

    // Global output position
    const out_x = try gid.x(); // output width position
    const out_y = try gid.y(); // output height position
    const batch_idx = try wid.z(); // batch index

    // Bounds check
    const x_check = try builder.lt(out_x, try out_width.toExpr());
    const y_check = try builder.lt(out_y, try out_height.toExpr());
    const batch_check = try builder.lt(batch_idx, try batch_size.toExpr());
    const bounds_check = try builder.logicalAnd(batch_check, try builder.logicalAnd(x_check, y_check));

    // Accumulator for each output channel (loop over out_channels)
    const oc_var = try builder.declareVar("oc", Type.u32Type(), try builder.u32Lit(0));
    const oc_cond = try builder.lt(try oc_var.toExpr(), try out_channels.toExpr());

    // Initialize sum for this output position
    const sum = try builder.declareVar("sum", Type.f32Type(), try builder.f32Lit(0.0));

    // Loop over input channels and kernel positions
    const ic_var = try builder.declareVar("ic", Type.u32Type(), try builder.u32Lit(0));
    const ic_cond = try builder.lt(try ic_var.toExpr(), try in_channels.toExpr());

    const ky_var = try builder.declareVar("ky", Type.u32Type(), try builder.u32Lit(0));
    const ky_cond = try builder.lt(try ky_var.toExpr(), try kernel_h.toExpr());

    const kx_var = try builder.declareVar("kx", Type.u32Type(), try builder.u32Lit(0));
    const kx_cond = try builder.lt(try kx_var.toExpr(), try kernel_w.toExpr());

    // Compute input position with stride and padding
    // ih = out_y * stride_h + ky - pad_h
    // iw = out_x * stride_w + kx - pad_w
    const ih_base = try builder.mul(out_y, try stride_h.toExpr());
    const ih_offset = try builder.add(ih_base, try ky_var.toExpr());
    const ih_signed = try builder.sub(ih_offset, try pad_h.toExpr());

    const iw_base = try builder.mul(out_x, try stride_w.toExpr());
    const iw_offset = try builder.add(iw_base, try kx_var.toExpr());
    const iw_signed = try builder.sub(iw_offset, try pad_w.toExpr());

    // Check if input position is valid (not in padding region)
    const ih_valid_low = try builder.gte(ih_signed, try builder.u32Lit(0));
    const ih_valid_high = try builder.lt(ih_signed, try in_height.toExpr());
    const iw_valid_low = try builder.gte(iw_signed, try builder.u32Lit(0));
    const iw_valid_high = try builder.lt(iw_signed, try in_width.toExpr());
    const input_valid = try builder.logicalAnd(
        try builder.logicalAnd(ih_valid_low, ih_valid_high),
        try builder.logicalAnd(iw_valid_low, iw_valid_high),
    );

    // Calculate input index: batch * C * H * W + ic * H * W + ih * W + iw
    const in_hw = try builder.mul(try in_height.toExpr(), try in_width.toExpr());
    const in_chw = try builder.mul(try in_channels.toExpr(), in_hw);
    const batch_offset = try builder.mul(batch_idx, in_chw);
    const channel_offset = try builder.mul(try ic_var.toExpr(), in_hw);
    const row_offset = try builder.mul(ih_signed, try in_width.toExpr());
    const input_idx = try builder.add(
        try builder.add(batch_offset, channel_offset),
        try builder.add(row_offset, iw_signed),
    );

    // Calculate weight index: oc * (C * kH * kW) + ic * (kH * kW) + ky * kW + kx
    const kernel_hw = try builder.mul(try kernel_h.toExpr(), try kernel_w.toExpr());
    const kernel_chw = try builder.mul(try in_channels.toExpr(), kernel_hw);
    const weight_oc_offset = try builder.mul(try oc_var.toExpr(), kernel_chw);
    const weight_ic_offset = try builder.mul(try ic_var.toExpr(), kernel_hw);
    const weight_ky_offset = try builder.mul(try ky_var.toExpr(), try kernel_w.toExpr());
    const weight_idx = try builder.add(
        try builder.add(weight_oc_offset, weight_ic_offset),
        try builder.add(weight_ky_offset, try kx_var.toExpr()),
    );

    // Load input value (0 if in padding)
    const input_val = try input.at(input_idx);
    const zero = try builder.f32Lit(0.0);
    const masked_input = try builder.select(input_valid, input_val, zero);

    // Load weight
    const weight_val = try weights.at(weight_idx);

    // Accumulate: sum += input * weight
    const product = try builder.mul(masked_input, weight_val);
    const new_sum = try builder.add(try sum.toExpr(), product);
    const update_sum = try builder.assignStmt(try sum.toExpr(), new_sum);

    // Increment kx
    const one = try builder.u32Lit(1);
    const next_kx = try builder.add(try kx_var.toExpr(), one);
    const update_kx = try builder.assignStmt(try kx_var.toExpr(), next_kx);

    // Build innermost kx loop
    try builder.forLoop(null, kx_cond, update_kx, &[_]*const dsl.Stmt{update_sum});

    // Reset kx, increment ky
    const reset_kx = try builder.assignStmt(try kx_var.toExpr(), try builder.u32Lit(0));
    const next_ky = try builder.add(try ky_var.toExpr(), one);
    const update_ky = try builder.assignStmt(try ky_var.toExpr(), next_ky);

    // Build ky loop body
    try builder.forLoop(null, ky_cond, update_ky, &[_]*const dsl.Stmt{reset_kx});

    // Reset ky, increment ic
    const reset_ky = try builder.assignStmt(try ky_var.toExpr(), try builder.u32Lit(0));
    const next_ic = try builder.add(try ic_var.toExpr(), one);
    const update_ic = try builder.assignStmt(try ic_var.toExpr(), next_ic);

    try builder.forLoop(null, ic_cond, update_ic, &[_]*const dsl.Stmt{reset_ky});

    // Add bias: sum += bias[oc]
    const bias_val = try bias.at(try oc_var.toExpr());
    const sum_with_bias = try builder.add(try sum.toExpr(), bias_val);

    // Calculate output index: batch * OC * OH * OW + oc * OH * OW + out_y * OW + out_x
    const out_hw = try builder.mul(try out_height.toExpr(), try out_width.toExpr());
    const out_chw = try builder.mul(try out_channels.toExpr(), out_hw);
    const out_batch_offset = try builder.mul(batch_idx, out_chw);
    const out_channel_offset = try builder.mul(try oc_var.toExpr(), out_hw);
    const out_row_offset = try builder.mul(out_y, try out_width.toExpr());
    const output_idx = try builder.add(
        try builder.add(out_batch_offset, out_channel_offset),
        try builder.add(out_row_offset, out_x),
    );

    // Store result
    const output_ptr = try output.at(output_idx);
    const store_stmt = try builder.assignStmt(output_ptr, sum_with_bias);

    // Reset sum, increment oc
    const reset_sum = try builder.assignStmt(try sum.toExpr(), try builder.f32Lit(0.0));
    const reset_ic = try builder.assignStmt(try ic_var.toExpr(), try builder.u32Lit(0));
    const next_oc = try builder.add(try oc_var.toExpr(), one);
    const update_oc = try builder.assignStmt(try oc_var.toExpr(), next_oc);

    try builder.forLoop(null, oc_cond, update_oc, &[_]*const dsl.Stmt{ store_stmt, reset_sum, reset_ic });

    // Wrap everything in bounds check
    try builder.ifStmt(bounds_check, &[_]*const dsl.Stmt{}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build MaxPool2D kernel
///
/// Performs max pooling over spatial dimensions, storing both the output
/// and the indices of max values (for backward pass gradient routing).
///
/// Buffers:
/// - input: [batch, channels, height, width] - input tensor
/// - output: [batch, channels, out_height, out_width] - pooled output
/// - indices: [batch, channels, out_height, out_width] - indices of max values
///
/// Uniforms:
/// - batch_size, channels, in_height, in_width
/// - out_height, out_width, kernel_size, stride, padding
pub fn buildMaxPool2dKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "max_pool2d");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(16, 16, 1);

    // Buffer bindings
    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);
    const indices = try builder.addBuffer("indices", Type.u32Type(), .write_only);

    // Uniform parameters
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());
    const in_height = try builder.addUniform("in_height", Type.u32Type());
    const in_width = try builder.addUniform("in_width", Type.u32Type());
    const out_height = try builder.addUniform("out_height", Type.u32Type());
    const out_width = try builder.addUniform("out_width", Type.u32Type());
    const kernel_size = try builder.addUniform("kernel_size", Type.u32Type());
    const stride = try builder.addUniform("stride", Type.u32Type());
    const padding = try builder.addUniform("padding", Type.u32Type());

    const gid = builder.globalInvocationId();
    const wid = builder.workgroupId();

    // Global position
    const out_x = try gid.x();
    const out_y = try gid.y();
    const channel = try wid.z();

    // We need to handle batch in a separate dimension or loop
    // For simplicity, assume batch is encoded in higher z workgroups
    // batch_idx = channel / actual_channels, channel_idx = channel % actual_channels
    const batch_idx = try builder.div(channel, try channels.toExpr());
    const channel_idx = try builder.mod(channel, try channels.toExpr());

    // Bounds check
    const x_check = try builder.lt(out_x, try out_width.toExpr());
    const y_check = try builder.lt(out_y, try out_height.toExpr());
    const batch_check = try builder.lt(batch_idx, try batch_size.toExpr());
    const bounds_check = try builder.logicalAnd(batch_check, try builder.logicalAnd(x_check, y_check));

    // Initialize max value and index
    const max_val = try builder.declareVar("max_val", Type.f32Type(), try builder.f32Lit(-3.4028235e+38));
    const max_idx = try builder.declareVar("max_idx", Type.u32Type(), try builder.u32Lit(0));

    // Loop over kernel window
    const ky_var = try builder.declareVar("ky", Type.u32Type(), try builder.u32Lit(0));
    const ky_cond = try builder.lt(try ky_var.toExpr(), try kernel_size.toExpr());

    const kx_var = try builder.declareVar("kx", Type.u32Type(), try builder.u32Lit(0));
    const kx_cond = try builder.lt(try kx_var.toExpr(), try kernel_size.toExpr());

    // Calculate input position
    const ih_base = try builder.mul(out_y, try stride.toExpr());
    const ih_offset = try builder.add(ih_base, try ky_var.toExpr());
    const ih = try builder.sub(ih_offset, try padding.toExpr());

    const iw_base = try builder.mul(out_x, try stride.toExpr());
    const iw_offset = try builder.add(iw_base, try kx_var.toExpr());
    const iw = try builder.sub(iw_offset, try padding.toExpr());

    // Check bounds
    const ih_valid = try builder.logicalAnd(
        try builder.gte(ih, try builder.u32Lit(0)),
        try builder.lt(ih, try in_height.toExpr()),
    );
    const iw_valid = try builder.logicalAnd(
        try builder.gte(iw, try builder.u32Lit(0)),
        try builder.lt(iw, try in_width.toExpr()),
    );
    const pos_valid = try builder.logicalAnd(ih_valid, iw_valid);

    // Calculate input index
    const in_hw = try builder.mul(try in_height.toExpr(), try in_width.toExpr());
    const in_chw = try builder.mul(try channels.toExpr(), in_hw);
    const input_idx = try builder.add(
        try builder.add(
            try builder.mul(batch_idx, in_chw),
            try builder.mul(channel_idx, in_hw),
        ),
        try builder.add(
            try builder.mul(ih, try in_width.toExpr()),
            iw,
        ),
    );

    // Load and compare
    const input_val = try input.at(input_idx);
    const is_greater = try builder.gt(input_val, try max_val.toExpr());
    const should_update = try builder.logicalAnd(pos_valid, is_greater);

    // Update max if greater
    const new_max = try builder.select(should_update, input_val, try max_val.toExpr());
    const new_idx = try builder.select(should_update, input_idx, try max_idx.toExpr());
    const update_max = try builder.assignStmt(try max_val.toExpr(), new_max);
    const update_idx = try builder.assignStmt(try max_idx.toExpr(), new_idx);

    // Increment kx
    const one = try builder.u32Lit(1);
    const next_kx = try builder.add(try kx_var.toExpr(), one);
    const update_kx = try builder.assignStmt(try kx_var.toExpr(), next_kx);

    try builder.forLoop(null, kx_cond, update_kx, &[_]*const dsl.Stmt{ update_max, update_idx });

    // Reset kx, increment ky
    const reset_kx = try builder.assignStmt(try kx_var.toExpr(), try builder.u32Lit(0));
    const next_ky = try builder.add(try ky_var.toExpr(), one);
    const update_ky = try builder.assignStmt(try ky_var.toExpr(), next_ky);

    try builder.forLoop(null, ky_cond, update_ky, &[_]*const dsl.Stmt{reset_kx});

    // Calculate output index
    const out_hw = try builder.mul(try out_height.toExpr(), try out_width.toExpr());
    const out_chw = try builder.mul(try channels.toExpr(), out_hw);
    const output_idx = try builder.add(
        try builder.add(
            try builder.mul(batch_idx, out_chw),
            try builder.mul(channel_idx, out_hw),
        ),
        try builder.add(
            try builder.mul(out_y, try out_width.toExpr()),
            out_x,
        ),
    );

    // Store results
    const output_ptr = try output.at(output_idx);
    const indices_ptr = try indices.at(output_idx);
    const store_output = try builder.assignStmt(output_ptr, try max_val.toExpr());
    const store_indices = try builder.assignStmt(indices_ptr, try max_idx.toExpr());

    try builder.ifStmt(bounds_check, &[_]*const dsl.Stmt{ store_output, store_indices }, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build AvgPool2D kernel
///
/// Performs average pooling over spatial dimensions.
///
/// Buffers:
/// - input: [batch, channels, height, width] - input tensor
/// - output: [batch, channels, out_height, out_width] - pooled output
///
/// Uniforms:
/// - batch_size, channels, in_height, in_width
/// - out_height, out_width, kernel_size, stride, padding
pub fn buildAvgPool2dKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "avg_pool2d");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(16, 16, 1);

    // Buffer bindings
    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);

    // Uniform parameters
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());
    const in_height = try builder.addUniform("in_height", Type.u32Type());
    const in_width = try builder.addUniform("in_width", Type.u32Type());
    const out_height = try builder.addUniform("out_height", Type.u32Type());
    const out_width = try builder.addUniform("out_width", Type.u32Type());
    const kernel_size = try builder.addUniform("kernel_size", Type.u32Type());
    const stride = try builder.addUniform("stride", Type.u32Type());
    const padding = try builder.addUniform("padding", Type.u32Type());

    const gid = builder.globalInvocationId();
    const wid = builder.workgroupId();

    const out_x = try gid.x();
    const out_y = try gid.y();
    const channel = try wid.z();

    const batch_idx = try builder.div(channel, try channels.toExpr());
    const channel_idx = try builder.mod(channel, try channels.toExpr());

    // Bounds check
    const x_check = try builder.lt(out_x, try out_width.toExpr());
    const y_check = try builder.lt(out_y, try out_height.toExpr());
    const batch_check = try builder.lt(batch_idx, try batch_size.toExpr());
    const bounds_check = try builder.logicalAnd(batch_check, try builder.logicalAnd(x_check, y_check));

    // Initialize sum
    const sum = try builder.declareVar("sum", Type.f32Type(), try builder.f32Lit(0.0));

    // Pool area for averaging (kernel_size * kernel_size)
    const pool_area = try builder.mul(try kernel_size.toExpr(), try kernel_size.toExpr());
    const pool_area_f32 = try builder.castToF32(pool_area);

    // Loop over kernel window
    const ky_var = try builder.declareVar("ky", Type.u32Type(), try builder.u32Lit(0));
    const ky_cond = try builder.lt(try ky_var.toExpr(), try kernel_size.toExpr());

    const kx_var = try builder.declareVar("kx", Type.u32Type(), try builder.u32Lit(0));
    const kx_cond = try builder.lt(try kx_var.toExpr(), try kernel_size.toExpr());

    // Calculate input position
    const ih_base = try builder.mul(out_y, try stride.toExpr());
    const ih_offset = try builder.add(ih_base, try ky_var.toExpr());
    const ih = try builder.sub(ih_offset, try padding.toExpr());

    const iw_base = try builder.mul(out_x, try stride.toExpr());
    const iw_offset = try builder.add(iw_base, try kx_var.toExpr());
    const iw = try builder.sub(iw_offset, try padding.toExpr());

    // Check bounds
    const ih_valid = try builder.logicalAnd(
        try builder.gte(ih, try builder.u32Lit(0)),
        try builder.lt(ih, try in_height.toExpr()),
    );
    const iw_valid = try builder.logicalAnd(
        try builder.gte(iw, try builder.u32Lit(0)),
        try builder.lt(iw, try in_width.toExpr()),
    );
    const pos_valid = try builder.logicalAnd(ih_valid, iw_valid);

    // Calculate input index
    const in_hw = try builder.mul(try in_height.toExpr(), try in_width.toExpr());
    const in_chw = try builder.mul(try channels.toExpr(), in_hw);
    const input_idx = try builder.add(
        try builder.add(
            try builder.mul(batch_idx, in_chw),
            try builder.mul(channel_idx, in_hw),
        ),
        try builder.add(
            try builder.mul(ih, try in_width.toExpr()),
            iw,
        ),
    );

    // Load input value (0 if padding)
    const input_val = try input.at(input_idx);
    const zero = try builder.f32Lit(0.0);
    const masked_input = try builder.select(pos_valid, input_val, zero);

    // Accumulate sum
    const new_sum = try builder.add(try sum.toExpr(), masked_input);
    const update_sum = try builder.assignStmt(try sum.toExpr(), new_sum);

    // Increment kx
    const one = try builder.u32Lit(1);
    const next_kx = try builder.add(try kx_var.toExpr(), one);
    const update_kx = try builder.assignStmt(try kx_var.toExpr(), next_kx);

    try builder.forLoop(null, kx_cond, update_kx, &[_]*const dsl.Stmt{update_sum});

    // Reset kx, increment ky
    const reset_kx = try builder.assignStmt(try kx_var.toExpr(), try builder.u32Lit(0));
    const next_ky = try builder.add(try ky_var.toExpr(), one);
    const update_ky = try builder.assignStmt(try ky_var.toExpr(), next_ky);

    try builder.forLoop(null, ky_cond, update_ky, &[_]*const dsl.Stmt{reset_kx});

    // Calculate average
    const avg = try builder.div(try sum.toExpr(), pool_area_f32);

    // Calculate output index
    const out_hw = try builder.mul(try out_height.toExpr(), try out_width.toExpr());
    const out_chw = try builder.mul(try channels.toExpr(), out_hw);
    const output_idx = try builder.add(
        try builder.add(
            try builder.mul(batch_idx, out_chw),
            try builder.mul(channel_idx, out_hw),
        ),
        try builder.add(
            try builder.mul(out_y, try out_width.toExpr()),
            out_x,
        ),
    );

    // Store result
    const output_ptr = try output.at(output_idx);
    const store_stmt = try builder.assignStmt(output_ptr, avg);

    try builder.ifStmt(bounds_check, &[_]*const dsl.Stmt{store_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build BatchNorm2D kernel (inference mode)
///
/// Performs batch normalization for 2D inputs (vision):
/// output = gamma * (input - running_mean) / sqrt(running_var + eps) + beta
///
/// Buffers:
/// - input: [batch, channels, height, width] - input tensor
/// - gamma: [channels] - scale parameters
/// - beta: [channels] - shift parameters
/// - running_mean: [channels] - running mean from training
/// - running_var: [channels] - running variance from training
/// - output: [batch, channels, height, width] - normalized output
///
/// Uniforms:
/// - batch_size, channels, height, width, epsilon
pub fn buildBatchNorm2dKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "batch_norm2d");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // Buffer bindings
    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const gamma = try builder.addBuffer("gamma", Type.f32Type(), .read_only);
    const beta = try builder.addBuffer("beta", Type.f32Type(), .read_only);
    const running_mean = try builder.addBuffer("running_mean", Type.f32Type(), .read_only);
    const running_var = try builder.addBuffer("running_var", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);

    // Uniform parameters
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());
    const height = try builder.addUniform("height", Type.u32Type());
    const width = try builder.addUniform("width", Type.u32Type());
    const epsilon = try builder.addUniform("epsilon", Type.f32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();

    // Total elements
    const hw = try builder.mul(try height.toExpr(), try width.toExpr());
    const chw = try builder.mul(try channels.toExpr(), hw);
    const total = try builder.mul(try batch_size.toExpr(), chw);

    const condition = try builder.lt(idx, total);

    // Compute channel index: channel = (idx / (H * W)) % C
    const idx_div_hw = try builder.div(idx, hw);
    const channel_idx = try builder.mod(idx_div_hw, try channels.toExpr());

    // Load per-channel parameters
    const x = try input.at(idx);
    const g = try gamma.at(channel_idx);
    const b = try beta.at(channel_idx);
    const mean = try running_mean.at(channel_idx);
    const var_val = try running_var.at(channel_idx);

    // Normalize: (x - mean) / sqrt(var + eps)
    const centered = try builder.sub(x, mean);
    const var_eps = try builder.add(var_val, try epsilon.toExpr());
    const std_dev = try builder.sqrt(var_eps);
    const normalized = try builder.div(centered, std_dev);

    // Scale and shift: gamma * normalized + beta
    const scaled = try builder.mul(g, normalized);
    const result = try builder.add(scaled, b);

    const output_ptr = try output.at(idx);
    const store_stmt = try builder.assignStmt(output_ptr, result);

    try builder.ifStmt(condition, &[_]*const dsl.Stmt{store_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build im2col kernel
///
/// Transforms image patches into columns for efficient convolution via GEMM.
/// Each output column contains one flattened patch from the input.
///
/// Buffers:
/// - input: [batch, channels, height, width] - input tensor
/// - output: [batch, channels * kernel_h * kernel_w, out_h * out_w] - column matrix
///
/// Uniforms:
/// - batch_size, channels, in_height, in_width
/// - out_height, out_width, kernel_h, kernel_w
/// - stride_h, stride_w, pad_h, pad_w
pub fn buildIm2colKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "im2col");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // Buffer bindings
    const input = try builder.addBuffer("input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .write_only);

    // Uniform parameters
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());
    const in_height = try builder.addUniform("in_height", Type.u32Type());
    const in_width = try builder.addUniform("in_width", Type.u32Type());
    const out_height = try builder.addUniform("out_height", Type.u32Type());
    const out_width = try builder.addUniform("out_width", Type.u32Type());
    const kernel_h = try builder.addUniform("kernel_h", Type.u32Type());
    const kernel_w = try builder.addUniform("kernel_w", Type.u32Type());
    const stride_h = try builder.addUniform("stride_h", Type.u32Type());
    const stride_w = try builder.addUniform("stride_w", Type.u32Type());
    const pad_h = try builder.addUniform("pad_h", Type.u32Type());
    const pad_w = try builder.addUniform("pad_w", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();

    // Output dimensions
    const kernel_hw = try builder.mul(try kernel_h.toExpr(), try kernel_w.toExpr());
    const col_h = try builder.mul(try channels.toExpr(), kernel_hw); // C * kH * kW
    const col_w = try builder.mul(try out_height.toExpr(), try out_width.toExpr()); // oH * oW
    const col_size = try builder.mul(col_h, col_w);
    const total = try builder.mul(try batch_size.toExpr(), col_size);

    const condition = try builder.lt(idx, total);

    // Decompose idx into (batch, row, col) in column matrix
    // idx = batch * col_size + row * col_w + col_in_row
    const batch_idx = try builder.div(idx, col_size);
    const idx_in_batch = try builder.mod(idx, col_size);
    const row = try builder.div(idx_in_batch, col_w);
    const col_in_row = try builder.mod(idx_in_batch, col_w);

    // Decompose row into (channel, ky, kx)
    // row = c * (kH * kW) + ky * kW + kx
    const c = try builder.div(row, kernel_hw);
    const row_in_kernel = try builder.mod(row, kernel_hw);
    const ky = try builder.div(row_in_kernel, try kernel_w.toExpr());
    const kx = try builder.mod(row_in_kernel, try kernel_w.toExpr());

    // Decompose col_in_row into (oh, ow)
    const oh = try builder.div(col_in_row, try out_width.toExpr());
    const ow = try builder.mod(col_in_row, try out_width.toExpr());

    // Calculate input position
    const ih_base = try builder.mul(oh, try stride_h.toExpr());
    const ih_with_kernel = try builder.add(ih_base, ky);
    const ih = try builder.sub(ih_with_kernel, try pad_h.toExpr());

    const iw_base = try builder.mul(ow, try stride_w.toExpr());
    const iw_with_kernel = try builder.add(iw_base, kx);
    const iw = try builder.sub(iw_with_kernel, try pad_w.toExpr());

    // Check if input position is valid
    const ih_valid = try builder.logicalAnd(
        try builder.gte(ih, try builder.u32Lit(0)),
        try builder.lt(ih, try in_height.toExpr()),
    );
    const iw_valid = try builder.logicalAnd(
        try builder.gte(iw, try builder.u32Lit(0)),
        try builder.lt(iw, try in_width.toExpr()),
    );
    const pos_valid = try builder.logicalAnd(ih_valid, iw_valid);

    // Calculate input index
    const in_hw = try builder.mul(try in_height.toExpr(), try in_width.toExpr());
    const in_chw = try builder.mul(try channels.toExpr(), in_hw);
    const input_idx = try builder.add(
        try builder.add(
            try builder.mul(batch_idx, in_chw),
            try builder.mul(c, in_hw),
        ),
        try builder.add(
            try builder.mul(ih, try in_width.toExpr()),
            iw,
        ),
    );

    // Load input value (0 if padding)
    const input_val = try input.at(input_idx);
    const zero = try builder.f32Lit(0.0);
    const col_val = try builder.select(pos_valid, input_val, zero);

    // Store to output
    const output_ptr = try output.at(idx);
    const store_stmt = try builder.assignStmt(output_ptr, col_val);

    try builder.ifStmt(condition, &[_]*const dsl.Stmt{store_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

/// Build col2im kernel
///
/// Transforms column matrix back to image (inverse of im2col).
/// Used during backward pass to compute input gradients.
/// Accumulates values at overlapping positions.
///
/// Buffers:
/// - col_input: [batch, channels * kernel_h * kernel_w, out_h * out_w] - column gradient
/// - output: [batch, channels, height, width] - image gradient (accumulated)
///
/// Uniforms:
/// - batch_size, channels, in_height, in_width
/// - out_height, out_width, kernel_h, kernel_w
/// - stride_h, stride_w, pad_h, pad_w
pub fn buildCol2imKernel(allocator: std.mem.Allocator) !*const KernelIR {
    var builder = KernelBuilder.init(allocator, "col2im");
    errdefer builder.deinit();

    _ = builder.setWorkgroupSize(256, 1, 1);

    // Buffer bindings
    const col_input = try builder.addBuffer("col_input", Type.f32Type(), .read_only);
    const output = try builder.addBuffer("output", Type.f32Type(), .read_write); // read_write for atomic add

    // Uniform parameters
    const batch_size = try builder.addUniform("batch_size", Type.u32Type());
    const channels = try builder.addUniform("channels", Type.u32Type());
    const in_height = try builder.addUniform("in_height", Type.u32Type());
    const in_width = try builder.addUniform("in_width", Type.u32Type());
    const out_height = try builder.addUniform("out_height", Type.u32Type());
    const out_width = try builder.addUniform("out_width", Type.u32Type());
    const kernel_h = try builder.addUniform("kernel_h", Type.u32Type());
    const kernel_w = try builder.addUniform("kernel_w", Type.u32Type());
    const stride_h = try builder.addUniform("stride_h", Type.u32Type());
    const stride_w = try builder.addUniform("stride_w", Type.u32Type());
    const pad_h = try builder.addUniform("pad_h", Type.u32Type());
    const pad_w = try builder.addUniform("pad_w", Type.u32Type());

    const gid = builder.globalInvocationId();
    const idx = try gid.x();

    // Column matrix dimensions
    const kernel_hw = try builder.mul(try kernel_h.toExpr(), try kernel_w.toExpr());
    const col_h = try builder.mul(try channels.toExpr(), kernel_hw);
    const col_w = try builder.mul(try out_height.toExpr(), try out_width.toExpr());
    const col_size = try builder.mul(col_h, col_w);
    const total = try builder.mul(try batch_size.toExpr(), col_size);

    const condition = try builder.lt(idx, total);

    // Decompose idx (same as im2col)
    const batch_idx = try builder.div(idx, col_size);
    const idx_in_batch = try builder.mod(idx, col_size);
    const row = try builder.div(idx_in_batch, col_w);
    const col_in_row = try builder.mod(idx_in_batch, col_w);

    const c = try builder.div(row, kernel_hw);
    const row_in_kernel = try builder.mod(row, kernel_hw);
    const ky = try builder.div(row_in_kernel, try kernel_w.toExpr());
    const kx = try builder.mod(row_in_kernel, try kernel_w.toExpr());

    const oh = try builder.div(col_in_row, try out_width.toExpr());
    const ow = try builder.mod(col_in_row, try out_width.toExpr());

    // Calculate input position
    const ih_base = try builder.mul(oh, try stride_h.toExpr());
    const ih_with_kernel = try builder.add(ih_base, ky);
    const ih = try builder.sub(ih_with_kernel, try pad_h.toExpr());

    const iw_base = try builder.mul(ow, try stride_w.toExpr());
    const iw_with_kernel = try builder.add(iw_base, kx);
    const iw = try builder.sub(iw_with_kernel, try pad_w.toExpr());

    // Check if input position is valid
    const ih_valid = try builder.logicalAnd(
        try builder.gte(ih, try builder.u32Lit(0)),
        try builder.lt(ih, try in_height.toExpr()),
    );
    const iw_valid = try builder.logicalAnd(
        try builder.gte(iw, try builder.u32Lit(0)),
        try builder.lt(iw, try in_width.toExpr()),
    );
    const pos_valid = try builder.logicalAnd(ih_valid, iw_valid);

    // Load column value
    const col_val = try col_input.at(idx);

    // Calculate output index
    const in_hw = try builder.mul(try in_height.toExpr(), try in_width.toExpr());
    const in_chw = try builder.mul(try channels.toExpr(), in_hw);
    const output_idx = try builder.add(
        try builder.add(
            try builder.mul(batch_idx, in_chw),
            try builder.mul(c, in_hw),
        ),
        try builder.add(
            try builder.mul(ih, try in_width.toExpr()),
            iw,
        ),
    );

    // Atomic add to handle overlapping regions
    const output_ptr = try output.at(output_idx);
    const atomic_add_expr = try builder.call(.atomic_add, &.{ output_ptr, col_val });
    const atomic_stmt = try dsl.exprStmt(allocator, atomic_add_expr);

    // Only accumulate if position is valid (not in padding)
    try builder.ifStmt(try builder.logicalAnd(condition, pos_valid), &[_]*const dsl.Stmt{atomic_stmt}, null);

    const ir = try allocator.create(KernelIR);
    ir.* = try builder.build();
    return ir;
}

// ============================================================================
// Tests
// ============================================================================

test "buildConv2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildConv2dKernel(allocator);
    try std.testing.expectEqualStrings("conv2d", ir.name);
    try std.testing.expectEqual(@as(usize, 4), ir.buffers.len); // input, weights, bias, output
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildMaxPool2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildMaxPool2dKernel(allocator);
    try std.testing.expectEqualStrings("max_pool2d", ir.name);
    try std.testing.expectEqual(@as(usize, 3), ir.buffers.len); // input, output, indices
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildAvgPool2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildAvgPool2dKernel(allocator);
    try std.testing.expectEqualStrings("avg_pool2d", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[0]);
    try std.testing.expectEqual(@as(u32, 16), ir.workgroup_size[1]);
}

test "buildBatchNorm2dKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildBatchNorm2dKernel(allocator);
    try std.testing.expectEqualStrings("batch_norm2d", ir.name);
    try std.testing.expectEqual(@as(usize, 6), ir.buffers.len); // input, gamma, beta, mean, var, output
    try std.testing.expectEqual(@as(usize, 5), ir.uniforms.len); // batch, channels, height, width, epsilon
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildIm2colKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildIm2colKernel(allocator);
    try std.testing.expectEqualStrings("im2col", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // input, output
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test "buildCol2imKernel" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const ir = try buildCol2imKernel(allocator);
    try std.testing.expectEqualStrings("col2im", ir.name);
    try std.testing.expectEqual(@as(usize, 2), ir.buffers.len); // col_input, output
    try std.testing.expectEqual(@as(u32, 256), ir.workgroup_size[0]);
}

test {
    std.testing.refAllDecls(@This());
}
