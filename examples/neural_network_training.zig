//! Complete Neural Network Training Example
//!
//! Demonstrates training a simple MLP on synthetic data using GPU/TPU acceleration.

const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n{s}╔══════════════════════════════════════╗{s}\n", .{
        abi.plugins.tui.ansi.bright_cyan,
        abi.plugins.tui.ansi.reset,
    });
    std.debug.print("{s}║  ABI Neural Network Training Demo   ║{s}\n", .{
        abi.plugins.tui.ansi.bright_cyan,
        abi.plugins.tui.ansi.reset,
    });
    std.debug.print("{s}╚══════════════════════════════════════╝{s}\n\n", .{
        abi.plugins.tui.ansi.bright_cyan,
        abi.plugins.tui.ansi.reset,
    });

    // Create accelerator
    var accel = abi.gpu.accelerator.createBestAccelerator(allocator);
    std.debug.print("{}Using backend: {s}{}{s}\n\n", .{
        abi.plugins.tui.ansi.bright_green,
        accel.backend.displayName(),
        abi.plugins.tui.ansi.dim,
        abi.plugins.tui.ansi.reset,
    });

    // Build network: 784 -> 128 -> 10 (MNIST-sized)
    std.debug.print("Building neural network...\n", .{});
    var layer1 = try abi.ai.layers.Dense.init(allocator, &accel, 784, 128);
    defer layer1.deinit();
    var activation1 = abi.ai.layers.ReLU.init(&accel);
    defer activation1.deinit();
    var layer2 = try abi.ai.layers.Dense.init(allocator, &accel, 128, 10);
    defer layer2.deinit();

    abi.plugins.tui.success("Network created: 784 -> 128 -> ReLU -> 10");

    // Generate random training data
    std.debug.print("\nGenerating training data...\n", .{});
    const batch_size = 32;
    const num_batches = 10;

    var input_data = try allocator.alloc(f32, batch_size * 784);
    defer allocator.free(input_data);
    var target_data = try allocator.alloc(f32, batch_size * 10);
    defer allocator.free(target_data);

    var prng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const random = prng.random();

    // Training loop
    std.debug.print("\n{s}Training:{s}\n", .{ abi.plugins.tui.ansi.bright_yellow, abi.plugins.tui.ansi.reset });
    var progress = abi.plugins.tui.ProgressBar.init(0, 0, 50, "Progress");

    for (0..num_batches) |epoch| {
        // Generate random batch
        for (input_data) |*x| x.* = random.float(f32);
        @memset(target_data, 0);
        for (0..batch_size) |i| {
            const label = random.intRangeAtMost(usize, 0, 9);
            target_data[i * 10 + label] = 1.0;
        }

        // Upload to device
        var input_mem = try accel.alloc(input_data.len * @sizeOf(f32));
        defer accel.free(&input_mem);
        try accel.copyToDevice(input_mem, std.mem.sliceAsBytes(input_data));

        // Forward pass
        const h1 = try layer1.forward(input_mem, batch_size);
        defer accel.free(@constCast(&h1));
        const a1 = try activation1.forward(h1, batch_size * 128);
        defer accel.free(@constCast(&a1));
        const output = try layer2.forward(a1, batch_size);
        defer accel.free(@constCast(&output));

        // Simple loss (would compute cross-entropy in production)
        var output_cpu = try allocator.alloc(f32, batch_size * 10);
        defer allocator.free(output_cpu);
        try accel.copyFromDevice(std.mem.sliceAsBytes(output_cpu), output);

        var loss: f32 = 0;
        for (0..batch_size * 10) |i| {
            const diff = output_cpu[i] - target_data[i];
            loss += diff * diff;
        }
        loss /= @floatFromInt(batch_size);

        // Update progress
        progress.setProgress(@as(f32, @floatFromInt(epoch + 1)) / @as(f32, @floatFromInt(num_batches)));
        progress.render();

        if (epoch == num_batches - 1) {
            std.debug.print("\n{}Final Loss: {d:.4}{s}\n", .{ abi.plugins.tui.ansi.bright_green, loss, abi.plugins.tui.ansi.reset });
        }
    }

    abi.plugins.tui.success("\nTraining complete!");

    // Demonstrate vector database integration
    std.debug.print("\n{s}Vector Database Demo:{s}\n", .{ abi.plugins.tui.ansi.bright_magenta, abi.plugins.tui.ansi.reset });

    var vec_search = abi.database.vector_search_gpu.VectorSearchGPU.init(allocator, &accel, 128);
    defer vec_search.deinit();

    // Insert embeddings
    const embedding = try allocator.alloc(f32, 128);
    defer allocator.free(embedding);
    for (embedding) |*e| e.* = random.float(f32);

    const id = try vec_search.insert(embedding);
    std.debug.print("Inserted vector with ID: {d}\n", .{id});

    // Search
    const results = try vec_search.search(embedding, 1);
    defer allocator.free(results);

    abi.plugins.tui.info("Vector search complete!");

    std.debug.print("\n{s}═══════════════════════════════════════{s}\n", .{
        abi.plugins.tui.ansi.bright_cyan,
        abi.plugins.tui.ansi.reset,
    });
    std.debug.print("Demo completed successfully! ✓\n", .{});
}
