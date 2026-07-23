const std = @import("std");
const types = @import("types.zig");
const point_neural_net = @import("point_neural_net.zig");
const incremental = @import("incremental.zig");
const identity = @import("identity.zig");
const weights = @import("router_weights.zig");
const sentiment = @import("router_sentiment.zig");

pub const ProfileWeights = weights.ProfileWeights;

/// Helper function to route to the appropriate profile based on profile selector.
/// Uses the same iterative template generator as streaming completions (without
/// a callback) so one-shot and incremental paths stay string-identical.
pub fn routeToProfile(allocator: std.mem.Allocator, profile_sel: types.AgentProfile, input: []const u8) ![]u8 {
    return incremental.generateProfileIncremental(allocator, profile_sel, input, null, null);
}

pub fn routeInput(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    const weights_val = sentiment.analyzeSentiment(input);
    const profile_sel = sentiment.selectBestProfile(weights_val);
    return routeToProfile(allocator, profile_sel, input);
}

/// Soul-aware routing: blends keyword-based sentiment with a
/// pre-trained 3-output PointNeuralNetwork (one output per profile:
/// abbey, aviva, abi). The network's output is softmax-normalized
/// and blended with keyword weights using `blend_alpha` (0.0 = keyword only,
/// 1.0 = neural only). Falls back to keyword-only if `net` is null
/// or doesn't have 3 outputs.
pub fn routeInputWithSoul(
    allocator: std.mem.Allocator,
    net: ?*point_neural_net.PointNeuralNetwork,
    blend_alpha: f32,
    input: []const u8,
) ![]u8 {
    if (sentiment.explicitProfileSelector(input)) |profile| {
        return routeToProfile(allocator, profile, input);
    }

    const keyword_weights = sentiment.analyzeSentiment(input);
    // Start from the keyword decision so a missing network or rejected output
    // shape/value preserves the documented fallback regardless of blend_alpha.
    var neural_weights = keyword_weights;

    if (net) |n| {
        if (n.layers.len > 0 and n.layers[n.layers.len - 1].output_size == 3) {
            const point = point_neural_net.Point.fromText(input);
            const output = try n.forward(&point.toArray());
            defer allocator.free(output);
            if (output.len == 3) {
                // Stable softmax: subtracting the largest finite logit avoids
                // overflow while non-finite output preserves keyword fallback.
                var logits_are_finite = true;
                var max_logit = output[0];
                for (output[1..]) |o| {
                    if (!std.math.isFinite(o)) {
                        logits_are_finite = false;
                        break;
                    }
                    max_logit = @max(max_logit, o);
                }
                if (!std.math.isFinite(max_logit)) logits_are_finite = false;

                if (logits_are_finite) {
                    var exps: [3]f32 = undefined;
                    var sum: f32 = 0;
                    for (output, 0..) |o, i| {
                        exps[i] = @exp(o - max_logit);
                        sum += exps[i];
                    }
                    if (sum > 0 and std.math.isFinite(sum)) {
                        neural_weights.w_abbey = exps[0] / sum;
                        neural_weights.w_aviva = exps[1] / sum;
                        neural_weights.w_abi = exps[2] / sum;
                    }
                }
            }
        }
    }

    const blended = weights.blendWeights(keyword_weights, neural_weights, blend_alpha);
    const profile_sel = sentiment.selectBestProfile(blended);
    return routeToProfile(allocator, profile_sel, input);
}

test "explicit profile requests override one-shot and soul routing" {
    const allocator = std.testing.allocator;

    const aviva_result = try routeInput(allocator, "Aviva, be direct.");
    defer allocator.free(aviva_result);
    try std.testing.expect(std.mem.startsWith(u8, aviva_result, identity.profileContract(.aviva).response_prefix));

    const abi_result = try routeInputWithSoul(allocator, null, 1.0, "ABI, orchestrate this.");
    defer allocator.free(abi_result);
    try std.testing.expect(std.mem.startsWith(u8, abi_result, identity.profileContract(.abi).response_prefix));
}

test "routeInputWithSoul preserves keyword routing without a network" {
    const allocator = std.testing.allocator;
    const input = "analyze the logical structure";

    const expected = try routeInput(allocator, input);
    defer allocator.free(expected);
    const actual = try routeInputWithSoul(allocator, null, 1.0, input);
    defer allocator.free(actual);

    try std.testing.expectEqualStrings(expected, actual);
}

test "routeInput returns response from selected profile" {
    const allocator = std.testing.allocator;
    const result = try routeInput(allocator, "analyze the logical structure");
    defer allocator.free(result);
    try std.testing.expect(result.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
