//! Utilities Demo
//! Demonstrates the comprehensive utilities available in the project

const std = @import("std");
const utils = @import("../src/utils.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.log.info("🚀 Starting Utilities Demo", .{});

    // JSON Utilities Demo
    try demoJsonUtils(allocator);

    // URL Utilities Demo
    try demoUrlUtils(allocator);

    // Base64 Utilities Demo
    try demoBase64Utils(allocator);

    // Validation Utilities Demo
    try demoValidationUtils();

    // Random Utilities Demo
    try demoRandomUtils(allocator);

    // Math Utilities Demo
    try demoMathUtils(allocator);

    // Memory Management Utilities Demo
    try demoMemoryUtils(allocator);

    // Error Handling Utilities Demo
    try demoErrorUtils(allocator);

    std.log.info("✅ Utilities Demo completed successfully!", .{});
}

fn demoJsonUtils(allocator: std.mem.Allocator) !void {
    std.log.info("📄 JSON Utilities Demo", .{});

    // Parse JSON
    const json_str = "{\"name\":\"Alice\",\"age\":30,\"active\":true}";
    var parsed = try utils.JsonUtils.parse(allocator, json_str);
    defer parsed.deinit(allocator);

    // Stringify back
    const stringified = try utils.JsonUtils.stringify(allocator, parsed);
    defer allocator.free(stringified);

    std.log.info("Parsed and stringified JSON: {s}", .{stringified});
}

fn demoUrlUtils(allocator: std.mem.Allocator) !void {
    std.log.info("🔗 URL Utilities Demo", .{});

    // URL encoding/decoding
    const original = "Hello World! 你好";
    const encoded = try utils.UrlUtils.encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try utils.UrlUtils.decode(allocator, encoded);
    defer allocator.free(decoded);

    std.log.info("Original: {s}", .{original});
    std.log.info("Encoded: {s}", .{encoded});
    std.log.info("Decoded: {s}", .{decoded});

    // Parse URL
    const url = "https://example.com:8080/path?param=value#section";
    var components = try utils.UrlUtils.parseUrl(allocator, url);
    defer components.deinit(allocator);

    std.log.info("URL Components:", .{});
    std.log.info("  Scheme: {s}", .{components.scheme});
    std.log.info("  Host: {s}", .{components.host});
    std.log.info("  Port: {}", .{components.port});
    std.log.info("  Path: {s}", .{components.path});
    std.log.info("  Query: {s}", .{components.query});
    std.log.info("  Fragment: {s}", .{components.fragment});
}

fn demoBase64Utils(allocator: std.mem.Allocator) !void {
    std.log.info("🔐 Base64 Utilities Demo", .{});

    const original = "Hello, World! 🌍";
    const encoded = try utils.Base64Utils.encode(allocator, original);
    defer allocator.free(encoded);

    const decoded = try utils.Base64Utils.decode(allocator, encoded);
    defer allocator.free(decoded);

    std.log.info("Original: {s}", .{original});
    std.log.info("Base64: {s}", .{encoded});
    std.log.info("Decoded: {s}", .{decoded});
}

fn demoValidationUtils() !void {
    std.log.info("✅ Validation Utilities Demo", .{});

    // Email validation
    const valid_email = "user@example.com";
    const invalid_email = "not-an-email";
    std.log.info("Email '{s}' is valid: {}", .{ valid_email, utils.ValidationUtils.isValidEmail(valid_email) });
    std.log.info("Email '{s}' is valid: {}", .{ invalid_email, utils.ValidationUtils.isValidEmail(invalid_email) });

    // UUID validation
    const valid_uuid = "550e8400-e29b-41d4-a716-446655440000";
    const invalid_uuid = "not-a-uuid";
    std.log.info("UUID '{s}' is valid: {}", .{ valid_uuid, utils.ValidationUtils.isValidUuid(valid_uuid) });
    std.log.info("UUID '{s}' is valid: {}", .{ invalid_uuid, utils.ValidationUtils.isValidUuid(invalid_uuid) });

    // Password validation
    const strong_password = "StrongP@ss123";
    const weak_password = "weak";
    const options = utils.ValidationUtils.PasswordOptions{};
    std.log.info("Password '{s}' is strong: {}", .{ strong_password, utils.ValidationUtils.isStrongPassword(strong_password, options) });
    std.log.info("Password '{s}' is strong: {}", .{ weak_password, utils.ValidationUtils.isStrongPassword(weak_password, options) });
}

fn demoRandomUtils(allocator: std.mem.Allocator) !void {
    std.log.info("🎲 Random Utilities Demo", .{});

    // Generate random strings
    const random_str = try utils.RandomUtils.randomAlphanumeric(allocator, 16);
    defer allocator.free(random_str);

    const url_safe_str = try utils.RandomUtils.randomUrlSafe(allocator, 20);
    defer allocator.free(url_safe_str);

    // Generate UUID
    const uuid = try utils.RandomUtils.generateUuid(allocator);
    defer allocator.free(uuid);

    // Generate secure token
    const token = try utils.RandomUtils.generateToken(allocator, 32);
    defer allocator.free(token);

    std.log.info("Random alphanumeric: {s}", .{random_str});
    std.log.info("Random URL-safe: {s}", .{url_safe_str});
    std.log.info("Generated UUID: {s}", .{uuid});
    std.log.info("Generated token: {s}", .{token});

    // Random numbers
    const random_int = utils.RandomUtils.randomInt(i32, 1, 100);
    const random_float = utils.RandomUtils.randomFloat();
    std.log.info("Random integer (1-100): {}", .{random_int});
    std.log.info("Random float (0-1): {d:.4}", .{random_float});
}

fn demoMathUtils(allocator: std.mem.Allocator) !void {
    std.log.info("🧮 Math Utilities Demo", .{});

    // Basic operations
    const clamped = utils.MathUtils.clamp(f64, 15.5, 0.0, 10.0);
    const interpolated = utils.MathUtils.lerp(0.0, 100.0, 0.75);
    const percentage = utils.MathUtils.percentage(25.0, 100.0);

    std.log.info("Clamped 15.5 to [0,10]: {d:.1}", .{clamped});
    std.log.info("Lerp(0,100,0.75): {d:.1}", .{interpolated});
    std.log.info("Percentage(25/100): {d:.1}%", .{percentage});

    // Power of 2 operations
    std.log.info("Is 16 power of 2: {}", .{utils.MathUtils.isPowerOfTwo(16)});
    std.log.info("Is 15 power of 2: {}", .{utils.MathUtils.isPowerOfTwo(15)});
    std.log.info("Next power of 2 after 15: {}", .{utils.MathUtils.nextPowerOfTwo(15)});

    // Statistics
    const values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const mean = utils.MathUtils.mean(&values);
    const std_dev = utils.MathUtils.standardDeviation(&values);
    const median = try utils.MathUtils.median(allocator, &values);

    std.log.info("Mean: {d:.2}", .{mean});
    std.log.info("Standard deviation: {d:.2}", .{std_dev});
    std.log.info("Median: {d:.2}", .{median});

    // Distance calculations
    const dist2d = utils.MathUtils.distance2D(0.0, 0.0, 3.0, 4.0);
    const dist3d = utils.MathUtils.distance3D(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);

    std.log.info("2D distance (0,0) to (3,4): {d:.2}", .{dist2d});
    std.log.info("3D distance (0,0,0) to (1,1,1): {d:.2}", .{dist3d});

    // Angle conversion
    const radians = utils.MathUtils.degreesToRadians(180.0);
    const degrees = utils.MathUtils.radiansToDegrees(std.math.pi);

    std.log.info("180° in radians: {d:.2}", .{radians});
    std.log.info("π in degrees: {d:.2}", .{degrees});
}

fn demoMemoryUtils(allocator: std.mem.Allocator) !void {
    std.log.info("💾 Memory Management Utilities Demo", .{});

    // Safe allocation
    const buffer = try utils.MemoryUtils.safeAlloc(allocator, u8, 100);
    defer allocator.free(buffer);
    std.log.info("Allocated buffer of size: {}", .{buffer.len});

    // Safe duplication
    const original = "Hello, Memory Utils!";
    const duplicated = try utils.MemoryUtils.safeDupe(allocator, u8, original);
    defer allocator.free(duplicated);
    std.log.info("Duplicated string: {s}", .{duplicated});

    // Managed buffer
    var managed_buffer = try utils.MemoryUtils.ManagedBuffer(u32).init(allocator, 50);
    defer managed_buffer.deinit();
    std.log.info("Managed buffer size: {}", .{managed_buffer.data.len});

    // Resize managed buffer
    try managed_buffer.resize(100);
    std.log.info("Resized managed buffer to: {}", .{managed_buffer.data.len});
}

fn demoErrorUtils(_: std.mem.Allocator) !void {
    std.log.info("⚠️ Error Handling Utilities Demo", .{});

    // Success result
    const success_result = utils.ErrorUtils.success(i32, 42);
    std.log.info("Success result: {}", .{success_result.unwrap()});
    std.log.info("Is success: {}", .{success_result.isSuccess()});

    // Failure result
    const failure_result = utils.ErrorUtils.failure(i32, "Test error", "demo.zig", 123, "demoFunction");
    std.log.info("Is failure: {}", .{failure_result.isFailure()});
    std.log.info("Default value: {}", .{failure_result.unwrapOr(99)});

    // Retry mechanism
    const retry_func = struct {
        fn call() !i32 {
            return 42;
        }
    }.call;

    const retry_result = try utils.ErrorUtils.retry(i32, retry_func, 3, 10);
    std.log.info("Retry result: {}", .{retry_result});

    // Common validation
    try utils.CommonValidationUtils.validateBounds(i32, 5, 0, 10);
    std.log.info("Validation passed: value 5 is within bounds [0, 10]", .{});

    try utils.CommonValidationUtils.validateString("hello", 100);
    std.log.info("String validation passed: 'hello' is valid", .{});
}
