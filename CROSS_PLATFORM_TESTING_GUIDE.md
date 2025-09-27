# Cross-Platform Testing Guide

This guide covers the comprehensive cross-platform testing strategy for the ABI AI Framework.

## Test Matrix

### Operating Systems
- **Windows**: Windows Server 2019, 2022
- **macOS**: macOS 13 (Ventura), macOS 14 (Sonoma)
- **Linux**: Ubuntu 18.04, 20.04, 22.04

### Architectures
- **x86_64**: Primary architecture for all platforms
- **aarch64**: ARM64 support (especially macOS Apple Silicon)

### Zig Versions
- **0.16.0-dev (master)**: Current baseline
- **Master nightly**: Tracks upstream commits for regression detection
- **master**: Nightly builds

## Platform-Specific Considerations

### Windows
- File paths use backslashes (`\`)
- Use Windows Sockets API (Winsock2)
- Consider Windows file attributes and permissions
- Test with different Windows versions (Server 2019/2022)

### macOS
- File paths use forward slashes (`/`)
- Use BSD socket API
- Consider macOS-specific frameworks (Foundation, CoreFoundation)
- Test on both Intel and Apple Silicon

### Linux
- Use epoll for efficient I/O multiplexing
- Consider different libc implementations (glibc, musl)
- Test with different kernel versions
- Consider containerized environments

## Testing Best Practices

### 1. Conditional Compilation
```zig
const builtin = @import("builtin");

if (builtin.os.tag == .windows) {
    // Windows-specific code
} else if (builtin.os.tag == .macos) {
    // macOS-specific code
} else if (builtin.os.tag == .linux) {
    // Linux-specific code
}
```

### 2. Platform Detection
```zig
const is_windows = builtin.os.tag == .windows;
const is_macos = builtin.os.tag == .macos;
const is_linux = builtin.os.tag == .linux;
```

### 3. Cross-Platform Path Handling
```zig
// Use std.fs.path for cross-platform paths
const path = try std.fs.path.join(allocator, &[_][]const u8{"dir", "file.txt"});
```

### 4. Network Testing
```zig
// Test both IPv4 and IPv6
const address = try std.net.Address.parseIp4("127.0.0.1", 8080);
// Also test IPv6: std.net.Address.parseIp6("::1", 8080)
```

## CI/CD Configuration

The CI pipeline tests multiple combinations of:
- Operating systems (Windows, macOS, Linux)
- Zig versions (dev, stable, master)
- Architectures (x86_64, aarch64)

## Running Cross-Platform Tests

```bash
# Run all tests
zig build test

# Run platform-specific tests
zig build test-cross-platform

# Run tests for specific OS
zig build test-windows
zig build test-macos
zig build test-linux
```

## Debugging Cross-Platform Issues

1. **Check platform detection**: Verify `builtin.os.tag` values
2. **Use conditional compilation**: Isolate platform-specific code
3. **Test path handling**: Ensure cross-platform path operations
4. **Verify network operations**: Test socket operations on each platform
5. **Check file permissions**: Verify file access patterns work across platforms

## Performance Considerations

- **Windows**: Consider I/O completion ports for high-performance networking
- **macOS**: Use kqueue for efficient event handling
- **Linux**: Leverage epoll for scalable I/O operations

## Container Testing

For Linux testing in containers:
```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y curl xz-utils
# Install Zig and test
```

## Continuous Integration

The CI pipeline automatically tests:
- Build compatibility across platforms
- Test execution on all supported platforms
- Cross-compilation to different targets
- Performance regression detection
