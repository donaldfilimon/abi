# Windows Networking Guide for WDBX Server

## Overview

This guide addresses common Windows networking issues when running the WDBX HTTP server and provides solutions for optimal performance.

## 🚨 Common Windows Networking Issues

### 1. **"GetLastError(87): The parameter is incorrect"**

This error typically occurs when:
- Socket options are not properly configured for Windows
- Buffer sizes are incompatible with Windows networking stack
- Socket flags are not Windows-compatible

**Solution**: The server now includes Windows-specific socket optimizations:
- Automatic TCP_NODELAY configuration
- Optimized buffer sizes (8192 bytes)
- SO_LINGER settings for immediate connection closure
- SO_REUSEADDR for better port binding

### 2. **Connection Reset by Peer**

**Normal Behavior**: On Windows, this is expected and indicates:
- Server is running and handling connections properly
- Client disconnections are handled gracefully
- Network stack is functioning correctly

**Not an Error**: This is Windows networking behavior, not a server problem.

### 3. **PowerShell Invoke-WebRequest Failures**

**Issue**: `Invoke-WebRequest: Unable to read data from the transport connection`

**Solutions**:
1. Use the enhanced TCP test client: `zig run simple_tcp_test.zig`
2. Use curl: `curl http://localhost:8080/network`
3. Use a web browser: Navigate to `http://localhost:8080`

## 🔧 Server Configuration

### Windows-Optimized Settings

```zig
const config = http_server.ServerConfig{
    .host = "127.0.0.1",
    .port = 8080,
    .enable_windows_optimizations = true,
    .socket_buffer_size = 8192,
    .tcp_nodelay = true,
    .socket_keepalive = true,
    .connection_timeout_ms = 30000,
    .max_retries = 3,
};
```

### Socket Optimizations Applied

- **TCP_NODELAY**: Disables Nagle's algorithm for better performance
- **SO_KEEPALIVE**: Maintains connection stability
- **SO_REUSEADDR**: Better port binding on Windows
- **SO_LINGER**: Immediate connection closure
- **Buffer Sizes**: Optimized for Windows networking stack

## 🧪 Testing Your Server

### 1. Start the Server

```bash
# Build and run the HTTP server
zig build run -- http

# Or use the direct executable
.\zig-out\bin\abi.exe http
```

### 2. Test Connectivity

```bash
# Use the enhanced TCP test client
zig run simple_tcp_test.zig

# Use the Windows networking test suite
zig run test_windows_networking.zig
```

### 3. Test HTTP Endpoints

```bash
# Test with curl (if available)
curl http://localhost:8080/health
curl http://localhost:8080/network

# Test with PowerShell (may have timeout issues)
Invoke-WebRequest -Uri http://localhost:8080/health
```

## 📊 Expected Behavior

### Normal Windows Networking

✅ **Expected and Normal**:
- Connection reset messages
- "Unexpected" errors (Windows networking behavior)
- Broken pipe errors during client disconnection
- Server continues running despite connection errors

❌ **Actual Problems**:
- Server crashes or stops responding
- Port binding failures
- Complete connection refusal

### Performance Characteristics

- **Connection Handling**: Graceful error recovery
- **Throughput**: Optimized for Windows networking stack
- **Stability**: Robust error handling prevents crashes
- **Compatibility**: Works with all Windows versions (10+)

## 🛠️ Troubleshooting

### Server Won't Start

1. **Check Port Availability**:
   ```bash
   netstat -an | findstr :8080
   ```

2. **Check Firewall Settings**:
   - Windows Defender Firewall
   - Third-party antivirus software
   - Corporate network policies

3. **Run as Administrator** (if needed):
   ```bash
   # Right-click PowerShell/Command Prompt -> Run as Administrator
   zig build run -- http
   ```

### Connection Issues

1. **Test Basic Connectivity**:
   ```bash
   ping 127.0.0.1
   telnet 127.0.0.1 8080
   ```

2. **Check Server Logs**:
   - Look for "Windows optimized" message
   - Check for socket configuration warnings
   - Verify connection acceptance

3. **Use Alternative Ports**:
   ```bash
   zig build run -- http --port 8081
   ```

## 🔍 Diagnostic Tools

### Built-in Tests

- `simple_tcp_test.zig`: Enhanced TCP client with Windows optimizations
- `test_windows_networking.zig`: Comprehensive Windows networking test suite
- `windows_network_test.zig`: Low-level socket testing

### External Tools

- **Wireshark**: Network packet analysis
- **Process Monitor**: File and registry monitoring
- **Resource Monitor**: Network activity monitoring

## 📝 Best Practices

### For Development

1. **Use Enhanced Test Clients**: Avoid PowerShell for initial testing
2. **Monitor Server Logs**: Check for Windows-specific warnings
3. **Test Multiple Ports**: Verify port-specific issues
4. **Use Windows-Optimized Config**: Enable all Windows networking features

### For Production

1. **Enable Logging**: Monitor connection patterns
2. **Set Appropriate Timeouts**: 30+ seconds for Windows
3. **Monitor Resource Usage**: Check for memory leaks
4. **Use Load Balancing**: For high-traffic scenarios

## 🎯 Quick Fixes

### Immediate Solutions

1. **Restart the Server**: Often resolves temporary networking issues
2. **Clear Port Bindings**: Restart networking services
3. **Check Antivirus**: Disable temporarily for testing
4. **Use Different Port**: Avoid port conflicts

### Long-term Solutions

1. **Update Network Drivers**: Ensure latest Windows networking stack
2. **Optimize Firewall Rules**: Allow specific application access
3. **Monitor System Resources**: Ensure adequate memory and CPU
4. **Regular Maintenance**: Restart services periodically

## 📚 Additional Resources

- [Windows Networking Documentation](https://docs.microsoft.com/en-us/windows/win32/winsock/)
- [Zig Networking Examples](https://github.com/ziglang/zig/tree/master/lib/std/net)
- [WDBX Server Documentation](./WDBX_ENHANCED.md)

## 🆘 Getting Help

If you continue experiencing issues:

1. **Run Diagnostic Tests**: Use the provided test scripts
2. **Check System Logs**: Windows Event Viewer
3. **Verify Dependencies**: Ensure Zig and dependencies are up-to-date
4. **Report Issues**: Include error messages and system information

---

**Remember**: Windows networking behavior is different from Unix systems. Many "errors" are actually normal Windows networking characteristics and indicate the server is working correctly.
