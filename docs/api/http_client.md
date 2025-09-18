# HTTP Client API

This document provides comprehensive API documentation for the `http_client` module.

## Table of Contents

- [Overview](#overview)
- [Core Types](#core-types)
- [Functions](#functions)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

Enhanced HTTP client with retry logic, timeouts, and proxy support.

### Features

- **Automatic retry** with exponential backoff
- **Configurable timeouts** for connection and reading
- **Proxy support** via environment variables
- **SSL/TLS** verification options

## Configuration

```zig
pub const HttpClientConfig = struct {
    connect_timeout_ms: u32 = 5000,
    read_timeout_ms: u32 = 10000,
    max_retries: u32 = 3,
    initial_backoff_ms: u32 = 500,
    max_backoff_ms: u32 = 4000,
    user_agent: []const u8 = "WDBX/1.0",
    follow_redirects: bool = true,
    verify_ssl: bool = true,
    verbose: bool = false,
};
```

