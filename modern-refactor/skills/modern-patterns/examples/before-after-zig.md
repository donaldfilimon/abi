# Zig Example: Modernizing Error Handling

**Before (legacy)**
```zig
pub fn loadConfig(path: []const u8) ?Config {
    // ...
    if (err) return null;
    return config;
}
```

**After (modern)**
```zig
pub const LoadError = error{ FileNotFound, ParseError, ValidationError };

pub fn loadConfig(path: []const u8) LoadError!Config {
    // ...
    return config;
}
```

The modern version makes failure modes explicit and forces callers to handle them.
