//! # Platform Abstractions
//!
//! OS-specific functionality abstracted for cross-platform portability.
//!
//! ## Supported Platforms
//!
//! | Platform | Status |
//! |----------|--------|
//! | Linux | Full support |
//! | Windows | Full support |
//! | macOS | Full support |
//! | WASM | Limited (no threads, no filesystem) |
//!
//! ## Features
//!
//! - **Thread Management**: Cross-platform thread spawning and affinity
//! - **File I/O**: Unified file operations
//! - **Memory Mapping**: Platform-specific mmap/VirtualAlloc wrappers
//! - **System Info**: CPU count, page size, memory stats
//!
//! ## Usage
//!
//! ### System Information
//!
//! ```zig
//! const platform = @import("shared").platform;
//!
//! const cpu_count = platform.getCpuCount();
//! const page_size = platform.getPageSize();
//! const total_memory = platform.getTotalMemory();
//! ```
//!
//! ### Thread Affinity
//!
//! ```zig
//! // Pin current thread to CPU 0
//! try platform.setThreadAffinity(0);
//!
//! // Get current CPU
//! const cpu_id = platform.getCurrentCpu();
//! ```
//!
//! ### Memory Mapping
//!
//! ```zig
//! const mapping = try platform.mmap(null, size, .read_write);
//! defer platform.munmap(mapping);
//! ```
//!
//! ## Platform-Specific Notes
//!
//! ### Windows
//!
//! Uses kernel32 APIs for file and memory operations. See
//! `src/features/ai/llm/io/mmap.zig` for memory-mapped file examples.
//!
//! ### WASM
//!
//! Many platform features are unavailable. Use feature detection:
//!
//! ```zig
//! if (builtin.target.isWasm()) {
//!     // WASM-specific fallback
//! }
//! ```
//!
//! ## See Also
//!
//! - [NUMA Support](../../compute/runtime/numa.zig)
