//! Hardware Acceleration Module
//!
//! Provides a unified interface for various hardware accelerators (CPU, GPU, etc.)
//! and tensor operations.

pub const CpuDriver = @import("backends/cpu_driver.zig").CpuDriver;
pub const driver = @import("driver.zig");
pub const AcceleratorType = driver.AcceleratorType;
pub const DeviceInfo = driver.DeviceInfo;
pub const Driver = driver.Driver;
pub const Tensor = @import("tensor.zig").Tensor;

// TODO: Export implementation factories
