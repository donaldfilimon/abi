//! FPGA Kernel Definitions
//!
//! Defines the hardware kernels available on the FPGA bitstream.
//! Based on internal FPGA research notes (see ROADMAP.md for status).

pub const KernelID = enum(u32) {
    QuantizedMatMul = 1,
    StreamingSoftmax = 2,
    AttentionMechanism = 3,
};

pub const QuantizedMatMulConfig = struct {
    m: u32,
    n: u32,
    k: u32,
    activation: u8, // 0=None, 1=ReLU, 2=GELU
};

pub const AttentionConfig = struct {
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
};
