//! Canonical error sets for WDBX.

pub const WdbxError = error{
    BlockNotFound,
    InvalidChecksum,
    StorageCorrupted,
    VectorDimensionMismatch,
    NodeOffline,
    OutOfMemory,
};
