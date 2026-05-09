//! Stubbed retrieval namespace for disabled database builds.

const core_db = @import("../stub.zig");

pub const hnsw = core_db.hnsw;
pub const distance = core_db.distance;
pub const simd = core_db.simd;
pub const quantize = core_db.quantize;
pub const batch = core_db.batch;
pub const fulltext = core_db.fulltext;
pub const hybrid = core_db.hybrid;
pub const filter = core_db.filter;
pub const clustering = core_db.clustering;
pub const formats = core_db.formats;
pub const index = core_db.index;
pub const quantization = core_db.quantization;
pub const parallel_hnsw = core_db.parallel_hnsw;
pub const parallel_search = core_db.parallel_search;
pub const diskann = core_db.diskann;
pub const scann = core_db.scann;

pub const KMeans = core_db.KMeans;
pub const ScalarQuantizer = core_db.ScalarQuantizer;
pub const ProductQuantizer = core_db.ProductQuantizer;

const std = @import("std");

test {
    std.testing.refAllDecls(@This());
}
