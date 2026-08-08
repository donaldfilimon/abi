#![feature(portable_simd)]

//! Shared compute contracts for ABI.
//!
//! This crate owns deterministic CPU vector primitives and the capability
//! vocabulary shared by storage and accelerator crates. It deliberately has
//! no dependency on either `abi-wdbx` or `abi-gpu`, so accelerator adapters can
//! implement [`Accelerator`] without creating a crate cycle.

mod accelerator;
mod backend;
mod cpu;

pub use accelerator::{Accelerator, ScoredIndex};
pub use backend::{
    Backend, Capability, CapabilityEvidence, CapabilityInvariantError, CapabilityState, Selection,
    ane_hardware_present, baseline_capability_states, best_cpu_backend, capabilities, select,
};
pub use cpu::{ComputeError, CpuBackend, cosine, dot, norm, simd_lanes, top_k};
