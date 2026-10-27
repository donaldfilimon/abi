//! Unsupported platforms refuse before allocating process or filesystem state.
use super::{ProcessLimits, ToolExecutionContext, ToolExecutionError, error};
use abi_capability::Digest;
use std::path::Path;
pub(super) struct Pinned {
    pub(super) cwd: String,
}
impl Pinned {
    pub(super) fn new(_: &Path) -> Result<Self, ToolExecutionError> {
        Err(error("process_platform_unavailable"))
    }
    pub(super) fn identity_digest(&self) -> Digest {
        Digest::default()
    }
    pub(super) fn verify(&self) -> Result<(), ToolExecutionError> {
        Err(error("process_platform_unavailable"))
    }
}
#[derive(Default)]
pub(super) struct Runner;
impl Runner {
    pub(super) fn run(
        &mut self,
        _: &Pinned,
        _: ProcessLimits,
        _: ToolExecutionContext<'_>,
    ) -> Result<Vec<u8>, ToolExecutionError> {
        Err(error("process_platform_unavailable"))
    }
}
