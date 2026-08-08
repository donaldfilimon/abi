//! Append-only MVCC audit chain.
//!
//! The Zig implementation keeps linked nodes stable while a snapshot records
//! the visible length. Rust can express the same contract more directly:
//! immutable blocks are reference counted and a snapshot owns a frozen list of
//! references. Appends therefore never invalidate an existing snapshot.

use std::fmt;
use std::sync::{Arc, RwLock};

use crate::{BlockRecord, Hash};

/// One immutable block and its MVCC version.
#[derive(Debug)]
pub struct MvccBlock {
    /// Persisted audit-chain record.
    pub record: BlockRecord,
    /// Version visible to snapshots; identical to the block sequence.
    pub version: u64,
}

#[derive(Debug, Default)]
struct ChainState {
    blocks: Vec<Arc<MvccBlock>>,
    next_sequence: u64,
}

/// A thread-safe append-only audit chain.
#[derive(Debug, Clone, Default)]
pub struct MvccChain {
    state: Arc<RwLock<ChainState>>,
}

/// A frozen, independently owned chain view.
#[derive(Debug, Clone, Default)]
pub struct MvccSnapshot {
    blocks: Arc<[Arc<MvccBlock>]>,
}

/// Errors produced by the in-memory MVCC chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MvccError {
    /// Profiles are required by the persisted block contract.
    EmptyProfile,
    /// Another thread panicked while holding the chain lock.
    LockPoisoned,
}

impl fmt::Display for MvccError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProfile => formatter.write_str("block profile must not be empty"),
            Self::LockPoisoned => formatter.write_str("MVCC chain lock was poisoned"),
        }
    }
}

impl std::error::Error for MvccError {}

impl MvccChain {
    /// Create an empty chain.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append using the current Unix millisecond timestamp.
    pub fn append(
        &self,
        profile: &str,
        query_id: u64,
        response_id: u64,
        metadata: &str,
    ) -> Result<Hash, MvccError> {
        self.append_at(
            profile,
            query_id,
            response_id,
            metadata,
            abi_foundation::time::unix_ms(),
        )
    }

    /// Append using an explicit timestamp, preserving deterministic restore.
    pub fn append_at(
        &self,
        profile: &str,
        query_id: u64,
        response_id: u64,
        metadata: &str,
        timestamp_ms: i64,
    ) -> Result<Hash, MvccError> {
        if profile.is_empty() {
            return Err(MvccError::EmptyProfile);
        }

        let mut state = self.state.write().map_err(|_| MvccError::LockPoisoned)?;
        let prev_hash = state
            .blocks
            .last()
            .map_or(Hash::GENESIS, |block| block.record.hash);
        let sequence = state.next_sequence;
        let hash =
            crate::wal::compute_block_hash(prev_hash, timestamp_ms, sequence, profile, metadata);
        let block = Arc::new(MvccBlock {
            record: BlockRecord {
                hash,
                prev_hash,
                timestamp_ms,
                sequence,
                profile: profile.to_owned(),
                query_id,
                response_id,
                metadata: metadata.to_owned(),
            },
            version: sequence,
        });
        state.blocks.push(block);
        state.next_sequence = state.next_sequence.saturating_add(1);
        Ok(hash)
    }

    /// Number of blocks visible in the current chain.
    pub fn len(&self) -> Result<usize, MvccError> {
        self.state
            .read()
            .map(|state| state.blocks.len())
            .map_err(|_| MvccError::LockPoisoned)
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> Result<bool, MvccError> {
        self.len().map(|length| length == 0)
    }

    /// Retrieve a stable shared reference by hash.
    pub fn get(&self, hash: Hash) -> Result<Option<Arc<MvccBlock>>, MvccError> {
        self.state
            .read()
            .map(|state| {
                state
                    .blocks
                    .iter()
                    .find(|block| block.record.hash == hash)
                    .cloned()
            })
            .map_err(|_| MvccError::LockPoisoned)
    }

    /// Hash of the newest block.
    pub fn tail_hash(&self) -> Result<Option<Hash>, MvccError> {
        self.state
            .read()
            .map(|state| state.blocks.last().map(|block| block.record.hash))
            .map_err(|_| MvccError::LockPoisoned)
    }

    /// Freeze the currently visible block references.
    pub fn snapshot(&self) -> Result<MvccSnapshot, MvccError> {
        self.state
            .read()
            .map(|state| MvccSnapshot {
                blocks: Arc::from(state.blocks.clone()),
            })
            .map_err(|_| MvccError::LockPoisoned)
    }

    /// Verify predecessor links and recompute every content hash.
    pub fn verify(&self) -> Result<bool, MvccError> {
        let state = self.state.read().map_err(|_| MvccError::LockPoisoned)?;
        Ok(verify_blocks(&state.blocks))
    }
}

impl MvccSnapshot {
    /// Number of blocks frozen into this view.
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether the snapshot contains no blocks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Retrieve a stable shared reference from this frozen view.
    #[must_use]
    pub fn get(&self, hash: Hash) -> Option<Arc<MvccBlock>> {
        self.blocks
            .iter()
            .find(|block| block.record.hash == hash)
            .cloned()
    }

    /// Iterate in append order without observing later writes.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &Arc<MvccBlock>> {
        self.blocks.iter()
    }

    /// Verify predecessor links and hashes within the frozen view.
    #[must_use]
    pub fn verify(&self) -> bool {
        verify_blocks(&self.blocks)
    }
}

fn verify_blocks(blocks: &[Arc<MvccBlock>]) -> bool {
    let mut expected_prev = Hash::GENESIS;
    for block in blocks {
        let record = &block.record;
        if record.prev_hash != expected_prev || record.sequence != block.version {
            return false;
        }
        let expected_hash = crate::wal::compute_block_hash(
            record.prev_hash,
            record.timestamp_ms,
            record.sequence,
            &record.profile,
            &record.metadata,
        );
        if record.hash != expected_hash {
            return false;
        }
        expected_prev = record.hash;
    }
    true
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::{MvccChain, MvccError};
    use crate::Hash;

    #[test]
    fn append_hash_linking_lookup_and_tail_match_the_oracle() {
        let chain = MvccChain::new();
        assert!(chain.is_empty().expect("empty"));
        let first = chain
            .append_at("abbey", 1, 2, "first", 1_000)
            .expect("first");
        let second = chain
            .append_at("aviva", 3, 4, "second", 1_001)
            .expect("second");

        assert_eq!(chain.len().expect("length"), 2);
        assert_eq!(chain.tail_hash().expect("tail"), Some(second));
        assert_eq!(
            chain
                .get(first)
                .expect("lookup")
                .expect("first")
                .record
                .prev_hash,
            Hash::GENESIS
        );
        assert_eq!(
            chain
                .get(second)
                .expect("lookup")
                .expect("second")
                .record
                .prev_hash,
            first
        );
        assert!(chain.verify().expect("verify"));
    }

    #[test]
    fn empty_profiles_are_rejected_without_burning_a_sequence() {
        let chain = MvccChain::new();
        assert_eq!(
            chain.append_at("", 0, 0, "invalid", 1),
            Err(MvccError::EmptyProfile)
        );
        let hash = chain.append_at("abi", 0, 0, "valid", 1).expect("valid");
        assert_eq!(
            chain
                .get(hash)
                .expect("lookup")
                .expect("block")
                .record
                .sequence,
            0
        );
    }

    #[test]
    fn snapshot_remains_frozen_after_later_appends() {
        let chain = MvccChain::new();
        let first = chain.append_at("abbey", 1, 1, "one", 1).expect("one");
        let second = chain.append_at("aviva", 2, 2, "two", 2).expect("two");
        let snapshot = chain.snapshot().expect("snapshot");

        let third = chain.append_at("abi", 3, 3, "three", 3).expect("three");
        assert_eq!(chain.len().expect("live length"), 3);
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot.get(first).is_some());
        assert!(snapshot.get(second).is_some());
        assert!(snapshot.get(third).is_none());
        assert_eq!(
            snapshot
                .iter()
                .map(|block| block.record.profile.as_str())
                .collect::<Vec<_>>(),
            ["abbey", "aviva"]
        );
        assert!(snapshot.verify());
    }

    #[test]
    fn snapshots_are_stable_while_another_thread_appends() {
        let chain = Arc::new(MvccChain::new());
        chain.append_at("abi", 0, 0, "base", 0).expect("base");
        let snapshot = chain.snapshot().expect("snapshot");
        let writer = {
            let chain = Arc::clone(&chain);
            thread::spawn(move || {
                for sequence in 1..=64 {
                    chain
                        .append_at(
                            "abi",
                            sequence,
                            sequence,
                            "later",
                            i64::try_from(sequence).expect("small sequence"),
                        )
                        .expect("append");
                }
            })
        };
        writer.join().expect("writer");

        assert_eq!(snapshot.len(), 1);
        assert_eq!(chain.len().expect("live length"), 65);
        assert!(snapshot.verify());
        assert!(chain.verify().expect("live verify"));
    }

    #[test]
    fn cloned_chain_handles_share_one_sequence() {
        let chain = MvccChain::new();
        let clone = chain.clone();
        let first = chain.append_at("abi", 1, 1, "one", 1).expect("one");
        let second = clone.append_at("abi", 2, 2, "two", 2).expect("two");
        assert_eq!(chain.get(first).expect("first").expect("block").version, 0);
        assert_eq!(
            chain.get(second).expect("second").expect("block").version,
            1
        );
    }
}
