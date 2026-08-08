//! Deterministic rendezvous shard placement.

use super::{NodeDescriptor, NodeState};
use sha2::{Digest as _, Sha256};
use uuid::Uuid;

/// Default replica count, capped by the number of active nodes.
pub const DEFAULT_REPLICATION_FACTOR: usize = 3;

/// Select replica UUIDs by descending rendezvous score and ascending UUID tie.
///
/// The membership generation is deliberately absent: only the shard key and
/// stable node identity influence placement, which minimizes movement when one
/// node joins or leaves.
#[must_use]
pub fn rendezvous_replicas(
    shard_key: &[u8],
    nodes: &[NodeDescriptor],
    replication_factor: usize,
) -> Vec<Uuid> {
    let mut scored: Vec<_> = nodes
        .iter()
        .filter(|node| node.state == NodeState::Active)
        .map(|node| {
            let mut hasher = Sha256::new();
            hasher.update(shard_key);
            hasher.update(node.id.as_bytes());
            let score: [u8; 32] = hasher.finalize().into();
            (score, node.id)
        })
        .collect();
    scored.sort_by(|(left_score, left_id), (right_score, right_id)| {
        right_score
            .cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    scored
        .into_iter()
        .take(replication_factor.min(nodes.len()))
        .map(|(_, id)| id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(count: usize) -> Vec<NodeDescriptor> {
        (1..=count)
            .map(|value| {
                NodeDescriptor::active(
                    Uuid::from_u128(u128::try_from(value).unwrap()),
                    format!("127.0.0.1:{}", 4100 + value),
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn placement_is_deterministic_distinct_and_capped() {
        let nodes = nodes(5);
        let first = rendezvous_replicas(b"project:key", &nodes, DEFAULT_REPLICATION_FACTOR);
        let second = rendezvous_replicas(b"project:key", &nodes, DEFAULT_REPLICATION_FACTOR);
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        let mut unique = first.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 3);
        assert_eq!(rendezvous_replicas(b"key", &nodes[..2], 3).len(), 2);
        assert_eq!(rendezvous_replicas(b"key", &nodes, 0), Vec::<Uuid>::new());
    }

    #[test]
    fn one_join_changes_only_keys_won_by_the_new_node() {
        let before = nodes(4);
        let after = nodes(5);
        let joined = after[4].id;
        for number in 0_u16..1_000 {
            let key = number.to_le_bytes();
            let old = rendezvous_replicas(&key, &before, 3);
            let new = rendezvous_replicas(&key, &after, 3);
            if new.contains(&joined) {
                assert_eq!(old.iter().filter(|id| new.contains(id)).count(), 2);
            } else {
                assert_eq!(old, new);
            }
        }
    }

    #[test]
    fn draining_and_removed_nodes_are_never_selected() {
        let mut nodes = nodes(4);
        let excluded = nodes[0].id;
        nodes[0].state = NodeState::Draining;
        for number in 0_u8..64 {
            assert!(!rendezvous_replicas(&[number], &nodes, 4).contains(&excluded));
        }
        nodes[0].state = NodeState::Removed;
        assert!(!rendezvous_replicas(b"key", &nodes, 4).contains(&excluded));
    }
}
