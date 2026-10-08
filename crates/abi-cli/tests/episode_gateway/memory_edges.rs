//! CLI read provenance over the parent test's real scratch gateway.

use super::*;
use abi_wdbx::v3::episode::{
    ActorKind, ActorRef, EdgeReason, EpisodeEvent, EpisodeWrite, MemoryEdge, MemoryEdgeKind,
};

fn digest_bytes(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    std::array::from_fn(|index| u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap())
}

fn append_edge(
    run: &impl Fn(&[&str]) -> Output,
    scratch: &Scratch,
    request: &str,
    kind: MemoryEdgeKind,
    target: &str,
) -> String {
    let mut write: EpisodeWrite = serde_json::from_slice(&memory_candidate_json(request)).unwrap();
    write.operation_id = format!("op_{request}");
    let resolves = kind == MemoryEdgeKind::Resolves;
    write.event = EpisodeEvent::MemoryEdge {
        recorded_by: ActorRef {
            principal_id: if resolves {
                "admin_ref"
            } else {
                "abbey_service"
            }
            .into(),
            kind: if resolves {
                ActorKind::GuildAdministrator
            } else {
                ActorKind::Service
            },
        },
        edge: MemoryEdge {
            kind,
            target: digest_bytes(target),
            counterpart: None,
            reason: if resolves {
                EdgeReason::ReviewedValid
            } else {
                EdgeReason::OperatorReport
            },
        },
    };
    let file = scratch.path(&format!("{request}.json"));
    std::fs::write(&file, serde_json::to_vec(&write).unwrap()).unwrap();
    let output = run(&["wdbx", "episode", "propose", file.to_str().unwrap()]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    field(&text(&output.stdout), "episode_digest").to_owned()
}

fn assert_edge(run: &impl Fn(&[&str]) -> Output, edge: &str, status: &str, resolver: &str) {
    for json in [false, true] {
        let mut args = vec!["wdbx", "episode", "verify", "guild_ref", edge];
        if json {
            args.push("--json");
        }
        let verified = run(&args);
        assert!(verified.status.success(), "{}", text(&verified.stderr));
        if json {
            let value: serde_json::Value = serde_json::from_slice(&verified.stdout).unwrap();
            assert_eq!(value["memory_edge_status"], status);
            assert_eq!(value["memory_edge_resolution"], resolver);
        } else {
            let line = text(&verified.stdout);
            assert_eq!(field(&line, "memory_edge_status"), status);
            assert_eq!(field(&line, "memory_edge_resolution"), resolver);
        }
    }
}

pub(super) fn assert_resolution_output(
    run: &impl Fn(&[&str]) -> Output,
    scratch: &Scratch,
    candidate: &str,
) {
    let edge = append_edge(
        run,
        scratch,
        "cli_flag",
        MemoryEdgeKind::Quarantines,
        candidate,
    );
    assert_edge(run, &edge, "open", "none");
    let resolution = append_edge(
        run,
        scratch,
        "cli_resolver",
        MemoryEdgeKind::Resolves,
        &edge,
    );
    assert_edge(run, &edge, "closed", &resolution);

    for non_edge in [candidate, resolution.as_str()] {
        let output = run(&["wdbx", "episode", "verify", "guild_ref", non_edge, "--json"]);
        assert!(output.status.success(), "{}", text(&output.stderr));
        let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(value.get("memory_edge_resolution").is_none());
    }
    let next_edge = append_edge(
        run,
        scratch,
        "cli_next_flag",
        MemoryEdgeKind::Quarantines,
        candidate,
    );
    assert_ne!(next_edge, edge);
    assert_edge(run, &next_edge, "open", "none");
    assert_edge(run, &edge, "closed", &resolution);
}
