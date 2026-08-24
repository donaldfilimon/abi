//! Semantic invariants for immutable changes and their human approvals.

use serde_json::{Map, Value};
use std::collections::BTreeSet;

pub(super) fn code(schema: &str, map: &Map<String, Value>) -> Option<&'static str> {
    if schema.ends_with("/authorization/change-approval.schema.json") {
        let requested = map.get("requested_by")?.as_object()?;
        let proposed = map.get("proposed_by")?.as_object()?;
        let approved = map.get("approved_by")?.as_object()?;
        let coapproved = map.get("coapproved_by").and_then(Value::as_object);
        let requested_id = requested.get("principal_id")?.as_str()?;
        let proposed_id = proposed.get("principal_id")?.as_str()?;
        let approved_id = approved.get("principal_id")?.as_str()?;
        let identities = BTreeSet::from([requested_id, proposed_id, approved_id]);
        if identities.len() != 3
            || proposed.get("kind").and_then(Value::as_str) != Some("service")
            || approved.get("kind").and_then(Value::as_str) == Some("service")
        {
            return Some("self_approval");
        }
        if let Some(coapprover) = coapproved {
            let id = coapprover.get("principal_id")?.as_str()?;
            if coapprover.get("kind").and_then(Value::as_str) == Some("service")
                || identities.contains(id)
            {
                return Some("self_approval");
            }
        }
        let level = map.get("approval_level")?.as_str()?;
        if (level == "A5DualControl") != coapproved.is_some() {
            return Some("approval_insufficient");
        }
        let kind_is_sufficient = |kind: Option<&str>| match level {
            "A0None" | "A1Actor" => matches!(
                kind,
                Some(
                    "human_subject"
                        | "organization_owner"
                        | "guild_owner"
                        | "guild_administrator"
                        | "guild_manager"
                )
            ),
            "A2Manager" => matches!(
                kind,
                Some(
                    "organization_owner" | "guild_owner" | "guild_administrator" | "guild_manager"
                )
            ),
            "A3Admin" => matches!(
                kind,
                Some("organization_owner" | "guild_owner" | "guild_administrator")
            ),
            "A4Owner" => matches!(kind, Some("organization_owner" | "guild_owner")),
            "A5DualControl" => matches!(
                kind,
                Some("organization_owner" | "guild_owner" | "guild_administrator")
            ),
            _ => false,
        };
        if !kind_is_sufficient(approved.get("kind").and_then(Value::as_str))
            || coapproved.is_some_and(|coapprover| {
                !kind_is_sufficient(coapprover.get("kind").and_then(Value::as_str))
            })
        {
            return Some("approval_insufficient");
        }
    }
    if schema.ends_with("/capability/change-set.schema.json") {
        let requested = map.get("requested_by")?.as_object()?;
        let proposed = map.get("proposed_by")?.as_object()?;
        if proposed.get("kind").and_then(Value::as_str) != Some("service")
            || proposed.get("principal_id") == requested.get("principal_id")
        {
            return Some("proposal_authority_invalid");
        }
        let created = map.get("created_at_ms")?.as_u64()?;
        let expires = map.get("expires_at_ms")?.as_u64()?;
        if expires <= created || expires > created.saturating_add(300_000) {
            return Some("proposal_expiry_invalid");
        }
        if map.get("compensation_class").and_then(Value::as_str) == Some("ExactRestore")
            && map
                .get("rollback_digest")
                .and_then(Value::as_str)
                .and_then(|digest| digest.strip_prefix("sha256:"))
                .is_some_and(|hex| hex.bytes().all(|byte| byte == b'0'))
        {
            return Some("rollback_required");
        }
    }
    None
}
